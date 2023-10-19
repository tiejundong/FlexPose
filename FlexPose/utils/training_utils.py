import os
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))

import numpy as np
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from common import save_idx_list, load_idx_list, delmkdir, is_tensor



def train_with_args(args, train):
    if args.use_multi_gpu:
        # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
        parallel_args = (args.world_size, args.port, args)
        torch.multiprocessing.spawn(train, args=parallel_args, nprocs=args.world_size, join=True)
    else:
        train(int(args.gpu_list), None, None, args)


def set_gpu_device(rank, world_size, port, args):
    if args.use_multi_gpu:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port
        torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)  # nccl gloo


def check_data_split(path='./'):
    return os.path.exists(f'{path}/train_list.txt') \
           and os.path.exists(f'{path}/val_list.txt') \
           and os.path.exists(f'{path}/test_list.txt')


def save_data_split(train_list, val_list, test_list, path='./'):
    delmkdir(path, remove_old=False)
    save_idx_list(train_list, f'{path}/train_list.txt')
    save_idx_list(val_list, f'{path}/val_list.txt')
    save_idx_list(test_list, f'{path}/test_list.txt')


def load_data_split(path='./', blind_training=False):
    train_list = load_idx_list(f'{path}/train_list.txt')
    val_list = load_idx_list(f'{path}/val_list.txt')
    test_list = load_idx_list(f'{path}/test_list.txt')

    if blind_training:
        train_list += val_list

    return train_list, val_list, test_list


def get_dataloader(args, train_dataset, val_dataset, world_size, collate_fn):
    if args.use_multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size,
                                                                        rank=args.rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                                   sampler=train_sampler, num_workers=args.num_workers,
                                                   pin_memory=False, persistent_workers=args.persistent_workers,
                                                   prefetch_factor=2, collate_fn=collate_fn)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size,
                                                                      rank=args.rank, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                                 sampler=val_sampler, num_workers=2, persistent_workers=args.persistent_workers,
                                                 collate_fn=collate_fn)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True,
                                                   pin_memory=False, persistent_workers=args.persistent_workers,
                                                   prefetch_factor=2, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                                 num_workers=2, shuffle=True, persistent_workers=args.persistent_workers,
                                                 collate_fn=collate_fn)
        train_sampler = val_sampler = None
    return train_loader, train_sampler, val_loader, val_sampler


class CustomOptimization():
    def __init__(self, args, dic_opt):
        '''
        WARNing: if len(list)>1, be aware of correct detach tensors
        '''
        self.use_amp_mix = args.use_amp_mix
        if args.use_amp_mix:  # might occur problem in some design
            self.scaler = torch.cuda.amp.GradScaler()

        self.optimizers = dic_opt['optimizer']
        self.schedulers = dic_opt['scheduler']
        self.lr_sche_batch_wise = [int(i) for i in args.lr_sche_batch_wise.split(',')] \
            if not isinstance(args.lr_sche_batch_wise, type(None)) and len(args.lr_sche_batch_wise) > 0 else []

        self.n_opt_step = len(self.optimizers)
        if args.rank == 0 or not args.use_multi_gpu:
            print(f'Optimizer groups: {self.n_opt_step}')

    def __call__(self, grad_loss, update_together=True, epoch=-1):
        '''
        opt model params
        :param grad_loss: grad_loss can be a tuple/list of loss, for single loss -> (loss_1, ) / [loss_1]
        :param update_together: If True, grad update after calc all grad. If False, update grad for each optimizer
        :return: None
        '''
        if is_tensor(grad_loss):
            grad_loss = [grad_loss]

        assert isinstance(grad_loss, tuple) or isinstance(grad_loss, list)
        assert len(grad_loss) == self.n_opt_step

        if update_together:
            for optimizer in self.optimizers:
                optimizer.zero_grad()

        if self.use_amp_mix:
            for opt_step_i in range(self.n_opt_step):
                retain_graph = True if opt_step_i < self.n_opt_step - 1 else False
                if not update_together:
                    self.optimizers[opt_step_i].zero_grad()

                self.scaler.scale(grad_loss[opt_step_i]).backward(retain_graph=retain_graph)

                if not update_together:
                    self.scaler.step(self.optimizers[opt_step_i])

            if update_together:
                for optimizer in self.optimizers:
                    self.scaler.step(optimizer)
            self.scaler.update()

        else:
            for opt_step_i in range(self.n_opt_step):
                retain_graph = True if opt_step_i < self.n_opt_step - 1 else False
                if not update_together:
                    self.optimizers[opt_step_i].zero_grad()

                grad_loss[opt_step_i].backward(retain_graph=retain_graph)

                if not update_together:
                    self.optimizers[opt_step_i].step()

            if update_together:
                for optimizer in self.optimizers:
                    optimizer.step()

        for opt_step_i in range(self.n_opt_step):
            if opt_step_i in self.lr_sche_batch_wise:
                self.schedulers[opt_step_i].step()

    def update_schedulers(self, epoch=-1):
        for opt_step_i in range(self.n_opt_step):
            if opt_step_i not in self.lr_sche_batch_wise:
                self.schedulers[opt_step_i].step()

    def save_state_dict(self):
        dic_save = {
            **{f'opt_{i}': o.state_dict() for i, o in enumerate(self.optimizers)},
            **{f'sche_{i}': s.state_dict() for i, s in enumerate(self.schedulers)},
        }
        if self.use_amp_mix:
            dic_save['amp'] = self.scaler.state_dict()
        return dic_save

    def load_state_dict(self, dic_save):
        for i in range(len(self.optimizers)):
            if i not in self.lr_sche_batch_wise:
                self.optimizers[i].load_state_dict(dic_save[f'opt_{i}'])
                self.schedulers[i].load_state_dict(dic_save[f'sche_{i}'])
        if self.use_amp_mix:
            self.scaler.load_state_dict(dic_save['amp'])


def weights_init(model, t=None, a=0, b=1, gain=1):
    if isinstance(t, type(None)):
        return
    assert t in ['random', 'xavier_normal_']
    for p in model.parameters():
        if t == 'random':
            torch.nn.init.uniform_(p, a=a, b=b)
        elif t == 'xavier_normal_':
            if len(p.shape) > 1:
                torch.nn.init.xavier_normal_(p, gain=gain)
        else:
            torch.nn.init.zeros_(p)


def save_params(args, my_model, loss_object, opt_object, dic_traj, save_path):
    torch.save(
        {'model_state_dict': my_model.module.state_dict() if args.use_multi_gpu else my_model.state_dict(),
         'loss_state_dict': loss_object.module.state_dict() if args.use_multi_gpu_for_loss_object else loss_object.state_dict(),
         'dic_traj': dic_traj,
         'args': args,
         **opt_object.save_state_dict(),
         },
        save_path)


def init_params(args, my_model, loss_object, opt_object):
    if args.restart == 0:
        if args.choose_start_weight != None:
            if args.rank == 0 or not args.use_multi_gpu:
                print(f'Loading params: from {args.choose_start_weight}')
            chk = torch.load(args.choose_start_weight, map_location=f'cuda:{args.rank}')

            strict = True
            if args.use_multi_gpu:
                my_model.module.load_state_dict(chk['model_state_dict'], strict=strict)
            else:
                my_model.load_state_dict(chk['model_state_dict'], strict=strict)
            if 'loss_state_dict' in chk.keys():
                if args.use_multi_gpu_for_loss_object:
                    loss_object.module.load_state_dict(chk['loss_state_dict'], strict=strict)
                else:
                    loss_object.load_state_dict(chk['loss_state_dict'], strict=strict)
            del chk
        else:
            weights_init(my_model)

        dic_traj = {'train': defaultdict(list), 'val': defaultdict(list)}
        if args.rank == 0 or not args.use_multi_gpu:
            delmkdir(args.weight_path)
            delmkdir(args.vis_path)

    else:
        if args.rank == 0 or not args.use_multi_gpu:
            print(f'Loading params from {args.weight_path}/state_{args.restart}.chk')
        chk = torch.load(f'{args.weight_path}/state_{args.restart}.chk', map_location=f'cuda:{args.rank}')

        if args.use_multi_gpu:
            my_model.module.load_state_dict(chk['model_state_dict'], strict=True)
        else:
            my_model.load_state_dict(chk['model_state_dict'], strict=True)
        if args.use_multi_gpu_for_loss_object:
            loss_object.module.load_state_dict(chk['loss_state_dict'], strict=True)
        else:
            loss_object.load_state_dict(chk['loss_state_dict'], strict=True)

        opt_object.load_state_dict(chk)

        dic_traj = chk['dic_traj']

        if args.rank == 0 or not args.use_multi_gpu:
            print('Restart at [{}/{}], current lr: {:.2e}'.format(args.restart, args.n_epoch, opt_object.optimizers[0].param_groups[0]['lr']))
        del chk
    return my_model, loss_object, opt_object, dic_traj


def init_tensorboard(log_dir, log_port, restart, dic_traj, start_new_tensorboard=True, drop_head_epoch=0, sele_env=None):
    '''
    # e.g. tensorboard --port 6007 --logdir /root/tf-logs/
    # dic_traj: {'train': defaultdict(list), 'val': defaultdict(list)}
    '''
    delmkdir(log_dir)
    if start_new_tensorboard:
        print(f'Kill old tensorboards and tried to start a new one on (port:{log_port}, logdir:{log_dir})')
        os.system('ps -ef | grep tensorboard | grep -v grep | awk \'{print "kill -9 "$2}\' | sh')
        cmd = [
            '#!/bin/bash',
            # if need select env
            f'conda activate {sele_env}' if not isinstance(sele_env, type(None)) else 'echo Tensorboard: Using current env',
            f'nohup tensorboard --port {log_port} --logdir {log_dir} > /dev/null 2>&1 &'
        ]
        sh_path = f'{log_dir}/start.sh'
        open(sh_path, 'w').write('\n'.join(cmd))
        os.system(f'nohup bash -i {sh_path} > /dev/null 2>&1 &')
    else:
        print(f'Tensorboard CMD: tensorboard --port {log_port} --logdir {log_dir}')
    writer = SummaryWriter(log_dir=log_dir)
    if restart != 0:
        for epoch in range(restart):
            update_tensorboard(writer, dic_traj, epoch, drop_head_epoch=drop_head_epoch)
    return writer


def update_tensorboard(writer, dic_traj, epoch, drop_head_epoch=0, add_single=False):
    if epoch < drop_head_epoch:
        return None

    if add_single:
        for k in dic_traj.keys():
            if k.endswith('_loss'):
                folder = 'Loss'
            elif k.endswith('_metric'):
                folder = 'Metric'
            else:
                folder = 'Other'

            writer.add_scalar(f'{folder}/{k}', dic_traj[k], global_step=epoch)

    else:
        for k in dic_traj['train'].keys():
            if k.endswith('_loss'):
                folder = 'Loss'
            elif k.endswith('_metric'):
                folder = 'Metric'
            else:
                folder = 'Other'
            writer.add_scalars(f'{folder}/{k}',
                               {'Train': np.array(dic_traj['train'][k][epoch]),
                                'Val': np.array(dic_traj['val'][k][epoch])},
                               global_step=epoch)


class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)






















