import torch
from collections import defaultdict


def get_default_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))


def filter_requires_grad(param_list):
    return list(filter(lambda p: p.requires_grad, param_list))


def get_FlexPose_dict(model):
    dic_param = defaultdict(list)
    for name, param in model.named_parameters():
        if name.startswith('p_encoder.') or name.startswith('l_feat_encoder.'):
            dic_param['encoder'].append(param)
        elif name.startswith('c_decoder.'):
            dic_param['decoder'].append(param)
        elif name.startswith('l_x_sca_embed.') \
                or name.startswith('l_x_vec_embed.') \
                or name.startswith('l_edge_sca_embed.') \
                or name.startswith('l_edge_vec_embed.') \
                or name.startswith('p_x_sca_embed.') \
                or name.startswith('p_x_vec_embed.') \
                or name.startswith('p_edge_sca_embed.') \
                or name.startswith('p_edge_vec_embed.') \
                or name.startswith('x_gate.') \
                or name.startswith('edge_gate.'):
            dic_param['extra_embed'].append(param)
        elif name.startswith('pred_'):
            dic_param['aux_pred'].append(param)
        else:
            raise KeyError(f'Unknown param, {name}')

    dic_param = {k: filter_requires_grad(v) for k, v in dic_param.items()}
    return dic_param


def get_FlexPose_params(model, loss_object, args):
    param = model.module if args.use_multi_gpu else model
    dic_param = get_FlexPose_dict(param)

    param_1 = [
        {'params': dic_param['decoder'] + dic_param['extra_embed'] + dic_param['aux_pred'],
         'lr': args.lr, 'betas': (0.9, 0.999)},
    ]

    # optimizer & scheduler
    optimizer_1 = torch.optim.Adam(param_1, lr=args.lr, weight_decay=args.weight_decay)
    lr_verbose = True if args.rank == 0 or not args.use_multi_gpu else False
    scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_1, args.lr_decay, last_epoch=-1, verbose=lr_verbose)
    return {'optimizer': [optimizer_1],
            'scheduler': [scheduler_1],
            }


def get_all_dict(model):
    dic_param = defaultdict(list)
    for name, param in model.named_parameters():
        dic_param['all'].append(param)

    dic_param = {k: filter_requires_grad(v) for k, v in dic_param.items()}
    return dic_param


def get_pocket_params(model, loss_object, args):
    param = model.module if args.use_multi_gpu else model
    dic_param = get_all_dict(param)

    param_1 = [
        {'params': dic_param['all'], 'lr': args.lr, 'betas': (0.9, 0.999)},
    ]

    # optimizer & scheduler
    optimizer_1 = torch.optim.Adam(param_1, lr=args.lr, weight_decay=args.weight_decay)
    lr_verbose = True if args.rank == 0 or not args.use_multi_gpu else False
    scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_1, args.lr_decay, last_epoch=-1, verbose=lr_verbose)
    return {'optimizer': [optimizer_1],
            'scheduler': [scheduler_1],
            }


def get_ligand_params(model, loss_object, args):
    param = model.module if args.use_multi_gpu else model
    dic_param = get_all_dict(param)

    param_1 = [
        {'params': dic_param['all'], 'lr': args.lr, 'betas': (0.9, 0.999)},
    ]

    # optimizer & scheduler
    optimizer_1 = torch.optim.Adam(param_1, lr=args.lr, weight_decay=args.weight_decay)
    lr_verbose = True if args.rank == 0 or not args.use_multi_gpu else False
    scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_1, args.lr_decay, last_epoch=-1, verbose=lr_verbose)
    return {'optimizer': [optimizer_1],
            'scheduler': [scheduler_1],
            }

