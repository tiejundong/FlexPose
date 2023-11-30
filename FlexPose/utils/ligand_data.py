import os

import numpy as np
import random
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import torch_geometric



def split_dataset(data_path, data_split_rate):
    idx_list = os.listdir(data_path)
    random.shuffle(idx_list)

    l = len(idx_list)
    cut_1 = int(data_split_rate[0] * l)
    cut_2 = cut_1 + int(data_split_rate[1] * l)
    train_list = idx_list[:cut_1]
    val_list = idx_list[cut_1:cut_2]
    test_list = idx_list[cut_2:]

    return train_list, val_list, test_list


def collate_fn(batch_list):
    max_len_ligand = 0
    for g in batch_list:
        max_len_ligand = max(max_len_ligand, g.len_ligand)
        g.l_x_mask = torch.ones((g.len_ligand))  # P.S. l_x_mask_bool for pretrain, l_x_mask for layer mask
        g.l_edge_mask = torch.ones((g.len_ligand, g.len_ligand))

    # max_len_ligand = 150

    # node
    l_x_sca_init = torch.stack([F.pad(g.l_x_sca_init,
                                      (0, 0, 0, max_len_ligand - g.l_x_sca_init.shape[0]),
                                      'constant', 0)
                                for g in batch_list], dim=0)

    # edge
    l_edge_sca_init = torch.stack([F.pad(g.l_edge_sca_init,
                                      (0, 0,
                                       0, max_len_ligand - g.l_edge_sca_init.shape[0],
                                       0, max_len_ligand - g.l_edge_sca_init.shape[0]),
                                      'constant', 0)
                                for g in batch_list], dim=0)

    # layer mask (not for pretrain)
    l_x_mask = torch.stack([F.pad(g.l_x_mask,
                                  (0, max_len_ligand - g.l_x_mask.shape[0]), 'constant', 0)
                            for g in batch_list], dim=0)
    l_edge_mask = torch.stack([F.pad(g.l_edge_mask,
                                  (0, max_len_ligand - g.l_edge_mask.shape[0],
                                   0, max_len_ligand - g.l_edge_mask.shape[0]), 'constant', 0)
                            for g in batch_list], dim=0)


    #####################################################################################################
    # for feat, coor and mask
    #####################################################################################################
    l_x_batch_info = repeat(torch.arange(len(batch_list)), 'b -> b n', n=max_len_ligand)
    l_edge_batch_info = repeat(torch.arange(len(batch_list)), 'b -> b i j', i=max_len_ligand, j=max_len_ligand)

    l_x_mask_bool = torch.stack([F.pad(g.l_x_mask_bool,
                                       (0, max_len_ligand - len(g.l_x_mask_bool)), 'constant', False)
                                 for g in batch_list], dim=0)
    l_x_mask_label = torch.stack([F.pad(g.l_x_mask_label,
                                        (0, 0, 0, max_len_ligand - len(g.l_x_mask_label)), 'constant', 0)
                                  for g in batch_list], dim=0)
    l_x_mask_label = rearrange(l_x_mask_label, 'b n d -> (b n) d')[l_x_mask_bool.reshape(-1)].squeeze(-1)
    l_x_mask_info = l_x_batch_info.reshape(-1)[l_x_mask_bool.reshape(-1)]

    l_edge_mask_bool = torch.stack(
        [F.pad(g.l_edge_mask_bool,
               (0, max_len_ligand - len(g.l_edge_mask_bool),
                0, max_len_ligand - len(g.l_edge_mask_bool)),
               'constant', False)
         for g in batch_list], dim=0)
    l_edge_mask_label = torch.stack(
        [F.pad(g.l_edge_mask_label,
               (0, 0,
                0, max_len_ligand - len(g.l_edge_mask_label),
                0, max_len_ligand - len(g.l_edge_mask_label)),
               'constant', 0)
         for g in batch_list], dim=0)
    l_edge_mask_label = rearrange(l_edge_mask_label, 'b i j d -> (b i j) d')[l_edge_mask_bool.reshape(-1)].squeeze(-1)
    l_edge_mask_info = l_edge_batch_info.reshape(-1)[l_edge_mask_bool.reshape(-1)]


    #####################################################################################################
    # for graph-level task
    #####################################################################################################
    edge_shift_index = []
    for g in batch_list:
        rand = random.sample(range(g.len_ligand), 1)[0]
        edge_shift_index.append(rand)
    edge_shift_index = torch.Tensor(edge_shift_index).reshape(-1, 1).long()

    #####################################################################################################
    # for coor task
    #####################################################################################################
    ligand_dismap_true = torch.stack(
        [F.pad(g.ligand_dismap_true,
               (0, max_len_ligand - len(g.ligand_dismap_true),
                0, max_len_ligand - len(g.ligand_dismap_true)),
               'constant', 0)
         for g in batch_list], dim=0)
    ligand_dismap_mask = torch.stack(
        [F.pad(g.ligand_dismap_mask,
               (0, max_len_ligand - len(g.ligand_dismap_mask),
                0, max_len_ligand - len(g.ligand_dismap_mask)),
               'constant', 0)
         for g in batch_list], dim=0)
    l_noise_dismap = torch.stack(
        [F.pad(g.l_noise_dismap,
               (0, max_len_ligand - len(g.l_noise_dismap),
                0, max_len_ligand - len(g.l_noise_dismap)),
               'constant', -1/g.coor_scale)
         for g in batch_list], dim=0)

    l_x_vec_init = torch.stack(
        [F.pad(g.l_x_vec_init,
               (0, 0,
                0, 0,
                0, max_len_ligand - len(g.l_x_vec_init)),
               'constant', 0)
         for g in batch_list], dim=0)
    l_edge_vec_init = torch.stack(
        [F.pad(g.l_edge_vec_init,
               (0, 0,
                0, 0,
                0, max_len_ligand - len(g.l_edge_vec_init),
                0, max_len_ligand - len(g.l_edge_vec_init)),
               'constant', 0)
         for g in batch_list], dim=0)
    l_coor_init = torch.stack(
        [F.pad(g.l_coor_init,
               (0, 0,
                0, max_len_ligand - len(g.l_coor_init)),
               'constant', 0)
         for g in batch_list], dim=0)
    l_coor_true = torch.stack(
        [F.pad(g.l_coor_true,
               (0, 0,
                0, max_len_ligand - len(g.l_coor_true)),
               'constant', 0)
         for g in batch_list], dim=0)
    l_x_coor_mask_bool = torch.stack(
        [F.pad(g.l_x_coor_mask_bool,
               (0, max_len_ligand - len(g.l_x_coor_mask_bool)),
               'constant', False)
         for g in batch_list], dim=0).bool().reshape(-1)

    l_coor_init_selected = rearrange(l_coor_init, 'b i c -> (b i) c')[l_x_coor_mask_bool]
    l_coor_true_selected = rearrange(l_coor_true, 'b i c -> (b i) c')[l_x_coor_mask_bool]

    l_x_batch_info = repeat(torch.arange(len(batch_list)), 'b -> b n', n=max_len_ligand)
    l_x_coor_mask_info = l_x_batch_info.reshape(-1)[l_x_coor_mask_bool]

    # supply
    len_ligand = torch.Tensor([g.len_ligand for g in batch_list])

    idx = [g.idx for g in batch_list]

    # use Data in PyG
    complex_graph = torch_geometric.data.Data(
        # node
        l_x_sca_init=l_x_sca_init,

        # edge
        l_edge_sca_init=l_edge_sca_init,

        # supply info
        l_x_mask=l_x_mask.to(torch.bool),
        l_edge_mask=l_edge_mask.to(torch.bool),

        max_len_ligand=max_len_ligand,
        len_ligand=len_ligand,

        idx=idx,

        # for mask
        l_x_mask_bool=l_x_mask_bool.to(torch.bool),
        l_x_mask_label=l_x_mask_label,
        l_x_mask_info=l_x_mask_info,
        l_edge_mask_bool=l_edge_mask_bool.to(torch.bool),
        l_edge_mask_label=l_edge_mask_label,
        l_edge_mask_info=l_edge_mask_info,

        edge_shift_index=edge_shift_index,

        # for coor
        ligand_dismap_true=ligand_dismap_true,
        ligand_dismap_mask=ligand_dismap_mask,
        l_noise_dismap=l_noise_dismap,

        l_x_vec_init=l_x_vec_init,
        l_edge_vec_init=l_edge_vec_init,
        l_coor_init=l_coor_init,
        l_coor_init_selected=l_coor_init_selected,
        l_coor_true_selected=l_coor_true_selected,
        l_x_coor_mask_bool=l_x_coor_mask_bool,
        l_x_coor_mask_info=l_x_coor_mask_info,
    )
    return complex_graph


class LigandDataset(torch.utils.data.Dataset):
    def __init__(self, mode, args, data_list, cache_path='./cache'):
        self.mode = mode
        self.data_list = data_list
        self.data_path = args.data_path

        self.data_path = args.data_path
        self.max_len_ligand = args.max_len_ligand
        self.coor_scale = args.coor_scale
        self.noise_dist = args.noise_dist / self.coor_scale

        if self.mode == 'train':
            self.n_batch = int(args.n_batch)
        elif self.mode == 'val':
            self.n_batch = 100  # int(args.n_batch // 100)

        # for mask
        self.mask_rate = args.mask_rate

        self.set_data(0)

    def set_data(self, epoch):
        self.epoch = epoch
        start = (epoch * self.n_batch) % len(self.data_list)

        if start + self.n_batch > len(self.data_list) - 1:
            pretrain_list = self.data_list + self.data_list
        else:
            pretrain_list = self.data_list
        self.sub_pretrain_list = pretrain_list[start:start + self.n_batch]

    def __getitem__(self, i):
        idx = self.sub_pretrain_list[i]
        complex_graph = self.get_complex(idx)
        return complex_graph

    def get_complex(self, idx):
        ##################################################################
        # get dict data
        ##################################################################
        dic_data = np.load(f'{self.data_path}/{idx}', allow_pickle=False)

        l_x_sca_init = dic_data['ligand_node_features']
        l_edge_sca_init = dic_data['ligand_edge_features']
        ligand_dismap = dic_data['ligand_dismap']
        ligand_coor = dic_data['ligand_coor']

        # get length
        len_ligand = len(l_x_sca_init)

        ##################################################################
        # to tensor
        ##################################################################
        l_x_sca_init = torch.from_numpy(l_x_sca_init).float()
        l_edge_sca_init = torch.from_numpy(l_edge_sca_init).float()
        ligand_dismap = torch.from_numpy(ligand_dismap).float()
        ligand_coor = torch.from_numpy(ligand_coor).float()


        ##################################################################
        # scale
        ##################################################################
        ligand_dismap = ligand_dismap / self.coor_scale
        ligand_coor = ligand_coor / self.coor_scale


        ##################################################################
        # for feat mask
        ##################################################################
        # ligand node feature mask
        l_atom_label = l_x_sca_init[:, :10]
        l_x_mask_bool = self.gen_mask_index([l_atom_label], self.mask_rate)
        l_x_mask_label = l_atom_label.argmax(dim=-1, keepdim=True)

        # ligand edge feature mask
        l_edge_label = l_edge_sca_init
        l_edge_mask_bool = self.gen_mask_index([l_edge_label], self.mask_rate)
        l_edge_mask_bool = (torch.triu(l_edge_mask_bool.float()) +
                            torch.triu(l_edge_mask_bool.float()).transpose(1, 0)).to(torch.bool)
        l_edge_mask_label = l_edge_label.argmax(dim=-1, keepdim=True)

        # mask origin feat
        l_x_sca_init = F.pad(l_x_sca_init, (0, 1), 'constant', 0)
        l_x_sca_init[l_x_mask_bool] = 0
        l_x_sca_init[l_x_mask_bool, -1] = 1

        l_edge_sca_init = F.pad(l_edge_sca_init, (0, 1), 'constant', 0)
        l_edge_sca_init = rearrange(l_edge_sca_init, 'i j d -> (i j) d')
        l_edge_sca_init[l_edge_mask_bool.reshape(-1)] = 0
        l_edge_sca_init[l_edge_mask_bool.reshape(-1), -1] = 1
        l_edge_sca_init = rearrange(l_edge_sca_init, '(i j) d -> i j d', i=len_ligand)


        ##################################################################
        # for coor pretrain
        ##################################################################
        ligand_dismap_true = ligand_dismap
        ligand_dismap_mask = (ligand_dismap != -1 / self.coor_scale).float()

        l_x_coor_mask_bool = torch.zeros(len_ligand).index_fill_(-1, torch.randperm(len_ligand)[:max(1, int(self.mask_rate*len_ligand))], 1).bool()
        l_coor_true = ligand_coor
        l_coor_init = ligand_coor
        l_coor_init = l_coor_init + torch.randn(l_coor_init.shape) * self.noise_dist * l_x_coor_mask_bool.float().unsqueeze(dim=-1)
        l_noise_dismap = (l_coor_init.unsqueeze(dim=-2) - l_coor_init.unsqueeze(dim=-3)).norm(p=2, dim=-1)


        ##################################################################
        # init ligand vec
        ##################################################################
        l_x_vec_init = (l_coor_init - l_coor_init.mean(dim=0, keepdims=True)).unsqueeze(dim=-2)
        l_edge_vec_init = (l_coor_init.unsqueeze(dim=-2) - l_coor_init.unsqueeze(dim=-3)).unsqueeze(dim=-2)


        ##################################################################
        # use Data in PyG
        ##################################################################
        complex_graph = torch_geometric.data.Data(
            x=None,
            # node
            l_x_sca_init=l_x_sca_init,

            # edge
            l_edge_sca_init=l_edge_sca_init,

            # supply info
            len_ligand=len_ligand,

            idx=idx,

            # for mask
            l_x_mask_bool=l_x_mask_bool,
            l_x_mask_label=l_x_mask_label,
            l_edge_mask_bool=l_edge_mask_bool,
            l_edge_mask_label=l_edge_mask_label,

            # for coor
            ligand_dismap_true=ligand_dismap_true,
            ligand_dismap_mask=ligand_dismap_mask,
            l_noise_dismap=l_noise_dismap,
            coor_scale=self.coor_scale,

            l_x_vec_init=l_x_vec_init,
            l_edge_vec_init=l_edge_vec_init,
            l_coor_init=l_coor_init,
            l_coor_true=l_coor_true,
            l_x_coor_mask_bool=l_x_coor_mask_bool,
        )
        return complex_graph

    def __len__(self):
        return self.n_batch

    def gen_mask_index(self, feat_label_list, mask_rate=0.15):
        allow_mask_pos = torch.cat([feat_label.sum(dim=-1, keepdim=True) == 1 for feat_label in feat_label_list],
                                   dim=-1).prod(dim=-1)
        origin_shape = allow_mask_pos.shape
        origin_shape_flat = torch.tensor(origin_shape).prod()
        n_mask = max(int(mask_rate * origin_shape_flat), 1)
        mask_index = torch.randperm(origin_shape_flat)[:n_mask]
        mask = torch.zeros(origin_shape_flat).index_fill_(-1, mask_index, 1)
        mask = mask.reshape(origin_shape).to(torch.bool)
        return mask



if __name__ == '__main__':
    # for testing
    pass
