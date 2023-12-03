import os
import random
import numpy as np

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
    max_len_pocket = 0
    for g in batch_list:
        max_len_pocket = max(max_len_pocket, g.len_pocket)
        g.p_x_mask = torch.ones((g.len_pocket))  # P.S. p_x_mask_bool for pretrain, p_x_mask for layer mask
        g.p_edge_mask = torch.ones((g.len_pocket, g.len_pocket))

    max_len_pocket = 170

    #####################################################################################################
    # for feat, coor and mask
    #####################################################################################################
    # node
    p_x_sca_init = torch.stack([F.pad(g.p_x_sca_init,
                                      (0, 0, 0, max_len_pocket - g.p_x_sca_init.shape[0]),
                                      'constant', 0)
                                for g in batch_list], dim=0)
    p_x_vec_init = torch.stack([F.pad(g.p_x_vec_init,
                                      (0, 0, 0, 0, 0, max_len_pocket - g.p_x_vec_init.shape[0]),
                                      'constant', 0)
                                for g in batch_list], dim=0)

    # edge
    p_edge_sca_init = torch.stack([F.pad(g.p_edge_sca_init,
                                         (0, 0,
                                          0, max_len_pocket - g.p_edge_sca_init.shape[0],
                                          0, max_len_pocket - g.p_edge_sca_init.shape[0]),
                                         'constant', 0)
                                   for g in batch_list], dim=0)
    p_edge_vec_init = torch.stack([F.pad(g.p_edge_vec_init,
                                         (0, 0, 0, 0,
                                          0, max_len_pocket - g.p_edge_vec_init.shape[0],
                                          0, max_len_pocket - g.p_edge_vec_init.shape[0]),
                                         'constant', 0)
                                   for g in batch_list], dim=0)

    # coor
    p_coor_init = torch.stack([F.pad(g.p_coor_init,
                                     (0, 0, 0, max_len_pocket - g.p_coor_init.shape[0]), 'constant', 0)
                               for g in batch_list], dim=0)

    # layer mask (not for pretrain)
    p_x_mask = torch.stack([F.pad(g.p_x_mask,
                                  (0, max_len_pocket - g.p_x_mask.shape[0]), 'constant', 0)
                            for g in batch_list], dim=0).to(torch.bool)
    p_edge_mask = torch.stack([F.pad(g.p_edge_mask,
                                     (0, max_len_pocket - g.p_edge_mask.shape[0],
                                      0, max_len_pocket - g.p_edge_mask.shape[0]), 'constant', 0)
                               for g in batch_list], dim=0).to(torch.bool)

    # for pretrain
    p_x_batch_info = repeat(torch.arange(len(batch_list)), 'b -> (b n)', n=max_len_pocket)
    p_coor_init_flat = rearrange(p_coor_init, 'b n c -> (b n) c')

    p_x_mask_AAtype_bool = torch.cat([F.pad(g.p_x_mask_AAtype_bool,
                                            (0, max_len_pocket - g.p_x_mask_AAtype_bool.size(0)),
                                            'constant', False)
                                      for g in batch_list], dim=0)
    p_x_mask_AAtype_label = torch.cat([g.p_x_mask_AAtype_label for g in batch_list], dim=0)
    p_x_mask_AAtype_scatter = p_x_batch_info[p_x_mask_AAtype_bool]

    p_x_mask_MCcoor_bool = torch.cat([F.pad(g.p_x_mask_MCcoor_bool,
                                            (0, max_len_pocket - g.p_x_mask_MCcoor_bool.size(0)),
                                            'constant', False)
                                      for g in batch_list], dim=0)
    p_x_mask_MCcoor_label = torch.cat([g.p_x_mask_MCcoor_label for g in batch_list], dim=0)
    p_x_mask_MCcoor_CAcoor = p_coor_init_flat[p_x_mask_MCcoor_bool]
    p_x_mask_MCcoor_scatter = p_x_batch_info[p_x_mask_MCcoor_bool]

    p_x_mask_SCcoor_bool = torch.cat([F.pad(g.p_x_mask_SCcoor_bool,
                                            (0, max_len_pocket - g.p_x_mask_SCcoor_bool.size(0)),
                                            'constant', False)
                                      for g in batch_list], dim=0)
    p_x_mask_SCcoor_label = torch.cat([g.p_x_mask_SCcoor_label for g in batch_list], dim=0)
    p_x_mask_SCcoor_label_SCmask = torch.cat([g.p_x_mask_SCcoor_label_SCmask for g in batch_list], dim=0)
    p_x_mask_SCcoor_CAcoor = p_coor_init_flat[p_x_mask_SCcoor_bool]
    p_x_mask_SCcoor_scatter = p_x_batch_info[p_x_mask_SCcoor_bool]

    p_x_mask_CAnoise_bool = torch.cat([F.pad(g.p_x_mask_CAnoise_bool,
                                             (0, max_len_pocket - g.p_x_mask_CAnoise_bool.size(0)),
                                             'constant', False)
                                       for g in batch_list], dim=0)
    p_x_mask_CAnoise_label = torch.cat([g.p_x_mask_CAnoise_label for g in batch_list], dim=0)
    p_x_mask_CAnoise_CAcoor = p_coor_init_flat[p_x_mask_CAnoise_bool]
    p_x_mask_CAnoise_scatter = p_x_batch_info[p_x_mask_CAnoise_bool]

    # supply
    len_pocket = torch.Tensor([g.len_pocket for g in batch_list])

    idx = [g.idx for g in batch_list]

    # use Data in PyG
    complex_graph = torch_geometric.data.Data(
        x=None,
        # node
        p_x_sca_init=p_x_sca_init,
        p_x_vec_init=p_x_vec_init,

        # edge
        p_edge_sca_init=p_edge_sca_init,
        p_edge_vec_init=p_edge_vec_init,

        # coor
        p_coor_init=p_coor_init,

        # supply info
        len_pocket=len_pocket,
        p_x_mask=p_x_mask,
        p_edge_mask=p_edge_mask,

        idx=idx,

        # for pretrain
        p_x_mask_AAtype_bool=p_x_mask_AAtype_bool,
        p_x_mask_AAtype_label=p_x_mask_AAtype_label,
        p_x_mask_AAtype_scatter=p_x_mask_AAtype_scatter,

        p_x_mask_MCcoor_bool=p_x_mask_MCcoor_bool,
        p_x_mask_MCcoor_label=p_x_mask_MCcoor_label,
        p_x_mask_MCcoor_CAcoor=p_x_mask_MCcoor_CAcoor,
        p_x_mask_MCcoor_scatter=p_x_mask_MCcoor_scatter,

        p_x_mask_SCcoor_bool=p_x_mask_SCcoor_bool,
        p_x_mask_SCcoor_label=p_x_mask_SCcoor_label,
        p_x_mask_SCcoor_label_SCmask=p_x_mask_SCcoor_label_SCmask,
        p_x_mask_SCcoor_CAcoor=p_x_mask_SCcoor_CAcoor,
        p_x_mask_SCcoor_scatter=p_x_mask_SCcoor_scatter,

        p_x_mask_CAnoise_bool=p_x_mask_CAnoise_bool,
        p_x_mask_CAnoise_label=p_x_mask_CAnoise_label,
        p_x_mask_CAnoise_CAcoor=p_x_mask_CAnoise_CAcoor,
        p_x_mask_CAnoise_scatter=p_x_mask_CAnoise_scatter,
    )
    return complex_graph


class PocketDataset(torch.utils.data.Dataset):
    def __init__(self, mode, args, data_list, cache_path='./cache'):
        self.mode = mode
        self.data_list = data_list
        self.data_path = args.data_path

        self.coor_scale = args.coor_scale
        self.max_len_pocket = args.max_len_pocket

        if self.mode == 'train':
            self.n_batch = int(args.n_batch)
        elif self.mode == 'val':
            self.n_batch = 100  # int(args.n_batch // 100)

        # for mask
        self.total_mask_rate = args.total_mask_rate
        self.mask_rate_AAtype = args.mask_rate_AAtype
        self.mask_rate_MCcoor = args.mask_rate_MCcoor
        self.mask_rate_SCcoor = args.mask_rate_SCcoor
        self.mask_rate_CAnoise = args.mask_rate_CAnoise

        self.CAnoise_dist = args.CAnoise_dist / self.coor_scale

        self.set_data(0)

    def set_data(self, epoch):
        self.epoch = epoch
        start = (epoch * self.n_batch) % len(self.data_list)
        if start + self.n_batch > len(self.data_list) - 1:
            pretrain_list = self.data_list + self.data_list
        else:
            pretrain_list = self.data_list
        self.sub_pretrain_list = pretrain_list[start:start + self.n_batch]

        assert len(self.sub_pretrain_list) == self.n_batch

    def __getitem__(self, i):
        idx = self.sub_pretrain_list[i]

        try:
            complex_graph = self.get_complex(idx)
        except:
            print(f'Fail to load {idx}')
            # os.remove(f'{self.data_path}/{idx}.npz')
            complex_graph = self.get_complex(self.sub_pretrain_list[0])
        return complex_graph

    def get_complex(self, idx):
        ##################################################################
        # get dict data
        ##################################################################
        dic_data = np.load(f'{self.data_path}/{idx}', allow_pickle=False)

        p_x_sca_init = dic_data['p_x_sca']  # AA_type
        p_x_vec_init = dic_data['p_x_vec']  # SC vec
        resi_connect = dic_data['resi_connect']
        p_coor = dic_data['p_coor']  # CA_coor
        CB_coor = dic_data['CB_coor']
        MCSC_coor = dic_data['MCSC_coor']  # MCSC_coor: [N_coor, C_coor, O_coor, SC_coor], [n_atom, type, 3]
        MCSC_mask = dic_data['MCSC_mask']

        ##################################################################
        # to tensor
        ##################################################################
        p_x_sca_init = torch.from_numpy(p_x_sca_init).float()
        p_x_vec_init = torch.from_numpy(p_x_vec_init).float()
        resi_connect = torch.from_numpy(resi_connect).float()
        p_coor_init = torch.from_numpy(p_coor).float()
        CB_coor = torch.from_numpy(CB_coor).float()
        MCSC_coor = torch.from_numpy(MCSC_coor).float()
        MCSC_mask = torch.from_numpy(MCSC_mask).long()

        ##################################################################
        # len info
        ##################################################################
        len_pocket = len(p_x_sca_init)

        ##################################################################
        # scale
        ##################################################################
        p_coor_init = p_coor_init / self.coor_scale
        CB_coor = CB_coor / self.coor_scale
        MCSC_coor = MCSC_coor / self.coor_scale
        p_x_vec_init = p_x_vec_init / self.coor_scale

        ##################################################################
        # init pocket info
        ##################################################################
        # CA CB N C and SC
        MCCACB_coor = torch.cat([p_coor_init.unsqueeze(-2), CB_coor.unsqueeze(-2), MCSC_coor[:, :2, :]], dim=1)
        SC_fromtor_vec = p_x_vec_init  # CB-CA, CG-CB, ...
        SC_fromCA_vec = torch.where(
            MCSC_coor[:, 3:] == 0, torch.zeros_like(MCSC_coor[:, 3:]), MCSC_coor[:, 3:] - p_coor_init.unsqueeze(-2)
        )

        # x vec
        x_MCCACB_vec = rearrange(MCCACB_coor, 'i t c -> i t () c') - rearrange(MCCACB_coor, 'i t c -> i () t c')
        x_MCCACB_vec_flat = rearrange(x_MCCACB_vec, 'i m n c -> i (m n) c')
        p_x_vec_init = torch.cat([x_MCCACB_vec_flat, SC_fromtor_vec, SC_fromCA_vec], dim=1)

        # edge sca
        edge_MCCACB_vec = rearrange(MCCACB_coor, 'n t c -> n () t () c') - rearrange(MCCACB_coor,
                                                                                     'n t c -> () n () t c')
        edge_MCCACB_dist_flat = rearrange(edge_MCCACB_vec.norm(p=2, dim=-1), 'i j m n -> i j (m n)')
        resi_connect_onehot = torch.zeros((resi_connect.size(0), resi_connect.size(0), 2))
        resi_connect_onehot[:, :, 0] = resi_connect.triu()
        resi_connect_onehot[:, :, 1] = - resi_connect.tril()
        p_edge_sca_init = torch.cat([edge_MCCACB_dist_flat, resi_connect_onehot], dim=-1)

        # edge vec
        edge_MCCACB_vec_flat = rearrange(edge_MCCACB_vec, 'i j m n c -> i j (m n) c')
        p_edge_vec_init = edge_MCCACB_vec_flat

        ##################################################################
        # for pretrain task
        ##################################################################
        allow_mask_pos = (p_x_sca_init.sum(dim=-1, keepdim=True) == 1).to(p_x_sca_init.dtype)
        p_x_sca_init = F.pad(p_x_sca_init, (0, 1), 'constant', 0)

        # CA noise
        p_x_mask_CAnoise_bool = self.gen_mask_index([allow_mask_pos], self.total_mask_rate * self.mask_rate_CAnoise)
        p_x_mask_CAnoise_label = p_coor_init[p_x_mask_CAnoise_bool]  # CA coor
        p_coor_init[p_x_mask_CAnoise_bool] = p_coor_init[p_x_mask_CAnoise_bool] + \
                                             torch.randn((p_x_mask_CAnoise_bool.sum(), 3)) * self.CAnoise_dist
        p_x_vec_init[p_x_mask_CAnoise_bool] = 0
        CAnoised_edge_MCCACB_coor = torch.cat([p_coor_init.unsqueeze(-2), CB_coor.unsqueeze(-2), MCSC_coor[:, :2, :]],
                                              dim=1)
        # p_edge_sca_init: -1 mask SC, keep connection info
        # p_edge_vec_init: 0 mask SC
        # left
        tmp_mask = torch.Tensor([1, 0, 0, 0]).unsqueeze(1) * torch.Tensor([1, 1, 1, 1]).unsqueeze(0)  # [CA CB N C]
        left_tmp = rearrange(CAnoised_edge_MCCACB_coor[p_x_mask_CAnoise_bool], 'i m c -> i () m () c') - rearrange(
            CAnoised_edge_MCCACB_coor, 'j n c -> () j () n c')
        left_tmp = left_tmp * rearrange(tmp_mask, 'm n -> () () m n ()')
        p_edge_sca_init[p_x_mask_CAnoise_bool, :, :-2] = rearrange(
            left_tmp.norm(p=2, dim=-1), 'i j m n -> i j (m n)') + rearrange(tmp_mask - 1, 'm n -> () () (m n)')
        p_edge_vec_init[p_x_mask_CAnoise_bool, :] = rearrange(left_tmp, 'i j m n c-> i j (m n) c')
        # right
        tmp_mask = torch.Tensor([1, 1, 1, 1]).unsqueeze(1) * torch.Tensor([1, 0, 0, 0]).unsqueeze(0)  # [CA CB N C]
        right_tmp = rearrange(CAnoised_edge_MCCACB_coor, 'i m c -> i () m () c') - rearrange(
            CAnoised_edge_MCCACB_coor[p_x_mask_CAnoise_bool], 'j n c -> () j () n c')
        right_tmp = right_tmp * rearrange(tmp_mask, 'm n -> () () m n ()')
        p_edge_sca_init[:, p_x_mask_CAnoise_bool, :-2] = rearrange(
            right_tmp.norm(p=2, dim=-1), 'i j m n -> i j (m n)') + rearrange(tmp_mask - 1, 'm n -> () () (m n)')
        p_edge_vec_init[:, p_x_mask_CAnoise_bool] = rearrange(right_tmp, 'i j m n c-> i j (m n) c')
        allow_mask_pos = allow_mask_pos + (p_x_mask_CAnoise_bool.unsqueeze(-1)).to(p_x_sca_init.dtype)

        # mask AAtype
        p_x_mask_AAtype_bool = self.gen_mask_index([allow_mask_pos], self.total_mask_rate * self.mask_rate_AAtype)
        p_x_mask_AAtype_label = p_x_sca_init.argmax(dim=-1)[p_x_mask_AAtype_bool]
        p_x_sca_init[p_x_mask_AAtype_bool] = 0
        p_x_sca_init[p_x_mask_AAtype_bool, -1] = 1
        allow_mask_pos = allow_mask_pos + (p_x_mask_AAtype_bool.unsqueeze(-1)).to(p_x_sca_init.dtype)

        # mask MC
        p_x_mask_MCcoor_bool = self.gen_mask_index([allow_mask_pos], self.total_mask_rate * self.mask_rate_MCcoor)
        p_x_mask_MCcoor_label = torch.stack([MCSC_coor[p_x_mask_MCcoor_bool, 0] - p_coor_init[p_x_mask_MCcoor_bool],
                                             MCSC_coor[p_x_mask_MCcoor_bool, 1] - p_coor_init[p_x_mask_MCcoor_bool],
                                             MCSC_coor[p_x_mask_MCcoor_bool, 2] - p_coor_init[p_x_mask_MCcoor_bool],
                                             MCSC_coor[p_x_mask_MCcoor_bool, 1] - MCSC_coor[p_x_mask_MCcoor_bool, 0],
                                             MCSC_coor[p_x_mask_MCcoor_bool, 2] - MCSC_coor[p_x_mask_MCcoor_bool, 0],
                                             MCSC_coor[p_x_mask_MCcoor_bool, 2] - MCSC_coor[p_x_mask_MCcoor_bool, 1]],
                                            dim=-2)
        p_x_vec_init = self.select_by_bool(p_x_vec_init, p_x_mask_MCcoor_bool,
                                           torch.Tensor([False, False, True, True]).to(torch.bool), 0, '1d',
                                           # [CA CB N C], mask N C
                                           additional_mask=torch.Tensor([False] * 8).to(torch.bool))  # keep all SC
        p_edge_sca_init = self.select_by_bool(p_edge_sca_init, p_x_mask_MCcoor_bool,
                                              torch.Tensor([False, False, True, True]).to(torch.bool), -1, '2d',
                                              # [CA CB N C], mask N C
                                              additional_mask=torch.Tensor([False, False]).to(
                                                  torch.bool))  # keep connection info
        p_edge_vec_init = self.select_by_bool(p_edge_vec_init, p_x_mask_MCcoor_bool,
                                              torch.Tensor([False, False, True, True]).to(torch.bool), 0, '2d',
                                              # [CA CB N C], mask N C
                                              additional_mask=None)
        allow_mask_pos = allow_mask_pos + (p_x_mask_MCcoor_bool.unsqueeze(-1)).to(p_x_sca_init.dtype)

        # mask SC
        p_x_mask_SCcoor_bool = self.gen_mask_index([allow_mask_pos], self.total_mask_rate * self.mask_rate_SCcoor)
        p_x_mask_SCcoor_label = torch.cat([SC_fromtor_vec[p_x_mask_SCcoor_bool], SC_fromCA_vec[p_x_mask_SCcoor_bool]],
                                          dim=-2)
        p_x_mask_SCcoor_label_SCmask = (p_x_mask_SCcoor_label.sum(-1) != 0).to(torch.float)
        p_x_vec_init = self.select_by_bool(p_x_vec_init, p_x_mask_SCcoor_bool,
                                           torch.Tensor([False, True, False, False]).to(torch.bool), 0, '1d',
                                           # [CA CB N C], mask CB
                                           additional_mask=torch.Tensor([True] * 8).to(torch.bool))  # mask all SC
        p_edge_sca_init = self.select_by_bool(p_edge_sca_init, p_x_mask_SCcoor_bool,
                                              torch.Tensor([False, True, False, False]).to(torch.bool), -1, '2d',
                                              # [CA CB N C], mask CB
                                              additional_mask=torch.Tensor([False, False]).to(
                                                  torch.bool))  # keep connection info
        p_edge_vec_init = self.select_by_bool(p_edge_vec_init, p_x_mask_SCcoor_bool,
                                              torch.Tensor([False, True, False, False]).to(torch.bool), 0, '2d',
                                              # [CA CB N C], mask CB
                                              additional_mask=None)
        allow_mask_pos = allow_mask_pos + (p_x_mask_SCcoor_bool.unsqueeze(-1)).to(p_x_sca_init.dtype)

        ##################################################################
        # use Data in PyG
        ##################################################################
        complex_graph = torch_geometric.data.Data(
            # node
            p_x_sca_init=p_x_sca_init,
            p_x_vec_init=p_x_vec_init,

            # edge
            p_edge_sca_init=p_edge_sca_init,
            p_edge_vec_init=p_edge_vec_init,

            # coor
            p_coor_init=p_coor_init,

            # supply info
            len_pocket=len_pocket,

            idx=idx,

            # for mask
            p_x_mask_AAtype_bool=p_x_mask_AAtype_bool,
            p_x_mask_AAtype_label=p_x_mask_AAtype_label,

            p_x_mask_MCcoor_bool=p_x_mask_MCcoor_bool,
            p_x_mask_MCcoor_label=p_x_mask_MCcoor_label,

            p_x_mask_SCcoor_bool=p_x_mask_SCcoor_bool,
            p_x_mask_SCcoor_label=p_x_mask_SCcoor_label,
            p_x_mask_SCcoor_label_SCmask=p_x_mask_SCcoor_label_SCmask,

            p_x_mask_CAnoise_bool=p_x_mask_CAnoise_bool,
            p_x_mask_CAnoise_label=p_x_mask_CAnoise_label,
        )
        return complex_graph

    def __len__(self):
        return self.n_batch

    def gen_mask_index(self, feat_label_list, mask_rate=0.15):
        allow_mask_pos = torch.cat([feat_label.sum(dim=-1, keepdim=True) == 1 for feat_label in feat_label_list],
                                   dim=-1).prod(dim=-1)
        n_allow = allow_mask_pos.sum()
        n_mask = min(max(int(mask_rate * allow_mask_pos.size(0)), 1), n_allow)
        mask_index = torch.randperm(n_allow)[:n_mask]  # pos in [0, n_allow]
        mask = torch.zeros(n_allow).index_fill_(-1, mask_index, 1)  # pos in [0, n_allow]
        new_mask = torch.zeros(allow_mask_pos.size(0))  # pos in [0, allow_mask_pos.size(0)]
        new_mask[allow_mask_pos.to(torch.bool)] = mask  # pos in [0, allow_mask_pos.size(0)]
        return new_mask.to(torch.bool)

    def select_by_bool(self, x, sample_index, mask_index_single, mask_v, select_type, additional_mask=None):
        # mask_index_single: [CA CB N C]
        # mask_index <- 1d: [mask_index_single, mask_index_single] or 2d: [mask_index_single, [True, ...](same shape with mask_index_single)]
        # mask_index <- cat([mask_index, additional_mask])
        # 1d [i, j, ...], sample_index->i, mask_index->j
        # 2d [i, j, k, ...], sample_index->ij, mask_index->k
        assert select_type in ['1d', '2d']
        assert sample_index.dtype == torch.bool and mask_index_single.dtype == torch.bool
        if select_type == '1d':
            mask_index = torch.zeros((mask_index_single.size(0), mask_index_single.size(0))).to(torch.bool)
            mask_index[mask_index_single, :] = True
            mask_index[:, mask_index_single] = True
            mask_index = rearrange(mask_index, 'i j -> (i j)')
            if not isinstance(additional_mask, type(None)):
                mask_index = torch.cat([mask_index, additional_mask])
            sampled = x[sample_index]
            sampled[:, mask_index] = mask_v
            x[sample_index] = sampled
        elif select_type == '2d':
            # left
            left_mask_index = rearrange(
                mask_index_single.unsqueeze(1) * torch.ones_like(mask_index_single).to(torch.bool).unsqueeze(0),
                'i j -> (i j)')
            if not isinstance(additional_mask, type(None)):
                left_mask_index = torch.cat([left_mask_index, additional_mask])
            left_sampled = x[sample_index, :]
            left_sampled[:, :, left_mask_index] = mask_v
            x[sample_index, :] = left_sampled
            # right
            right_mask_index = rearrange(
                torch.ones_like(mask_index_single).to(torch.bool).unsqueeze(1) * mask_index_single.unsqueeze(0),
                'i j -> (i j)')
            if not isinstance(additional_mask, type(None)):
                right_mask_index = torch.cat([right_mask_index, additional_mask])
            right_sampled = x[:, sample_index]
            right_sampled[:, :, right_mask_index] = mask_v
            x[:, sample_index] = right_sampled
        return x


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target, bce=False):
        if bce:
            ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        else:
            ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss



if __name__ == '__main__':
    # for testing
    pass
