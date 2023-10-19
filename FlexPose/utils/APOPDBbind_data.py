import os
import random

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from utils.common import *
from utils.data_utils import pad_zeros
from model.MMFF import *


def split_dataset(data_path, data_split_rate, test_list_path=None):
    if test_list_path is None:
        idx_list = os.listdir(data_path)
        random.shuffle(idx_list)

        l = len(idx_list)
        cut_1 = int(data_split_rate[0] * l)
        cut_2 = cut_1 + int(data_split_rate[1] * l)
        train_list = idx_list[:cut_1]
        val_list = idx_list[cut_1:cut_2]
        test_list = idx_list[cut_2:]
    else:
        idx_list = os.listdir(data_path)
        test_list = [f"{i.split('.')[0]}.pkl" for i in load_idx_list(test_list_path) if f"{i.split('.')[0]}.pkl" in idx_list]
        rest_list = [i for i in idx_list if i not in test_list]
        random.shuffle(rest_list)

        l = len(rest_list)
        cut_2 = int(data_split_rate[1] * l)
        val_list = rest_list[:cut_2]
        train_list = rest_list[cut_2:]

    return train_list, val_list, test_list


def get_aff(info_path):
    dic_aff = {line[:4]: float(line[18:23])
               for line in open(info_path, 'r').readlines() if not line.startswith('#')}
    return dic_aff


def collate_fn(batch_list, return_batch_list=False):
    max_len_pocket = 0
    max_len_ligand = 0
    for g in batch_list:
        max_len_pocket = max(max_len_pocket, g['len_pocket'])
        g['p_x_mask'] = torch.ones((g['len_pocket']))
        g['p_edge_mask'] = torch.ones((g['len_pocket'], g['len_pocket']))
        max_len_ligand = max(max_len_ligand, g['len_ligand'])
        g['l_x_mask'] = torch.ones((g['len_ligand']))
        g['l_edge_mask'] = torch.ones((g['len_ligand'], g['len_ligand']))

    # for mem testing
    # max_len_pocket = 150
    # max_len_ligand = 150

    # force batch_size to 1, if data to large
    if 'drop_huge_data' in batch_list[0].keys():
        if batch_list[0]['drop_huge_data'] \
                and max_len_pocket + max_len_ligand > batch_list[0]['drop_huge_data_max_len'] \
                and batch_list[0]['training']:
            random.shuffle(batch_list)
            batch_list = [batch_list[0]]


    dic_data = {}

    #####################################################################################################
    # for feat, coor and mask
    #####################################################################################################
    # protein
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'p_x_sca_init', 'p_coor_init',
                              ],
                              max_len_pocket,
                              collect_dim=-3, data_type='1d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'p_x_vec_init',
                              ],
                              max_len_pocket,
                              collect_dim=-4, data_type='1d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'p_edge_sca_init',
                              ],
                              max_len_pocket,
                              collect_dim=-4, data_type='2d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'p_edge_vec_init',
                              ],
                              max_len_pocket,
                              collect_dim=-5, data_type='2d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'p_x_mask',
                              ],
                              max_len_pocket,
                              collect_dim=-2, data_type='1d', output_dtype=torch.bool, value=False))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'p_edge_mask',
                              ],
                              max_len_pocket,
                              collect_dim=-3, data_type='2d', output_dtype=torch.bool, value=False))

    # ligand
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'l_x_sca_init', 'l_coor_init',
                              ],
                              max_len_ligand,
                              collect_dim=-3, data_type='1d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'l_x_vec_init',
                              ],
                              max_len_ligand,
                              collect_dim=-4, data_type='1d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'l_edge_sca_init',
                              ],
                              max_len_ligand,
                              collect_dim=-4, data_type='2d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'l_edge_vec_init',
                              ],
                              max_len_ligand,
                              collect_dim=-5, data_type='2d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'l_dismap',
                              ],
                              max_len_ligand,
                              collect_dim=-3, data_type='2d', output_dtype=torch.float, value=-1/batch_list[0]['coor_scale']))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'l_dismap_mask',
                              ],
                              max_len_ligand,
                              collect_dim=-3, data_type='2d', output_dtype=torch.float, value=0))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'l_x_mask',
                              ],
                              max_len_ligand,
                              collect_dim=-2, data_type='1d', output_dtype=torch.bool, value=False))
    dic_data.update(pad_zeros(batch_list,
                              [
                                  'l_edge_mask',
                              ],
                              max_len_ligand,
                              collect_dim=-3, data_type='2d', output_dtype=torch.bool, value=False))


    dic_data['l_coor_update_mask'] = F.pad(dic_data['l_x_mask'], (max_len_pocket, 0), 'constant', 0).float()
    dic_data['l_coor_true'] = torch.cat([g['l_coor_true'] for g in batch_list], dim=0)
    dic_data['l_coor_true_dense'] = pad_zeros(
        batch_list,
        [
            'l_coor_true',
        ], max_len_ligand, collect_dim=-3, data_type='1d', output_dtype=torch.float)['l_coor_true']


    # ligand coor indexes
    len_tmp_complex = 0
    len_tmp_ligand = 0
    len_tmp_batch_match = 0
    ligand_node_loc_in_complex = []
    l_match = []
    l_nomatch = []
    scatter_ligand_1 = []
    scatter_ligand_2 = []
    for i, g in enumerate(batch_list):
        ligand_node_loc_in_complex.append(torch.arange(g['len_ligand']) + max_len_pocket + len_tmp_complex)
        l_match.append(g['l_match'] + len_tmp_ligand)
        l_nomatch.append(g['l_nomatch'] + len_tmp_ligand)
        scatter_ligand_1.append(repeat(torch.arange(0, len(g['l_match']) // g['len_ligand']),
                                       'i -> (i m)', m=g['len_ligand']) + len_tmp_batch_match)
        scatter_ligand_2.append(torch.zeros(len(g['l_match']) // g['len_ligand']) + i)
        len_tmp_complex += max_len_pocket + max_len_ligand
        len_tmp_ligand += g['len_ligand']
        len_tmp_batch_match += len(g['l_match']) // g['len_ligand']
    dic_data['ligand_node_loc_in_complex_flat'] = torch.cat(ligand_node_loc_in_complex, dim=0).long()
    dic_data['l_match'] = torch.cat(l_match, dim=0).long()
    dic_data['l_nomatch'] = torch.cat(l_nomatch, dim=0).long()
    dic_data['scatter_ligand_1'] = torch.cat(scatter_ligand_1, dim=0).long()
    dic_data['scatter_ligand_2'] = torch.cat(scatter_ligand_2, dim=0).long()

    # protein ground truth
    dic_data['p_tor_true'] = torch.cat([g['p_tor_true'] for g in batch_list], dim=0)
    dic_data['p_tor_vec_true'] = torch.stack([dic_data['p_tor_true'].sin(), dic_data['p_tor_true'].cos()], dim=-1)
    dic_data['p_tor_alt_true'] = torch.cat([g['p_tor_alt_true'] for g in batch_list], dim=0)
    dic_data['p_tor_vec_alt_true'] = torch.stack([dic_data['p_tor_alt_true'].sin(), dic_data['p_tor_alt_true'].cos()], dim=-1)
    dic_data['p_tor_mask'] = torch.cat([g['p_tor_mask'] for g in batch_list], dim=0).float()

    dic_data['p_coor_true'] = torch.cat([g['p_coor_true'] for g in batch_list], dim=0)
    dic_data['p_CB_coor_true'] = torch.cat([g['p_CB_coor_true'] for g in batch_list], dim=0)

    dic_data['p_partial_select_mask'] = torch.cat([
        F.pad(g['p_partial_select_mask'], (0, max_len_pocket + max_len_ligand - g['p_partial_select_mask'].shape[0]),
              'constant', False) for g in batch_list], dim=0).bool()
    dic_data['scatter_pocket'] = torch.cat([torch.zeros(g['p_tor_true'].size(0)) + i for i, g in enumerate(batch_list)]).long()

    # additional tor
    # for input torsion
    input_SCtorsion = torch.stack([
        F.pad(g['input_SCtorsion'], (0, 0, 0, max_len_pocket - g['input_SCtorsion'].shape[0]),
              'constant', 0) for g in batch_list], dim=0)
    sc_in = input_SCtorsion
    sc_in_mask = (sc_in != 0).float().unsqueeze(dim=-2)
    sc_in = torch.stack([sc_in, sc_in.sin(), sc_in.cos()], dim=-2)
    sc_in = sc_in * sc_in_mask  # + (1 - sc_in_mask) * (-99)
    dic_data['sc_in'] = rearrange(sc_in, 'b n d c -> b n (d c)')   # sc_in[:, :, :4] is input tor

    sc_in_flat = rearrange(dic_data['sc_in'], 'b n d -> (b n) d')
    tmp_mask = torch.cat([
        F.pad(g['p_partial_select_mask'], (0, max_len_pocket - g['p_partial_select_mask'].shape[0]),
              'constant', False) for g in batch_list], dim=0).bool()
    dic_data['sc_in_partial_select'] = sc_in_flat[tmp_mask]

    # for torsion delta
    p_tor_input = dic_data['sc_in_partial_select'][:, :4]
    dic_data['delta_p_tor_true'] = dic_data['p_tor_true'] - p_tor_input
    dic_data['delta_p_tor_alt_true'] = dic_data['p_tor_alt_true'] - p_tor_input
    dic_data['delta_p_tor_vec_true'] = torch.stack([dic_data['delta_p_tor_true'].sin(), dic_data['delta_p_tor_true'].cos()], dim=-1)
    dic_data['delta_p_tor_vec_alt_true'] = torch.stack([dic_data['delta_p_tor_alt_true'].sin(), dic_data['delta_p_tor_alt_true'].cos()], dim=-1)

    # for CA, CB mask
    dic_data['p_CA_mask'] = torch.cat([g['p_CA_mask'] for g in batch_list])
    dic_data['p_CB_mask'] = torch.cat([g['p_CB_mask'] for g in batch_list])

    # affinity
    dic_data['aff_true'] = torch.Tensor([g['aff_true'] for g in batch_list])
    dic_data['aff_mask'] = torch.Tensor([g['aff_mask'] for g in batch_list])
    
    # for MMFF
    dic_MMFF_param = {}
    for k in batch_list[0]['MMFF_keys']:
        len_tmp_ligand = 0
        MMFF_index_list = []
        MMFF_param_list = []
        MMFF_batch_list = []
        for i, g in enumerate(batch_list):
            MMFF_index_list.append(g[k + '_index'] + len_tmp_ligand)
            MMFF_param_list.append(g[k + '_param'])
            MMFF_batch_list.append(g[k + '_batch'] + i)
            len_tmp_ligand += g['len_ligand']
        dic_MMFF_param[k + '_index'] = torch.cat(MMFF_index_list, dim=0)
        dic_MMFF_param[k + '_param'] = torch.cat(MMFF_param_list, dim=0)
        dic_MMFF_param[k + '_batch'] = torch.cat(MMFF_batch_list, dim=0)
    dic_data.update(dic_MMFF_param)


    # supply
    dic_data['len_pocket'] = torch.Tensor([g['len_pocket'] for g in batch_list])
    dic_data['len_ligand'] = torch.Tensor([g['len_ligand'] for g in batch_list])

    dic_data['idx'] = [g['idx'] for g in batch_list]

    # use Data in PyG
    complex_graph = torch_geometric.data.Data(**dic_data)

    if return_batch_list:
        return complex_graph, batch_list, max_len_pocket, max_len_ligand
    else:
        return complex_graph


class ComplexDataset(torch.utils.data.Dataset):
    def __init__(self, mode, args, data_list, cache_path='./cache'):
        self.mode = mode
        if args.do_pregen_data:
            self.mode = 'pregen'
            self.pregen_path = args.pregen_path
            delmkdir(self.pregen_path)
        assert mode in ['train', 'val', 'test', 'pregen']
        self.data_list = data_list
        self.data_path = args.data_path
        self.select_MC_type = None

        self.coor_scale = args.coor_scale
        self.aff_scale = args.aff_scale

        self.max_len_pocket = args.max_len_pocket
        self.max_len_ligand = args.max_len_ligand
        self.drop_huge_data = args.drop_huge_data
        self.drop_huge_data_max_len = args.drop_huge_data_max_len

        # for apo or MC data selection
        self.MC_type = ['apo', 'rand_pert', 'fix_pack', 'flex_pack', 'holo']
        self.MC_rate = [args.apo_rate, args.rand_pert_rate, args.fix_pack_rate, args.flex_pack_rate, args.holo_rate]

        self.training = False
        self.dropout_pocket = args.dropout_pocket
        self.l_init_sigma = args.l_init_sigma / self.coor_scale
        self.CA_cutoff = args.CA_cutoff

        # for MMFF
        self.MMFF_keys = MMFF_keys
        self.MMFF_pad_dim = MMFF_pad_dim

        # for pregen
        self.use_pregen_data = args.use_pregen_data
        self.pregen_path = args.pregen_path

    def shuffle_data(self):
        random.shuffle(self.data_list)

    def __getitem__(self, i):
        f_name = self.data_list[i]
        if self.use_pregen_data:
            complex_graph = pickle.load(open(f'{self.cache}/{f_name}.pkl', 'rb'))
        else:
           complex_graph = self.get_complex(f_name, i_num=i)
        return complex_graph

    def get_complex(self, f_name, i_num=0):
        ##################################################################
        # get dict data
        ##################################################################
        dic_data = pickle.load(open(f'{self.data_path}/{f_name}', 'rb'))
        ligand_data = dic_data['ligand_data']
        holo_true_data = dic_data['holo_true_data']
        apo_data = dic_data['apo_data']
        rand_pert_data_list = dic_data['rand_pert_data_list']
        fixbb_data_list = dic_data['fixbb_data_list']
        flexbb_data_list = dic_data['flexbb_data_list']
        aff_true = dic_data['aff'] if 'aff' in dict(dic_data).keys() else -1
        dic_MMFF_param = dic_data['dic_MMFF_param'] if 'dic_MMFF_param' in dict(dic_data).keys() else None


        have_apo = True if not isinstance(apo_data, type(None)) else False
        MC_type = np.random.choice(self.MC_type, p=self.MC_rate, size=1)[0] if self.mode != 'pregen' else 'apo'
        if MC_type == 'apo':
            if isinstance(apo_data, type(None)):
                MC_type = 'holo'
        elif MC_type == 'rand_pert':
            if len(rand_pert_data_list) == 0:
                MC_type = 'holo'
        elif MC_type == 'fix_pack':
            if len(fixbb_data_list) == 0:
                MC_type = 'holo'
        elif MC_type == 'flex_pack':
            if len(flexbb_data_list) == 0:
                MC_type = 'holo'

        if self.select_MC_type is not None:
            MC_type = self.select_MC_type


        ##################################################################
        # choose data
        ##################################################################
        # input
        l_x_sca_init, l_edge_sca_init, l_coor_true, l_match, l_dismap = ligand_data
        # if self.training and len(l_match)>10 and len(l_x_sca_init)>100:
        #     l_match_index = np.random.choice(np.arange(len(l_match)), size=10, replace=False)
        #     l_match = l_match[l_match_index]
        l_match = l_match.reshape(-1)
        n_match = len(l_match) // len(l_x_sca_init)
        l_nomatch = repeat(torch.arange(0, len(l_x_sca_init)), 'm -> (n m)', n=n_match)

        # if have_apo: input_pocket (all types) = [resi in apobind match, additional resi], true data = [resi in apobind match]
        if MC_type == 'apo':
            input_pocket = apo_data[:-1]
        elif MC_type == 'rand_pert':
            input_pocket = random.choice(rand_pert_data_list)
        elif MC_type == 'fix_pack':
            input_pocket = random.choice(fixbb_data_list)
        elif MC_type == 'flex_pack':
            input_pocket = random.choice(flexbb_data_list)
        elif MC_type == 'holo':
            input_pocket = holo_true_data[:-1]
        p_x_sca_init, p_x_vec_init, resi_connect, p_coor_init, CB_coor, MCSC_coor, MCSC_mask, tor_data = input_pocket
        input_SCtorsion, _, _ = tor_data  # additional init tor input

        # pocket dropout
        if self.training:
            num_random_drop = int(np.random.uniform(low=0, high=self.dropout_pocket, size=1) * len(p_x_sca_init))
            if len(p_x_sca_init) > self.max_len_pocket:  # For OOM
                num_random_drop = len(p_x_sca_init) - self.max_len_pocket
            resi_drop = np.random.choice(np.arange(len(p_x_sca_init)), size=num_random_drop, replace=False)

            input_retain_index = np.array([i for i in range(len(p_x_sca_init)) if i not in resi_drop])
            p_x_sca_init = p_x_sca_init[input_retain_index]
            p_x_vec_init = p_x_vec_init[input_retain_index]
            p_coor_init = p_coor_init[input_retain_index]
            CB_coor = CB_coor[input_retain_index]
            MCSC_coor = MCSC_coor[input_retain_index]
            MCSC_mask = MCSC_mask[input_retain_index]
            resi_connect = resi_connect[input_retain_index, :][:, input_retain_index]
            input_SCtorsion = input_SCtorsion[input_retain_index]

        # gound truth
        _, _, _, p_coor_true, p_CB_coor_true, _, _, holo_pocket_SCtorsion_data = holo_true_data[:-1]
        # different length in p_x_sca_init, if have_apo = True (only consider residues in apobind mapping)
        p_tor_true, p_tor_alt_true, p_tor_mask = holo_pocket_SCtorsion_data
        if have_apo:
            # no gap in holo
            p_tor_true = p_tor_true[:len(apo_data[-1])]
            p_tor_alt_true = p_tor_alt_true[:len(apo_data[-1])]
            p_tor_mask = p_tor_mask[:len(apo_data[-1])]
            p_coor_true = p_coor_true[:len(apo_data[-1])]
            p_CB_coor_true = p_CB_coor_true[:len(apo_data[-1])]
            if self.training:
                label_retain_mask_2 = [False if i in resi_drop else True for i in range(len(p_tor_true))]
                if not any(label_retain_mask_2):  # i.e. len(p_tor_true[label_retain_mask_2]) == 0
                    p_tor_true = np.zeros((1, p_tor_true.shape[-1]))
                    p_tor_alt_true = np.zeros((1, p_tor_alt_true.shape[-1]))
                    p_tor_mask = np.zeros((1, p_tor_mask.shape[-1]))
                    p_coor_true = np.zeros((1, p_coor_true.shape[-1]))
                    p_CB_coor_true = np.zeros((1, p_CB_coor_true.shape[-1]))
                    p_CA_mask = np.zeros(p_coor_true.shape[0])
                    p_CB_mask = np.zeros(p_CB_coor_true.shape[0])
                else:
                    p_tor_true = p_tor_true[label_retain_mask_2]
                    p_tor_alt_true = p_tor_alt_true[label_retain_mask_2]
                    p_tor_mask = p_tor_mask[label_retain_mask_2]
                    p_coor_true = p_coor_true[label_retain_mask_2]
                    p_CB_coor_true = p_CB_coor_true[label_retain_mask_2]
                    p_CA_mask = np.ones(p_coor_true.shape[0])
                    p_CB_mask = np.ones(p_CB_coor_true.shape[0])
            else:
                p_CA_mask = np.ones(p_coor_true.shape[0])
                p_CB_mask = np.ones(p_CB_coor_true.shape[0])
            p_partial_select_mask = np.array([True] * len(p_tor_true) + [False] * (len(p_x_sca_init) - len(p_tor_true)))
            # assert (p_tor_mask - MCSC_mask[:, 3:][p_partial_select_mask]).sum() == 0  # check mapping, Cys-Cys not correspond in rosetta SC num
        else:
            if self.training:
                p_tor_true = p_tor_true[input_retain_index]
                p_tor_alt_true = p_tor_alt_true[input_retain_index]
                p_tor_mask = p_tor_mask[input_retain_index]
                p_coor_true = p_coor_true[input_retain_index]
                p_CB_coor_true = p_CB_coor_true[input_retain_index]
            p_CA_mask = np.ones(p_coor_true.shape[0])
            p_CB_mask = np.ones(p_CB_coor_true.shape[0])
            p_partial_select_mask = np.array([True] * len(p_tor_mask))


        # del big CA shift
        # CA_shift = np.linalg.norm(p_coor_init[p_partial_select_mask] - p_coor_true, ord=2, axis=-1)
        # p_partial_select_mask = np.array([True if i == True and j < self.CA_cutoff else False
        #                                   for i, j in zip(p_partial_select_mask, CA_shift)])

        # get ligand MMFF (if exists)
        if dic_MMFF_param is not None:
            dic_MMFF_param = self.repad_MMFFparam(dic_MMFF_param, self.MMFF_keys, self.MMFF_pad_dim)
        else:
            dic_MMFF_param = {}
            for k, pad_dim in zip(self.MMFF_keys, self.MMFF_pad_dim):
                dic_MMFF_param[k + '_index'] = np.zeros((1, pad_dim[0]))
                dic_MMFF_param[k + '_param'] = np.zeros((1, pad_dim[1]))
                dic_MMFF_param[k + '_batch'] = np.zeros(1)


        ##################################################################
        # to tensor
        ##################################################################
        # pocekt
        p_x_sca_init = torch.from_numpy(p_x_sca_init).float()
        p_x_vec_init = torch.from_numpy(p_x_vec_init).float()
        p_coor_init = torch.from_numpy(p_coor_init).float()
        p_coor_true = torch.from_numpy(p_coor_true).float()
        p_CB_coor_true = torch.from_numpy(p_CB_coor_true).float()
        p_tor_true = torch.from_numpy(p_tor_true).float()
        p_tor_alt_true = torch.from_numpy(p_tor_alt_true).float()
        p_tor_mask = torch.from_numpy(p_tor_mask).long()
        p_partial_select_mask = torch.from_numpy(p_partial_select_mask).bool()
        p_CA_mask = torch.from_numpy(p_CA_mask).float()
        p_CB_mask = torch.from_numpy(p_CB_mask).float()


        resi_connect = torch.from_numpy(resi_connect).float()
        CB_coor = torch.from_numpy(CB_coor).float()
        MCSC_coor = torch.from_numpy(MCSC_coor).float()
        MCSC_mask = torch.from_numpy(MCSC_mask).long()
        input_SCtorsion = torch.from_numpy(input_SCtorsion).float()

        # ligand
        l_x_sca_init = torch.from_numpy(l_x_sca_init).float()
        l_edge_sca_init = torch.from_numpy(l_edge_sca_init).float()
        l_coor_true = torch.from_numpy(l_coor_true).float()
        l_dismap = torch.from_numpy(l_dismap).float()
        l_match = torch.from_numpy(l_match).long()

        dic_MMFF_param = {k: torch.from_numpy(v).long() if 'index' in k or 'batch' in k else torch.from_numpy(v).float()
                      for k, v in dic_MMFF_param.items()}


        ##################################################################
        # len info
        ##################################################################
        len_pocket = len(p_x_sca_init)
        len_ligand = len(l_x_sca_init)


        ##################################################################
        # scale
        ##################################################################
        p_coor_init = p_coor_init / self.coor_scale
        p_coor_true = p_coor_true / self.coor_scale
        p_CB_coor_true = p_CB_coor_true / self.coor_scale  # holo CB (true)
        CB_coor = CB_coor / self.coor_scale  # input CB
        MCSC_coor = MCSC_coor / self.coor_scale
        p_x_vec_init = p_x_vec_init / self.coor_scale
        l_coor_true = l_coor_true / self.coor_scale
        l_dismap = l_dismap / self.coor_scale

        p_tor_true = p_tor_true * np.pi / 180
        p_tor_alt_true = p_tor_alt_true * np.pi / 180
        input_SCtorsion = input_SCtorsion * np.pi / 180

        ##################################################################
        # init pocket info
        ##################################################################
        # CA CB N C and SC
        MCCACB_coor = torch.cat([p_coor_init.unsqueeze(-2), CB_coor.unsqueeze(-2), MCSC_coor[:, :2, :]], dim=1)
        SC_fromtor_vec = p_x_vec_init  # CB-CA, CG-CB, ...
        SC_fromCA_vec = torch.where(MCSC_coor[:, 3:] == 0, 0 * MCSC_coor[:, 3:], MCSC_coor[:, 3:] - p_coor_init.unsqueeze(-2))

        # x vec
        x_MCCACB_vec = rearrange(MCCACB_coor, 'i t c -> i t () c') - rearrange(MCCACB_coor, 'i t c -> i () t c')
        x_MCCACB_vec_flat = rearrange(x_MCCACB_vec, 'i m n c -> i (m n) c')
        p_x_vec_init = torch.cat([x_MCCACB_vec_flat, SC_fromtor_vec, SC_fromCA_vec], dim=1)

        # edge sca
        edge_MCCACB_vec = rearrange(MCCACB_coor, 'n t c -> n () t () c') - rearrange(MCCACB_coor, 'n t c -> () n () t c')
        edge_MCCACB_dist_flat = rearrange(edge_MCCACB_vec.norm(p=2, dim=-1), 'i j m n -> i j (m n)')
        resi_connect_onehot = torch.zeros((resi_connect.size(0), resi_connect.size(0), 2))
        resi_connect_onehot[:, :, 0] = resi_connect.triu()
        resi_connect_onehot[:, :, 1] = - resi_connect.tril()
        p_edge_sca_init = torch.cat([edge_MCCACB_dist_flat, resi_connect_onehot], dim=-1)

        # edge vec
        edge_MCCACB_vec_flat = rearrange(edge_MCCACB_vec, 'i j m n c -> i j (m n) c')
        p_edge_vec_init = edge_MCCACB_vec_flat


        ##################################################################
        # init ligand info
        ##################################################################
        l_coor_init = p_coor_init.mean(dim=0) + self.l_init_sigma * torch.randn_like(l_coor_true)
        l_x_vec_init = (l_coor_init - l_coor_init.mean(dim=0, keepdims=True)).unsqueeze(dim=-2)
        l_edge_vec_init = (l_coor_init.unsqueeze(dim=-2) - l_coor_init.unsqueeze(dim=-3)).unsqueeze(dim=-2)


        ##################################################################
        # aff
        ##################################################################
        aff_mask = 0 if aff_true == -1 else 1
        aff_true = aff_true / self.aff_scale


        ##################################################################
        # pretrain mask
        ##################################################################
        l_x_sca_init = F.pad(l_x_sca_init, (0, 1), 'constant', 0)
        l_edge_sca_init = F.pad(l_edge_sca_init, (0, 1), 'constant', 0)
        p_x_sca_init = F.pad(p_x_sca_init, (0, 1), 'constant', 0)

        l_dismap_mask = (l_dismap != (-1 / self.coor_scale)).float()

        ##################################################################
        # use Data in PyG
        ##################################################################
        dic_data = dict(
            ######################################################################
            # input
            ######################################################################
            # node
            p_x_sca_init=p_x_sca_init,
            p_x_vec_init=p_x_vec_init,
            l_x_sca_init=l_x_sca_init,
            l_x_vec_init=l_x_vec_init,

            # edge
            p_edge_sca_init=p_edge_sca_init,
            p_edge_vec_init=p_edge_vec_init,
            l_edge_sca_init=l_edge_sca_init,
            l_edge_vec_init=l_edge_vec_init,
            l_dismap=l_dismap,
            l_dismap_mask=l_dismap_mask,

            # coor
            p_coor_init=p_coor_init,
            l_coor_init=l_coor_init,

            ######################################################################
            # ground truth
            ######################################################################
            l_coor_true=l_coor_true,
            l_match=l_match,
            l_nomatch=l_nomatch,

            p_tor_true=p_tor_true,
            p_tor_alt_true=p_tor_alt_true,
            p_coor_true=p_coor_true,
            p_CB_coor_true=p_CB_coor_true,
            p_partial_select_mask=p_partial_select_mask,
            p_tor_mask=p_tor_mask,
            p_CA_mask=p_CA_mask,
            p_CB_mask=p_CB_mask,

            ######################################################################
            # supply info
            ######################################################################
            input_SCtorsion=input_SCtorsion,
            aff_true=aff_true,
            aff_mask=aff_mask,

            len_pocket=len_pocket,
            len_ligand=len_ligand,

            coor_scale=self.coor_scale,

            idx=f_name,
            i_num=i_num,

            drop_huge_data=self.drop_huge_data,
            drop_huge_data_max_len=self.drop_huge_data_max_len,
            training=self.training,

            MMFF_keys=self.MMFF_keys,
            **dic_MMFF_param,
        )
        return dic_data

    def __len__(self):
        return len(self.data_list)

    def repad_MMFFparam(self, dic_MMFF_param, MMFF_keys, pad_dim):
        # repad param, if shape==0 (remove by pocket_sub_index or Hs)
        for k, p in zip(MMFF_keys, pad_dim):
            if len(dic_MMFF_param[k+'_index']) == 0:
                dic_MMFF_param[k+'_index'] = np.array([list(range(p[0]))])
                dic_MMFF_param[k+'_param'] = np.array([[0] * p[1]])
                dic_MMFF_param[k+'_batch'] = np.array([0])
        return dic_MMFF_param


@torch.no_grad()
def calc_rmsd(coor_pred, coor_true, match=None):
    if isinstance(match, type(None)):
        match = torch.arange(len(coor_true))
    n_atom = coor_true.size(-2)
    n_match = len(match) // n_atom
    nomatch = repeat(torch.arange(0, coor_true.size(-2)), 'n -> (m n)', m=n_match)

    coor_pred = rearrange(rearrange(coor_pred, 'e n c -> n e c')[match], '(m n) e c -> m n e c', m=n_match)
    coor_true = rearrange(rearrange(coor_true, 'e n c -> n e c')[nomatch], '(m n) e c -> m n e c', m=n_match)

    coor_loss = torch.einsum('m n e c -> m e', (coor_pred - coor_true)**2)
    rmsd_loss = (coor_loss / n_atom)**0.5

    return rmsd_loss, coor_pred, n_match


@torch.no_grad()
def pred_ens(coor_pred, dic_data, return_raw=False):
    # coor_pred: [ens, n_atom, 3]
    ens = coor_pred.shape[0]
    l_match = dic_data.l_match.reshape(ens, -1)[0]

    if ens > 1:
        ens_pred = coor_pred[0]
        first_pred = coor_pred[0]

        rest_pred = coor_pred[1:]

        rmsd_match_ens, tmp_pred, n_match = calc_rmsd(rest_pred,
                                                      repeat(first_pred, 'n c -> e n c', e=rest_pred.size(0)),
                                                      match=l_match)  # return [match, ens]
        min_index = rmsd_match_ens.min(dim=0, keepdims=True)[1]
        rest_ens_matched_pred = torch.gather(tmp_pred, dim=0,
                                             index=repeat(min_index, 'm e -> m n e c', n=rest_pred.size(1),
                                                          c=3)).squeeze(0)  # to [n_atom, ens-1, 3]
        ens_pred = torch.cat([first_pred.unsqueeze(1), rest_ens_matched_pred], dim=1)
        if return_raw:
            return ens_pred
        else:
            ens_pred = ens_pred.mean(dim=1)
    else:
        ens_pred = coor_pred[0]

    return ens_pred




if __name__ == '__main__':
    # for testing
    pass
