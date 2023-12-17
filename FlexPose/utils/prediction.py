import os
import shutil
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))

import argparse
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from collections import defaultdict
from ray.util.multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_scatter import scatter_min, scatter_add

import pyrosetta
opts = '-mute true -ignore_unrecognized_res true'
pyrosetta.distributed.init(opts)


from FlexPose.model.layers import FlexPose
from FlexPose.utils.common import *
from FlexPose.preprocess.prepare_for_training import try_prepare_task
from FlexPose.utils.APOPDBbind_data import pred_ens
from FlexPose.utils.pdbbind_preprocess import *
from FlexPose.utils.data_utils import *
from FlexPose.model.MMFF import MMFF_keys, MMFF_pad_dim, get_MMFF_param
if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange



def set_device(device):
    if device == 'cpu':
        torch.set_num_threads(16)
    else:
        torch.cuda.set_device(device)


def get_torsion_from_pose(pose):
    bb_torsion = []
    sc_torsion = []
    for i in range(1, pose.size() + 1):
        try:
            res = pose.residue(i)
            assert res.name3() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
            phi_psi = [pose.phi(i), pose.psi(i)]
            chi = [c for c in res.chi()]
            bb_torsion.append(phi_psi)
            sc_torsion.append(chi)
        except:
            bb_torsion.append([None])
            sc_torsion.append([None])
    return {'bb_torsion': bb_torsion, 'sc_torsion': sc_torsion}


def prepare_single_input(tupin):
    f_name_list, idx, cache_path = tupin
    p_path, l_path, ref_path = f_name_list

    max_len_ligand = 150
    max_len_pocket = 150

    # =========== ligand encoding ===========
    ligand_mol = read_rdkit_mol(l_path)
    if l_path.endswith('mol2'):
        ligand_template = ligand_mol
    else:
        mol2 = '.'.join(l_path.split('.')[:-1]) + '.mol2'
        if os.path.exists(mol2):
            try:
                ligand_template = Chem.MolFromMol2File(mol2)
                ligand_mol = AllChem.AssignBondOrdersFromTemplate(ligand_template, ligand_mol)
                print(f'Found mol2 {mol2} as input.')
            except:
                ligand_template = ligand_mol
        else:
            ligand_template = ligand_mol
    if ligand_mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(ligand_mol, maxAttempts=10, useRandomCoords=True, clearConfs=False)
        ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
            ligand_mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(ligand_mol))
        for atom_i in range(ligand_mol.GetNumAtoms()):
            ff.MMFFAddPositionConstraint(atom_i, 1, 100)  # maxDispl: maximum displacement
        ff.Minimize(maxIts=20)

    try:
        dic_MMFF_param = get_MMFF_param(ligand_template)
    except:
        dic_MMFF_param = None


    ligand_node_features = get_node_feature(ligand_template, 'ligand')
    ligand_edge, ligand_edge_features = get_ligand_edge_feature(ligand_template)
    ligand_match = get_ligand_match(ligand_template)
    ligand_dismap = get_ligand_unrotable_distance(ligand_template)  # not use in our model
    ligand_coor_true = get_true_posi(ligand_mol)
    ligand_coor_true = ligand_coor_true[get_ligand_match(ligand_mol, ligand_template)[0]]

    ligand_data = [ligand_node_features, ligand_edge_features, ligand_coor_true, ligand_match, ligand_dismap]
    assert len(ligand_node_features) <= max_len_ligand, 'ligand atoms need less than 150'

    # =========== protein encoding ===========
    # load modeller again for ray
    from modeller import Environ
    from modeller.scripts import complete_pdb
    with suppress_stdout_stderr():
        env_ = Environ()
        env_.libs.topology.read(file='$(LIB)/top_heav.lib')
        env_.libs.parameters.read(file='$(LIB)/par.lib')
    import pyrosetta

    fixed_protein_path = cache_path + f'/{idx}_protein_tmp.pdb'
    pdb_m = complete_pdb(env_, p_path)
    pdb_m.write(fixed_protein_path)

    opts = '-mute true -ignore_unrecognized_res true'
    pyrosetta.distributed.init(opts)
    pose = pyrosetta.io.pose_from_pdb(fixed_protein_path)
    dic_tor = get_torsion_from_pose(pose)

    ref_mol = read_rdkit_mol(ref_path, silence=True)
    ref_coor = get_true_posi(ref_mol)

    biodf_protein = PandasPdb().read_pdb(fixed_protein_path)
    df_protein = biodf_protein.df['ATOM']
    df_protein['chain_resi'] = df_protein['chain_id'].astype(str) + '_' + df_protein['residue_number'].astype(str)
    df_pocket, sele_res = get_pocket(df_protein, ref_coor, max_len_protein=max_len_pocket)
    SCtorsion_data = get_torsion(dic_tor, df_protein, df_pocket)
    protein_data = encode_pocket(df_pocket) + [SCtorsion_data]
    assert protein_data[0].shape[0] == SCtorsion_data[0].shape[0]
    assert protein_data[0].shape[0] <= max_len_pocket, 'pocket residues need less than 150'

    # os.remove(fixed_protein_path)

    dic_data = dict(
        ligand_data=ligand_data,
        protein_data=protein_data,
        protein_path=fixed_protein_path,
        ligand_path=l_path,
        sele_res=sele_res,
        dic_MMFF_param=dic_MMFF_param,
    )
    pickle.dump(dic_data, open(cache_path + '/{}.pkl'.format(idx), 'wb'))
    return True


def preprare_input_data(input_list, cache_path, prepare_data_with_multi_cpu):
    delmkdir(cache_path)

    tasks = []
    for idx, f_name_list in enumerate(input_list):
        tasks.append((prepare_single_input, (f_name_list, idx, cache_path)))

    fail = 0
    if prepare_data_with_multi_cpu:
        pool = Pool()
        print('Preparing input data...')
        for r in pool.map(try_prepare_task, tasks):
            if not r:
                fail += 1
    else:
        for task in tqdm(tasks, desc='Preparing input data'):
            r = try_prepare_task(task)
            if not r:
                fail += 1

    print(f'Prepared data: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')


def read_input(protein, ligand, ref_pocket_center, batch_csv):
    if batch_csv is not None:
        df_input = pd.read_csv(batch_csv)
        protein_list = df_input['protein'].values
        ligand_list = df_input['ligand'].values
        ref_pocket_center_list = df_input['ref_pocket_center'].values
    else:
        assert protein is not None and ligand is not None and ref_pocket_center is not None
        if not isinstance(protein, list):
            protein_list = [protein]
        else:
            protein_list = protein
        if not isinstance(ligand, list):
            ligand_list = [ligand]
        else:
            ligand_list = ligand
        if not isinstance(ref_pocket_center, list):
            ref_pocket_center_list = [ref_pocket_center]
        else:
            ref_pocket_center_list = ref_pocket_center

    input_list = [(i, j, k) for i, j, k in zip(protein_list, ligand_list, ref_pocket_center_list)]
    return input_list


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, args, cache_path, ens=1):
        self.data_path = cache_path
        self.data_list = [i.split('.')[0] for i in os.listdir(cache_path) if i.endswith('.pkl')]
        self.ens = ens

        self.coor_scale = args.coor_scale

        self.max_len_pocket = args.max_len_pocket
        self.max_len_ligand = args.max_len_ligand

        self.l_init_sigma = args.l_init_sigma / self.coor_scale

    def __getitem__(self, i):
        if self.ens > 1:
            complex_graph = []
            for e in range(self.ens):
                complex_graph.append(self.get_complex(self.data_list[i]))
            complex_graph = collate_input(complex_graph)  # use collate_dummy in loader
        else:
            complex_graph = self.get_complex(self.data_list[i])
        return complex_graph

    def get_complex(self, idx):
        # =============== get dict data ===============
        dic_data = pickle.load(open(f'{self.data_path}/{idx}.pkl', 'rb'))
        ligand_data = dic_data['ligand_data']
        protein_data = dic_data['protein_data']
        protein_path = dic_data['protein_path']
        ligand_path = dic_data['ligand_path']
        sele_res = dic_data['sele_res']
        dic_MMFF_param = dic_data['dic_MMFF_param']

        # =============== ligand ===============
        l_x_sca_init, l_edge_sca_init, l_coor_ref, l_match, l_dismap = ligand_data
        l_match = l_match.reshape(-1)
        n_match = len(l_match) // len(l_x_sca_init)
        l_nomatch = repeat(torch.arange(0, len(l_x_sca_init)), 'm -> (n m)', n=n_match)

        # get ligand MMFF (if exists)
        if dic_MMFF_param is not None:
            dic_MMFF_param = self.repad_MMFFparam(dic_MMFF_param, MMFF_keys, MMFF_pad_dim)
        else:
            dic_MMFF_param = {}
            for k, pad_dim in zip(MMFF_keys, MMFF_pad_dim):
                dic_MMFF_param[k + '_index'] = np.zeros((1, pad_dim[0]))
                dic_MMFF_param[k + '_param'] = np.zeros((1, pad_dim[1]))
                dic_MMFF_param[k + '_batch'] = np.zeros(1)

        # =============== protein ===============
        p_x_sca_init, p_x_vec_init, resi_connect, p_coor_init, CB_coor, MCSC_coor, MCSC_mask, tor_data = protein_data
        input_SCtorsion, _, _ = tor_data  # additional init tor input

        # =============== to tensor ===============
        # pocekt
        p_x_sca_init = torch.from_numpy(p_x_sca_init).float()
        p_x_vec_init = torch.from_numpy(p_x_vec_init).float()
        p_coor_init = torch.from_numpy(p_coor_init).float()

        resi_connect = torch.from_numpy(resi_connect).float()
        CB_coor = torch.from_numpy(CB_coor).float()
        MCSC_coor = torch.from_numpy(MCSC_coor).float()
        input_SCtorsion = torch.from_numpy(input_SCtorsion).float()

        # ligand
        l_x_sca_init = torch.from_numpy(l_x_sca_init).float()
        l_edge_sca_init = torch.from_numpy(l_edge_sca_init).float()
        l_dismap = torch.from_numpy(l_dismap).float()
        l_match = torch.from_numpy(l_match).long()
        l_coor_ref = torch.from_numpy(l_coor_ref).float()

        dic_MMFF_param = {k: torch.from_numpy(v).long() if 'index' in k or 'batch' in k else torch.from_numpy(v).float()
                      for k, v in dic_MMFF_param.items()}

        # =============== length ===============
        len_pocket = len(p_x_sca_init)
        len_ligand = len(l_x_sca_init)

        # =============== scale ===============
        p_coor_init = p_coor_init / self.coor_scale
        CB_coor = CB_coor / self.coor_scale  # input CB
        MCSC_coor = MCSC_coor / self.coor_scale
        p_x_vec_init = p_x_vec_init / self.coor_scale
        l_dismap = l_dismap / self.coor_scale
        input_SCtorsion = input_SCtorsion * np.pi / 180

        # =============== init pocket info ===============
        # CA CB N C and SC
        MCCACB_coor = torch.cat([p_coor_init.unsqueeze(-2), CB_coor.unsqueeze(-2), MCSC_coor[:, :2, :]], dim=1)
        SC_fromtor_vec = p_x_vec_init  # CB-CA, CG-CB, ...
        SC_fromCA_vec = torch.where(MCSC_coor[:, 3:] == 0, 0 * MCSC_coor[:, 3:],
                                    MCSC_coor[:, 3:] - p_coor_init.unsqueeze(-2))

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

        # =============== init ligand info ===============
        l_coor_init = p_coor_init.mean(dim=0) + self.l_init_sigma * torch.randn(len_ligand, 3)
        l_x_vec_init = (l_coor_init - l_coor_init.mean(dim=0, keepdims=True)).unsqueeze(dim=-2)
        l_edge_vec_init = (l_coor_init.unsqueeze(dim=-2) - l_coor_init.unsqueeze(dim=-3)).unsqueeze(dim=-2)

        # =============== pretrain mask ===============
        l_x_sca_init = F.pad(l_x_sca_init, (0, 1), 'constant', 0)
        l_edge_sca_init = F.pad(l_edge_sca_init, (0, 1), 'constant', 0)
        p_x_sca_init = F.pad(p_x_sca_init, (0, 1), 'constant', 0)

        # =============== output ===============
        dic_data = dict(
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

            # coor
            p_coor_init=p_coor_init,
            l_coor_init=l_coor_init,

            # suppl
            l_match=l_match,
            l_nomatch=l_nomatch,
            input_SCtorsion=input_SCtorsion,
            l_coor_ref=l_coor_ref,
            len_pocket=len_pocket,
            len_ligand=len_ligand,

            coor_scale=self.coor_scale,

            idx=idx,
            protein_path=protein_path,
            ligand_path=ligand_path,
            sele_res=sele_res,

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



def collate_input(batch_list):
    max_len_pocket = 0
    max_len_ligand = 0
    for g in batch_list:
        max_len_pocket = max(max_len_pocket, g['len_pocket'])
        g['p_x_mask'] = torch.ones((g['len_pocket']))
        g['p_edge_mask'] = torch.ones((g['len_pocket'], g['len_pocket']))
        max_len_ligand = max(max_len_ligand, g['len_ligand'])
        g['l_x_mask'] = torch.ones((g['len_ligand']))
        g['l_edge_mask'] = torch.ones((g['len_ligand'], g['len_ligand']))

    dic_data = {}

    # feat, coor and mask
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
                                  'l_x_sca_init', 'l_coor_init', 'l_coor_ref',
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
                              collect_dim=-3, data_type='2d', output_dtype=torch.float,
                              value=-1 / batch_list[0]['coor_scale']))
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

    # additional tor
    # for input torsion
    input_SCtorsion = torch.stack([
        F.pad(g['input_SCtorsion'], (0, 0, 0, max_len_pocket - g['input_SCtorsion'].shape[0]),
              'constant', 0) for g in batch_list], dim=0)
    sc_in = input_SCtorsion
    sc_in_mask = (sc_in != 0).float().unsqueeze(dim=-2)
    sc_in = torch.stack([sc_in, sc_in.sin(), sc_in.cos()], dim=-2)
    sc_in = sc_in * sc_in_mask  # + (1 - sc_in_mask) * (-99)
    dic_data['sc_in'] = rearrange(sc_in, 'b n d c -> b n (d c)')  # sc_in[:, :, :4] is input tor

    # for MMFF
    dic_MMFF_param = {}
    for k in MMFF_keys:
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

    # suppl
    dic_data['len_pocket'] = torch.Tensor([g['len_pocket'] for g in batch_list])
    dic_data['len_ligand'] = torch.Tensor([g['len_ligand'] for g in batch_list])

    dic_data['idx'] = [g['idx'] for g in batch_list]
    dic_data['protein_path'] = [g['protein_path'] for g in batch_list]
    dic_data['ligand_path'] = [g['ligand_path'] for g in batch_list]
    dic_data['sele_res'] = [g['sele_res'] for g in batch_list]
    dic_data['coor_scale'] = batch_list[0]['coor_scale']

    # use Data in PyG
    complex_graph = torch_geometric.data.Data(**dic_data)
    return complex_graph


def pred_model_conf(ens_pred_coor):
    n_ens = ens_pred_coor.shape[1]
    top_k = 10

    atom_d_calc = rearrange(
        (rearrange(ens_pred_coor, 'n e c -> n e () c') - rearrange(ens_pred_coor, 'n e c -> n () e c')) ** 2,
        'n i j c -> n (i j) c').sum(-1)
    atom_d_calc = np.triu(atom_d_calc, 1)
    index_array = np.argpartition(-atom_d_calc, top_k - 1, axis=1)
    topk_indices = np.take(index_array, np.arange(n_ens * n_ens), axis=1)
    topk_values = np.take_along_axis(atom_d_calc, topk_indices, axis=1)
    atom_d_calc = topk_values[:, :top_k].mean(-1)
    atom_d_calc = np.exp(-atom_d_calc)
    mol_d_calc = atom_d_calc.mean()

    return (mol_d_calc, atom_d_calc)


@torch.no_grad()
def get_rmsd_to_ref(tup_pred, g, ens):
    if ens == 1:
        _, coor_pred = (tup_pred[0][:, :, -1, :] * g.coor_scale).split(
            [g.len_pocket.max(dim=-1)[0].int(), g.len_ligand.max(dim=-1)[0].int()], dim=1)
        coor_pred = rearrange(coor_pred, 'b n c -> (b n) c')[g.l_x_mask.reshape(-1).bool()][g.l_match]  # to (batch*n_atom*match, 3)
        coor_ref = rearrange(g.l_coor_ref, 'b n c -> (b n) c')[g.l_x_mask.reshape(-1).bool()][g.l_nomatch]  # to (batch*n_atom*match, 3)
        coor_loss = ((coor_pred - coor_ref) ** 2).sum(dim=-1)  # to (batch*n_atom*match,)
        coor_loss = scatter_add(coor_loss, g.scatter_ligand_1, dim=0)  # to (batch*match,)
        coor_loss = scatter_min(coor_loss, g.scatter_ligand_2, dim=0)[0]  # to (batch,)
        rmsd_value = (coor_loss / g.len_ligand) ** 0.5

    else:
        coor_pred = g.ens_pred
        coor_ref = g.l_coor_ref[0]
        coor_pred = coor_pred[g.l_match.reshape(ens, -1)[0]]
        coor_ref = coor_ref[g.l_nomatch.reshape(ens, -1)[0]]
        coor_loss = ((coor_pred - coor_ref) ** 2).sum(dim=-1)
        coor_loss = scatter_add(coor_loss, g.scatter_ligand_1.reshape(ens, -1)[0], dim=0)
        coor_loss = scatter_min(coor_loss, g.scatter_ligand_2.reshape(ens, -1)[0], dim=0)[0]
        rmsd_value = (coor_loss / g.len_ligand[[0]]) ** 0.5

    return rmsd_value


@torch.no_grad()
def save_struct(
        l_coor,
        SC_pred,
        sc_in,
        protein_path,
        ligand_path,
        sele_res,
        len_pocket,
        len_ligand,
        idx,
        structure_output_path,
        cache_path='./cache',
        conf_atom=None,
        rdkit_min=False,
        env_min=False,
        save_mol=False,
        save_only_pocket=False,
):
    len_pocket = len_pocket.int()
    len_ligand = len_ligand.int()
    l_coor = l_coor.detach().cpu().numpy()
    SC_pred = SC_pred.detach().cpu().numpy()

    output_struct_file = f'{structure_output_path}/{idx}.pdb'

    # =============== protein ===============
    sc_num = (sc_in[..., :4] != 0).sum(-1)
    sc_deg_pred_list = [[tan_2_deg(s[i, 0], s[i, 1]) * 180 / np.pi
                         for i in range(n)] for s, n in zip(SC_pred[:len_pocket], sc_num)]

    df_protein = PandasPdb().read_pdb(protein_path).df['ATOM']
    df_protein['chain_res'] = [str(df_protein.loc[i, 'residue_number']) + ',' + str(df_protein.loc[i, 'chain_id']) for i
                               in range(len(df_protein))]
    sele_res = df_protein[df_protein['chain_res'].apply(lambda x: True if x in sele_res else False) &
                          (df_protein['atom_name'] == 'CA')]['chain_res']
    pose = pyrosetta.io.pose_from_pdb(protein_path)
    for chain_res, sc_deg_pred in zip(sele_res, sc_deg_pred_list):
        res_idx, chain_idx = chain_res.split(',')
        res = pose.residue(pose.pdb_info().pdb2pose(chain_idx, int(res_idx)))
        for i, deg in enumerate(sc_deg_pred):
            res.set_chi(i + 1, deg)
    pose.dump_pdb(f'{cache_path}/protein.pdb')

    # =============== ligand ===============
    ligand_mol = read_rdkit_mol(ligand_path)
    ligand_mol = assign_struct(ligand_mol, l_coor, min=rdkit_min)

    if save_only_pocket:
        biodf_protein = PandasPdb().read_pdb(f'{cache_path}/protein.pdb')
        df_protein = biodf_protein.df['ATOM']
        df_protein['chain_resi'] = df_protein['chain_id'].astype(str) + '_' + df_protein['residue_number'].astype(str)
        df_pocket, _ = get_pocket(df_protein, get_true_posi(ligand_mol), max_len_protein=50)
        del df_pocket['chain_resi']
        biodf_protein.df['ATOM'] = df_pocket
        biodf_protein.to_pdb(f'{cache_path}/protein.pdb', records=['ATOM'])

    complex_mol = Chem.CombineMols(Chem.MolFromPDBFile(f'{cache_path}/protein.pdb', sanitize=False), ligand_mol)
    if env_min:
        try:
            complex_mol = min_struct(complex_mol, len_ligand=ligand_mol.GetNumAtoms())
        except:
            print(f'Fail: env minimization for {idx}')

    lines = Chem.MolToPDBBlock(complex_mol).split('\n')
    if conf_atom is None:
        conf_atom = np.zeros(len_ligand)
    new_lines = []
    c = 0
    for line in lines:
        if 'UNL' in line:
            line = line[:62] + f'{conf_atom[c]:.2f}' + line[66:]
            c += 1
        new_lines.append(line)
    open(output_struct_file, 'w').write('\n'.join(new_lines))

    if save_mol:
        pickle.dump(complex_mol, open('.'.join(output_struct_file.split('.')[:-1]) + '.pkl', 'wb'))


@torch.no_grad()
def predict(
        param_path=None,  # model parameter path
        protein=None,  # protein path
        ligand=None,  # ligand path
        ref_pocket_center=None,  # ligand-like file path
        batch_csv=None,  # batch prediction
        model_conf=False,  # predict model confidence
        device='cuda:0',
        ens=1,  # ensemble number
        seed=42,
        batch_size=12,  # only work with ens=1
        num_workers=4,  # only work with ens=1
        prepare_data_with_multi_cpu=True,  # prepare inputs with multi-processing
        min=True,  # torch build-in minimization
        min_type='GD',  # ['GD', 'LBFGS']
        rdkit_min=False,  # rdkit minimization
        env_min=False,
        cache_path='./cache',
        output_structure=True,
        save_mol=False,
        save_only_pocket=False,
        structure_output_path='./structure_output',
        output_result_path=None,
        calc_rmsd=False,
        min_loop=100,  # minimization loop
        min_constraint=5,  # minimization constraint factor
):
    print('='*20, 'Running FlexPose', '='*20)

    if model_conf:
        assert ens >= 10, 'Model confidence requires ens > 1, Recommend ens >= 10'
    set_all_seed(seed)
    set_device(device)

    input_list = read_input(protein, ligand, ref_pocket_center, batch_csv)
    preprare_input_data(input_list, cache_path, prepare_data_with_multi_cpu)

    model = FlexPose(param_path=param_path).to(device)
    model.train(False)
    model.use_min = min
    infer_dataset = InferDataset(model.args, cache_path, ens=ens)
    infer_loader = torch.utils.data.DataLoader(
        dataset=infer_dataset, batch_size=batch_size if ens == 1 else 1,
        num_workers=num_workers, shuffle=False,
        pin_memory=False, persistent_workers=6, prefetch_factor=2,
        collate_fn=collate_input if ens == 1 else collate_dummy)

    dic_result = defaultdict(list)
    with torch.no_grad():
        for g in tqdm(infer_loader, desc='Predicting'):
            g = g.to(device)
            tup_pred = model.infer(g)

            if ens > 1:
                _, coor_pred = tup_pred[0][:, :, -1, :].split([
                    g.len_pocket.max(dim=-1)[0].int(), g.len_ligand.max(dim=-1)[0].int()], dim=1)
                aligned_pred = pred_ens(coor_pred, g, return_raw=True) * model.args.coor_scale
                ens_pred = aligned_pred.mean(dim=1)
                g.ens_pred = ens_pred

                if min:
                    ens_pred = repeat(ens_pred, 'n c -> (b n) c', b=ens)
                    ens_pred = model.coor_min_object(ens_pred, g,
                                                     loop=min_loop, constraint=min_constraint, min_type=min_type)

                dic_result['coor_pred'].append(ens_pred)
                dic_result['SC_pred'].append(tup_pred[3][0])
                dic_result['sc_in'].append(g.sc_in[0])
                dic_result['protein_path'].append(g.protein_path[0])
                dic_result['ligand_path'].append(g.ligand_path[0])
                dic_result['sele_res'].append(g.sele_res[0])
                dic_result['len_pocket'].append(g.len_pocket[0])
                dic_result['len_ligand'].append(g.len_ligand[0])
                dic_result['idx'].append(int(g.idx[0]))
                dic_result['aff'].append(tup_pred[2].mean().item() * model.args.aff_scale)
                if model_conf:
                    conf_mol, conf_atom = pred_model_conf(aligned_pred.detach().cpu().numpy())
                    dic_result['conf_atom'].append(conf_atom)
                    dic_result['conf_mol'].append(conf_mol.item())

            else:
                if min:
                    g.coor = tup_pred[0][:, :, -1, :]
                    g = model.energy_min(g, loop=min_loop, constraint=min_constraint, min_type=min_type)
                    last_frame = g.coor
                else:
                    last_frame = tup_pred[0][:, :, -1, :]

                for j in range(len(g.idx)):
                    _, coor_pred = last_frame[j].split([
                        g.len_pocket.max(dim=-1)[0].int(), g.len_ligand.max(dim=-1)[0].int()], dim=0)
                    dic_result['coor_pred'].append(coor_pred * model.args.coor_scale)
                    dic_result['SC_pred'].append(tup_pred[3][j])
                    dic_result['sc_in'].append(g.sc_in[j])
                    dic_result['protein_path'].append(g.protein_path[j])
                    dic_result['ligand_path'].append(g.ligand_path[j])
                    dic_result['sele_res'].append(g.sele_res[j])
                    dic_result['len_pocket'].append(g.len_pocket[j])
                    dic_result['len_ligand'].append(g.len_ligand[j])
                    dic_result['idx'].append(int(g.idx[j]))
                    dic_result['aff'].append(tup_pred[2][j].item() * model.args.aff_scale)

            if calc_rmsd:
                dic_result['rmsd'] += get_rmsd_to_ref(tup_pred, g, ens).detach().cpu().tolist()

    if output_structure:
        delmkdir(structure_output_path)
        for i in trange(len(dic_result['idx']), desc='Saving files'):
            try:
                save_struct(
                    dic_result['coor_pred'][i],
                    dic_result['SC_pred'][i],
                    dic_result['sc_in'][i],
                    dic_result['protein_path'][i],
                    dic_result['ligand_path'][i],
                    dic_result['sele_res'][i],
                    dic_result['len_pocket'][i],
                    dic_result['len_ligand'][i],
                    dic_result['idx'][i],
                    structure_output_path,
                    cache_path=cache_path,
                    conf_atom=dic_result['conf_atom'][i] if 'conf_atom' in dic_result.keys() and model_conf else None,
                    rdkit_min=rdkit_min,
                    env_min=env_min,
                    save_mol=save_mol,
                    save_only_pocket=save_only_pocket,
                )
            except:
                print(f"Failure in saving structure: {dic_result['idx'][i]}")

    dic_output = defaultdict(list)
    for i, pl in enumerate(input_list):
        p, l, _ = pl
        dic_output['protein'].append(p)
        dic_output['ligand'].append(l)
    df_output = pd.DataFrame.from_dict(dic_output)

    loc = dic_result['idx']
    df_output['predicted_affinity'] = np.nan
    df_output.loc[loc, 'predicted_affinity'] = dic_result['aff']
    if model_conf:
        df_output['model_confidence'] = np.nan
        df_output.loc[loc, 'model_confidence'] = dic_result['conf_mol']
    if calc_rmsd:
        df_output['RMSD'] = np.nan
        df_output.loc[loc, 'RMSD'] = dic_result['rmsd']

    if output_result_path is not None:
        df_output.to_csv(output_result_path, index=True)

    shutil.rmtree(cache_path)

    print('DONE!')
    print('=' * 20, 'Exit FlexPose', '=' * 20)



if __name__ == '__main__':
    pass










