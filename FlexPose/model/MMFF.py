import os
import sys

import random

sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter_min, scatter_mean, scatter_add
from einops import rearrange, repeat
from rdkit import Chem
from rdkit.Chem import AllChem


from model.MMFF_supply import dic_atom2lin


MMFF_keys = ['BS', 'AB', 'SB', 'OOP', 'TOR', 'VDW', 'ELE']
MMFF_pad_dim = [[2, 5], [3, 8], [3, 5], [4, 1], [4, 4], [2, 5], [2, 2]]  # index param


def pad_zero_param(index, param, index_dim, param_dim):
    if len(index) == 0:
        index = [list(range(index_dim))]
        param = [[0]*param_dim]
    return index, param


def get_BS_param(mol, props, dic_MMFF_param, dic_tmp):
    # bond stretching
    bond_list = np.array([[x.GetBeginAtomIdx(), x.GetEndAtomIdx()] for x in mol.GetBonds()])
    BS_index = bond_list
    # kb, rij0
    BS_param = []
    for i, j in BS_index:
        p = props.GetMMFFBondStretchParams(mol, int(i), int(j))
        BS_param.append([p[1], p[2]])
    '''
    drij = rij - rij0
    energy = A*drij**2 * (B + C*drij + D*drij**2)
    '''
    rij0 = np.array([rij0 for kb, rij0 in BS_param])
    A = np.array([143.9325*kb/2 for kb, r0 in BS_param])
    B = np.array([1 for kb, r0 in BS_param])
    C = np.array([-2 for kb, r0 in BS_param])
    D = np.array([7/12*(-2)**2 for kb, r0 in BS_param])
    BS_param = np.stack([rij0, A, B, C, D], axis=-1)

    BS_index, BS_param = pad_zero_param(BS_index, BS_param, 2, 5)
    dic_MMFF_param['BS_index'] = BS_index
    dic_MMFF_param['BS_param'] = BS_param
    dic_tmp['bond_list'] = bond_list
    return dic_MMFF_param, dic_tmp


def get_angle_by_bond(bond_list):
    tmp = []
    n_bond = len(bond_list)
    for l in range(n_bond):
        i, j = bond_list[l]
        for r in range(l+1, n_bond):
            p, q = bond_list[r]
            if i == p:
                tmp.append([q, i, j])
            elif i == q:
                tmp.append([p, i, j])
            elif j == p:
                tmp.append([i, j, q])
            elif j == q:
                tmp.append([i, j, p])
    angle_list = []
    for angle in tmp:
        if (angle not in angle_list) and (angle[::-1] not in angle_list):
            angle_list.append(angle)
    return np.array(angle_list)


def get_AB_param(mol, props, dic_MMFF_param, dic_tmp):
    # angle bending
    angle_list = get_angle_by_bond(dic_tmp['bond_list'])
    AB_index = angle_list
    # lin, ka, theta0
    AB_param = []
    for i, j, k in AB_index:
        p = props.GetMMFFAngleBendParams(mol, int(i), int(j), int(k))
        AB_param.append([dic_atom2lin[props.GetMMFFAtomType(int(j))], p[1], p[2]])
    '''
    dtheta = theta - theta0 (degree)
    lin1 = 1 if center atom is not linear else 0
    lin2 = 1 if center atom is linear else 0
    energy = lin1 * (A*dtheta**2 * (B + C*dtheta)) + lin2 * (D + E*cos(dtheta*pi/180))
    '''
    lin_1_list = np.array([1 if lin==0 else 0 for lin, ka, theta0 in AB_param])
    lin_2_list = np.array([1 if lin!=0 else 0 for lin, ka, theta0 in AB_param])
    theta0 = np.array([theta0 for lin, ka, theta0 in AB_param])
    A = np.array([0.043844*ka/2 for lin, ka, theta0 in AB_param])
    B = np.array([1 for lin, ka, theta0 in AB_param])
    C = np.array([-0.006981317 for lin, ka, theta0 in AB_param])
    D = np.array([143.9325*ka for lin, ka, theta0 in AB_param])
    E = np.array([143.9325*ka for lin, ka, theta0 in AB_param])
    AB_param = np.stack([lin_1_list, lin_2_list, theta0, A, B, C, D, E], axis=-1)

    AB_index, AB_param = pad_zero_param(AB_index, AB_param, 3, 8)
    dic_MMFF_param['AB_index'] = AB_index
    dic_MMFF_param['AB_param'] = AB_param
    dic_tmp['angle_list'] = angle_list
    return dic_MMFF_param, dic_tmp


def filter_SB(mol, props, SB_index):
    new_index = []
    for i, j, k in SB_index:
        flag = props.GetMMFFStretchBendParams(mol, int(i), int(j), int(k))
        if flag != None:
            new_index.append([i, j, k])
    return new_index


def get_SB_param(mol, props, dic_MMFF_param, dic_tmp):
    # stretch-bend
    angle_list = dic_tmp['angle_list']
    SB_index = filter_SB(mol, props, angle_list)
    # kbaIJK, kbaKJI, r0ij, r0kj, theta0
    SB_param = []
    for i, j, k in SB_index:
        p1 = props.GetMMFFStretchBendParams(mol, int(i), int(j), int(k))
        p2 = props.GetMMFFBondStretchParams(mol, int(i), int(j))
        p3 = props.GetMMFFBondStretchParams(mol, int(k), int(j))
        p4 = props.GetMMFFAngleBendParams(mol, int(i), int(j), int(k))
        SB_param.append([p1[1], p1[2], p2[2], p3[2], p4[2]])
    '''
    drij = rij - r0ij
    drkj = rkj - r0kj
    dtheta = theta - theta0 (degree)
    energy = (A*drij + B*drkj) * dtheta
    '''
    r0ij = np.array([r0ij for kbaIJK, kbaKJI, r0ij, r0kj, theta0 in SB_param])
    r0kj = np.array([r0kj for kbaIJK, kbaKJI, r0ij, r0kj, theta0 in SB_param])
    theta0 = np.array([theta0 for kbaIJK, kbaKJI, r0ij, r0kj, theta0 in SB_param])
    A = np.array([2.51210*kbaIJK for kbaIJK, kbaKJI, r0ij, r0kj, theta0 in SB_param])
    B = np.array([2.51210*kbaKJI for kbaIJK, kbaKJI, r0ij, r0kj, theta0 in SB_param])
    SB_param = np.stack([r0ij, r0kj, theta0, A, B], axis=-1)

    SB_index, SB_param = pad_zero_param(SB_index, SB_param, 3, 5)
    dic_MMFF_param['SB_index'] = SB_index
    dic_MMFF_param['SB_param'] = SB_param
    return dic_MMFF_param, dic_tmp


def get_dict_bond(bond_list):
    dic_bond = {}
    for i, j in bond_list:
        if i not in dic_bond.keys():
            dic_bond[i] = []
        if j not in dic_bond.keys():
            dic_bond[j] = []
        dic_bond[i].append(j)
        dic_bond[j].append(i)
    return dic_bond


def get_oop(mol, props, dic_bond, angle_list):
    oop_list_repeat = []
    for i, j, k in angle_list:
        if len(dic_bond[j]) < 3:
            continue
        for l in dic_bond[j]:
            if l == i or l == k:
                continue
            else:
                try_param = props.GetMMFFOopBendParams(mol, int(i), int(j), int(k), int(l))
                if try_param != None:
                    oop_list_repeat.append([i, j, k, l])
    oop_list = []
    for i, j, k, l in oop_list_repeat:
        if ([i, j, k, l] not in oop_list) and ([k, j, i, l] not in oop_list):
            oop_list.append([i, j, k, l])
    return oop_list


def get_OOP_param(mol, props, dic_MMFF_param, dic_tmp):
    # out-of-plane bending
    dic_bond = get_dict_bond(dic_tmp['bond_list'])
    oop_list = get_oop(mol, props, dic_bond, dic_tmp['angle_list'])
    OOP_index = oop_list
    # koop
    OOP_param = [props.GetMMFFOopBendParams(mol, int(i), int(j), int(k), int(l)) for i, j, k, l in OOP_index]
    '''
    chi: (degree)
    energy = A*chi**2
    '''
    A = np.array([0.043844*koop/2 for koop in OOP_param])
    OOP_param = np.stack([A], axis=-1)

    OOP_index, OOP_param = pad_zero_param(OOP_index, OOP_param, 4, 1)
    dic_MMFF_param['OOP_index'] = OOP_index
    dic_MMFF_param['OOP_param'] = OOP_param
    dic_tmp['dic_bond'] = dic_bond
    dic_tmp['oop_list'] = oop_list
    return dic_MMFF_param, dic_tmp


def get_torsion(mol, props, dic_bond, angle_list):
    torsion_list_repeat = []
    for i, j, k in angle_list:
        if len(dic_bond[i]) > 1:
            for l in dic_bond[i]:
                if l == j:
                    continue
                else:
                    torsion_list_repeat.append([l, i, j, k])
        if len(dic_bond[k]) > 1:
            for l in dic_bond[k]:
                if l == j:
                    continue
                else:
                    torsion_list_repeat.append([i, j, k, l])
    torsion_list = []
    for torsion in torsion_list_repeat:
        if (torsion not in torsion_list) and (torsion[::-1] not in torsion_list):
            i, j, k, l = torsion
            try_param = props.GetMMFFTorsionParams(mol, int(i), int(j), int(k), int(l))
            if try_param != None:
                torsion_list.append(torsion)
    return torsion_list


def get_TOR_param(mol, props, dic_MMFF_param, dic_tmp):
    # torsion
    torsion_list = get_torsion(mol, props, dic_tmp['dic_bond'], dic_tmp['angle_list'])
    TOR_index = torsion_list
    # V1, V2, V3
    TOR_param = []
    for i, j, k, l in TOR_index:
        p = props.GetMMFFTorsionParams(mol, int(i), int(j), int(k), int(l))
        TOR_param.append([p[1], p[2], p[3]])
    '''
    cos1: cos 1 dihedral
    cos2: cos 2 dihedral
    cos3: cos 3 dihedral
    energy = A + B*cos1 + C*cos2 + D*cos3
    '''
    A = np.array([0.5*(V1+V2+V3) for V1, V2, V3 in TOR_param])
    B = np.array([0.5*V1 for V1, V2, V3 in TOR_param])
    C = np.array([0.5*(-V2) for V1, V2, V3 in TOR_param])
    D = np.array([0.5*V3 for V1, V2, V3 in TOR_param])
    TOR_param = np.stack([A, B, C, D], axis=-1)

    TOR_index, TOR_param = pad_zero_param(TOR_index, TOR_param, 4, 4)
    dic_MMFF_param['TOR_index'] = TOR_index
    dic_MMFF_param['TOR_param'] = TOR_param
    dic_tmp['torsion_list'] = torsion_list
    return dic_MMFF_param, dic_tmp


def get_14(mol):
    top_dismap = Chem.rdmolops.GetDistanceMatrix(mol)
    tmp = [[i, l] for i in range(mol.GetNumAtoms())
           for l in range(i+1, mol.GetNumAtoms())]
    pair_14_list = []
    for i, l in tmp:
        if top_dismap[i, l] == 3:
            pair_14_list.append([i, l])
    return pair_14_list


def get_noncov_pair(mol, bond_list, angle_list, pair_14_list):
    all_pair = [[i, j] for i in range(mol.GetNumAtoms()) for j in range(i+1, mol.GetNumAtoms())]
    pair_12 = [[i, j] for i, j in bond_list]
    pair_13 = [[i, k] for i, j, k in angle_list]
    pair_12_13 = pair_12 + pair_13
    pair_14 = pair_14_list
    nonconv_pair = []
    flag_14_list = []
    for pair in all_pair:
        if pair in pair_12_13 or pair[::-1] in pair_12_13:
            continue
        nonconv_pair.append(pair)
        if pair in pair_14 or pair[::-1] in pair_14:
            flag_14_list.append(True)
        else:
            flag_14_list.append(False)
    return nonconv_pair, flag_14_list


def get_VDW_param(mol, props, dic_MMFF_param, dic_tmp, select_pair_index=None):
    # van der Waals
    if select_pair_index==None:
        pair_14_list = get_14(mol)
        nonconv_pair, flag_14_list = get_noncov_pair(mol,
                                                     dic_tmp['bond_list'],
                                                     dic_tmp['angle_list'],
                                                     pair_14_list)
    else:
        nonconv_pair = select_pair_index
        flag_14_list = [False] * len(select_pair_index)
    VDW_index = nonconv_pair
    # (R_ij_starUnscaled), (epsilonUnscaled), R_ij_star, epsilon
    VDW_param = []
    for i, j in VDW_index:
        p = props.GetMMFFVdWParams(int(i), int(j))
        VDW_param.append([p[2], p[3]])
    '''
    rij: ...
    energy = A / (rij + B)**7 * (C / (rij**7 + D) + E)
    '''
    A = np.array([epsilon*(1.07*R_ij_star)**7 for R_ij_star, epsilon in VDW_param])
    B = np.array([0.07*R_ij_star for R_ij_star, epsilon in VDW_param])
    C = np.array([1.12*R_ij_star**7 for R_ij_star, epsilon in VDW_param])
    D = np.array([0.12*R_ij_star**7 for R_ij_star, epsilon in VDW_param])
    E = np.array([-2 for R_ij_star, epsilon in VDW_param])
    VDW_param = np.stack([A, B, C, D, E], axis=-1)

    VDW_index, VDW_param = pad_zero_param(VDW_index, VDW_param, 2, 5)
    dic_MMFF_param['VDW_index'] = VDW_index
    dic_MMFF_param['VDW_param'] = VDW_param
    dic_tmp['nonconv_pair'] = nonconv_pair
    dic_tmp['flag_14_list'] = flag_14_list
    return dic_MMFF_param, dic_tmp


def get_ELE_param(mol, props, dic_MMFF_param, dic_tmp):
    # electrostatic
    ELE_index = dic_tmp['nonconv_pair']
    flag_14_list = dic_tmp['flag_14_list']
    # flag_14, qi, qj (partial charge)
    ELE_param = [[0.75 if flag_14 else 1.,
                  props.GetMMFFPartialCharge(int(i)),
                  props.GetMMFFPartialCharge(int(j))] for (i, j), flag_14 in zip(ELE_index, flag_14_list)]
    '''
    qi: partial charge of node i
    qi: partial charge of node j
    rij: ...
    energy = A / (rij + B)
    '''
    A = np.array([332.07169*flag_14*qi*qj for flag_14, qi, qj in ELE_param])
    B = np.array([0.05 for flag_14, qi, qj in ELE_param])
    ELE_param = np.stack([A, B], axis=-1)

    ELE_index, ELE_param = pad_zero_param(ELE_index, ELE_param, 2, 2)
    dic_MMFF_param['ELE_index'] = ELE_index
    dic_MMFF_param['ELE_param'] = ELE_param
    return dic_MMFF_param, dic_tmp


def add_batch_info(dic_MMFF_param):
    dic_tmp = {}
    for k in dic_MMFF_param.keys():
        if 'index' in k:
            t = k.split('_')[0]
            len_t = len(dic_MMFF_param[k])
            dic_tmp[t+'_batch'] = [0]*len_t
    return {**dic_MMFF_param, **dic_tmp}


def get_MMFF_param(mol, props=None, strict=False):
    # mol = Chem.AddHs(mol)  # H is needed for formal charge estimation
    # mol = Chem.RemoveHs(mol)
    # AllChem.EmbedMolecule(mol)
    # AllChem.MMFFOptimizeMolecule(mol)

    props = AllChem.MMFFGetMoleculeProperties(mol) if props == None else props
    assert props != None
    # ignoreInterfragInteractions :
    # if true, nonbonded terms between
    # fragments will not be added to the forcefield
    # nonBondedThreshused to exclude long-range non-bonded
    # interactions (defaults to 100.0)
    if strict:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props,
                                               nonBondedThresh=100., confId=-1, ignoreInterfragInteractions=False)
        assert not isinstance(ff, type(None))
        # tol_e = ff.CalcEnergy()

    dic_MMFF_param = {}
    dic_tmp = {}
    dic_MMFF_param, dic_tmp = get_BS_param(mol, props, dic_MMFF_param, dic_tmp)
    dic_MMFF_param, dic_tmp = get_AB_param(mol, props, dic_MMFF_param, dic_tmp)
    dic_MMFF_param, dic_tmp = get_SB_param(mol, props, dic_MMFF_param, dic_tmp)
    dic_MMFF_param, dic_tmp = get_OOP_param(mol, props, dic_MMFF_param, dic_tmp)
    dic_MMFF_param, dic_tmp = get_TOR_param(mol, props, dic_MMFF_param, dic_tmp)
    dic_MMFF_param, dic_tmp = get_VDW_param(mol, props, dic_MMFF_param, dic_tmp)
    dic_MMFF_param, dic_tmp = get_ELE_param(mol, props, dic_MMFF_param, dic_tmp)
    dic_MMFF_param = add_batch_info(dic_MMFF_param)
    dic_MMFF_param = {k: np.array(v) for k, v in dic_MMFF_param.items()}
    return dic_MMFF_param


def get_MMFF_param_for_complex(protein_mol, ligand_mol):
    Combine_mol = Chem.CombineMols(protein_mol, ligand_mol)
    # Pre-condition Violation RingInfo not initialized
    try:
        props = AllChem.MMFFGetMoleculeProperties(Combine_mol)
    except:
        Chem.SanitizeMol(Combine_mol)
        props = AllChem.MMFFGetMoleculeProperties(Combine_mol)

    dic_MMFF_param = {}
    dic_tmp = {}
    select_pair_index = [[i, j] for i in range(protein_mol.GetNumAtoms())
                         for j in range(protein_mol.GetNumAtoms(), Combine_mol.GetNumAtoms())]
    dic_MMFF_param, dic_tmp = get_VDW_param(Combine_mol, props, dic_MMFF_param, dic_tmp,
                                            select_pair_index=select_pair_index)
    dic_MMFF_param, dic_tmp = get_ELE_param(Combine_mol, props, dic_MMFF_param, dic_tmp)
    dic_MMFF_param = add_batch_info(dic_MMFF_param)
    dic_MMFF_param = {k: np.array(v) for k, v in dic_MMFF_param.items()}
    return dic_MMFF_param


class MMFFLoss():
    def __init__(self, split_interact=False, warm=False):
        super(MMFFLoss, self).__init__()
        self.MMFF_keys = ['BS', 'AB', 'SB', 'OOP', 'TOR', 'VDW', 'ELE']
        self.split_interact = split_interact
        if split_interact:  # VDW1: p2l; VDW2: l2l, same for ELE
            self.MMFF_keys = ['BS', 'AB', 'SB', 'OOP', 'TOR', 'VDW1', 'VDW2', 'ELE1', 'ELE2']

        self.warm = warm

    def __call__(self, coor, MMFF_param, return_sum=True):
        '''
        get energy by coordinates, with pre-calculated parameters
        :param coor: batched ligand coor
        :param MMFF_param: batched MMFF_param
        :return: total energy
        '''

        dic_e = {}
        dic_e['BS'] = self.get_BS(coor, MMFF_param.BS_index, MMFF_param.BS_param, MMFF_param.BS_batch)
        dic_e['AB'] = self.get_AB(coor, MMFF_param.AB_index, MMFF_param.AB_param, MMFF_param.AB_batch)
        dic_e['SB'] = self.get_SB(coor, MMFF_param.SB_index, MMFF_param.SB_param, MMFF_param.SB_batch)
        dic_e['OOP'] = self.get_OOP(coor, MMFF_param.OOP_index, MMFF_param.OOP_param, MMFF_param.OOP_batch)
        dic_e['TOR'] = self.get_TOR(coor, MMFF_param.TOR_index, MMFF_param.TOR_param, MMFF_param.TOR_batch)
        if self.split_interact:
            dic_e['VDW1'] = self.get_VDW(coor, MMFF_param.VDW1_index, MMFF_param.VDW1_param, MMFF_param.VDW1_batch)
            dic_e['VDW2'] = self.get_VDW(coor, MMFF_param.VDW2_index, MMFF_param.VDW2_param, MMFF_param.VDW2_batch)
            dic_e['ELE1'] = self.get_ELE(coor, MMFF_param.ELE1_index, MMFF_param.ELE1_param, MMFF_param.ELE1_batch)
            dic_e['ELE2'] = self.get_ELE(coor, MMFF_param.ELE2_index, MMFF_param.ELE2_param, MMFF_param.ELE2_batch)
        else:
            dic_e['VDW'] = self.get_VDW(coor, MMFF_param.VDW_index, MMFF_param.VDW_param, MMFF_param.VDW_batch)
            dic_e['ELE'] = self.get_ELE(coor, MMFF_param.ELE_index, MMFF_param.ELE_param, MMFF_param.ELE_batch)
        '''
        # for testing
        print(coor.shape)
        i = 0
        print(f'BS: {e_BS[i].sum().detach().cpu().numpy():.3f}; AB: {e_AB[i].sum().detach().cpu().numpy():.3f}; '
              f'SB: {e_SB[i].sum().detach().cpu().numpy():.3f}; OOP: {e_OOP[i].sum().detach().cpu().numpy():.3f}; '
              f'TOR: {e_TOR[i].sum().detach().cpu().numpy():.3f}; VDW: {e_VDW[i].sum().detach().cpu().numpy():.3f}; '
              f'ELE: {e_ELE[i].sum().detach().cpu().numpy():.3f}')
        '''
        if return_sum:
            tol_e = dic_e['BS'] + dic_e['AB'] + dic_e['SB'] + dic_e['OOP'] + dic_e['TOR']
            if self.split_interact:
                tol_e = tol_e + dic_e['VDW1'] + dic_e['VDW2'] + dic_e['ELE1'] + dic_e['ELE2']
            else:
                tol_e = tol_e + dic_e['VDW'] + dic_e['ELE']
            return tol_e
        else:
            return dic_e

    def get_BS(self, coor, BS_index, BS_param, BS_batch):
        '''
        bond stretching
        drij = rij - rij0
        energy = A*drij**2 * (B + C*drij + D*drij**2)
        :param coor: [n_atom, 3]
        :param BS_index: [n_bond, 2], i, j
        :param BS_param: [n_bond, 5], r0, A, B, C, D
        :param BS_batch: [n_bond]
        :return: energy
        '''
        i, j = self.split_to_single_dim(BS_index)
        r0, A, B, C, D = BS_param.split(1, dim=-1)
        rij = (coor[i] - coor[j]).norm(p=2, dim=-1, keepdim=True)
        drij = rij - r0
        if self.warm:
            e = drij ** 2 * A
            e = scatter_mean(e, BS_batch, dim=0)
        else:
            e = A*drij**2 * (B + C*drij + D*drij**2)
            e = scatter_add(e, BS_batch, dim=0)
        return e

    def get_AB(self, coor, AB_index, AB_param, AB_batch):
        '''
        angle bending
        dtheta = theta - theta0 (degree)
        lin1 = 1 if center atom is not linear else 0
        lin2 = 1 if center atom is linear else 0
        energy = lin1 * (A*dtheta**2 * (B + C*dtheta)) + lin2 * (D + E*cos(dtheta*pi/180))
        :param coor: [n_atom, 3]
        :param AB_index: [n_angle, 3], i, j, k
        :param AB_param: [n_angle, 5], lin1, lin2, theta0, A, B, C, D, E
        :param AB_batch: [n_angle]
        :return: energy
        '''
        theta, theta_cos = self.get_angle(coor, AB_index)
        lin1, lin2, theta0, A, B, C, D, E = AB_param.split(1, dim=-1)
        dtheta = theta - theta0
        if self.warm:
            e = dtheta ** 2 * A
            e = scatter_mean(e, AB_batch, dim=0)
        else:
            e = lin1 * (A*dtheta**2 * (B + C*dtheta)) + lin2 * (D + E*theta_cos)
            e = scatter_add(e, AB_batch, dim=0)
        return e

    def get_SB(self, coor, SB_index, SB_param, SB_batch):
        '''
        stretch-bend
        drij = rij - r0ij
        drkj = rkj - r0kj
        dtheta = theta - theta0 (degree)
        energy = (A*drij + B*drkj) * dtheta
        :param coor: [n_atom, 3]
        :param SB_index: [n_angle, 3], n_oop
        :param SB_param: [n_angle, 5], r0ij, r0kj, theta0, A, B
        :param SB_batch: [n_angle]
        :return: energy
        '''
        i, j, k = self.split_to_single_dim(SB_index)
        r0ij, r0kj, theta0, A, B = SB_param.split(1, dim=-1)
        rij = (coor[i] - coor[j]).norm(p=2, dim=-1, keepdim=True)
        rkj = (coor[k] - coor[j]).norm(p=2, dim=-1, keepdim=True)
        theta, _ = self.get_angle(coor, SB_index)
        drij = rij - r0ij
        drkj = rkj - r0kj
        dtheta = theta - theta0
        e = (A*drij + B*drkj) * dtheta
        e = scatter_add(e, SB_batch, dim=0)
        if self.warm:
            e = e * 0
        return e

    def get_OOP(self, coor, OOP_index, OOP_param, OOP_batch):
        '''
        out-of-plane bending,
        chi: (degree)
        energy = A*chi**2
        :param coor: [n_atom, 3]
        :param OOP_index: [n_oop, 4], i, j, k, l
        :param OOP_param: [n_oop, 1], A
        :param OOP_batch: [n_oop]
        :return: energy
        '''
        A = OOP_param
        chi = self.get_oop(coor, OOP_index)
        e = A*chi**2
        e = scatter_add(e, OOP_batch, dim=0)
        return e

    def get_TOR(self, coor, TOR_index, TOR_param, TOR_batch):
        '''
        torsion
        cos1: cos 1 dihedral
        cos2: cos 2 dihedral
        cos3: cos 3 dihedral
        energy = A + B*cos1 + C*cos2 + D*cos3
        :param coor: [n_atom, 3]
        :param TOR_index: [n_torsion, 4], i, j, k, l
        :param TOR_param: [n_torsion, 4], A, B, C, D
        :param TOR_batch: [n_torsion]
        :return: energy
        '''
        A, B, C, D = TOR_param.split(1, dim=-1)
        cos1 = self.get_dihedral_angle(coor, TOR_index)
        cos2 = 2.0 * cos1 * cos1 - 1.0
        cos3 = cos1 * (2.0 * cos2 - 1.0)  # or 4*cos1**3 - 3*cos1
        e = A + B*cos1 + C*cos2 + D*cos3
        e = scatter_add(e, TOR_batch, dim=0)
        return e

    def get_VDW(self, coor, VDW_index, VDW_param, VDW_batch):
        '''
        van der Waals
        rij: ...
        energy = A / (rij + B)**7 * (C / (rij**7 + D) + E)
        :param coor: [n_atom, 3]
        :param VDW_index: [n_pair, 2], i, j
        :param VDW_param: [n_pair, 5], A, B, C, D, E
        :param VDW_batch: [n_pair]
        :return: energy
        '''
        i, j = self.split_to_single_dim(VDW_index)
        A, B, C, D, E = VDW_param.split(1, dim=-1)
        rij = (coor[i] - coor[j]).norm(p=2, dim=-1, keepdim=True)
        if self.warm:
            e = F.relu(1 - rij)
            e = scatter_add(e, VDW_batch, dim=0)
        else:
            e = A / (rij + B)**7 * (C / (rij**7 + D) + E)
            e = scatter_add(e, VDW_batch, dim=0)
        return e

    def get_ELE(self, coor, ELE_index, ELE_param, ELE_batch):
        '''
        electrostatic
        qi: partial charge of node i
        qi: partial charge of node j
        rij: ...
        energy = A / (rij + B)
        :param coor: [n_atom, 3]
        :param ELE_index: [n_pair, 2], i, j
        :param ELE_param: [n_pair, 2], A, B
        :param ELE_batch: [n_pair]
        :return: energy
        '''
        i, j = self.split_to_single_dim(ELE_index)
        A, B = ELE_param.split(1, dim=-1)
        rij = (coor[i] - coor[j]).norm(p=2, dim=-1, keepdim=True)
        e = A / (rij + B)
        e = scatter_add(e, ELE_batch, dim=0)
        if self.warm:
            e = e * 0
        return e

    def get_angle(self, coor, angle_index):
        i, j, k = self.split_to_single_dim(angle_index)
        a = coor[i] - coor[j]
        b = coor[k] - coor[j]
        inner_product = (a * b).sum(dim=-1, keepdim=True)
        a_norm = a.norm(p=2, dim=-1, keepdim=True)
        b_norm = b.norm(p=2, dim=-1, keepdim=True)
        angle_cos = inner_product / (a_norm * b_norm + 1e-8)
        angle_deg = angle_cos.clamp(min=-1+1e-6, max=1-1e-6).acos() * 180/np.pi
        return angle_deg, angle_cos

    def get_oop(self, coor, oop_index):
        i, j, k, l = self.split_to_single_dim(oop_index)
        i = coor[i]
        j = coor[j]
        k = coor[k]
        l = coor[l]
        ji = i - j
        jk = k - j
        jl = l - j
        ji = ji / (ji.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        jk = jk / (jk.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        jl = jl / (jl.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        plane_v = self.get_cross(ji, jk)
        plane_v = plane_v / plane_v.norm(p=2, dim=-1, keepdim=True)
        chi_sin = (plane_v * jl).sum(dim=-1, keepdim=True)
        chi = chi_sin.clamp(min=-1+1e-6, max=1-1e-6).arcsin() * 180/np.pi
        return chi

    def get_dihedral_angle(self, coor, dihedral_index):
        i, j, k, l = self.split_to_single_dim(dihedral_index)
        i = coor[i]
        j = coor[j]
        k = coor[k]
        l = coor[l]
        ij = i - j
        kj = k - j
        jk = j - k
        lk = l - k
        t1 = self.get_cross(ij, kj)
        t2 = self.get_cross(jk, lk)
        cos_dih = (t1 * t2).sum(dim=-1, keepdim=True) / (t1.norm(p=2, dim=-1, keepdim=True) * t2.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return cos_dih

    def get_cross(self, v1, v2):
        if v1.dim()==2:
            return torch.cross(v1, v2, dim=-1)
        elif v1.dim()==3:
            h = v1.shape[1]
            v1, v2 = map(lambda t: rearrange(t, 'n h c -> (n h) c'), (v1, v2))
            cross_product = torch.cross(v1, v2, dim=-1)
            return rearrange(cross_product, '(n h) c -> n h c', h=h)
        else:
            raise 'cross dim error'

    def split_to_single_dim(self, x):
        x_list = x.split(1, dim=-1)
        return [x.squeeze(dim=-1) for x in x_list]



if __name__=='__main__':
    # for testing
    pass
    # print('tol_e_rdkit:', tol_e_rdkit)
    # print('tol_e_torch:', tol_e_torch.detach().cpu().numpy()[0, 0])
    # print('=' * 20 + 'TEST DONE' + '=' * 20)
'''
# test, consistent to rdkit
BS---9.819
AB---12.039
SB---1.555
OOP---0.044
TOR---3.575
'''


