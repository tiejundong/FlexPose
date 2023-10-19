import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from einops import rearrange


MMFF_keys = ['BS', 'AB', 'SB', 'OOP', 'TOR', 'VDW', 'ELE']
MMFF_pad_dim = [[2, 5], [3, 8], [3, 5], [4, 1], [4, 4], [2, 5], [2, 2]]  # index param


class MMFFLoss():
    def __init__(self, split_interact=False, warm=False):
        super(MMFFLoss, self).__init__()
        self.MMFF_keys = ['BS', 'AB', 'SB', 'OOP', 'TOR', 'VDW', 'ELE']
        self.split_interact = split_interact
        if split_interact:  # VDW1: p2l; VDW2: l2l, same for ELE
            self.MMFF_keys = ['BS', 'AB', 'SB', 'OOP', 'TOR', 'VDW1', 'VDW2', 'ELE1', 'ELE2']

        self.warm = warm

    def __call__(self, coor, MMFF_param, return_sum=True):
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
        A = OOP_param
        chi = self.get_oop(coor, OOP_index)
        e = A*chi**2
        e = scatter_add(e, OOP_batch, dim=0)
        return e

    def get_TOR(self, coor, TOR_index, TOR_param, TOR_batch):
        A, B, C, D = TOR_param.split(1, dim=-1)
        cos1 = self.get_dihedral_angle(coor, TOR_index)
        cos2 = 2.0 * cos1 * cos1 - 1.0
        cos3 = cos1 * (2.0 * cos2 - 1.0)  # or 4*cos1**3 - 3*cos1
        e = A + B*cos1 + C*cos2 + D*cos3
        e = scatter_add(e, TOR_batch, dim=0)
        return e

    def get_VDW(self, coor, VDW_index, VDW_param, VDW_batch):
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
        ji = ji / ji.norm(p=2, dim=-1, keepdim=True)
        jk = jk / jk.norm(p=2, dim=-1, keepdim=True)
        jl = jl / jl.norm(p=2, dim=-1, keepdim=True)
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
