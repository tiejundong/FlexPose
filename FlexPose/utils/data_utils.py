import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn.functional as F


def pad_zeros(batch_list, keys, max_len, collect_dim=-3, data_type='1d', cat=False, value=0, output_dtype=None):
    # 1d: set of [..., pad_dim, ...], 2d: set of [..., pad_dim, pad_dim + 1, ...]
    # To:
    # 1d: [..., collect_dim, pad_dim, ...], 2d: [..., collect_dim, pad_dim, pad_dim + 1, ...]
    assert collect_dim < 0
    pad_dim = collect_dim + 1

    collect = torch.concat if cat else torch.stack

    dic_data = {}
    for k in keys:
        if data_type == '1d':
            collection = collect([F.pad(g[k],
                   tuple([0] * (np.abs(pad_dim) - 1) * 2 + [0, max_len - g[k].shape[pad_dim]]),
                   'constant', value)
                   for g in batch_list], dim=collect_dim)
        if data_type == '2d':
            collection = collect([F.pad(g[k],
                   tuple([0] * (np.abs(pad_dim) - 2) * 2 + [0, max_len - g[k].shape[pad_dim]]*2),
                   'constant', value)
                   for g in batch_list], dim=collect_dim)
        else:
            assert data_type in ['1d', '2d']

        if not isinstance(output_dtype, type(None)):
            collection = collection.to(output_dtype)
        dic_data[k] = collection

    return dic_data


def collate_dummy(batch_list):
    return batch_list[0]


def assign_struct(mol, coor, min=False):
    AllChem.EmbedMolecule(mol, maxAttempts=10, useRandomCoords=True, clearConfs=True)
    mol_conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        mol_conf.SetAtomPosition(i, coor[i].astype(float))
    if min:
        try:
            ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
                mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(mol))
            for atom_i in range(mol.GetNumAtoms()):
                ff.MMFFAddPositionConstraint(atom_i, 1, 5)  # maxDispl: maximum displacement
            ff.Minimize(maxIts=10)
        except:
            print('Failure: energy minimization')
    return mol


def min_struct(mol, len_ligand=None):
    Chem.SanitizeMol(mol)
    ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
        mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(mol), ignoreInterfragInteractions=False)
    if len_ligand is None:
        len_ligand = mol.GetNumAtoms()
    for atom_i in range(mol.GetNumAtoms() - len_ligand):
        ff.MMFFAddPositionConstraint(atom_i, 1, 5)
    for atom_i in range(mol.GetNumAtoms() - len_ligand, mol.GetNumAtoms()):
        ff.MMFFAddPositionConstraint(atom_i, 1, 5)
    ff.Minimize(maxIts=10)

    # AllChem.MMFFOptimizeMolecule(mol, maxIters=10)
    return mol


def tan_2_deg(sin_pred, cos_pred):
    deg_pred = np.arctan(sin_pred/cos_pred)
    if np.sin(deg_pred) * sin_pred > 0 and np.cos(deg_pred) * cos_pred > 0:
        return deg_pred
    else:
        deg_pred = deg_pred + np.pi if deg_pred < 0 else deg_pred - np.pi
        assert np.sin(deg_pred) * sin_pred > 0 and np.cos(deg_pred) * cos_pred > 0
        return deg_pred




