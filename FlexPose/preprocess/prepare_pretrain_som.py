import os
import sys
import shutil
from ray.util.multiprocessing import Pool


from utils.pdbbind_preprocess import *
from utils.common import *


def prepare(smi):
    ligand_mol = Chem.MolFromSmiles(smi)
    AllChem.EmbedMolecule(ligand_mol, maxAttempts=10, useRandomCoords=True, clearConfs=False)

    # encode
    ligand_node_features = get_node_feature(ligand_mol, 'ligand')
    ligand_edge, ligand_edge_features = get_ligand_edge_feature(ligand_mol)
    ligand_dismap = get_ligand_unrotable_distance(ligand_mol)
    ligand_coor = get_true_posi(ligand_mol)

    assert len(ligand_node_features) <= 150

    return dict(ligand_node_features=ligand_node_features,
                ligand_edge=ligand_edge,
                ligand_edge_features=ligand_edge_features,
                ligand_smiles=smi,
                ligand_dismap=ligand_dismap,
                ligand_coor=ligand_coor
                )


def save(tup_in):
    save_path, smi, idx = tup_in
    dic_data = prepare(smi)
    np.savez_compressed(save_path + '/{}.npz'.format(idx), **dic_data)


def try_prepare(*args, **kwargs):
    try:
        save(*args, **kwargs)
    except:
        pass


if __name__ == '__main__':
    lines = open('pretrain_smiles.txt', 'r').readlines()
    smi_list = [line[:-1] for line in lines]
    print('smi num:', len(smi_list))

    save_path = '/home/dongtj/work_site/drug/data/pocket_for_pretrain/prepare_SOM_npz'
    delmkdir(save_path)

    task = []
    for idx, smi in enumerate(smi_list):
        task.append((save_path, smi, idx))


    pool = Pool()
    for _ in pool.map(try_prepare, task):
        pass

    print('DONE')


































