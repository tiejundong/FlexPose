import os
import sys
import shutil
from ray.util.multiprocessing import Pool

from tqdm import tqdm, trange
import numpy as np

import torch
import scipy
import scipy.spatial
import scipy.sparse
from keras_progbar import Progbar
import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from utils.pdbbind_preprocess import *
from utils.common import *


def get_dismap(df_protein):
    chain_resi_list = list(df_protein['chain_resi'].unique())

    dic_data = {}
    dic_data['CB_coor'] = []
    for i in range(len(chain_resi_list)):
        df_i = df_protein[df_protein['chain_resi'] == chain_resi_list[i]]
        resi_type = df_i['residue_name'].values[0]

        if resi_type == 'GLY':
            CB_coor = df_i[(df_i['atom_name'] == 'CA')][['x_coord', 'y_coord', 'z_coord']].values[0]
        else:
            CB_coor = df_i[(df_i['atom_name'] == 'CB')][['x_coord', 'y_coord', 'z_coord']].values[0]

        dic_data['CB_coor'].append(CB_coor)

    CB_coor = np.stack(dic_data['CB_coor'], axis=0)

    return CB_coor


def prepare(df_protein, df_STP, STP_id):
    # get contact protein
    df_center = df_STP[df_STP['residue_number'] == int(STP_id)]
    df_contact, _ = get_pocket(df_protein, df_center[['x_coord', 'y_coord', 'z_coord']].values)
    df_contact['chain_resi'] = df_contact['chain_id'].astype(str) + '_' + df_contact['residue_number'].astype(str)
    df_CA = df_contact[(df_contact['atom_name'] == 'CA')]

    assert list(df_CA['chain_resi'].unique()) == list(df_contact['chain_resi'].unique())  # make sure sequential correct

    # get protein feature
    p_x_sca, p_x_vec = get_protein_x_sca_vec(df_contact)
    # p_edge_sca, p_edge_vec = get_protein_edge_sca_vec(df_contact)
    p_coor = df_CA[['x_coord', 'y_coord', 'z_coord']].values

    # get pretrain target
    CB_coor = get_dismap(df_contact)
    MCSC_coor, MCSC_mask = get_AA_coor(df_contact)
    resi_connect = get_protein_edge_connect_info(df_contact)

    return dict(p_x_sca=p_x_sca,
                p_x_vec=p_x_vec,
                # p_edge_sca=p_edge_sca,
                # p_edge_vec=p_edge_vec,
                resi_connect=resi_connect,
                p_coor=p_coor,
                CB_coor=CB_coor,
                MCSC_coor=MCSC_coor,
                MCSC_mask=MCSC_mask,
                center_coor=df_center[['x_coord', 'y_coord', 'z_coord']].values,
                )


def save(tup_in):
    save_path, fpocket_path, pdb_id, pocket_id_list = tup_in

    # get fpocket output (add metal future ?)
    biodf_protein = PandasPdb().read_pdb(f'{fpocket_path}/{pdb_id}/{pdb_id}_fixed_out.pdb')
    df_protein = biodf_protein.df['ATOM']
    df_protein = df_protein[df_protein['element_symbol'] != 'H']
    df_HETATM = biodf_protein.df['HETATM']
    df_STP = df_HETATM[df_HETATM['residue_name'] == 'STP']

    for STP_id in pocket_id_list:
        pocket_id = f'{pdb_id}_{STP_id}'
        try:
            dic_data = prepare(df_protein, df_STP, STP_id)
            np.savez_compressed(save_path + '/{}.npz'.format(pocket_id), **dic_data)
        except:
            pass


def try_prepare(*args, **kwargs):
    try:
        save(*args, **kwargs)
    except:
        pass


if __name__ == '__main__':
    fpocket_path = '/home/dongtj/work_site/drug/data/pocket_for_pretrain/small_fpocket_output'
    fpocket_csv_path = '/home/dongtj/work_site/drug/data/pocket_for_pretrain/pocket_filter.csv'

    save_path = '/home/dongtj/work_site/drug/data/pocket_for_pretrain/prepare_pocket_npz'
    delmkdir(save_path)


    df_pocket = pd.read_csv(fpocket_csv_path)
    pocket_list = df_pocket['pocket_id'].values
    print('pocket num', len(pocket_list))

    dic_pocket_id = {}
    pdb_id_list = os.listdir(fpocket_path)
    for i in tqdm(pocket_list):
        if os.path.exists(f'{save_path}/{i}.npz'):
            continue
        i_split = i.split('_')
        pdb_id = i_split[0]
        pocket_id = i_split[1]

        if pdb_id not in pdb_id_list:
            continue

        if pdb_id not in dic_pocket_id.keys():
            dic_pocket_id[pdb_id] = [pocket_id]
        else:
            dic_pocket_id[pdb_id].append(pocket_id)


    task = []
    for idx, pdb_id in enumerate(dic_pocket_id.keys()):
        task.append((save_path, fpocket_path, pdb_id, dic_pocket_id[pdb_id]))
    print('task num', len(task))


    print('Begin')
    # for i in tqdm(task):
    #     save(i)
    pool = Pool()
    for _ in pool.map(try_prepare, task):
        pass
    print('DONE')




