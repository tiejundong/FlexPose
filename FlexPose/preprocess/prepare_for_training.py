import os
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
from ray.util.multiprocessing import Pool

from tqdm import tqdm, trange
import numpy as np
import pickle

import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from FlexPose.utils.pdbbind_preprocess import *



def prepare(tup_in):
    npz_save_path, pdb_id, df_apo_sub, apo_path, pdbbind_path, aff_path, MCaug_path, df_apo_sub, have_apo, max_len_pocket, max_len_ligand, tmp_path = tup_in


    ####################################################################################################################
    # ligand encoding
    ####################################################################################################################
    ligand_mol = read_mol_from_pdbbind(pdbbind_path, pdb_id)
    ligand_node_features = get_node_feature(ligand_mol, 'ligand')
    ligand_edge, ligand_edge_features = get_ligand_edge_feature(ligand_mol)
    ligand_coor_true = get_true_posi(ligand_mol)
    ligand_match = get_ligand_match(ligand_mol)
    ligand_dismap = get_ligand_unrotable_distance(ligand_mol)
    ligand_data = [ligand_node_features, ligand_edge_features, ligand_coor_true, ligand_match, ligand_dismap]
    assert len(ligand_node_features) <= max_len_ligand


    ####################################################################################################################
    # pocket dataframe
    ####################################################################################################################
    # load modeller again for ray
    from modeller import Environ
    from modeller.scripts import complete_pdb
    env_ = Environ()
    env_.libs.topology.read(file='$(LIB)/top_heav.lib')
    env_.libs.parameters.read(file='$(LIB)/par.lib')

    if isinstance(have_apo, type(None)):
        try:
            tmscore = df_apo_sub['tmscore'].values[0]
            backbone_rmsd = df_apo_sub['backbone_rmsd'].values[0]
            assert tmscore > 0.3 and backbone_rmsd < 15
            have_apo = True
        except:
            have_apo = False

    # for holo
    if have_apo:
        holo_pdb_path = f'{apo_path}/{pdb_id}/{pdb_id}_protein_nowat.pdb'
    else:
        holo_pdb_path = f'{pdbbind_path}/{pdb_id}/{pdb_id}_protein.pdb'
    fixed_holo_path = f'{tmp_path}/{pdb_id}_holo.pdb'
    pdb_m = complete_pdb(env_, holo_pdb_path)
    pdb_m.write(fixed_holo_path)

    biodf_protein_holo = PandasPdb().read_pdb(fixed_holo_path)
    df_protein_holo = biodf_protein_holo.df['ATOM']
    df_protein_holo['chain_resi'] = df_protein_holo['chain_id'].astype(str) + '_' + df_protein_holo['residue_number'].astype(str)

    if have_apo:
        holo_idx = [int(s) for s in df_apo_sub['holo_bind_indices'].values[0].split(' ')]
        holo_idx = get_pocket_idx_without_HETATM(holo_pdb_path, fixed_holo_path, holo_idx)
        holo_idx_mask = [True if s != '-' else False for s in df_apo_sub['holo_bind_indices'].values[0].split(' ')]
        holo_bind_chain_resi = df_protein_holo['chain_resi'].unique()[holo_idx]
        df_protein_holo['is_pocket'] = df_protein_holo['chain_resi'].apply(lambda x: x in holo_bind_chain_resi)
        df_pocket_holo = df_protein_holo[df_protein_holo['is_pocket'] == True]

        biodf_protein_holo.df['ATOM'] = df_pocket_holo
        aa_holo = biodf_protein_holo.amino3to1()['residue_name'].values.reshape(-1)
        holo_bind_res = np.array(df_apo_sub['holo_bind_res'].values[0].split(' '))[holo_idx_mask]
        assert ''.join(aa_holo) == ''.join(holo_bind_res), ''.join(aa_holo) + '---' + ''.join(holo_bind_res)

        # for apo
        apo_pdb_path = f'{apo_path}/{pdb_id}/{pdb_id}_apo_added.pdb'
        fixed_apo_path = f'{tmp_path}/{pdb_id}_apo.pdb'
        pdb_m = complete_pdb(env_, apo_pdb_path)
        pdb_m.write(fixed_apo_path)

        biodf_protein_apo = PandasPdb().read_pdb(fixed_apo_path)
        df_protein_apo = biodf_protein_apo.df['ATOM']
        df_protein_apo['chain_resi'] = df_protein_apo['chain_id'].astype(str) + '_' + df_protein_apo['residue_number'].astype(str)

        apo_idx = [int(s) for s in df_apo_sub['apo_bind_indices'].values[0].split(' ') if s != '-']
        apo_idx = get_pocket_idx_without_HETATM(apo_pdb_path, fixed_apo_path, apo_idx)
        apo_idx_mask = [True if s != '-' else False for s in df_apo_sub['apo_bind_indices'].values[0].split(' ')]
        apo_bind_chain_resi = df_protein_apo['chain_resi'].unique()[apo_idx]
        df_protein_apo['is_pocket'] = df_protein_apo['chain_resi'].apply(lambda x: x in apo_bind_chain_resi)
        df_pocket_apo = df_protein_apo[df_protein_apo['is_pocket'] == True]  # note that df_pocket_apo contains no metals

        biodf_protein_apo.df['ATOM'] = df_pocket_apo
        aa_apo = biodf_protein_apo.amino3to1()['residue_name'].values.reshape(-1)
        apo_bind_res = np.array(df_apo_sub['apo_bind_res'].values[0].split(' '))[apo_idx_mask]
        assert ''.join(aa_apo) == ''.join(apo_bind_res), ''.join(aa_apo) + '---' + ''.join(apo_bind_res)

        # for apo (add metal), del HETATM resi with n_atom>1
        # df_HETATM_apo = PandasPdb().read_pdb(apo_pdb_path).df['HETATM']
        # if len(df_HETATM_apo) > 0:
        #     df_HETATM_apo['resi_cout'] = [np.sum(df_HETATM_apo['residue_number'] == r) for r in
        #                                   df_HETATM_apo['residue_number']]
        #     df_HETATM_apo = df_HETATM_apo[df_HETATM_apo['resi_cout'] == 1]
        #     df_HETATM_apo['chain_id'] = [i for i in range(len(df_HETATM_apo))]
        #     df_HETATM_apo['chain_resi'] = df_HETATM_apo['chain_id'].astype(str) + '_' + df_HETATM_apo[
        #         'residue_number'].astype(str)
        #     metal2pocket = np.linalg.norm(np.expand_dims(df_HETATM_apo[['x_coord', 'y_coord', 'z_coord']].values, -2) -
        #                                   np.expand_dims(df_pocket_apo[['x_coord', 'y_coord', 'z_coord']].values, -0),
        #                                   ord=2, axis=-1)
        #     df_HETATM_apo['is_pocket'] = metal2pocket.min(-1) < 6  # add metal near pocket within 6A
        #     df_HETATM_apo = df_HETATM_apo[df_HETATM_apo['is_pocket'] == True]
        #     df_pocket_apo = pd.concat([df_pocket_apo, df_HETATM_apo], axis=0, join='outer', ignore_index=True)

        # check mapping residues are same AA seq
        common_idx_mask = [True if i != '-' and j != '-' else False
                           for i, j in zip(df_apo_sub['apo_bind_indices'].values[0].split(' '),
                                           df_apo_sub['holo_bind_indices'].values[0].split(' '))]
        apo_common_part = np.array(df_apo_sub['apo_bind_res'].values[0].split(' '))[common_idx_mask]
        holo_common_part = np.array(df_apo_sub['holo_bind_res'].values[0].split(' '))[common_idx_mask]
        assert all(apo_common_part == holo_common_part)

        # add additional residue
        df_pocket_manual_select, _ = get_pocket(df_protein_apo, ligand_coor_true, max_len_protein=max_len_pocket)
        additional_chain_resi = []
        for chain_res in df_pocket_manual_select['chain_resi'].unique():
            if chain_res not in df_pocket_apo['chain_resi'].unique():
                additional_chain_resi.append(chain_res)
        df_pocket_manual_select['is_addition'] = df_pocket_manual_select['chain_resi'].apply(
                                                        lambda x: True if x in additional_chain_resi else False)
        df_addition = df_pocket_manual_select[df_pocket_manual_select['is_addition'] == True]
        df_pocket_apo = pd.concat([df_pocket_apo, df_addition], join='outer', axis=0)

        # add additional residue
        df_pocket_manual_select, _ = get_pocket(df_protein_holo, ligand_coor_true, max_len_protein=max_len_pocket)
        additional_chain_resi = []
        for chain_res in df_pocket_manual_select['chain_resi'].unique():
            if chain_res not in df_pocket_holo['chain_resi'].unique():
                additional_chain_resi.append(chain_res)
        df_pocket_manual_select['is_addition'] = df_pocket_manual_select['chain_resi'].apply(
            lambda x: True if x in additional_chain_resi else False)
        df_addition = df_pocket_manual_select[df_pocket_manual_select['is_addition'] == True]
        df_pocket_holo = pd.concat([df_pocket_holo, df_addition], join='outer', axis=0)

    else:  # have_apo = False
        df_pocket_holo, sele_res_without_apo = get_pocket(df_protein_holo, ligand_coor_true, max_len_protein=max_len_pocket)
        holo_idx_mask = [True] * len(sele_res_without_apo)
        df_pocket_apo = None
        apo_idx_mask = None

    df_pocket_holo_true = df_pocket_holo

    ####################################################################################################################
    # MCaug pocekt, pseudo apo
    ####################################################################################################################
    sub_MC_path = f'{MCaug_path}/{pdb_id}'

    def get_pseudo(f):
        holo_pdb_path = f'{sub_MC_path}/{f}'
        fixed_holo_path = f'{tmp_path}/{pdb_id}_holo.pdb'
        pdb_m = complete_pdb(env_, holo_pdb_path)
        pdb_m.write(fixed_holo_path)

        biodf_protein_holo = PandasPdb().read_pdb(fixed_holo_path)
        df_protein_holo = biodf_protein_holo.df['ATOM']
        df_protein_holo['chain_resi'] = df_protein_holo['chain_id'].astype(str) + '_' + df_protein_holo['residue_number'].astype(str)

        if have_apo:
            holo_idx = [int(s) for s in df_apo_sub['holo_bind_indices'].values[0].split(' ')]
            holo_idx = get_pocket_idx_without_HETATM(holo_pdb_path, fixed_holo_path, holo_idx)
            holo_idx_mask = [True if s != '-' else False for s in df_apo_sub['holo_bind_indices'].values[0].split(' ')]
            holo_bind_chain_resi = df_protein_holo['chain_resi'].unique()[holo_idx]
            df_protein_holo['is_pocket'] = df_protein_holo['chain_resi'].apply(lambda x: x in holo_bind_chain_resi)
            df_pocket_holo = df_protein_holo[df_protein_holo['is_pocket'] == True]

            biodf_protein_holo.df['ATOM'] = df_pocket_holo
            aa_holo = biodf_protein_holo.amino3to1()['residue_name'].values.reshape(-1)
            holo_bind_res = np.array(df_apo_sub['holo_bind_res'].values[0].split(' '))[holo_idx_mask]
            assert ''.join(aa_holo) == ''.join(holo_bind_res), ''.join(aa_holo) + '---' + ''.join(holo_bind_res)

            # add additional residue
            df_pocket_manual_select, _ = get_pocket(df_protein_holo, ligand_coor_true, max_len_protein=max_len_pocket)
            additional_chain_resi = []
            for chain_res in df_pocket_manual_select['chain_resi'].unique():
                if chain_res not in df_pocket_holo['chain_resi'].unique():
                    additional_chain_resi.append(chain_res)
            df_pocket_manual_select['is_addition'] = df_pocket_manual_select['chain_resi'].apply(
                lambda x: True if x in additional_chain_resi else False)
            df_addition = df_pocket_manual_select[df_pocket_manual_select['is_addition'] == True]
            df_pocket_holo = pd.concat([df_pocket_holo, df_addition], join='outer', axis=0)
        else:
            df_protein_holo['pocket_flag'] = [
                True if str(df_protein_holo.loc[i, 'residue_number']) + ',' + str(
                    df_protein_holo.loc[i, 'chain_id']) in sele_res_without_apo else False
                for i in range(len(df_protein_holo))]
            df_pocket_holo = df_protein_holo[df_protein_holo['pocket_flag'] == True][df_protein_holo.columns[:-1]]

        return df_pocket_holo, df_protein_holo

    # random pert
    rand_pert_file_list = [i for i in os.listdir(sub_MC_path) if i.startswith('rand_pert')]
    rand_pert_df_list = []
    rand_pert_tor_list = []
    for f in rand_pert_file_list:
        df_pocket_holo, df_protein_holo = get_pseudo(f)
        rand_pert_df_list.append(df_pocket_holo)
        rand_pert_tor_list.append([f"{sub_MC_path}/torsion_{f.split('.')[0]}.npz", df_protein_holo, df_pocket_holo])


    # fixbb repacking
    fixbb_repack_file_list = [i for i in os.listdir(sub_MC_path) if i.startswith('fixbb_repack_')]
    fixbb_repack_df_list = []
    fixbb_repack_tor_list = []
    for f in fixbb_repack_file_list:
        df_pocket_holo, df_protein_holo = get_pseudo(f)
        fixbb_repack_df_list.append(df_pocket_holo)
        fixbb_repack_tor_list.append([f"{sub_MC_path}/torsion_{f.split('.')[0]}.npz", df_protein_holo, df_pocket_holo])


    # flexbb repacking
    flexbb_repack_file_list = [i for i in os.listdir(sub_MC_path) if i.startswith('flexbb_repack_')]
    flexbb_repack_df_list = []
    flexbb_repack_tor_list = []
    for f in flexbb_repack_file_list:
        df_pocket_holo, df_protein_holo = get_pseudo(f)
        flexbb_repack_df_list.append(df_pocket_holo)
        flexbb_repack_tor_list.append([f"{sub_MC_path}/torsion_{f.split('.')[0]}.npz", df_protein_holo, df_pocket_holo])


    ####################################################################################################################
    # pocket encoding
    ####################################################################################################################
    # apo
    holo_pocket_SCtorsion_data = get_torsion(f'{sub_MC_path}/torsion.npz', df_protein_holo, df_pocket_holo_true)
    holo_true_data = encode_pocket(df_pocket_holo_true) + [holo_pocket_SCtorsion_data, holo_idx_mask]
    if have_apo:
        apo_pocket_SCtorsion_data = get_torsion(f'{sub_MC_path}/torsion_apo.npz', df_protein_apo, df_pocket_apo)
        apo_data = encode_pocket(df_pocket_apo) + [apo_pocket_SCtorsion_data, apo_idx_mask]
    else:
        apo_data = None

    # pseudo apo
    rand_pert_data_list = [encode_pocket(df) + [get_torsion(*tor)] for df, tor in zip(rand_pert_df_list, rand_pert_tor_list)]
    fixbb_data_list = [encode_pocket(df) + [get_torsion(*tor)] for df, tor in zip(fixbb_repack_df_list, fixbb_repack_tor_list)]
    flexbb_data_list = [encode_pocket(df) + [get_torsion(*tor)] for df, tor in zip(flexbb_repack_df_list, flexbb_repack_tor_list)]

    assert len(df_pocket_holo_true['chain_resi'].unique()) <= max_len_pocket

    aff = get_aff(aff_path)[pdb_id]


    try:
        from model.MMFF import get_MMFF_param
        dic_MMFF_param = get_MMFF_param(ligand_mol)
    except:
        dic_MMFF_param = None


    return dict(ligand_data=ligand_data,
                holo_true_data=holo_true_data,
                apo_data=apo_data,
                rand_pert_data_list=rand_pert_data_list,
                fixbb_data_list=fixbb_data_list,
                flexbb_data_list=flexbb_data_list,
                aff=aff,
                dic_MMFF_param=dic_MMFF_param,
                )


def save_APOPDBbind(tup_in):
    dic_data = prepare(tup_in)
    save_path, pdb_id, df_apo_sub, apo_path, pdbbind_path, aff_path, MCaug_path, df_apo_sub, have_apo, max_len_pocket, max_len_ligand, tmp_path = tup_in
    # np.savez_compressed(npz_save_path + '/{}.npz'.format(pdb_id), **dic_data)
    pickle.dump(dic_data, open(save_path + '/{}.pkl'.format(pdb_id), 'wb'))


def try_prepare_APOPDBbind(*args, **kwargs):
    try:
        save_APOPDBbind(*args, **kwargs)
    except:
        pass


def try_prepare_task(intup):
    f, task = intup
    try:
        f(task)
        return True
    except:
        return False


if __name__ == '__main__':
    pass

































