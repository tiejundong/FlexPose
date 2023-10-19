import pickle
import numpy as np
import scipy
import scipy.spatial
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from FlexPose.model.pyrosetta_SC import dic_SC



def get_file_lines(f_path):
    with open(f_path, 'r') as f:
        lines = f.readlines()
    return lines

def read_mol_with_pdb_smi(pdb_path, smiles):
    ligand_mol = Chem.MolFromPDBFile(pdb_path)
    ligand_template = Chem.MolFromSmiles(smiles)
    ligand_mol = AllChem.AssignBondOrdersFromTemplate(ligand_template, ligand_mol)
    assert ligand_mol != None
    return ligand_mol


def read_mol_from_pdbbind(data_path, pdb_id):
    ligand_mol2_path = f'{data_path}/{pdb_id}/{pdb_id}_ligand.mol2'
    ligand_mol = Chem.MolFromMol2File(ligand_mol2_path)
    if ligand_mol == None:
        ligand_pdbpath = f'{data_path}/{pdb_id}/{pdb_id}_ligand.pdb'
        ligand_smiles_path = f'{data_path}/{pdb_id}/{pdb_id}_ligand.smi'
        ligand_smiles = open(ligand_smiles_path, 'r').readlines()[0].split('\t')[0]
        ligand_mol = read_mol_with_pdb_smi(ligand_pdbpath, ligand_smiles)
    return ligand_mol

def get_true_posi(mol):
    mol_conf = mol.GetConformer()
    node_posi = np.array([mol_conf.GetAtomPosition(int(idx)) for idx in range(mol.GetNumAtoms())])
    return node_posi


def get_aff(info_path):
    lines = open(info_path, 'r').readlines()
    dic_aff = {line[:4]: float(line[18:23]) for line in lines if not line.startswith('#')}
    return dic_aff


def onehot_with_allowset(x, allowset, with_unk=True):
    if x not in allowset and with_unk == True:
        x = allowset[0]  # UNK
    return list(map(lambda s: x == s, allowset))


def get_ligand_node_feature(atom, idx, ring_info, canonical_rank):
    # encode with rich features
    atom_features = \
        onehot_with_allowset(atom.GetSymbol(), ['UNK', 'C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'B', 'I'], with_unk=True) + \
        onehot_with_allowset(atom.GetTotalDegree(), ['UNK', 1, 2, 3, 4, 5], with_unk=True) + \
        onehot_with_allowset(atom.GetFormalCharge(), ['UNK', -1, -2, 0, 1, 2], with_unk=True) + \
        onehot_with_allowset(atom.GetImplicitValence(), ['UNK', 0, 1, 2, 3], with_unk=True) + \
        onehot_with_allowset(atom.GetTotalNumHs(), ['UNK', 0, 1, 2, 3], with_unk=True) + \
        onehot_with_allowset(atom.GetHybridization(), \
                             ['UNK',
                              Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2], with_unk=True) + \
        [atom.GetIsAromatic()] + \
        [ring_info.IsAtomInRingOfSize(idx, 3),
         ring_info.IsAtomInRingOfSize(idx, 4),
         ring_info.IsAtomInRingOfSize(idx, 5),
         ring_info.IsAtomInRingOfSize(idx, 6),
         ring_info.IsAtomInRingOfSize(idx, 7),
         ring_info.IsAtomInRingOfSize(idx, 8)]
    atom_features = np.array(atom_features).astype(np.float32)
    return atom_features


def get_node_feature(mol, mol_type):
    if mol_type == 'ligand':
        ring_info = mol.GetRingInfo()
        canonical_rank = Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=True)
        node_features = [get_ligand_node_feature(atom, idx, ring_info, canonical_rank) for idx, atom in
                         zip(range(mol.GetNumAtoms()), mol.GetAtoms())]
    else:
        assert mol_type in ['ligand']
    node_features = np.stack(node_features)
    return node_features


def get_full_connectedge(mol):  # include a->b, a<-b and a->a
    edge = np.array([[i, j] for i in range(mol.GetNumAtoms()) for j in range(mol.GetNumAtoms())])
    return edge


def get_havebond(mol, idx_1, idx_2):
    bond = mol.GetBondBetweenAtoms(int(idx_1), int(idx_2))
    if bond == None:
        return 0
    else:
        return 1


def get_atomdistance(mol_conf, idx_1, idx_2):
    coor_1 = mol_conf.GetAtomPosition(int(idx_1))
    coor_2 = mol_conf.GetAtomPosition(int(idx_2))
    return coor_1.Distance(coor_2)


def has_common_neighbor_atom(mol, idx_1, idx_2):
    flag = False
    for idx in range(mol.GetNumAtoms()):
        if idx == idx_1 or idx == idx_2:
            continue
        else:
            bond_1 = mol.GetBondBetweenAtoms(int(idx), int(idx_1))
            bond_2 = mol.GetBondBetweenAtoms(int(idx), int(idx_2))
            if bond_1 != None and bond_2 != None:
                flag = True
                break
    return flag


def get_ligand_bond_feature(mol, idx_1, idx_2):
    bond = mol.GetBondBetweenAtoms(int(idx_1), int(idx_2))
    if bond == None:
        edge_feature = [1] + [0] * 5
    else:
        edge_feature = [0]
        edge_feature += onehot_with_allowset(bond.GetBondType(), ['UNK', \
                                                                  Chem.rdchem.BondType.SINGLE, \
                                                                  Chem.rdchem.BondType.DOUBLE, \
                                                                  Chem.rdchem.BondType.TRIPLE, \
                                                                  Chem.rdchem.BondType.AROMATIC], with_unk=True)
    edge_feature = np.array(edge_feature).astype(np.float32)
    return edge_feature


def get_ligand_edge_feature(mol):
    edge = get_full_connectedge(mol)
    edge_features = np.array([get_ligand_bond_feature(mol, i, j) for i, j in edge])
    edge_features = edge_features.reshape(mol.GetNumAtoms(), mol.GetNumAtoms(), -1)
    return edge, edge_features


def get_ligand_match(mol1, mol2=None):
    if mol2 is None:
        matches = mol1.GetSubstructMatches(mol1, uniquify=False)
    else:
        matches = mol1.GetSubstructMatches(mol2, uniquify=False)
    return np.array(matches)


def get_ligand_unrotable_distance(ligand_mol):
    rot_patt = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'  # '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]' for rotable bond
    patt = Chem.MolFromSmarts(rot_patt)
    hit_bonds = ligand_mol.GetSubstructMatches(patt)
    em = Chem.EditableMol(ligand_mol)
    for (idx1, idx2) in hit_bonds:
        em.RemoveBond(idx1, idx2)
    p = em.GetMol()
    part_list = Chem.GetMolFrags(p, asMols=False)

    added_part_list = []
    for part in part_list:
        tmp = list(part)
        for bonds in hit_bonds:
            i, j = bonds
            if i in part:
                tmp.append(j)
            elif j in part:
                tmp.append(i)
        added_part_list.append(tmp)

    n_atoms = ligand_mol.GetNumAtoms()
    dist_map = np.zeros((n_atoms, n_atoms)) + -1
    mol_conf = ligand_mol.GetConformer()
    for part in added_part_list:
        for i in part:
            for j in part:
                dist_map[i, j] = get_atomdistance(mol_conf, i, j)
    return dist_map


def get_pocket_idx_without_HETATM(pdb_file, fixed_file, pocket_idx):
    # map original index with modeller processed
    lines = get_file_lines(pdb_file)
    chain_resi_repeat = [line[20:28] for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
    ATOMHETATM_repeat = [line[:4] for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
    aa_repeat = [line[17:20] for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
    atom_repeat = [line[13:15] for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]
    info_repeat = [line[31:56] for line in lines if line.startswith('HETATM') or line.startswith('ATOM')]

    dic_chain_resi = {}
    for i in range(len(chain_resi_repeat)):
        chain_resi = chain_resi_repeat[i]
        ATOMHETATM_type = ATOMHETATM_repeat[i]
        aa = aa_repeat[i]
        atom = atom_repeat[i]
        info = info_repeat[i]

        if chain_resi not in dic_chain_resi.keys():
            dic_chain_resi[chain_resi] = {}
            dic_chain_resi[chain_resi]['ATOMHETATM_type'] = ATOMHETATM_type
            dic_chain_resi[chain_resi]['aa'] = aa

        if atom == 'CA':
            dic_chain_resi[chain_resi]['atom'] = atom
            dic_chain_resi[chain_resi]['info'] = info

    dic_map = {}
    cur_i = 0
    decay_i = 0
    fixed_plain_str = ''.join(get_file_lines(fixed_file))
    for chain_resi in dic_chain_resi.keys():
        if 'info' not in dic_chain_resi[chain_resi].keys():
            dic_chain_resi[chain_resi]['info'] = 'xxxxx'

        info = dic_chain_resi[chain_resi]['info']
        if info in fixed_plain_str:
            dic_map[cur_i] = decay_i
            decay_i += 1
        else:
            dic_map[cur_i] = decay_i
        cur_i += 1
    pocket_idx = [dic_map[i] for i in pocket_idx]
    return pocket_idx


def get_pocket(df_protein, center_coor, any_atom=True, dis=10, max_len_protein=150):
    df_pocket = df_protein.copy()
    if not any_atom:
        df_tmp = df_pocket[df_pocket['atom_name'] == 'CA'].copy()
    else:
        df_tmp = df_pocket.copy()

    p_coor = df_tmp[['x_coord', 'y_coord', 'z_coord']].values
    p2l_dismap = scipy.spatial.distance.cdist(p_coor, center_coor, metric='euclidean')
    p2l_dis = p2l_dismap.min(-1)
    df_tmp['pocket_flag'] = (p2l_dis < dis)
    df_tmp['dis'] = p2l_dis

    dic_sele_res = {'res_chain': [], 'dis': []}
    for res_i in df_tmp['residue_number'].unique():
        for chain_i in df_tmp['chain_id'].unique():
            df_sub = df_tmp[(df_tmp['residue_number'] == res_i) & (df_tmp['chain_id'] == chain_i)]
            tmp_dis = df_sub['pocket_flag'].values
            if any(tmp_dis):
                dic_sele_res['res_chain'].append(f'{res_i},{chain_i}')
                dic_sele_res['dis'].append(df_sub['dis'].min())
    df_sele_res = pd.DataFrame.from_dict(dic_sele_res)
    df_sele_res = df_sele_res.sort_values(by=['dis'], ascending=[True])
    sele_res = df_sele_res['res_chain'].values[:max_len_protein]

    df_pocket['pocket_flag'] = [
        True if str(df_pocket.loc[i, 'residue_number']) + ',' + str(df_pocket.loc[i, 'chain_id']) in sele_res else False
        for i in range(len(df_pocket))]
    df_pocket = df_pocket[df_pocket['pocket_flag'] == True][df_pocket.columns[:-1]]
    return df_pocket, sele_res


def get_torsion_alt(torsion):
    if torsion > 0:
        return torsion - 180
    elif torsion < 0:
        return torsion + 180
    else:
        return torsion


def get_protein_x_sca(df_resi):
    x_sca = \
        onehot_with_allowset(df_resi['residue_name'].unique()[0], \
                             ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', \
                              'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'], \
                             with_unk=False)
    x_sca = np.array(x_sca).astype(np.float32)
    return x_sca


def get_protein_x_vec(df_resi):
    resi_type = df_resi['residue_name'].unique()[0]
    if resi_type in dic_SC.keys():
        dic_resi_SC = dic_SC[resi_type]
    else:
        dic_resi_SC = {}

    x_vec = []
    for SC_i in range(1, 5):
        if SC_i in dic_resi_SC.keys():
            v = dic_resi_SC[SC_i]
            vec = df_resi[df_resi['atom_name'] == v[2]][['x_coord', 'y_coord', 'z_coord']].values - \
                  df_resi[df_resi['atom_name'] == v[1]][['x_coord', 'y_coord', 'z_coord']].values
            vec = vec.squeeze(0)
        else:
            vec = [0, 0, 0]
        x_vec.append(vec)
    x_vec = np.array(x_vec).astype(np.float32)
    return x_vec


def get_protein_x_sca_vec(df_protein):
    x_sca = []
    x_vec = []
    chain_resi_list = list(df_protein['chain_resi'].unique())
    for chain_resi_i in chain_resi_list:
        df_sub = df_protein[df_protein['chain_resi'] == chain_resi_i]
        x_sca.append(get_protein_x_sca(df_sub))
        x_vec.append(get_protein_x_vec(df_sub))
    x_sca = np.stack(x_sca, axis=0)
    x_vec = np.stack(x_vec, axis=0)
    return x_sca, x_vec


def get_protein_edge_sca_vec(df_protein):
    chain_resi_list = list(df_protein['chain_resi'].unique())

    dic_resi = {}
    for i in range(len(chain_resi_list)):
        dic_resi[i] = {}
        df_i = df_protein[df_protein['chain_resi'] == chain_resi_list[i]]
        dic_resi[i]['df'] = df_i
        chain_resi_i = chain_resi_list[i].split('_')
        dic_resi[i]['chain'] = chain_resi_i[0]
        dic_resi[i]['resi'] = int(chain_resi_i[1])
        dic_resi[i]['CA'] = df_i[(df_i['atom_name'] == 'CA')][['x_coord', 'y_coord', 'z_coord']].values[0]

    edge_sca = np.zeros((len(chain_resi_list), len(chain_resi_list), 3))
    edge_vec = np.zeros((len(chain_resi_list), len(chain_resi_list), 4, 3))
    for i in range(len(chain_resi_list)):
        df_i = dic_resi[i]['df']
        chain_i = dic_resi[i]['chain']
        resi_i = dic_resi[i]['resi']

        for j in range(i, len(chain_resi_list)):
            if i == j:
                continue
            else:
                df_j = dic_resi[j]['df']
                chain_j = dic_resi[j]['chain']
                resi_j = dic_resi[j]['resi']

                if chain_i == chain_j and abs(resi_i - resi_j) == 1:
                    # connected resi
                    # assert resi_i - resi_j == -1 # ensure AA direction
                    CA1 = dic_resi[i]['CA']
                    C = df_i[df_i['atom_name'] == 'C'][['x_coord', 'y_coord', 'z_coord']].values[0]
                    N = df_j[df_j['atom_name'] == 'N'][['x_coord', 'y_coord', 'z_coord']].values[0]
                    CA2 = dic_resi[j]['CA']

                    edge_sca[i, j, 0] = 1
                    edge_sca[j, i, 1] = 1
                    edge_sca[i, j, 2] = np.linalg.norm(CA1 - CA2, ord=2, axis=-1)
                    edge_sca[j, i, 2] = edge_sca[i, j, 2]

                    edge_vec[i, j, 0] = CA1 - CA2
                    edge_vec[i, j, 1] = CA1 - C
                    edge_vec[i, j, 2] = C - N
                    edge_vec[i, j, 3] = N - CA2
                    edge_vec[j, i, 0] = -edge_vec[i, j, 0]
                    edge_vec[j, i, 1] = -edge_vec[i, j, 1]
                    edge_vec[j, i, 2] = -edge_vec[i, j, 2]
                    edge_vec[j, i, 3] = -edge_vec[i, j, 3]
                else:
                    # separated resi
                    CA1 = dic_resi[i]['CA']
                    CA2 = dic_resi[j]['CA']

                    edge_sca[i, j, 2] = np.linalg.norm(CA1 - CA2, ord=2, axis=-1)
                    edge_sca[j, i, 2] = edge_sca[i, j, 2]

                    edge_vec[i, j, 0] = CA1 - CA2
                    edge_vec[j, i, 0] = -edge_vec[i, j, 0]
    return edge_sca, edge_vec


def get_protein_edge_connect_info(df_protein):
    chain_resi_list = list(df_protein['chain_resi'].unique())

    dic_resi = {}
    for i in range(len(chain_resi_list)):
        dic_resi[i] = {}
        df_i = df_protein[df_protein['chain_resi'] == chain_resi_list[i]]
        dic_resi[i]['df'] = df_i
        chain_resi_i = chain_resi_list[i].split('_')
        dic_resi[i]['chain'] = chain_resi_i[0]
        dic_resi[i]['resi'] = int(chain_resi_i[1])
        dic_resi[i]['CA'] = df_i[(df_i['atom_name'] == 'CA')][['x_coord', 'y_coord', 'z_coord']].values[0]

    resi_connect = np.zeros((len(chain_resi_list), len(chain_resi_list)))
    for i in range(len(chain_resi_list)):
        chain_i = dic_resi[i]['chain']
        resi_i = dic_resi[i]['resi']

        for j in range(i, len(chain_resi_list)):
            if i == j:
                continue
            else:
                chain_j = dic_resi[j]['chain']
                resi_j = dic_resi[j]['resi']

                if chain_i == chain_j and abs(resi_i - resi_j) == 1:
                    resi_connect[i, j] = 1
                    resi_connect[j, i] = 1
    return resi_connect


def get_protein_dismap(df_protein):
    chain_resi_list = list(df_protein['chain_resi'].unique())

    dic_data = {}
    dic_data['CA_coor'] = []
    dic_data['CB_coor'] = []
    for i in range(len(chain_resi_list)):
        df_i = df_protein[df_protein['chain_resi'] == chain_resi_list[i]]
        resi_type = df_i['residue_name'].values[0]

        CA_coor = df_i[(df_i['atom_name'] == 'CA')][['x_coord', 'y_coord', 'z_coord']].values[0]
        if resi_type == 'GLY':
            CB_coor = df_i[(df_i['atom_name'] == 'CA')][['x_coord', 'y_coord', 'z_coord']].values[0]
        else:
            CB_coor = df_i[(df_i['atom_name'] == 'CB')][['x_coord', 'y_coord', 'z_coord']].values[0]

        dic_data['CA_coor'].append(CA_coor)
        dic_data['CB_coor'].append(CB_coor)

    CA_coor = np.stack(dic_data['CA_coor'], axis=0)
    CB_coor = np.stack(dic_data['CB_coor'], axis=0)

    # CACB_dismap = np.linalg.norm(CA_coor - CB_coor, ord=2, axis=-1)
    # CBCB_dismap = np.linalg.norm(CB_coor - CB_coor, ord=2, axis=-1)
    # CACB_dismap = np.stack([CACB_dismap, CBCB_dismap], axis=0)
    return CB_coor


def get_AA_coor(df_protein):
    chain_resi_list = list(df_protein['chain_resi'].unique())

    # for MC
    dic_MC_coor = {}
    dic_MC_coor['N_coor'] = []
    dic_MC_coor['C_coor'] = []
    dic_MC_coor['O_coor'] = []
    for i in range(len(chain_resi_list)):
        df_i = df_protein[df_protein['chain_resi'] == chain_resi_list[i]]

        N_coor = df_i[df_i['atom_name'] == 'N'][['x_coord', 'y_coord', 'z_coord']].values[0]
        C_coor = df_i[df_i['atom_name'] == 'C'][['x_coord', 'y_coord', 'z_coord']].values[0]
        O_coor = df_i[df_i['atom_name'] == 'O'][['x_coord', 'y_coord', 'z_coord']].values[0]

        dic_MC_coor['N_coor'].append(N_coor)
        dic_MC_coor['C_coor'].append(C_coor)
        dic_MC_coor['O_coor'].append(O_coor)
    N_coor = np.stack(dic_MC_coor['N_coor'], axis=0)
    C_coor = np.stack(dic_MC_coor['C_coor'], axis=0)
    O_coor = np.stack(dic_MC_coor['O_coor'], axis=0)

    # for SC
    SC_coor = []
    SC_mask = []
    for i in range(len(chain_resi_list)):
        df_i = df_protein[df_protein['chain_resi'] == chain_resi_list[i]]
        resi_type = df_i['residue_name'].unique()[0]
        dic_resi_SC = dic_SC[resi_type]

        SC_tmp = []
        mask_tmp = []
        for SC_i in range(1, 5):
            if SC_i in dic_resi_SC.keys():
                a = dic_resi_SC[SC_i][2]
                SC_tmp.append(df_i[df_i['atom_name'] == a][['x_coord', 'y_coord', 'z_coord']].values[0])
                mask_tmp.append(True)
            else:
                SC_tmp.append(np.zeros(3))
                mask_tmp.append(False)
        SC_tmp = np.stack(SC_tmp, axis=0)

        SC_coor.append(SC_tmp)
        SC_mask.append(mask_tmp)

    SC_coor = np.stack(SC_coor, axis=0)
    SC_mask = np.array(SC_mask)

    # cat all
    MC_coor = np.stack([N_coor, C_coor, O_coor], axis=1)
    MC_msak = np.ones((MC_coor.shape[0], MC_coor.shape[1])).astype(np.bool_)
    MCSC_coor = np.concatenate([MC_coor, SC_coor], axis=1)
    MCSC_mask = np.concatenate([MC_msak, SC_mask], axis=1)

    return MCSC_coor, MCSC_mask


def encode_pocket(df_pocket):
    df_CA = df_pocket[(df_pocket['atom_name'] == 'CA')]
    p_x_sca, p_x_vec = get_protein_x_sca_vec(df_pocket)
    # p_edge_sca, p_edge_vec = get_protein_edge_sca_vec(df_pocket_holo_true)
    p_coor = df_CA[['x_coord', 'y_coord', 'z_coord']].values
    CB_coor = get_protein_dismap(df_pocket)
    MCSC_coor, MCSC_mask = get_AA_coor(df_pocket)
    resi_connect = get_protein_edge_connect_info(df_pocket)
    return [p_x_sca, p_x_vec, resi_connect, p_coor, CB_coor, MCSC_coor, MCSC_mask]


def get_torsion(tor_path, df_protein, df_pocket):
    if isinstance(tor_path, str):
        torsion = pickle.load(open(tor_path, 'rb'))
    else:
        torsion = tor_path
    SCtorsion = torsion['sc_torsion']
    SCtorsion = [i for i in SCtorsion if not (len(i) > 0 and isinstance(i[0], type(None)))]
    assert len(SCtorsion) == len(df_protein['chain_resi'].unique()), \
        f"{len(SCtorsion)}, {len(df_protein['chain_resi'].unique())}"

    protein_chainres = df_protein['chain_resi'].unique()
    pocket_chainres = df_pocket['chain_resi'].unique()
    dic_tmp = {k: v for k, v in zip(protein_chainres, SCtorsion)}
    pocket_SCtorsion_nopad = [dic_tmp[chain_resi] for chain_resi in pocket_chainres]
    pocket_aa = df_pocket[df_pocket['atom_name'] == 'CA']['residue_name'].values
    pocket_SCtorsion = []
    pocket_SCtorsion_alt = []
    pocket_SCtorsion_mask = []
    for i in range(len(pocket_SCtorsion_nopad)):
        res_torsion = pocket_SCtorsion_nopad[i]
        # assert len(res_torsion) == len(dic_SC[res_type].keys()) # mute for CYS S-S
        if pocket_aa[i] == 'ASP':
            res_torsion_alt = [res_torsion[0], get_torsion_alt(res_torsion[1])]
        elif pocket_aa[i] == 'GLU':
            res_torsion_alt = [res_torsion[0], res_torsion[1], get_torsion_alt(res_torsion[2])]
        elif pocket_aa[i] == 'PHE':
            res_torsion_alt = [res_torsion[0], get_torsion_alt(res_torsion[1])]
        elif pocket_aa[i] == 'TYR':
            res_torsion_alt = [res_torsion[0], get_torsion_alt(res_torsion[1])]
        else:
            res_torsion_alt = res_torsion

        res_torsion_mask = [1] * len(res_torsion) + [0] * (4 - len(res_torsion))
        res_torsion = res_torsion + [0] * (4 - len(res_torsion))
        res_torsion_alt = res_torsion_alt + [0] * (4 - len(res_torsion_alt))
        pocket_SCtorsion.append(res_torsion)
        pocket_SCtorsion_alt.append(res_torsion_alt)
        pocket_SCtorsion_mask.append(res_torsion_mask)
    pocket_SCtorsion = np.array(pocket_SCtorsion)
    pocket_SCtorsion_alt = np.array(pocket_SCtorsion_alt)
    pocket_SCtorsion_mask = np.array(pocket_SCtorsion_mask)
    pocket_SCtorsion_data = [pocket_SCtorsion, pocket_SCtorsion_alt, pocket_SCtorsion_mask]

    return pocket_SCtorsion_data


if __name__ == '__main__':
    pass





















