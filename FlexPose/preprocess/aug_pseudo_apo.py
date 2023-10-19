import os
import shutil
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))

import argparse
from ray.util.multiprocessing import Pool
import numpy as np
import scipy.spatial
from einops import rearrange
import random
import pickle

import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta import core

from modeller import environ
from modeller.scripts import complete_pdb

from FlexPose.utils.common import delmkdir
from FlexPose.utils.pdbbind_preprocess import read_mol_from_pdbbind, get_true_posi



def random_sc(pose, res_list=None, pert=180):
    # random chi
    if isinstance(res_list, type(None)):
        res_list = range(1, pose.size() + 1)
    for i in res_list:
        res = pose.residue(i)
        for chino, chi in enumerate(res.chi(), start=1):
            res.set_chi(chino, chi + random.uniform(-pert, pert))

def get_FastRelax(pose, res_list=None, flexbb=True):
    if isinstance(res_list, type(None)):
        res_list = range(1, pose.size() + 1)
    res_selector = core.select.residue_selector.ResidueIndexSelector(','.join([str(i) for i in res_list]))

    # get TaskFactory
    tf = core.pack.task.TaskFactory()
    tf.push_back(core.pack.task.operation.InitializeFromCommandline())
    # tf.push_back(core.pack.task.operation.IncludeCurrent())
    tf.push_back(core.pack.task.operation.NoRepackDisulfides())
    tf.push_back(core.pack.task.operation.RestrictToRepacking())
    restrict_to_focus = core.pack.task.operation.OperateOnResidueSubset(core.pack.task.operation.PreventRepackingRLT(),
                                                                        res_selector,
                                                                        True)  # True indicates flipping the selection
    tf.push_back(restrict_to_focus)
    # pyrosetta.toolbox.generate_resfile.generate_resfile_from_pose(original_pose, f'{sub_MC_path}/protein_resfile',
    #                                                               pack=True, design=False, input_sc=False)
    # tf.push_back(core.pack.task.operation.ReadResfile(f'{sub_MC_path}/protein_resfile'))
    # print(tf.create_task_and_apply_taskoperations(pose))

    # test tf
    # packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
    # packer = pyrosetta.rosetta.protocols.minimization_packing.MinPackMover()
    # packer.task_factory(tf)
    # packer.apply(pose)

    # get FastRelax
    mm = core.kinematics.MoveMap()
    mm.set_jump(False)
    for i in range(1, pose.size()+1):
        if i in res_list:
            mm.set_chi(i, True)
            mm.set_bb(i, flexbb)
        else:
            mm.set_chi(i, False)
            mm.set_bb(i, False)
    # mmf = core.select.movemap.MoveMapFactory()
    # mmf.all_bb(False)
    # mmf.all_bondangles(False)
    # mmf.all_bondlengths(False)
    # mmf.all_branches(False)
    # mmf.all_chi(False)
    # mmf.all_jumps(False)
    # mmf.all_nu(False)
    # mmf.set_cartesian(False)
    # mmf.add_bb_action(core.select.movemap.move_map_action.mm_enable, pocket_selector)
    # mmf.add_chi_action(core.select.movemap.move_map_action.mm_enable, pocket_selector)
    # mm = mmf.create_movemap_from_pose(pose)

    fr = pyrosetta.rosetta.protocols.relax.FastRelax()
    # fr.max_iter(100)
    fr.constrain_relax_to_start_coords(False)
    fr.set_movemap_disables_packing_of_fixed_chi_positions(True)
    fr.set_task_factory(tf)
    fr.set_movemap(mm)
    # fr.set_movemap_factory(mmf)
    fr.cartesian(False)
    fr.set_scorefxn(core.scoring.ScoreFunctionFactory.create_score_function('ref2015_cart'))
    fr.min_type('dfpmin_armijo_nonmonotone')  # For non-Cartesian scorefunctions, use "dfpmin_armijo_nonmonotone", else lbfgs_armijo_nonmonotone

    return fr

def get_torsion(pose):
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


def try_gen_pose(task, pose):
    try:
        task.apply(pose)
        return True
    except:
        return False


def run_single_pdbbind(tup_in):
    pdbbind_path, apobind_path, MC_path, pdb_id, n_rand_pert, n_fixbb_repack, n_flexbb_repack, rand_pert_range = tup_in
    sub_pdbbind_path = f'{pdbbind_path}/{pdb_id}'
    sub_apobind_path = f'{apobind_path}/{pdb_id}'
    sub_MC_path = f'{MC_path}/{pdb_id}'
    delmkdir(sub_MC_path)

    ####################################################################################################################
    # repaire protein
    ####################################################################################################################
    env_ = environ()
    env_.libs.topology.read(file='$(LIB)/top_heav.lib')
    env_.libs.parameters.read(file='$(LIB)/par.lib')
    pdb_m = complete_pdb(env_, f'{sub_pdbbind_path}/{pdb_id}_protein.pdb')
    pdb_m.write(f'{sub_MC_path}/protein_modeller.pdb')
    os.system(f'grep HETATM {sub_pdbbind_path}/{pdb_id}_protein.pdb >> {sub_MC_path}/protein_modeller.pdb')  # add ion
    os.system(f'grep -v END {sub_MC_path}/protein_modeller.pdb > {sub_MC_path}/protein_repaired.pdb')

    if os.path.exists(f'{sub_apobind_path}/{pdb_id}_apo_added.pdb'):
        have_apo = True
    else:
        have_apo = False
    if have_apo:
        pdb_m = complete_pdb(env_, f'{sub_apobind_path}/{pdb_id}_apo_added.pdb')
        pdb_m.write(f'{sub_MC_path}/protein_modeller.pdb')
        os.system(f'grep -v END {sub_MC_path}/protein_modeller.pdb > {sub_MC_path}/protein_repaired_apo.pdb')

    ####################################################################################################################
    # init rosetta
    ####################################################################################################################
    # check https://new.rosettacommons.org/docs/latest/full-options-list for opts
    # -ex3 -ex4 -ex1aro -ex2aro
    opts = '-ex1 true -packing:ex1:level 1 -ex2 true -packing:ex2:level 1 -extrachi_cutoff 0 -ignore_unrecognized_res true -relax:default_repeats 3'
    pyrosetta.distributed.init(opts)

    original_pose = pyrosetta.io.pose_from_pdb(f'{sub_MC_path}/protein_repaired.pdb')
    original_pose.dump_pdb(f'{sub_MC_path}/origin.pdb')
    if have_apo:
        apo_pose = pyrosetta.io.pose_from_pdb(f'{sub_MC_path}/protein_repaired_apo.pdb')

    # select local res
    ligand_mol = read_mol_from_pdbbind(pdbbind_path, pdb_id)
    ligand_coor = get_true_posi(ligand_mol)
    neighbor = []
    for i in range(1, original_pose.size() + 1):
        res = original_pose.residue(i)
        if not res.is_polymer():
            continue
        CA_xyz = res.atom('CA').xyz()
        CA_coor = np.array([CA_xyz.x, CA_xyz.y, CA_xyz.z])
        CA_lig_dist = scipy.spatial.distance.cdist(rearrange(CA_coor, 'c -> () c'), ligand_coor, metric='euclidean')
        CA_dist = CA_lig_dist.min()
        if CA_dist < 10:
            neighbor.append(i)


    ####################################################################################################################
    # get apo/holo torsion
    ####################################################################################################################
    dic_torsion = get_torsion(original_pose)
    pickle.dump(dic_torsion, open(f'{sub_MC_path}/torsion.npz', 'wb'))

    if have_apo:
        dic_torsion = get_torsion(apo_pose)
        pickle.dump(dic_torsion, open(f'{sub_MC_path}/torsion_apo.npz', 'wb'))


    ####################################################################################################################
    # random chi
    ####################################################################################################################
    for i in range(n_rand_pert):
        pose = original_pose.clone()
        random_sc(pose, res_list=neighbor, pert=rand_pert_range)
        pose.dump_pdb(f'{sub_MC_path}/rand_pert_{i}.pdb')

        dic_torsion = get_torsion(pose)
        pickle.dump(dic_torsion, open(f'{sub_MC_path}/torsion_rand_pert_{i}.npz', 'wb'))


    ####################################################################################################################
    # FastRelax Fixbb
    ####################################################################################################################
    # get TaskFactory
    fr = get_FastRelax(original_pose, res_list=neighbor, flexbb=False)
    for i in range(n_fixbb_repack):
        pose = original_pose.clone()
        success_gen = try_gen_pose(fr, pose)
        if success_gen:
            pose.dump_pdb(f'{sub_MC_path}/fixbb_repack_{i}.pdb')
            
            dic_torsion = get_torsion(pose)
            pickle.dump(dic_torsion, open(f'{sub_MC_path}/torsion_fixbb_repack_{i}.npz', 'wb'))


    ####################################################################################################################
    # FastRelax Flexbb
    ####################################################################################################################
    # get TaskFactory
    fr = get_FastRelax(original_pose, res_list=neighbor, flexbb=True)
    for i in range(n_flexbb_repack):
        pose = original_pose.clone()
        success_gen = try_gen_pose(fr, pose)
        if success_gen:
            # rmsd = pyrosetta.rosetta.core.scoring.calpha_superimpose_pose(pose, original_pose)
            pose.dump_pdb(f'{sub_MC_path}/flexbb_repack_{i}.pdb')
            
            dic_torsion = get_torsion(pose)
            pickle.dump(dic_torsion, open(f'{sub_MC_path}/torsion_flexbb_repack_{i}.npz', 'wb'))


def try_prepare(*args, **kwargs):
    try:
        run_single_pdbbind(*args, **kwargs)
    except:
        pass


if __name__ == '__main__':
    # main args
    parser = argparse.ArgumentParser()

    # data source
    parser.add_argument('--apobind_path', type=str,
                        default='/root/autodl-tmp/drug/data/apobind', help='APObind dataset path')
    parser.add_argument('--pdbbind_path', type=str,
                        default='/root/autodl-tmp/drug/data/v2020-PL', help='PDBbind dataset path')

    # parameters
    parser.add_argument('--n_rand_pert', type=int,
                        default=6, help='number of random perturbation decoy')
    parser.add_argument('--rand_pert_range', type=int,
                        default=30, help='max degree for random perturbation')
    parser.add_argument('--n_fixbb_repack', type=int,
                        default=3, help='number of repacking decoy with fixed bb')
    parser.add_argument('--n_flexbb_repack', type=int,
                        default=3, help='number of repacking decoy with flexible bb')

    # output
    parser.add_argument('--save_path', type=str,
                        default='/root/autodl-tmp/drug/data/pdbbind_MC', help='output path')
          
    args = parser.parse_args()
    
    print('Preparing tasks...')
    pdb_list = os.listdir(args.pdbbind_path)
    delmkdir(args.save_path)
    task = []
    for idx, pdb_id in enumerate(pdb_list):
        sub_path = f'{args.save_path}/{pdb_id}'
        task.append((args.pdbbind_path, args.apobind_path, args.save_path, pdb_id,
                 args.n_rand_pert, args.n_fixbb_repack, args.n_flexbb_repack,
                 args.rand_pert_range))
    print('task num', len(task))


    print('Begin')
    # from tqdm import tqdm
    # for i in tqdm(task[:5]):
    #     run_single_pdbbind(i)
    #     # break
    # sys.exit()
    pool = Pool()
    for _ in pool.map(try_prepare, task):
        pass
    print('DONE')




















