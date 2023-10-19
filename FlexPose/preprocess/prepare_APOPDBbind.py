import os
import shutil
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))

import argparse
import pandas as pd
from ray.util.multiprocessing import Pool
from tqdm import tqdm

from FlexPose.utils.common import print_args, delmkdir
from FlexPose.preprocess.prepare_for_training import try_prepare_APOPDBbind, save_APOPDBbind


if __name__ == '__main__':
    # main args
    parser = argparse.ArgumentParser()

    # data source
    parser.add_argument('--apobind_path', type=str,
                        default='/home/dtj/work_site/test/tmp/data/apobind', help='APObind dataset path')
    parser.add_argument('--pdbbind_path', type=str,
                        default='/home/dtj/work_site/test/tmp/data/v2020-PL', help='PDBbind dataset path')
    parser.add_argument('--apo_info_path', type=str,
                        default='/home/dtj/work_site/test/tmp/data/apobind_all.csv', help='APObind apo-holo mapping csv path (provided by APObind)')
    parser.add_argument('--aff_info_path', type=str,
                        default='/home/dtj/work_site/test/tmp/data/index/INDEX_general_PL_data.2020',
                        help='PDBbind affinity data path')
    parser.add_argument('--aug_path', type=str,
                        default='/home/dtj/work_site/test/tmp/data/pdbbind_MC', help='Rosetta decoys (pseudo apo structures)')

    # parameters
    parser.add_argument('--max_len_pocket', type=int,
                        default=50, help='max number of protein pocket residues')
    parser.add_argument('--max_len_ligand', type=int,
                        default=50, help='max number of ligand atoms')

    # other
    parser.add_argument('--tmp_path', type=str,
                        default='./tmp', help='tmp file for temporary saving')

    # output
    parser.add_argument('--save_path', type=str,
                        default='/home/dtj/work_site/test/tmp/data/processed_data_maxp50_maxl50', help='output path (preprocessed), npz or pkl')


    args = parser.parse_args()
    print_args(args)

    delmkdir(args.tmp_path)
    delmkdir(args.save_path)

    pdb_list = [i.split('.')[0] for i in os.listdir(args.pdbbind_path)]

    pdb_list = [i.split('.')[0] for i in os.listdir('/root/autodl-tmp/drug/qita/demo_test_decoy/')]

    df_apo = pd.read_csv(args.apo_info_path, index_col=0)
    df_apo['apo_bind_res'] = df_apo['apo_bind_res'].apply(lambda x: x[1:] if x.startswith('=') else x)

    ####################################################################################################################
    # 1. try preparing all data
    ####################################################################################################################
    print('Preparing tasks...')
    task = []
    for pdb_id in pdb_list:
        df_apo_sub = df_apo[df_apo['holo_id'] == pdb_id]
        have_apo = None
        if not os.path.exists(args.save_path + '/{}.pkl'.format(pdb_id)):
            task.append((args.save_path, pdb_id, df_apo_sub, args.apobind_path, args.pdbbind_path, args.aff_info_path,
                         args.aug_path, df_apo_sub, have_apo, args.max_len_pocket, args.max_len_ligand, args.tmp_path))
    print(f'Task num: {len(task)}')

    print(f'Begin...')
    # for t in tqdm(task[2:5]):
    #     save_APOPDBbind(t)
    # sys.exit()
    pool = Pool()
    for _ in pool.map(try_prepare_APOPDBbind, task):
        pass
    print('DONE')

    ####################################################################################################################
    # 2. re-preparing failed data
    ####################################################################################################################
    print('Preparing task...')
    task = []
    for pdb_id in pdb_list:
        df_apo_sub = df_apo[df_apo['holo_id'] == pdb_id]
        if not os.path.exists(args.save_path + '/{}.pkl'.format(pdb_id)):
            have_apo = False
            task.append(
                (args.save_path, pdb_id, df_apo_sub, args.apobind_path, args.pdbbind_path, args.aff_info_path,
                 args.aug_path, df_apo_sub, have_apo, args.max_len_pocket, args.max_len_ligand, args.tmp_path))
    print(f'Task num: {len(task)}')

    print(f'Begin...')
    # for t in tqdm(task[:5]):
    #     save_APOPDBbind(t)
    # sys.exit()
    pool = Pool()
    for _ in pool.map(try_prepare_APOPDBbind, task):
        pass
    shutil.rmtree(args.tmp_path)
    print('DONE')







