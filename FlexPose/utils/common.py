import os
import shutil
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))

import argparse
import numpy as np
import random
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib widget
import seaborn as sns
import pickle
from rdkit import Chem

import torch
import torch_geometric


EPS = 1e-8

def delmkdir(path, remove_old=True):
    isexist = os.path.exists(path)
    if not isexist:
        os.makedirs(path)
    if isexist == True and remove_old:
        shutil.rmtree(path)
        os.makedirs(path)


def try_do(intup):
    f, task = intup
    try:
        f(task)
        return True
    except:
        return False


def summarize_model(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


class info_recorder():
    def __init__(self, name):
        self.name = name
        self.trj_dic = {}
        self.batch_dic = {}

    def reset_trj(self, k):
        self.trj_dic[k] = []
        self.batch_dic[k] = []

    def update_trj(self, batch_size=1):
        for k in self.trj_dic.keys():
            self.trj_dic[k].append(np.nanmean(self.batch_dic[k]) / batch_size)
            self.batch_dic[k] = []

    def __call__(self, k, x):
        self.batch_dic[k].append(x)

    def save_trj(self):
        np.savez('./{}.npz'.format(self.name), **self.trj_dic)

    def load_history(self, restart=None):
        history = np.load('./{}.npz'.format(self.name))
        if restart == None:
            for k in history.keys():
                self.trj_dic[k] = list(history[k])
        else:
            for k in history.keys():
                self.trj_dic[k] = list(history[k])[:restart]

    def plot(self, y_lim=None, plot_flag=True):
        text_size = 17
        ticklabel_size = 15
        legend_size = 15
        color_palette = sns.color_palette()
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        for i, k in enumerate(self.trj_dic.keys()):
            x = np.arange(len(self.trj_dic[k]))
            ax.plot(x, self.trj_dic[k], label=k, color=color_palette[i])

        ax.set_title(self.name, fontsize=text_size)
        ax.grid(False)
        if y_lim != None:
            ax.set_ylim((y_lim[0], y_lim[1]))
        # ax.set_ylabel('loss', fontsize=text_size)
        ax.set_xlabel('Iterations', fontsize=text_size)
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)
        ax.axvline(x=0, color='grey', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        # x_major_locator=plt.MultipleLocator(20000)
        # ax.xaxis.set_major_locator(x_major_locator)
        ax.tick_params(labelsize=ticklabel_size)
        [ax.spines[ax_j].set_color('black') for ax_j in ax.spines.keys()]
        ax.tick_params(bottom=True, left=True, direction='out', width=2, length=5, color='black')
        fig.tight_layout()
        ax.legend(frameon=False, prop={'size': legend_size}) #, bbox_to_anchor=(1, 1))  # loc='upper right', markerscale=1000)
        if plot_flag:
            plt.show()
        else:
            plt.savefig('./' + self.name + '.svg', bbox_inches='tight', dpi=600)
        plt.close()

    def print_info(self):
        print(self.name)
        for k in self.trj_dic.keys():
            print(k.ljust(20, '.') + '{:.6f}'.format(self.trj_dic[k][-1]).rjust(20, '.'))

def set_all_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def fix_bool(args):
    args_dict = vars(args)
    new_dic = {}
    for k, v in args_dict.items():
        if isinstance(v, str):
            if v.upper() == 'TRUE':
                v = True
            elif v.upper() == 'FALSE':
                v = False
            elif v.upper() == 'NONE':
                v = None
        new_dic[k] = v
    return argparse.Namespace(**new_dic)


def split_rate(data_split_rate):
    split_str = None
    for i in ['-', '_', ',']:
        if i in data_split_rate:
            split_str = i
    assert split_str != None
    data_split_rate = list(map(lambda x: float(x), data_split_rate.split(split_str)))
    assert np.array(data_split_rate).sum() == 1
    return data_split_rate


def print_args(args):
    print('=' * 30 + ' Current settings ' + '=' * 30)
    for k, v in args.__dict__.items():
        print(k.ljust(40, '.'), v)
    print('=' * (60 + len(' Current settings ')))


def load_idx_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line[:-1] if line[-1] == '\n' else line for line in lines]
    return lines


def save_idx_list(idx_list, file_path):
    with open(file_path, 'w') as f:
        f.write('\n'.join(idx_list) + '\n')


def save_val(data, rank, f_name='tmp_val'):
    with open(f'{f_name}_{rank}.pkl', 'wb') as f:
        pickle.dump(data, f)


def load_val(f_name='tmp_val'):
    f_list = [f for f in os.listdir() if f.startswith(f_name)]

    for i, f_name in enumerate(f_list):
        with open(f_name, 'rb') as f:
            dic_tmp = pickle.load(f)
        if i == 0:
            dic = dic_tmp
        else:
            for k in dic.keys():
                dic[k] = dic[k] + dic_tmp[k]
        os.remove(f_name)
    return dic


def is_tensor(x):
    if isinstance(x, torch.Tensor) or \
       isinstance(x, torch.LongTensor) or \
       isinstance(x, torch.FloatTensor) or \
       isinstance(x, torch.BoolTensor) or \
       isinstance(x, torch.HalfTensor) or \
       isinstance(x, torch_geometric.data.data.Data):
        return True
    else:
        return False


def is_notebook():
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


class suppress_stdout_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])




def read_rdkit_mol(mol, silence=False):
    if mol.endswith('mol2'):
        mol = Chem.MolFromMol2File(mol)
    else:
        if not silence:
            print('Strongly recommend using mol2 format, model was trained with mol2 files !!!')
        if mol.endswith('pdb'):
            mol = Chem.MolFromPDBFile(mol)
        elif mol.endswith('mol'):
            mol = Chem.MolFromMolFile(mol)
        elif mol.endswith('sdf'):
            SD = Chem.SDMolSupplier(mol)
            mol = [x for x in SD][0]
        else:
            mol = Chem.MolFromSmiles(mol)
    return mol



if __name__ == '__main__':
    # for testing
    pass
