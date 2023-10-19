import os
import sys
import re


pyrosetta_SC_path = os.path.join(os.path.dirname(__file__), './residue_types/')
aa_type = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
for aa in aa_type:
    assert aa+'.params' in os.listdir(pyrosetta_SC_path)

def get_sc_from_param(param_file):
    with open(param_file, 'r') as f:
        lines = f.readlines()
    dic_tmp = {}
    for line in lines:
        if line.startswith('CHI'):
            line_split = re.split('[ \n]+', line)
            idx = int(line_split[1])
            atoms = [line_split[i] for i in range(2,6)]
            dic_tmp[idx] = atoms
    return dic_tmp

dic_SC = {}
for aa in aa_type:
    param_file = pyrosetta_SC_path + aa + '.params'
    dic_SC[aa] = get_sc_from_param(param_file)


if __name__=='__main__':
    print(dic_SC)




