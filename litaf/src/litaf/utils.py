'''
Submodule containing utility functions

'''

import logging
from absl import logging as absl_logging
import yaml
import os

import py3Dmol

import numpy as np
import matplotlib.pyplot as plt
from alphafold.data import parsers
from colabfold.alphafold.models import load_models_and_params
from colabfold.colabfold import pymol_color_list, alphabet_list
from alphapulldown.utils import parse_fasta
from alphafold.common import residue_constants


def make_dir_monomer_dictionary(monomer_objects_dir):
    """
    a function to gather all monomers across different monomer_objects_dir

    args
    monomer_objects_dir: a list of directories where monomer objects are stored, given by FLAGS.monomer_objects_dir
    """
    output_dict = dict()
    for m_dir in monomer_objects_dir:
        monomers = os.listdir(m_dir)
        for m in monomers:
            if m.endswith('.pkl') or m.endswith('.pkl.bz2'):
                output_dict[m.split('.')[0]] = m_dir
    return output_dict

def create_colabfold_runners(
    model_suffix: str,
    num_models: int,
    use_templates: bool,
    num_recycles: int,
    data_dir: str,
    max_seq: int,
    max_extra_seq: str,
    num_pred: int,
    use_dropout: bool,
    use_cluster_profile: bool,
    save_all: bool,
    ) -> list:

    if model_suffix == '_ptm':
        rank_by = 'plddt'
    elif model_suffix == '_multimer_v3':
        rank_by = 'multimer'
    else:
        raise ValueError(f"Unrecognized model type with suffix: {model_suffix}")
    
    model_runners_tmp = load_models_and_params(num_models = num_models,
                                                use_templates = use_templates,
                                                model_suffix = model_suffix,
                                                data_dir=data_dir,
                                                num_recycles=num_recycles,
                                                max_seq=max_seq,
                                                max_extra_seq=max_extra_seq,
                                                use_dropout=use_dropout,
                                                use_cluster_profile=use_cluster_profile,
                                                save_all=save_all,
                                                rank_by=rank_by,)
    model_runners = {}
    for mr in model_runners_tmp:
        for i in range(num_pred):
            model_runners[f'{mr[0]}_pred_{i}'] = mr[1:]

    return model_runners

def load_mutation_dict(file: str) -> dict:
    logging.info(f'Reading mutation information from {file}')
    with open(file, 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.loader.SafeLoader)

    return yaml_dict

def remove_msa_for_template_aligned_regions(feature_dict):
    mask = np.zeros(feature_dict['seq_length'][0], dtype=bool)
    for templ in feature_dict['template_sequence']:
        for i,aa in enumerate(templ.decode("utf-8")):
            if aa != '-':
                mask[i] = True              
    feature_dict['deletion_matrix_int'][:,mask] = 0
    feature_dict['msa'][:,mask] = 21


def setup_logging(log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename = log_file,
                        format="%(name)s - %(levelname)s - %(message)s",
                        level = logging.INFO,
                        filemode='a',)
    absl_logging.set_verbosity(absl_logging.INFO)


def iter_seqs(fasta_fns):
    for fasta_path in fasta_fns:
        with open(fasta_path, "r") as f:
            sequences, descriptions = parse_fasta(f.read())
            for seq, desc in zip(sequences, descriptions):
                yield seq, desc

def encode_seqs(seqs):
    arr = np.zeros([len(seqs), seqs.shape[1], 22])
    for i,seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            arr[i,j,aa] += 1

    return arr.reshape([len(seqs), seqs.shape[1]*22])


def decode_ohe_seqs(ohe_seqs):
    ohe_seqs = ohe_seqs.reshape([len(ohe_seqs), ohe_seqs.shape[1]/22, 22])
    arr = np.zeros([ohe_seqs[0], ohe_seqs[1]])
    for i, ohe_seq in enumerate(ohe_seqs):
        for j, ohe_aa in enumerate(ohe_seq):
            arr[i, j] = np.argmax(ohe_aa)

    return arr

def consensusVoting(seqs):
    ## Find the consensus sequence
    ## Modified from https://github.com/HWaymentSteele/AF_Cluster/blob/main/scripts/utils.py
    consensus = ""
    n_chars = len(seqs[0])
    for i in range(n_chars):
        baseArray = [x[i] for x in seqs]
        baseCount = np.zeros(22)
        for a, idx in residue_constants.HHBLITS_AA_TO_ID.items():
            baseCount[idx] += baseArray.count(a)
        vote = np.argmax(baseCount)
        consensus += residue_constants.ID_TO_HHBLITS_AA[vote]

    return consensus

def to_string_seq(seq):
    str_seq = ''
    for aa in seq:
        str_seq += residue_constants.ID_TO_HHBLITS_AA[aa]

    return str_seq

def cluster_analysis(feature_dict):
    nl = '\n'
    L = feature_dict['seq_length'][0]
    cluster_msa = [to_string_seq(x) for x in feature_dict['msa']]
    cs = consensusVoting(encode_seqs(feature_dict['msa']))
    avg_dist_to_cs = np.mean([1-levenshtein(x,cs)/L for x in cluster_msa])
    avg_dist_to_query = np.mean([1-levenshtein(x,feature_dict['sequence'][0].decode("utf-8"))/L for x in cluster_msa])
    logging.info(f"Generated cluster {clst} with consensus sequence:{nl}{cs}{nl}"
        f"Average distance from consensus sequence: {avg_dist_to_cs}{nl}"
        f"Average distance from query sequence: {avg_dist_to_query}{nl}")



def plot_msa_landscape(x, y, qx, qy, labels, ax_labels):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x[labels == -1], y[labels == -1], color='lightgray', marker='x', label='unclustered')
    ax.scatter(x[labels != -1], y[labels != -1], c=labels[labels != -1], marker='o')
    ax.scatter(qx,qy, color='red', marker='*', s=150, label='Ref Seq')
    #ax.scatter(x[0],y[0], color='blue', marker='*', s=50, label='Best Seq')
    plt.legend()
    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    plt.tight_layout()

    return fig

def show_pdb(pdb_file, n_chains, show_sidechains=False, show_mainchains=False, color="lDDT"):
    #The function show_pdb is adapted from ColabFold
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js',)
    view.addModel(open(pdb_file,'r').read(),'pdb')

    if color == "lDDT":
        view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
    elif color == "rainbow":
        view.setStyle({'cartoon': {'color':'spectrum'}})
    elif color == "chain":
        for n,chain,color in zip(range(n_chains),alphabet_list,pymol_color_list):
           view.setStyle({'chain':chain},{'cartoon': {'color':color}})

    if show_sidechains:
        BB = ['C','O','N']
        view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                        {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        
    if show_mainchains:
        BB = ['C','O','N','CA']
        view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

    view.setClickable({},
                        'true',
                        "function(atom, viewer, event, container){"\
                        "if(atom.label){viewer.removeLabel(atom.label);delete atom.label;}" \
                        "else{atom.label=viewer.addLabel(atom.resn+atom.resi+':'+atom.b, {'position': atom});}\n}")

    view.zoomTo()
    return view