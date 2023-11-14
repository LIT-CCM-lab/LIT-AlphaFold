'''
Submodule containing utility functions

'''

import os
import logging
import pickle

import numpy as np
from alphafold.data import parsers
from colabfold.alphafold.models import load_models_and_params
from alphapulldown.utils import check_empty_templates
from litaf_development.objects import MultimericObject


def read_all_proteins(fasta_path) -> list:
    """
    A function to read all proteins in the file
    Modified version of the original alphapulldown function

    Args:
    fasta_path: path to the fasta file where all proteins are in one file
    """
    all_proteins = []
    with open(fasta_path, "r") as f:
        lines = list(f.readlines())
        if any(l.startswith(">") for l in lines):
            # this mean the file is a fasta file
            with open(fasta_path, "r") as input_file:
                sequences, descriptions = parsers.parse_fasta(input_file.read())
                for desc in descriptions:
                    all_proteins.append({'protein_name': desc})
        else:
            for l in lines:
                if len(l.strip()) > 0:
                    all_proteins.append(obtain_options(l))
    return all_proteins


def read_custom(line) -> list:
    """
    A function to input file under the mode: custom

    Args:
    line: each individual line in the custom input file
    """
    all_proteins = []
    curr_list = line.rstrip().split(";")
    for substring in curr_list:
        all_proteins.append(obtain_options(substring))

    return all_proteins


def obtain_options(input_string) -> dict:
    '''
    Read the custom input file and converts all the options to a dictionary
    '''
    opt_list = input_string.strip().split(' ')
    opt_dict = {'protein_name': opt_list[0]}
    bool_commands = {'remove_msa_templates',
                    'remove_monomer_msa',
                    'remove_templates',
                    'shuffle_templates'}

    if len(opt_list) == 1:
        return opt_dict

    logging.info(f'Reading options for monomer {opt_list[0]}')
    for opt in opt_list[1:]:
        if opt.startswith('selected_residues'):
            logging.info(f'Reading residues for chopped monomer')
            opt = opt.split('=')[1]
            regions = opt.split(',')
            output_region = []
            for r in regions:
                output_region.append(
                    (int(r.split("-")[0]), int(r.split("-")[1]))
                    )
            opt_dict['selected_residues'] = output_region
        elif opt.startswith('remove_msa_region'):
            logging.info(f'Reading regions to remove MSA')
            opt = opt.split('=')[1]
            regions = opt.split(',')
            output_region = []
            for r in regions:
                output_region.append(
                    (int(r.split("-")[0])-1, int(r.split("-")[1])-1)
                    )
            opt_dict['remove_msa_region'] = output_region
        elif opt.startswith('mutate_msa'):
            logging.info(f'Reading point mutations to the MSA')
            opt = opt.split('=')[1]
            mutation_dict = {}
            point_mut = opt.split(',')
            for pm in point_mut:
                idx, mut = pm.split(':')
                mutation_dict[int(idx)-1] = mut
            opt_dict['mutate_msa'] = mutation_dict
        else:
            for cmd in bool_commands:
                if opt.startswith(cmd):
                    logging.info(f'Reading {cmd}')
                    opt_dict[cmd] = True
                    break

    return opt_dict

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
    
    model_runners_tmp = load_models_and_params(num_models = num_models,
                                                use_templates = use_templates,
                                                model_suffix = model_suffix,
                                                data_dir=data_dir,
                                                num_recycles=num_recycles,
                                                max_seq=max_seq,
                                                max_extra_seq=max_extra_seq,
                                                use_dropout=use_dropout,
                                                use_cluster_profile=use_cluster_profile,
                                                save_all=save_all,)
    model_runners = {}
    for mr in model_runners_tmp:
        for i in range(num_pred):
            model_runners[f'{mr[0]}_pred_{i}'] = mr[1:]

    return model_runners

def load_mutation_dict(file: str) -> dict:
    logger.info(f'Reading mutation information from {file}')
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
    return feature_dict

import logging
from absl import logging as absl_logging

def setup_logging(log_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename = log_file,
                        format="%(name)s - %(levelname)s - %(message)s",
                        level = logging.INFO,
                        filemode='a',)
    absl_logging.set_verbosity(absl_logging.INFO)

def load_monomer_objects(monomer_dir_dict, protein_name):
    """
    a function to load monomer an object from its pickle

    args
    monomer_dir_dict: a dictionary recording protein_name and its directory. created by make_dir_monomer_dictionary()
    """
    target_path = monomer_dir_dict[f"{protein_name}.pkl"]
    target_path = os.path.join(target_path, f"{protein_name}.pkl")
    monomer = pickle.load(open(target_path, "rb"))
    if isinstance(monomer, MultimericObject):
        return monomer
    if check_empty_templates(monomer.feature_dict):
        monomer.feature_dict = mk_mock_template(monomer.feature_dict)
    return monomer