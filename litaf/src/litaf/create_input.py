'''
Functions to generate input objects for AlphaFold prediction

* *create_interactors*, generates groups of monomers
* *create_multimer_objects*, generate multimers
* *create_pulldown*, generate input for pulldown calculations
* *create_all_vs_all*, generate input for all vs all calculations
* *create_homooligomers*, generate input for homooligomers calulations
* *create_custom*, generate input for custom calculations

'''

import itertools
import copy

import logging
import yaml

from litaf.utils import (read_custom,
                        read_all_proteins,
                        obtain_options,
                        make_dir_monomer_dictionary)

from litaf.objects import MultimericObject, ChoppedObject, load_monomer_objects
from litaf.rename import *

def create_interactors_colab(data,
            monomer_objects_dict,
            remove_msa,
            remove_template_msa,
            remove_templates,
            shuffle_templates,
            paired_msa = False,
            unpaired_msa = True) -> list:

    """
    Generates and modifies monomeric objects used for prediction by the colab
    version of LIT-AlphaFold

    Parameters
    ----------
    data: dict
        Dictionary contining the inputs
    pair_msa: bool, default = True
        Generates paired MSA for multimer prediction
    monomer_objects_dir: str or list
        Direcotries containing the monomer pickle objects
    remove_msa: bool
        Remove MSA features from all interactors
    remove_template_msa: bool
        Remove MSA in template aligned regions for all interactors
    remove_templates: bool
        Remove template features from all interactors
    mutate_msa: bool
        Mutate the monomer MSA in specific positions for all interactors
    remove_msa_region: bool
        Remove a specific region of the monomeric MSA for all interactors
    remove_templates: bool
        Remove template featrues for all interactors 

    Return
    ------
    Monomers prepared for prediction and processing: list of MonomericObject
    """
    
    interactors = []
    for d in data:
        logging.info(f"Processing {d['protein_name']}")
        monomer = monomer_objects_dict[d['protein_name']]

        if isinstance(monomer, MultimericObject):
            return [monomer]

        interactors.append(
            modify_monomer(d,
                    copy.deepcopy(monomer),
                    remove_msa,
                    remove_template_msa,
                    remove_templates,
                    False,
                    False,
                    shuffle_templates,
                    paired_msa,
                    unpaired_msa)
            )
    return interactors

def create_interactors(entries,
            monomer_objects_dir,
            remove_msa,
            remove_template_msa,
            remove_templates,
            mutate_msa,
            remove_msa_region,
            shuffle_templates,
            paired_msa = False,
            unpaired_msa = True) -> list:
    """
    Generates and modifies monomeric objects used for prediction

    Parameters
    ----------
    data: dict
        Dictionary contining the inputs
    pair_msa: bool, default = True
        Generates paired MSA for multimer prediction
    monomer_objects_dir: str or list
        Direcotries containing the monomer pickle objects
    remove_msa: bool
        Remove MSA features from all interactors
    remove_template_msa: bool
        Remove MSA in template aligned regions for all interactors
    remove_templates: bool
        Remove template features from all interactors
    mutate_msa: bool
        Mutate the monomer MSA in specific positions for all interactors
    remove_msa_region: bool
        Remove a specific region of the monomeric MSA for all interactors
    remove_templates: bool
        Remove template featrues for all interactors 

    Return
    ------
    Monomers prepared for prediction and processing: list of MonomericObject
    """
    interactors_obj = []
    monomer_dir_dict = make_dir_monomer_dictionary(monomer_objects_dir)
    for entry in entries:
        for e in entry:
            e['remove_msa'] = e.get('remove_msa', remove_msa)
            e['remove_msa_templates'] =  e.get('remove_msa_templates', remove_template_msa)
            e['remove_templates'] =  e.get('remove_templates', remove_templates)
            e['mutate_msa'] =  e.get('mutate_msa', mutate_msa)
            e['remove_msa_region'] = e.get('remove_msa_region', remove_msa_region)
            e['shuffle_templates'] = e.get('shuffle_templates', shuffle_templates)
            e['paired'] =  e.get('paired', paired_msa)
            e['unpaired'] =  e.get('unpaired', unpaired_msa)

    all_monomers, interactors_list = load_monomers(entries, monomer_dir_dict)
    for i,interactors in enumerate(interactors_list):
        interactors_obj.append([])
        for interactor, e in zip(interactors, entries[i]):
            interactors_obj[-1].append(copy.deepcopy(all_monomers[interactor]))
            if e.get('monomer'):
                interactors_obj[-1][-1].description = e.get('monomer')

    return interactors_obj



def read_yaml(file):
    with open(file, 'r') as f:
        data = f.read()
    return [y for y in yaml.load_all(data, Loader=yaml.Loader)]

def read_homomer(file):
    yaml_data = read_yaml(file)
    monomers = []
    for homo in yaml_data:
        if isinstance(homo['monomer'], str):
            homo['monomer'] = {'protein_name': homo['monomer']}

    return [[y] for y in yaml_data]

def read_proteins(file):
    yaml_data = read_yaml(file)
    return [[{'protein': y}] if isinstance(y, str) else [y] for y in yaml_data]

def read_custom(file):
    yaml_data = read_yaml(file)
    for i, multi in enumerate(yaml_data):
        for j, mono in enumerate(multi):
            if isinstance(mono, str):
                yaml_data[i][j] = {'protein_name': mono}

    return yaml_data

def load_monomers(entries, monomer_dir_dict):
    interactors = {}
    structure_list = []

    for entry in entries:
        structure_list.append([])
        for e in entry:
            e_name = get_interactor_name(e)
            if e_name not in interactors:
                logging.info(f"Processing {e['protein_name']}")
                monomer = load_monomer_objects(monomer_dir_dict, e['protein_name'])
                if not isinstance(monomer, MultimericObject):
                    monomer = modify_monomer(e, monomer)
                interactors[e_name] = monomer

            structure_list[-1].append(e_name)

    return interactors, structure_list


def get_interactor_name(interactor):
    name = interactor['protein_name']
    if interactor.get('remove_msa_region'):
        name = rename_remove_msa_region(name,
                                        interactor.get('remove_msa_region'),
                                        interactor.get('paired', False),
                                        interactor.get('unpaired', True),)
    if interactor.get('mutate_msa'):
        name = rename_mutate_msa(name,
                                        interactor.get('mutate_msa'),
                                        interactor.get('paired', False),
                                        interactor.get('unpaired', True),)  
    if interactor.get('remove_template_msa'):
        name = rename_remove_template_from_msa(name)
    if interactor.get('remove_monomer_msa'):
        name = rename_remove_msa_features(name)
    if interactor.get('remove_templates'):
        name = rename_remove_templates(name)
    if interactor.get('shuffle_templates'):
        name = rename_remove_templates(name, interactor.get('shuffle_templates'))
    return name


def modify_monomer(d, monomer):

    if d.get('selected_residues'):
        logging.info(
            f"creating chopped monomer with residues" \
            f"{d.get('selected_residues')}"
            )
        monomer = ChoppedObject(
                    monomer.description,
                    monomer.sequence,
                    monomer.feature_dict,
                    d.get('selected_residues'),
                )
        monomer.prepare_final_sliced_feature_dict()

    if d.get('remove_msa_region'):
        logging.info(
            f"removing MSA data from the regions " \
            f"{d.get('remove_msa_region')}")
        monomer.remove_msa_region(d.get('remove_msa_region'),
            inplace = True,
            paired = paired_msa,
            unpaired = unpaired_msa)

    if d.get('mutate_msa'):
        logging.info(f"Mutating MSA as: {d.get('mutate_msa')}")
        monomer.mutate_msa(d.get('mutate_msa'),
                            inplace = True,
                            paired = d.get('paired'),
                            unpaired = d.get('unpaired'))

    if d.get('remove_msa_templates'):
        logging.info('Removing template information from the MSA')
        monomer.remove_template_from_msa(inplace = True)

    if d.get('remove_monomer_msa'):
        logging.info('Removing monomer MSA')
        monomer.remove_msa_features(inplace = True)

    if d.get('remove_templates'):
        logging.info('Removing template data')
        monomer.remove_templates(inplace = True)

    if d.get('shuffle_templates'):
        logging.info('Shuffling templates')
        monomer.shuffle_templates(inplace = True)

    return monomer


def create_multimer_objects(
    data: dict,
    monomer_objects_dir: str,
    pair_msa: bool = True,
    remove_msa: bool = False,
    remove_template_msa: bool = False,
    mutate_msa: str = None,
    remove_msa_region: str = None,
    remove_templates: bool = False,
    shuffle_templates: bool = False,
    paired_msa=False,
    unpaired_msa=True,
    ) -> list:
    """
    A function to create multimer objects

    Parameters
    ----------
    data: dict
        Dictionary contining the inputs
    pair_msa: bool, default = True
        Generates paired MSA for multimer prediction
    monomer_objects_dir: str or list
        Direcotries containing the monomer pickle objects
    remove_msa: bool
        Remove MSA features from all interactors
    remove_template_msa: bool
        Remove MSA in template aligned regions for all interactors
    remove_templates: bool
        Remove template features from all interactors
    mutate_msa: bool
        Mutate the monomer MSA in specific positions for all interactors
    remove_msa_region: bool
        Remove a specific region of the monomeric MSA for all interactors
    remove_templates: bool
        Remove template featrues for all interactors 

    Return
    ------
    MultimericObject and MonomericObject for prediction: list
    """

    multimers = []

    interactors_list = create_interactors(
        data,
        monomer_objects_dir,
        remove_msa=remove_msa,
        remove_template_msa=remove_template_msa,
        remove_templates = remove_templates,
        mutate_msa = mutate_msa,
        remove_msa_region = remove_msa_region,
        shuffle_templates=shuffle_templates,
        unpaired_msa=unpaired_msa,
        paired_msa=paired_msa)
    for interactors in interactors_list:
        if len(interactors) > 1:
            multimer = MultimericObject(
                interactors=interactors,
                pair_msa=pair_msa)
            logging.info(f"done creating multimer {multimer.description}")
            multimers.append(multimer)
        else:
            logging.info(f"done loading monomer {interactors[0].description}")
            multimers.append(interactors[0])

    return multimers


def create_pulldown(
    proteins_list: list,
    monomer_objects_dir: int,
    pair_msa: bool = False,
    remove_msa: bool = False,
    remove_template_msa: bool = False,
    mutate_msa: dict = None,
    remove_msa_region: dict = None,
    remove_templates: bool = False,
    shuffle_templates: bool = False,
    paired_msa=False,
    unpaired_msa=True,
    ) -> list:
    '''
    Create MultimericObjects from pulldown style input
  
    Parameters
    ----------
    proteins_list: list
        file with bait proteins is the first in the list, candidate proteins
         files from position 1 to the end
    pair_msa: bool, default = True
        Generates paired MSA for multimer prediction
    monomer_objects_dir: str or list
        Direcotries containing the monomer pickle objects
    remove_msa: bool
        Remove MSA features from all interactors
    remove_template_msa: bool
        Remove MSA in template aligned regions for all interactors
    remove_templates: bool
        Remove template features from all interactors
    mutate_msa: bool
        Mutate the monomer MSA in specific positions for all interactors
    remove_msa_region: bool
        Remove a specific region of the monomeric MSA for all interactors
    remove_templates: bool
        Remove template featrues for all interactors 

    Return
    ------
    MultimericObject and MonomericObject for prediction: list
    '''
    bait_proteins = read_proteins(proteins_list[0])
    candidate_proteins = []
    for file in proteins_list[1:]:
        candidate_proteins.append(read_proteins(file))

    all_protein_pairs = list(
        itertools.product(*[bait_proteins, *candidate_proteins]))

    return create_multimer_objects(
            data = all_protein_pairs,
            monomer_objects_dir = monomer_objects_dir,
            pair_msa = pair_msa,
            remove_msa=remove_msa,
            remove_template_msa=remove_template_msa,
            mutate_msa = mutate_msa,
            remove_msa_region = remove_msa_region,
            remove_templates = remove_templates,
            shuffle_templates=shuffle_templates,
            unpaired_msa=unpaired_msa,
            paired_msa=paired_msa)


def create_all_vs_all(
    proteins_list: list,
    monomer_objects_dir: int,
    pair_msa: bool = False,
    remove_msa: bool = False,
    remove_template_msa: bool = False,
    mutate_msa: dict = None,
    remove_msa_region: dict = None,
    remove_templates: bool = False,
    shuffle_templates: bool = False,
    paired_msa=False,
    unpaired_msa=True,
    ) -> dict:
    """Create MultimericObjects from all vs all input
    i.e all pairings of proteins are evaluated

    Parameters
    ----------
    proteins_list: list
        proteins to include in the all vs all analysis
    pair_msa: bool, default = True
        Generates paired MSA for multimer prediction
    monomer_objects_dir: str or list
        Direcotries containing the monomer pickle objects
    remove_msa: bool
        Remove MSA features from all interactors
    remove_template_msa: bool
        Remove MSA in template aligned regions for all interactors
    remove_templates: bool
        Remove template features from all interactors
    mutate_msa: bool
        Mutate the monomer MSA in specific positions for all interactors
    remove_msa_region: bool
        Remove a specific region of the monomeric MSA for all interactors
    remove_templates: bool
        Remove template featrues for all interactors 

    Return
    ------
    MultimericObject and MonomericObject for prediction: list
    """
    all_proteins = []
    for file in proteins_list:
        all_proteins = all_proteins+read_proteins(file)
    all_possible_pairs = list(itertools.combinations(all_proteins, 2))

    return create_multimer_objects(
                            all_possible_pairs,
                            monomer_objects_dir,
                            pair_msa = pair_msa,
                            remove_msa=remove_msa,
                            remove_template_msa=remove_template_msa,
                            mutate_msa = mutate_msa,
                            remove_msa_region = remove_msa_region,
                            remove_templates = remove_templates,
                            shuffle_templates=shuffle_templates,
                            unpaired_msa=unpaired_msa,
                            paired_msa=paired_msa)



def create_homooligomers(
    oligomer_state_file: str,
    monomer_objects_dir: int,
    pair_msa: bool = False,
    remove_msa: bool = False,
    remove_template_msa: bool = False,
    mutate_msa: dict = None,
    remove_msa_region: dict = None,
    remove_templates: bool = False,
    shuffle_templates: bool = False,
    paired_msa=False,
    unpaired_msa=True,
    ) -> list:
    """Create MultimericObjects for homoligomers

    Parameters
    ----------
    oligomer_state_file: str
        description of the homoligomers given as the protein and the 
        stochiometry
    pair_msa: bool, default = True
        Generates paired MSA for multimer prediction
    monomer_objects_dir: str or list
        Direcotries containing the monomer pickle objects
    remove_msa: bool
        Remove MSA features from all interactors
    remove_template_msa: bool
        Remove MSA in template aligned regions for all interactors
    remove_templates: bool
        Remove template features from all interactors
    mutate_msa: bool
        Mutate the monomer MSA in specific positions for all interactors
    remove_msa_region: bool
        Remove a specific region of the monomeric MSA for all interactors
    remove_templates: bool
        Remove template featrues for all interactors 

    Return
    ------
    MultimericObject and MonomericObject for prediction: list
    """
    multimers = []
    data = read_homomer(oligomer_state_file)

    interactors = create_interactors(data,
                                monomer_objects_dir,
                                remove_msa=remove_msa,
                                remove_template_msa=remove_template_msa,
                                remove_templates = remove_templates,
                                mutate_msa = mutate_msa,
                                remove_msa_region = remove_msa_region,
                                shuffle_templates=shuffle_templates,
                                unpaired_msa=unpaired_msa,
                                paired_msa=paired_msa)[0]

    for interactors, d in zip(interactors, data):
        if d['n_units'] > 1:
            interactors = [monomer] * d['n_units']
            homooligomer = MultimericObject(interactors,pair_msa=pair_msa)
            homooligomer.description = f"{monomer.description}_homo_{num_units}er"
            multimers.append(homooligomer)
            logging.info(
                f"finished creating homooligomer {homooligomer.description}"
            )
        elif num_units == 1:
            multimers.append(monomer)
            logging.info(f"finished loading monomer: {monomer.description}")

    return multimers


def create_custom_jobs(
    custom_input_file: str,
    monomer_objects_dir: str,
    pair_msa: bool = True,
    remove_msa: bool = False,
    remove_template_msa: bool = False,
    mutate_msa: dict = None,
    remove_msa_region: dict = None,
    remove_templates: bool = False,
    shuffle_templates: bool = False,
    paired_msa=False,
    unpaired_msa=True,
    ) -> list:
    """
    Create MultimericObjects using custom multimer input

    Parameters
    ----------
    custom_input_file: str
        A list of input_files from FLAGS.protein_lists
    pair_msa: bool, default = True
        Generates paired MSA for multimer prediction
    monomer_objects_dir: str or list
        Direcotries containing the monomer pickle objects
    remove_msa: bool
        Remove MSA features from all interactors
    remove_template_msa: bool
        Remove MSA in template aligned regions for all interactors
    remove_templates: bool
        Remove template features from all interactors
    mutate_msa: bool
        Mutate the monomer MSA in specific positions for all interactors
    remove_msa_region: bool
        Remove a specific region of the monomeric MSA for all interactors
    remove_templates: bool
        Remove template featrues for all interactors 

    Return
    ------
    MultimericObject and MonomericObject for prediction: list

    """
    all_files = []
    for file in custom_input_file:
        all_files= all_files + read_custom(file)

    return create_multimer_objects(
            all_files,
            monomer_objects_dir,
            pair_msa=pair_msa,
            remove_msa=remove_msa,
            remove_template_msa=remove_template_msa,
            mutate_msa = mutate_msa,
            remove_msa_region = remove_msa_region,
            remove_templates = remove_templates,
            shuffle_templates=shuffle_templates,
            unpaired_msa=unpaired_msa,
            paired_msa=paired_msa)