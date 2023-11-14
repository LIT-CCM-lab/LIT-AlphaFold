'''
Functions to generate input objects for AlphaFold prediction

* *create_interactors*, generates groups of monomers
* *create_multimer_objects*, generate multimers
* *create_pulldown*, generate input for pulldown calculations
* *create_all_vs_all*, generate input for all vs all calculations
* *create_homooligomers*, generate input for homooligomers calulations
* *create_custom*, generate input for custom calculations

'''

import pickle
import itertools

import logging

from alphapulldown.utils import (make_dir_monomer_dictionary,
                                    check_empty_templates)

from litaf_development.utils import (create_colabfold_runners,
                                    read_custom,
                                    read_all_proteins,
                                    obtain_options,
                                    load_monomer_objects,)
from litaf_development.objects import MultimericObject, ChoppedObject



def create_interactors(data,
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
    interactors = []
    monomer_dir_dict = make_dir_monomer_dictionary(monomer_objects_dir)
    for d in data:
        logging.info(f"Processing {d['protein_name']}")
        monomer = load_monomer_objects(monomer_dir_dict, d['protein_name'])

        if isinstance(monomer, MultimericObject):
            return [monomer]

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
        elif remove_msa_region:
            msa_region_string = ','.join(
                [f'{i}-{j}' for i,j in remove_msa_region]
                )
            logging.info(
                f'removing MSA data from the regions {msa_region_string}'
                )
            monomer.remove_msa_region(remove_msa_region,
                                        inplace = True,
                                        paired = paired_msa,
                                        unpaired = unpaired_msa)

        if d.get('mutate_msa'):
            logging.info(f"Mutating MSA as: {d.get('mutate_msa')}")
            monomer.mutate_msa(d.get('mutate_msa'),
                                inplace = True,
                                paired = paired_msa,
                                unpaired = unpaired_msa)
        elif mutate_msa:
            logging.info(f'Mutating MSA as: {mutate_msa}')
            monomer.mutate_msa(mutate_msa,
                                inplace = True,
                                paired = paired_msa,
                                unpaired = unpaired_msa)

        if d.get('remove_msa_templates') or remove_template_msa:
            logging.info(f'Removing template information from the MSA')
            monomer.remove_template_from_msa(inplace = True)

        if d.get('remove_monomer_msa') or remove_msa:
            logging.info(f'Removing monomer MSA')
            monomer.remove_msa_features(inplace = True)

        if d.get('remove_templates') or remove_templates:
            logging.info(f'Removing template data')
            monomer.remove_templates(inplace = True)

        if d.get('shuffle_templates') or shuffle_templates:
            logging.info(f'Shuffling templates')
            monomer.shuffle_templates(inplace = True)

        interactors.append(monomer)
    return interactors


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

    for multi in data:
        interactors = create_interactors(
            multi,
            monomer_objects_dir,
            remove_msa=remove_msa,
            remove_template_msa=remove_template_msa,
            remove_templates = remove_templates,
            mutate_msa = mutate_msa,
            remove_msa_region = remove_msa_region,
            shuffle_templates=shuffle_templates,
            unpaired_msa=unpaired_msa,
            paired_msa=paired_msa)
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
    bait_proteins = read_all_proteins(proteins_list[0])
    candidate_proteins = []
    for file in proteins_list[1:]:
        candidate_proteins.append(read_all_proteins(file))

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
        all_proteins = all_proteins+read_all_proteins(file)
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
    monomer_dir_dict = make_dir_monomer_dictionary(monomer_objects_dir)
    lines = []
    for file in oligomer_state_file:
        with open(file) as f:
            lines = lines + list(f.readlines())
            f.close()
    
    for l in lines:
        if len(l.strip()) == 0:
            continue
        if len(l.rstrip().split(";")) > 1:
            data = [obtain_options(l.rstrip().split(";")[0])]
            num_units = int(l.rstrip().split(";")[1])
        else:
            data = [obtain_options(l.rstrip().split(";")[0])]
            num_units = 1

        monomer = create_interactors(data,
                                    monomer_objects_dir,
                                    remove_msa=remove_msa,
                                    remove_template_msa=remove_template_msa,
                                    remove_templates = remove_templates,
                                    mutate_msa = mutate_msa,
                                    remove_msa_region = remove_msa_region,
                                    shuffle_templates=shuffle_templates,
                                    unpaired_msa=unpaired_msa,
                                    paired_msa=paired_msa)[0]

        if num_units > 1:
            interactors = [monomer] * num_units
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
    lines = []
    for file in custom_input_file:
        with open(file) as f:
            lines = lines + list(f.readlines())
            f.close()

    all_files = []
    for l in lines:
        if len(l.strip()) == 0:
            continue
        all_files.append(read_custom(l))


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