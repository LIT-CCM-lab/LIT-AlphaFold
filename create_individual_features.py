#!/usr/bin/env python3

# Original Author Dingquan Yu
# Modified by Luca Chiesa
# This script is just to create msa and structural features
#for each sequences and store them in pickle
# #
import sys
import os
import bz2
import logging
from datetime import datetime
from pathlib import Path
import contextlib

try:
   import cPickle as pickle
except:
   import pickle

import yaml
import hydra
import shutil
from omegaconf import OmegaConf

from alphafold.data.tools import hmmsearch, hhsearch
from alphafold.data import templates

from colabfold.utils import DEFAULT_API_SERVER
from colabfold.batch import mk_hhsearch_db

from alphapulldown.utils import (
    save_meta_data,
    create_uniprot_runner,
)

from litaf.objects import (MonomericObject,
                            MonomericObjectMmseqs2,
                            load_monomer_objects,
                            check_existing_objects)
from litaf.filterpdb import load_template_filter
from litaf.utils import setup_logging, iter_seqs
from litaf.alphafold.data.pipeline import DataPipeline

@contextlib.contextmanager
def output_meta_file(file_path):
    """function that create temp file"""
    with open(file_path, "w") as outfile:
        yield outfile.name


def create_soft(cfg):
    defaults = ['jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign']
    for k,v in cfg.soft.items():
        if v in defaults:
            cfg['soft'][k] = shutil.which(v)

def create_template_dbs(cfg):
    if cfg.db.pdb_seqres_database_path is None:
        cfg.db.pdb_seqres_database_path = os.path.join(
            cfg.db.data_dir, "pdb_seqres", "pdb_seqres.txt"
        )

    # Path to a directory with template mmCIF structures, each named <pdb_id>.cif.
    if cfg.db.template_mmcif_dir is None:
        cfg.db.template_mmcif_dir = os.path.join(cfg.db.data_dir, "pdb_mmcif", "mmcif_files")

    # Path to a file mapping obsolete PDB IDs to their replacements.
    if cfg.db.obsolete_pdbs_path is None:
        cfg.db.obsolete_pdbs_path = os.path.join(cfg.db.data_dir, "pdb_mmcif", "obsolete.dat")

    # Path to pdb70 database
    if cfg.db.pdb70_database_path is None:
        cfg.db.pdb70_database_path = os.path.join(cfg.db.data_dir, "pdb70", "pdb70")

def create_msa_dbs(cfg):

    # Path to the Uniref30 database for use by HHblits.
    if cfg.db.uniref30_database_path is None:
        cfg.db.uniref30_database_path = os.path.join(
            cfg.db.data_dir, "uniref30", "UniRef30_2021_03"
        )

    if cfg.db.uniref90_database_path is None:
        cfg.db.uniref90_database_path = os.path.join(
            cfg.db.data_dir, "uniref90", "uniref90.fasta"
        )

    # Path to the MGnify database for use by JackHMMER.
    if cfg.db.mgnify_database_path is None:
        cfg.db.mgnify_database_path = os.path.join(
            cfg.db.data_dir, "mgnify", "mgy_clusters_2022_05.fa"
        )

    # Path to the BFD database for use by HHblits.
    if cfg.db.bfd_database_path is None:
        cfg.db.bfd_database_path = os.path.join(
            cfg.db.data_dir,
            "bfd",
            "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
        )

    # Path to the Small BFD database for use by JackHMMER.
    if cfg.db.small_bfd_database_path is None:
        cfg.db.small_bfd_database_path = os.path.join(
            cfg.db.data_dir, "small_bfd", "bfd-first_non_consensus_sequences.fasta"
        )

    cfg.db.use_small_bfd = cfg.run.db_preset == "reduced_dbs"

def create_pipeline(cfg) -> DataPipeline:
    '''Create the pipeline for MSA and template search

    The functions geenrates the pipeline for the search in sequence and
     structure databases.
    The HHSearch method is used for search in an user defined template 
    database. The method was taken from the ColabFold features for user
     defined templates.
    Hmmsearch is used for the AlphaFold PDB database as in the original 
    implementation.

    Return
    ------
    DataPipeline: An AlphaFold DataPipeline with the specified binarypaths
        and file paths
    '''
    if cfg.db.custom_template_path:
        template_searcher = hhsearch.HHSearch(
                                binary_path=cfg.soft.hhsearch_binary_path,
                                databases=[os.path.join(
                                            cfg.db.custom_template_path,
                                            'pdb70')]
                                )
        template_featurizer = templates.HhsearchHitFeaturizer(
                                mmcif_dir=cfg.db.custom_template_path,
                                max_template_date=cfg.max_template_date,
                                max_hits=cfg.max_template_hits,
                                kalign_binary_path=cfg.soft.kalign_binary_path,
                                release_dates_path=None,
                                obsolete_pdbs_path=None,
                                )
    else:
        template_searcher = hmmsearch.Hmmsearch(
                                binary_path=cfg.soft.hmmsearch_binary_path,
                                hmmbuild_binary_path=cfg.soft.hmmbuild_binary_path,
                                database_path=cfg.db.pdb_seqres_database_path,
                            )
        template_featurizer=templates.HmmsearchHitFeaturizer(
                                mmcif_dir=cfg.db.template_mmcif_dir,
                                max_template_date=cfg.max_template_date,
                                max_hits=cfg.max_template_hits,
                                kalign_binary_path=cfg.soft.kalign_binary_path,
                                obsolete_pdbs_path=cfg.db.obsolete_pdbs_path,
                                release_dates_path=None,
                            )
    monomer_data_pipeline = DataPipeline(
                                jackhmmer_binary_path=cfg.soft.jackhmmer_binary_path,
                                hhblits_binary_path=cfg.soft.hhblits_binary_path,
                                uniref90_database_path=cfg.db.uniref90_database_path,
                                mgnify_database_path=cfg.db.mgnify_database_path,
                                bfd_database_path=cfg.db.bfd_database_path,
                                uniref30_database_path=cfg.db.uniref30_database_path,
                                small_bfd_database_path=cfg.db.small_bfd_database_path,
                                use_small_bfd=cfg.db.use_small_bfd,
                                use_precomputed_msas=cfg.use_precomputed_msa,
                                template_searcher=template_searcher,
                                template_featurizer=template_featurizer,
                                )
    return monomer_data_pipeline


def create_and_save_monomer_objects(m: MonomericObject, pipeline: DataPipeline,
        cfg: dict,use_mmseqs2: bool =False) -> None:
    '''Create and save the passed MonomericObject object

    The function takes an empty monomer object and adds MSA and template
     features.
    The function check if a file with the same name as the proposed one already 
    exists, in case the existing file object is loaded and the features added 
    to the passed object. It is possible to still perform the searchdespite a 
    file with the desired name already existing by setting FLAGS.skip_existing 
    to False.
    If the file does not already exist the features are generated by searching 
    the databses and added to the object. The search can be performed on both 
    local databases and using the mmseqs2 web server. When using mmseqs2 it is 
    possible to use local template databses.
    The newly generated files are saved as pickle objects.

    Parameters
    ----------
    m : MonomericObject
        empty monomer object
    pipeline : DataPipeline
        pipeline object used for database search
    flags_dict : dict
        dictionary containing the FLAGS arguments
    use_mmseqs2: bool
        use the mmseqs2 webserver for MSA and template search

    Returns
    -------
    None
    '''
    if cfg.db.custom_template_path:
        custom_path = os.path.basename(
                        os.path.normpath(cfg.db.custom_template_path))
        outfile = f'{m.description}_{custom_path}'
    else:
        outfile = f'{m.description}'

    if cfg.skip_existing and check_existing_objects(cfg.output_dir, outfile) :
        logging.info(f"Already found {outfile} in {cfg.output_dir} Skipped")
        #load old file and add all features of the old object in the passed one
        monomer = load_monomer_objects({outfile: cfg.output_dir},
                                         outfile)
        for attr, val in monomer.__dict__.items():
            m.__dict__[attr] = val
    else:
        timing = datetime.date(datetime.now())
        metadata_output_path = os.path.join(
            cfg.output_dir,
            f"{m.description}_feature_metadata_{timing}.txt",
        )
        #with output_meta_file(metadata_output_path) as meta_data_outfile:
        #    save_meta_data(flags_dict, meta_data_outfile)

        if not use_mmseqs2:
            m.make_features(
                pipeline,
                output_dir=cfg.output_dir,
                use_precomputed_msa=cfg.use_precomputed_msa,
                save_msa=cfg.save_msa_files,
                paired_msa=cfg.paired_msa
            )
        else:
            if cfg.db.custom_template_path:
                templates_type = cfg.db.custom_template_path
            elif cfg.run.use_mmseqs2_templates:
                templates_type = 'mmseqs2'
            else:
                templates_type = 'local'
            logging.info("running mmseqs2 now")
            m.make_features(
                DEFAULT_API_SERVER=DEFAULT_API_SERVER,
                output_dir=cfg.output_dir,
                templates_path=templates_type,
                max_template_date=cfg.max_template_date,
                pdb70_database_path=cfg.db.pdb70_database_path,
                template_mmcif_dir=cfg.db.template_mmcif_dir,
                )
        if cfg.db.custom_template_path:
            m.description = f'{m.description}_{custom_path}'
        output_file = os.path.join(cfg.output_dir, m.description)
        logging.info(f'Saving monomer object {output_file}')
        if cfg.compress:
            pickle.dump(m, bz2.BZ2File(f"{output_file}.pkl.bz2", "w"))
        else:
            pickle.dump(m, open(f"{output_file}.pkl", "wb"))


def filter_and_save_monomer_object(monomer: MonomericObject, filters: dict,
    pipeline: DataPipeline, cfg) -> None:
    '''Modify templates used for AlphaFold calculations

    Modify an existing MonomericObject filtering the templates based on a 
    search on online annotated structural databases, or by forcing the 
    exclusion or inclusion of specific templates.
    The results of the query search are saved in a new query file for 
    reusability, and for further analysis. The file is named: 
    {output_file}_query_results.yaml

    Parameters
    ----------
    monomer : MonomericObject
        Original MonomericObject object
    queries : dict
        Dictionary containing the name of the query file and a the querys
    pipeline: DataPipeline
        Pipeline for search in structural databse

    Return
    ------
    None

    '''
    if pipeline is not None:
        template_featuriser = pipeline.template_featuriser
    elif hasattr(monomer, 'template_featuriser_mmseqs2'):
        template_featuriser = None
    else:
        raise Exception('Something went wrong with the pipeline, contact the developers')

    for filter_file,filter_t in filters.items():
        logging.info(f'Generating custom input with template filter: ' \
                    f'{filter_file}')
        filter_name = Path(filter_file).stem
        file_name = f"{monomer.description}_{filter_name}"
        if cfg.skip_existing and check_existing_objects(cfg.output_dir, file_name):
            logging.info(f"{file_name} exists in {cfg.output_dir} Skipped")
            continue
        monomer.make_template_features(
            template_featuriser,
            filter_t = filter_t,
            inplace = True
        )
        old_description = monomer.description
        monomer.description = f'{monomer.description}_{filter_name}'
        output_file = os.path.join(cfg.output_dir, f'{monomer.description}')
        logging.info(f'Saving monomer object {output_file}')
        if cfg.compress:
            pickle.dump(monomer, bz2.BZ2File(f"{output_file}.pkl.bz2", "w"))
        else:
            pickle.dump(monomer, open(f"{output_file}.pkl", "wb"))
        logging.info(f'Dumping search results in a filter file ' \
                        f'{output_file}_filter_results.yaml')
        with open(f'{output_file}_filter_results.yaml', 'w') as yaml_file:
            yaml.dump(monomer.filter_pdb_results, yaml_file)
        monomer.description = old_description


@hydra.main(version_base=None, config_path="conf/features", config_name="config")
def main(cfg):

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config: {'  '.join(missing_keys)}")

    try:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(os.path.join(cfg.output_dir,
                                    f'{cfg.logger_file}.log'))
    except FileExistsError:
        logging.info("Multiple processes are trying to create" \
                    " the same folder now.")

    create_soft(cfg)

    if cfg.db.data_dir is not None:
        create_template_dbs(cfg)
        create_msa_dbs(cfg)

    elif cfg.run.use_mmseqs2 is False:
        raise ValueError('No indication for genetic database, \
                                please set a value for data_dir or \
                                set use_mmseqs2 to True')
    elif cfg.run.use_mmseqs2_templates is False:
        raise ValueError('No indication for template database, \
                                please set a value for data_dir or \
                                set use_mmseqs2_templates to True')
        

    if cfg.db.custom_template_path:
        ff_appendix = ['_a3m.ffdata', 
                        '_a3m.ffindex', 
                        '_cs219.ffdata', 
                        '_cs219.ffindex']
        ff_files = [os.path.join(cfg.db.custom_template_path,
                                    f'pdb70_{ff}') for ff in ff_appendix]
        for ff in ff_files:
            if not os.path.isfile(os.path.join(cfg.db.custom_template_path, ff)):
                mk_hhsearch_db(cfg.db.custom_template_path)
                break

    if not cfg.max_template_date and not cfg.run.use_mmseqs2_templates:
        # The pipeline for the template requires a limit date, not 
        # providing it closes the script.
        logging.info("You have not provided a max_template_date." \
            "Please specify a date and run again.")
        sys.exit()
    if cfg.run.use_mmseqs2:
        #mmseqs2 uses a specific pipeline created by a different function.
        #a serach function for uniprot is not required by using mmseqs2
        pipeline=None
        uniprot_runner=None
    else:
        pipeline = create_pipeline(cfg)
        uniprot_database_path = os.path.join(cfg.db.data_dir,
            "uniprot/uniprot.fasta")
        if os.path.isfile(uniprot_database_path):
            uniprot_runner = create_uniprot_runner(
                cfg.soft.jackhmmer_binary_path, uniprot_database_path
            )
        else:
            # Missing the uniprot.fasta file does not allow the script to 
            # properly work
            logging.info(
                f"Failed to find uniprot.fasta in {uniprot_database_path}."\
                " Please make sure data_dir has been configured correctly."
            )
            sys.exit()
        
    filters = {qf: load_template_filter(qf) for qf in cfg.template_filters}

    seqs = iter_seqs(cfg.fasta_paths)
    for seq_idx, (curr_seq, curr_desc) in enumerate(seqs, 1):
        if cfg.seq_index != seq_idx and cfg.seq_index:
            continue
        elif curr_desc is None or curr_desc.isspace():
            continue
        else:
            if cfg.run.use_mmseqs2:
                curr_monomer = MonomericObjectMmseqs2(curr_desc.replace(' ', '_'), curr_seq)
            else:
                curr_monomer = MonomericObject(curr_desc.replace(' ', '_'), curr_seq)
                curr_monomer.uniprot_runner = uniprot_runner
            
            create_and_save_monomer_objects(curr_monomer,
                                            pipeline,
                                            cfg,
                                            use_mmseqs2=cfg.run.use_mmseqs2)
            filter_and_save_monomer_object(curr_monomer, filters, pipeline, cfg)

            del curr_monomer


if __name__ == "__main__":
    main()
