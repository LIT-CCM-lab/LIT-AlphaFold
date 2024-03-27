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

from absl import app
from absl import flags
import yaml

from alphafold.data.tools import hmmsearch, hhsearch
from alphafold.data import templates
from alphafold import run_alphafold as run_af

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
from litaf.pipeline import DataPipeline
from litaf.utils import setup_logging, iter_seqs

@contextlib.contextmanager
def output_meta_file(file_path):
    """function that create temp file"""
    with open(file_path, "w") as outfile:
        yield outfile.name


flags = run_af.flags
flags.DEFINE_bool("save_msa_files", False,
                "save msa output or not"
)
flags.DEFINE_bool("skip_existing", False,
                "skip existing monomer feature pickles or not"
)
flags.DEFINE_integer("seq_index", None,
                    "index of sequence in the fasta file, starting from 1"
)
flags.DEFINE_string("new_uniclust_dir", None,
    "directory where new version of uniclust is stored"
)
flags.DEFINE_bool("use_mmseqs2",False,
                "Use mmseqs2 remotely or not. Default is False"
)
flags.DEFINE_bool("use_mmseqs2_templates",False,
                "Use mmseqs2 templates remotely or not. Default is False"
)
flags.DEFINE_list('template_filters', [],
                'List of query for template selection'
)
flags.DEFINE_string('custom_template_path', None,
                'Template containing pdb structure of custom templates'
)
flags.DEFINE_string('logger_file', 'feature_generation',
                'File where to store the results'
)
flags.DEFINE_bool('paired_msa', True,
                  "Search Uniprot for sequences for paired MSA generation"
)
flags.DEFINE_bool('compress', True,
                  "Compress the pkl file to save space"
)

delattr(flags.FLAGS, "data_dir")
flags.DEFINE_string("data_dir", None, "Path to database directory")

FLAGS = flags.FLAGS
MAX_TEMPLATE_HITS = 20

flags_dict = FLAGS.flag_values_dict()

global pdb70_database_path
global template_mmcif_dir
pdb70_database_path = None
template_mmcif_dir = None

def create_global_arguments(flags_dict):
    global uniref90_database_path
    global mgnify_database_path
    global bfd_database_path
    global small_bfd_database_path
    global pdb_seqres_database_path
    global template_mmcif_dir
    global obsolete_pdbs_path
    global pdb70_database_path
    global use_small_bfd
    global uniref30_database_path

    # Path to the Uniref30 database for use by HHblits.
    if FLAGS.uniref30_database_path is None:
        uniref30_database_path = os.path.join(
            FLAGS.data_dir, "uniref30", "UniRef30_2021_03"
        )
    else:
        uniref30_database_path = FLAGS.uniref30_database_path
    flags_dict.update({"uniref30_database_path": uniref30_database_path})

    if FLAGS.uniref90_database_path is None:
        uniref90_database_path = os.path.join(
            FLAGS.data_dir, "uniref90", "uniref90.fasta"
        )
    else:
        uniref90_database_path = FLAGS.uniref90_database_path

    flags_dict.update({"uniref90_database_path": uniref90_database_path})

    # Path to the MGnify database for use by JackHMMER.
    if FLAGS.mgnify_database_path is None:
        mgnify_database_path = os.path.join(
            FLAGS.data_dir, "mgnify", "mgy_clusters_2022_05.fa"
        )
    else:
        mgnify_database_path = FLAGS.mgnify_database_path
    flags_dict.update({"mgnify_database_path": mgnify_database_path})

    # Path to the BFD database for use by HHblits.
    if FLAGS.bfd_database_path is None:
        bfd_database_path = os.path.join(
            FLAGS.data_dir,
            "bfd",
            "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
        )
    else:
        bfd_database_path = FLAGS.bfd_database_path
    flags_dict.update({"bfd_database_path": bfd_database_path})

    # Path to the Small BFD database for use by JackHMMER.
    if FLAGS.small_bfd_database_path is None:
        small_bfd_database_path = os.path.join(
            FLAGS.data_dir, "small_bfd", "bfd-first_non_consensus_sequences.fasta"
        )
    else:
        small_bfd_database_path = FLAGS.small_bfd_database_path
    flags_dict.update({"small_bfd_database_path": small_bfd_database_path})

    if FLAGS.pdb_seqres_database_path is None:
        pdb_seqres_database_path = os.path.join(
            FLAGS.data_dir, "pdb_seqres", "pdb_seqres.txt"
        )
    else:
        pdb_seqres_database_path = FLAGS.pdb_seqres_database_path
    flags_dict.update({"pdb_seqres_database_path": pdb_seqres_database_path})

    # Path to a directory with template mmCIF structures, each named <pdb_id>.cif.
    if FLAGS.template_mmcif_dir is None:
        template_mmcif_dir = os.path.join(FLAGS.data_dir, "pdb_mmcif", "mmcif_files")
    else:
        template_mmcif_dir = FLAGS.template_mmcif_dir
    flags_dict.update({"template_mmcif_dir": template_mmcif_dir})

    # Path to a file mapping obsolete PDB IDs to their replacements.
    if FLAGS.obsolete_pdbs_path is None:
        obsolete_pdbs_path = os.path.join(FLAGS.data_dir, "pdb_mmcif", "obsolete.dat")
    else:
        obsolete_pdbs_path = FLAGS.obsolete_pdbs_path
    flags_dict.update({"obsolete_pdbs_path": obsolete_pdbs_path})

    # Path to pdb70 database
    if FLAGS.pdb70_database_path is None:
        pdb70_database_path = os.path.join(FLAGS.data_dir, "pdb70", "pdb70")
    else:
        pdb70_database_path = FLAGS.pdb70_database_path
    flags_dict.update({"pdb70_database_path": pdb70_database_path})
    use_small_bfd = FLAGS.db_preset == "reduced_dbs"

def create_pipeline() -> DataPipeline:
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
    if FLAGS.custom_template_path:
        template_searcher = hhsearch.HHSearch(
                                binary_path=FLAGS.hhsearch_binary_path,
                                databases=[os.path.join(
                                            FLAGS.custom_template_path,
                                            'pdb70')]
                                )
        template_featurizer = templates.HhsearchHitFeaturizer(
                                mmcif_dir=FLAGS.custom_template_path,
                                max_template_date=FLAGS.max_template_date,
                                max_hits=MAX_TEMPLATE_HITS,
                                kalign_binary_path=FLAGS.kalign_binary_path,
                                release_dates_path=None,
                                obsolete_pdbs_path=None,
                                )
    else:
        template_searcher = hmmsearch.Hmmsearch(
                                binary_path=FLAGS.hmmsearch_binary_path,
                                hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
                                database_path=pdb_seqres_database_path,
                            )
        template_featurizer=templates.HmmsearchHitFeaturizer(
                                mmcif_dir=template_mmcif_dir,
                                max_template_date=FLAGS.max_template_date,
                                max_hits=MAX_TEMPLATE_HITS,
                                kalign_binary_path=FLAGS.kalign_binary_path,
                                obsolete_pdbs_path=obsolete_pdbs_path,
                                release_dates_path=None,
                            )
    monomer_data_pipeline = DataPipeline(
                                jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
                                hhblits_binary_path=FLAGS.hhblits_binary_path,
                                uniref90_database_path=uniref90_database_path,
                                mgnify_database_path=mgnify_database_path,
                                bfd_database_path=bfd_database_path,
                                uniref30_database_path=uniref30_database_path,
                                small_bfd_database_path=small_bfd_database_path,
                                use_small_bfd=use_small_bfd,
                                use_precomputed_msas=FLAGS.use_precomputed_msas,
                                template_searcher=template_searcher,
                                template_featurizer=template_featurizer,
                                )
    return monomer_data_pipeline


def create_and_save_monomer_objects(m: MonomericObject, pipeline: DataPipeline,
        flags_dict: dict,use_mmseqs2: bool =False) -> None:
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
    if FLAGS.custom_template_path:
        custom_path = os.path.basename(
                        os.path.normpath(FLAGS.custom_template_path))
        outfile = f'{m.description}_{custom_path}'
    else:
        outfile = f'{m.description}'

    if FLAGS.skip_existing and check_existing_objects(FLAGS.output_dir, outfile) :
        logging.info(f"Already found {outfile} in {FLAGS.output_dir} Skipped")
        #load old file and add all features of the old object in the passed one
        monomer = load_monomer_objects({outfile: FLAGS.output_dir},
                                         outfile)
        for attr, val in monomer.__dict__.items():
            m.__dict__[attr] = val
    else:
        timing = datetime.date(datetime.now())
        metadata_output_path = os.path.join(
            FLAGS.output_dir,
            f"{m.description}_feature_metadata_{timing}.txt",
        )
        with output_meta_file(metadata_output_path) as meta_data_outfile:
            save_meta_data(flags_dict, meta_data_outfile)

        if not use_mmseqs2:
            m.make_features(
                pipeline,
                output_dir=FLAGS.output_dir,
                use_precomputed_msa=FLAGS.use_precomputed_msas,
                save_msa=FLAGS.save_msa_files,
                paired_msa=FLAGS.paired_msa
            )
        else:
            if FLAGS.custom_template_path:
                templates_type = FLAGS.custom_template_path
            elif FLAGS.use_mmseqs2_templates:
                templates_type = 'mmseqs2'
            else:
                templates_type = 'local'
            logging.info("running mmseqs2 now")
            m.make_features(
                DEFAULT_API_SERVER=DEFAULT_API_SERVER,
                output_dir=FLAGS.output_dir,
                templates_path=templates_type,
                max_template_date=FLAGS.max_template_date,
                pdb70_database_path=pdb70_database_path,
                template_mmcif_dir=template_mmcif_dir,
                )
        if FLAGS.custom_template_path:
            m.description = f'{m.description}_{custom_path}'
        output_file = os.path.join(FLAGS.output_dir, m.description)
        logging.info(f'Saving monomer object {output_file}')
        if FLAGS.compress:
            pickle.dump(m, bz2.BZ2File(f"{output_file}.pkl.bz2", "w"))
        else:
            pickle.dump(m, open(f"{output_file}.pkl", "wb"))


def filter_and_save_monomer_object(monomer: MonomericObject, filters: dict,
    pipeline: DataPipeline) -> None:
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
        if FLAGS.skip_existing and check_existing_objects(FLAGS.output_dir, file_name):
            logging.info(f"{file_name} exists in {FLAGS.output_dir} Skipped")
            continue
        monomer.make_template_features(
            template_featuriser,
            filter_t = filter_t,
            inplace = True
        )
        old_description = monomer.description
        monomer.description = f'{monomer.description}_{filter_name}'
        output_file = os.path.join(FLAGS.output_dir, f'{monomer.description}')
        logging.info(f'Saving monomer object {output_file}')
        if FLAGS.compress:
            pickle.dump(monomer, bz2.BZ2File(f"{output_file}.pkl.bz2", "w"))
        else:
            pickle.dump(monomer, open(f"{output_file}.pkl", "wb"))
        logging.info(f'Dumping search results in a filter file ' \
                        f'{output_file}_filter_results.yaml')
        with open(f'{output_file}_filter_results.yaml', 'w') as yaml_file:
            yaml.dump(monomer.filter_pdb_results, yaml_file)
        monomer.description = old_description



def main(argv):
    try:
        Path(FLAGS.output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(os.path.join(FLAGS.output_dir,
                                    f'{FLAGS.logger_file}.log'))
    except FileExistsError:
        logging.info("Multiple processes are trying to create" \
                    " the same folder now.")

    flags_dict = FLAGS.flag_values_dict()
    if FLAGS.data_dir is not None:
        create_global_arguments(flags_dict)
    elif FLAGS.use_mmseqs2 is False:
        raise ValueError('No indication for genetic database, \
                                please set a value for data_dir or \
                                set use_mmseqs2 to True')
    elif FLAGS.use_mmseqs2_templates is False:
        raise ValueError('No indication for template database, \
                                please set a value for data_dir or \
                                set use_mmseqs2_templates to True')
        

    if FLAGS.custom_template_path:
        ff_appendix = ['_a3m.ffdata', 
                        '_a3m.ffindex', 
                        '_cs219.ffdata', 
                        '_cs219.ffindex']
        ff_files = [os.path.join(FLAGS.custom_template_path,
                                    f'pdb70_{ff}') for ff in ff_appendix]
        for ff in ff_files:
            if not os.path.isfile(os.path.join(FLAGS.custom_template_path, ff)):
                mk_hhsearch_db(FLAGS.custom_template_path)
                break

    if not FLAGS.use_mmseqs2:
        if not FLAGS.max_template_date:
            # The pipeline for the template requires a limit date, not 
            # providing it closes the script.
            logging.info("You have not provided a max_template_date." \
                "Please specify a date and run again.")
            sys.exit()
        else:
            pipeline = create_pipeline()
            uniprot_database_path = os.path.join(FLAGS.data_dir,
                "uniprot/uniprot.fasta")
            flags_dict.update({"uniprot_database_path": uniprot_database_path})
            if os.path.isfile(uniprot_database_path):
                uniprot_runner = create_uniprot_runner(
                    FLAGS.jackhmmer_binary_path, uniprot_database_path
                )
            else:
                # Missing the uniprot.fasta file does not allow the script to 
                # properly work
                logging.info(
                    f"Failed to find uniprot.fasta in {uniprot_database_path}."\
                    " Please make sure data_dir has been configured correctly."
                )
                sys.exit()
    else:
        #mmseqs2 uses a specific pipeline created by a different function.
        #a serach function for uniprot is not required by using mmseqs2
        pipeline=None
        uniprot_runner=None
        flags_dict=FLAGS.flag_values_dict()

    filters = {qf: load_template_filter(qf) for qf in FLAGS.template_filters}

    seqs = iter_seqs(FLAGS.fasta_paths)
    for seq_idx, (curr_seq, curr_desc) in enumerate(seqs, 1):
        if FLAGS.seq_index != seq_idx and FLAGS.seq_index:
            continue
        elif curr_desc is None or curr_desc.isspace():
            continue
        else:
            if FLAGS.use_mmseqs2:
                curr_monomer = MonomericObjectMmseqs2(curr_desc.replace(' ', '_'), curr_seq)
            else:
                curr_monomer = MonomericObject(curr_desc.replace(' ', '_'), curr_seq)
                curr_monomer.uniprot_runner = uniprot_runner
            
            create_and_save_monomer_objects(curr_monomer,
                                            pipeline,
                                            flags_dict,
                                            use_mmseqs2=FLAGS.use_mmseqs2)
            filter_and_save_monomer_object(curr_monomer, filters, pipeline)

            del curr_monomer


if __name__ == "__main__":
    flags.mark_flags_as_required(
        ["fasta_paths", "output_dir","max_template_date"]
    )
    app.run(main)
