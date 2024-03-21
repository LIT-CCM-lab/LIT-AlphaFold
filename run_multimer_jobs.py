#!/usr/bin/env python3


# Original Author: Dingquan Yu
# Modified by Luca Chiesa
# A script to create region information for create_multimer_features.py
# #
import os
import sys
import pickle
import random
from pathlib import Path
from absl import app, flags, logging

import openmm
import jax

from alphafold import run_alphafold as run_af

#run_af = get_run_alphafold()


from litaf.objects import MultimericObject
from litaf.predict_structure import predict, ModelsToRelax
from litaf.create_input import (create_pulldown,
                                            create_homooligomers,
                                            create_all_vs_all,
                                            create_custom_jobs)
from litaf.utils import setup_logging, create_colabfold_runners, load_mutation_dict




flags = run_af.flags
# Basic i/o parameter on how to run the calculation
flags.DEFINE_enum("mode","pulldown",
                ["pulldown", "all_vs_all", "homo-oligomer", "custom"],
                "choose the mode of running multimer jobs",
                )
flags.DEFINE_string("output_path", None,
                    "output directory where the region data is going to be stored")
flags.DEFINE_list("monomer_objects_dir", None,
                "a list of directories where monomer objects are stored",)
flags.DEFINE_list("input_file", None, "path to input file")
delattr(flags.FLAGS, "data_dir")
flags.DEFINE_string("data_dir", None, "Path to params directory")
flags.DEFINE_string('logger_file', 'alphafold_prediction',
                'File where to store the results')
flags.DEFINE_boolean("save_multimers", False,
                    "Save the MultimerObject as pkl files")

#Inference settings
flags.DEFINE_integer("num_cycle_mono", 5,
                    "number of recycles for monomer prediction")
flags.DEFINE_integer("num_cycle_multi", 20,
                    "number of recycles for multimer prediction")
flags.DEFINE_integer("max_seq", None, "Maximum number of sequences in the msa")
flags.DEFINE_integer("max_extra_seq", None,
                    "Maximum number of extra sequences in the msa")
flags.DEFINE_boolean("allow_resume", True, "resume previous predictions")
flags.DEFINE_integer("num_predictions_per_model", 1,
                    "How many predictions per model. Default is 1")
flags.DEFINE_boolean("dropout", False, "Use dropout at inference time")
flags.DEFINE_boolean("cluster_profile", True, "Use cluster profile")
flags.DEFINE_boolean("save_all", False, 
                    "Save additional prediction information: distogram, \
                    masked_msa, experimentally resolved weights")
delattr(flags.FLAGS, "models_to_relax")
flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.NONE, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')

#Input customization settings
flags.DEFINE_boolean("only_template", False,
                    "Use only models with template features")
flags.DEFINE_boolean("use_templates", True,
                    "Use template features during prediction")
flags.DEFINE_boolean("remove_pair_msa", False,
                    "Do not generate paired MSA when constructing multimer objects")
flags.DEFINE_boolean("remove_unpaired_msa", False,
                    "Remove single chain MSA when constructing multimer objects")
flags.DEFINE_boolean("remove_template_msa", False,
                     "Remove single chain MSAs if template information is availalbe")
flags.DEFINE_boolean("shuffle_templates", False,
                    "Shuffle templates to use random templates during prediction")
flags.DEFINE_string('mutate_msa_file', None,
                    "yaml file containing the residues to mutate in the MSA")
flags.DEFINE_string('remove_msa_region', None,
                    "Regions where to remove MSA features")
flags.DEFINE_boolean("modify_unpaired_msa", True, "allow modification on the unpaired portion of the MSA")
flags.DEFINE_boolean("modify_paired_msa", False, "allow modification on the paired portion of the MSA")


flags.mark_flag_as_required("output_path")
flags.mark_flag_as_required('input_file')
flags.mark_flag_as_required('monomer_objects_dir')
flags.mark_flag_as_required('data_dir')

unused_flags = (
    'bfd_database_path',
    'db_preset',
    'fasta_paths',
    'hhblits_binary_path',
    'hhsearch_binary_path',
    'hmmbuild_binary_path',
    'hmmsearch_binary_path',
    'jackhmmer_binary_path',
    'kalign_binary_path',
    'max_template_date',
    'mgnify_database_path',
    'num_multimer_predictions_per_model',
    'obsolete_pdbs_path',
    'output_dir',
    'pdb70_database_path',
    'pdb_seqres_database_path',
    'small_bfd_database_path',
    'template_mmcif_dir',
    'uniprot_database_path',
    'uniref30_database_path',
    'uniref90_database_path',
)

for flag in unused_flags:
    delattr(flags.FLAGS, flag)


FLAGS = flags.FLAGS


def predict_individual_jobs(
    multimer_object,
    output_path: str,
    model_runners: list,
    random_seed: int,
    save_multimer: bool = False,
    ) -> None:
    '''Run AlphaFold predictions on a single object 

    Parameters
    ----------
    multimer_object: MultimericObject or MonomericObject
        Object containing the MSA and template featrues necessary for prediction
    output_path: str
        Path where to store the results of the calculations
    model_runners: list
        Runners to use for prediction
    random_seed: int
        Value used as seed for random number generation
    save_multimers: bool, default = False
        Save the passed objects for further analysis or to reuse for new
         calculations
    pae_plots: bool, default=True
        Create and save PAE plots generated by AlphaFold

    Returns
    -------
    None
    '''
    output_path = os.path.join(output_path, multimer_object.description)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"now running prediction on {multimer_object.description}")

    if not isinstance(multimer_object, MultimericObject):
        multimer_object.input_seqs = [multimer_object.sequence]

    if save_multimer:
        pickle.dump(multimer_object, open(f"{output_path}.pkl", "wb"))

    predict(
        model_runners,
        output_path,
        multimer_object.feature_dict,
        random_seed,
        FLAGS.benchmark,
        fasta_name=multimer_object.description,
        models_to_relax=FLAGS.models_to_relax,
        seqs=multimer_object.input_seqs,
        allow_resume=FLAGS.allow_resume
    )


def predict_multimers(
    multimers: list,
    save_multimers: bool =False
    ) -> None:
    """
    Final function to predict multimers

    Parameters
    ----------
    multimers: list
        Multimeric and monomeric object to use for structure prediction
    save_multimers: bool, default=False
        Save the passed objects for further analysis or to reuse for new
         calculations

    Returns
    -------
    None
    """
    
    t_multi = []
    t_mono = []
    run_description = ''
    if FLAGS.dropout:
        logging.info("Run prediction using dropout for enhanced sampling")
        run_description = run_description+'_dropout'
    if not FLAGS.cluster_profile:
        logging.info("Run preddiction without cluster profiling")
        run_description = run_description+'_noclusterprofile'
    if FLAGS.max_seq is not None and FLAGS.max_extra_seq is not None:
        run_description = run_description+f'_MSA-subsampling-{FLAGS.max_seq}:{FLAGS.max_extra_seq}'
    for obj in multimers:
        obj.description = obj.description+run_description
        if isinstance(obj, MultimericObject):
            t_multi.append(obj)
        else:
            t_mono.append(obj)

    if len(t_multi) > 0:
        n = 5
        model_runners = create_colabfold_runners(
                            '_multimer_v3',
                            n,
                            True,
                            FLAGS.num_cycle_multi,
                            FLAGS.data_dir,
                            FLAGS.max_seq,
                            FLAGS.max_extra_seq,
                            FLAGS.num_predictions_per_model,
                            FLAGS.dropout,
                            FLAGS.cluster_profile,
                            FLAGS.save_all)
        random_seed = random.randrange(sys.maxsize // len(model_runners))
        logging.info(
            f"Using base random seed {random_seed} for the predictions"
            )

        for obj in t_multi:
            logging.info('Multimer object: '+obj.description)
            predict_individual_jobs(
                obj,
                FLAGS.output_path,
                model_runners=model_runners,
                random_seed=random_seed,
                save_multimer=save_multimers,
            )
        
    if len(t_mono) > 0:
        n = 2 if FLAGS.only_template else 5
        model_runners = create_colabfold_runners(
                            '_ptm',
                            n,
                            FLAGS.use_templates,
                            FLAGS.num_cycle_mono,
                            FLAGS.data_dir,
                            FLAGS.max_seq,
                            FLAGS.max_extra_seq,
                            FLAGS.num_predictions_per_model,
                            FLAGS.dropout,
                            FLAGS.cluster_profile,
                            FLAGS.save_all)
        random_seed = random.randrange(sys.maxsize // len(model_runners))
        logging.info(
            f"Using base random seed {random_seed} for the predictions"
            )
        for obj in t_mono:
            logging.info('Monomer object: '+obj.description)
            predict_individual_jobs(
                obj,
                FLAGS.output_path,
                model_runners=model_runners,
                random_seed=random_seed,
                save_multimer=save_multimers,
            )

def main(argv):
    if not os.path.isdir(FLAGS.output_path):
        Path(FLAGS.output_path).mkdir(parents=True, exist_ok=True)

    setup_logging(os.path.join(FLAGS.output_path,
                                f'{FLAGS.logger_file}.log'))

    # check what device is available
    try:
        # check if TPU is available
        import jax.tools.colab_tpu
        jax.tools.colab_tpu.setup_tpu()
        logging.info('Running on TPU')
    except:
        if jax.local_devices()[0].platform == 'cpu':
            logging.info("WARNING: no GPU detected, will be using CPU")
        else:
            logging.info('Running on GPU')

    if FLAGS.mutate_msa_file:
        mutate_msa = load_mutation_dict(FLAGS.mutate_msa_file)
    else:
        mutate_msa = None

    if FLAGS.remove_msa_region:
        regions = FLAGS.remove_msa_region.split(',')
        output_region = []
        for r in regions:
            output_region.append((int(r.split("-")[0]), int(r.split("-")[1])))
    else:
        output_region = None

    create_functions = {'pulldown': create_pulldown,
                        'all_vs_all': create_all_vs_all,
                        'homo-oligomer': create_homooligomers,
                        'custom': create_custom_jobs,
                        }

    multimers = create_functions[FLAGS.mode](
                    FLAGS.input_file,
                    FLAGS.monomer_objects_dir,
                    pair_msa=not FLAGS.remove_pair_msa,
                    remove_msa=FLAGS.remove_unpaired_msa,
                    remove_template_msa=FLAGS.remove_template_msa,
                    mutate_msa = mutate_msa,
                    remove_msa_region = output_region,
                    remove_templates = not FLAGS.use_templates,
                    shuffle_templates = FLAGS.shuffle_templates,
                    paired_msa = FLAGS.modify_paired_msa,
                    unpaired_msa = FLAGS.modify_unpaired_msa,
                )


    predict_multimers(multimers, save_multimers = FLAGS.save_multimers)


if __name__ == "__main__":
    app.run(main)
