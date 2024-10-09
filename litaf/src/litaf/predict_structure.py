'''
Functions for predictions and results handling

* *get_score_from_result_pkl*, read confidence score
* *get_existing_model_info*, read model info
* *predict*, launch calculations
'''

#
# This script is
# based on run_alphafold.py by DeepMind from https://github.com/deepmind/alphafold
# and contains code copied from the script run_alphafold.py
# The script presents ulterior modifications from the script with the same name in AlphaPulldown
# #
import json
import os
try:
   import cPickle as pickle
except:
   import pickle
import time
import logging
import bz2
import numpy as np

from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.relax import relax
from alphafold import run_alphafold as run_af

#import jax
#import jax.numpy as jnp
#import optax

RELAX_MAX_ITERATIONS = run_af.RELAX_MAX_ITERATIONS
RELAX_ENERGY_TOLERANCE = run_af.RELAX_ENERGY_TOLERANCE
RELAX_STIFFNESS = run_af.RELAX_STIFFNESS
RELAX_EXCLUDE_RESIDUES = run_af.RELAX_EXCLUDE_RESIDUES
RELAX_MAX_OUTER_ITERATIONS = run_af.RELAX_MAX_OUTER_ITERATIONS

def get_score_from_result_pkl(pkl_path):
    """Get confidence score from the model result pkl file

    Parameters
    ----------
    pkl_path: str
        path containing the pkl results

    Return
    ------
    type of score (iptm+ptm for monomers, mean_plddt for multimers), and structure confidence score: str, list
    """

    if pkl_path.endswith('.bz2'):
        in_file = bz2.BZ2File(pkl_path, "rb")
    else:
        in_file = open(pkl_path, 'rb')
    result = pickle.load(in_file)
    if "iptm" in result:
        score_type = "iptm+ptm"
        score = 0.8 * result["iptm"] + 0.2 * result["ptm"]
    else:
        score_type = "plddt"
        score = result["mean_plddt"]

    return score_type, score

def get_existing_model_info(output_dir, model_runners ):
    '''Extract model info

    Parameters
    ----------
    output_dir: str
        directory containing the AlphaFold outputs
    model_runners: list
        runners used for the predictions

    Retrun
    ------
    ranking confidences scores, unrelaxed protein object, unrelaxed pdb files, number of processed models, type of confidence scores: dict, dict, dict, int, str
    '''
    ranking_confidences = {}
    unrelaxed_proteins = {}
    unrelaxed_pdbs = {}
    processed_models = 0
    score_name = None

    for model_name in model_runners.keys():
        pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
        pkl_path = os.path.join(output_dir, f"result_{model_name}.pkl")

        if not os.path.exists(pdb_path):
            break
        if os.path.exists(pkl_path):
            score_name, score = get_score_from_result_pkl(pkl_path)
        elif os.path.exists(pkl_path+'.bz2'):
            score_name, score = get_score_from_result_pkl(pkl_path+'.bz2')
        else:
            break

        
        ranking_confidences[model_name] = score.tolist()

        with open(pdb_path, "r") as f:
            unrelaxed_pdb_str = f.read()
        unrelaxed_proteins[model_name] = protein.from_pdb_string(unrelaxed_pdb_str)
        unrelaxed_pdbs[model_name] = unrelaxed_pdb_str

        processed_models += 1

    return ranking_confidences, unrelaxed_proteins, unrelaxed_pdbs, processed_models, score_name


def predict(
    model_runners_and_params,
    output_dir,
    feature_dict,
    random_seed,
    benchmark,
    models_to_relax: str,
    fasta_name,
    allow_resume=True,
    seqs=[],
    use_gpu_relax=True,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    save_recycles: bool = False,
    stop_at_score: float = 100,
    save_all: bool = False,
    compress_results: bool = True,
) -> None:
    '''Run AlphaFold prediction
        Modified version of the ColabFold predict function to include the
        additionally ColabFold featrues

    Parameters
    ----------
    model_runner_and_params: list
        list of model runners and deep learning parameters
    output_dir: str
        Directory where to save the results 
    feature_dict: FeatureDict
        Features of the dtructure to predict
    random_seed: int
        The random seed to use for calculation
    benchmark: bool
        Remove compilation time for timing calculation
    models_to_relax: ModelToRelax
        Determine on which models to perform relaxation
    fasta_name: str
        Name of the object
    allow_resume: bool
        Allow to resume calculation if the previous calculations were not completed
    seqs: list
    use_gpu_relax: bool
        Use GPU to perform relaxation calculations
    save_single_representations: bool
    save_pair representation: bool
    save_recycles: bool
        Save the results at each recycle phase
    stop_at_score: float
        Early stopping of the calculation when confidence is above the given threshold
    save_all: bool
        Save all data

    Return
    ------
    None
    '''
    timings = {}
    relaxed_pdbs = {}
    unrelaxed_pdbs = {}
    relax_metrics = {}
    ranking_confidences = {}
    unrelaxed_proteins = {}
    START = 0
    ranking_output_path = os.path.join(output_dir, "ranking_debug.json")
    temp_timings_output_path = os.path.join(output_dir, "timings_temp.json")
    #To keep track of timings in case of crash and resume

    if allow_resume:
        logging.info("Checking for existing results")
        ranking_confidences, unrelaxed_proteins, unrelaxed_pdbs, START, ranking_confidence_score_type = get_existing_model_info(output_dir, model_runners_and_params)
        #ADD A WAY TO RESUME ALL TIMINGS
        #pdb.set_trace()
        if (os.path.exists(ranking_output_path) and 
            len(unrelaxed_pdbs) == len(model_runners_and_params)):
                logging.info(
                    "ranking_debug.json exists. Skipping prediction." \
                    " Restoring unrelaxed predictions and ranked order"
                )
                START = len(model_runners_and_params)
        elif START > 0:
            logging.info("Found existing results, continuing from there.")

    num_models = len(model_runners_and_params)
    for (model_index,
        (model_name, (model_runner, params))) in enumerate(
                            model_runners_and_params.items()):
        if model_index < START:
            continue
        model_runner.params = params
        logging.info("Running model %s on %s", model_name, fasta_name)
        t_0 = time.time()
        model_random_seed = model_index + random_seed * num_models
        logging.info(f"Using random seed {model_random_seed} for predictions")
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=model_random_seed
        )
        timings[f"process_features_{model_name}"] = time.time() - t_0

        #monitor intermediate results
        def callback(result, recycles):
            if recycles == 0:
                result.pop("tol",None)
            #if not is_complex: result.pop("iptm",None)
            print_line = ""
            scores = [["mean_plddt","pLDDT"],
                    ["ptm","pTM"],
                    ["iptm","ipTM"],
                    ["tol","tol"]]
            for x,y in scores:
              if x in result:
                print_line += f" {y}={result.get(x):.3g}"
            logging.info(f"{model_name} recycle={recycles}{print_line}")
            if save_recycles:
                write_prediction_output(output_dir, result, processed_feature_dict,
                                        model_name+f'_recycle_{recycles}', compress_results,
                                        (len(seqs) == 1), save_all = save_all)

        return_representations = save_all or \
                                save_single_representations or \
                                save_pair_representations

        t_0 = time.time()
        unrelaxed_protein,ranking_confidence,ranking_confidence_score_type = run_prediction(model_runner, model_name,
                                                                                processed_feature_dict,
                                                                                model_random_seed,
                                                                                return_representations,
                                                                                callback,
                                                                                seqs, output_dir,
                                                                                compress_results,
                                                                                (len(seqs) == 1))

        unrelaxed_proteins.update(unrelaxed_protein)
        ranking_confidences.update(ranking_confidence)

        t_diff = time.time() - t_0
        timings[f"predict_and_compile_{model_name}"] = t_diff
        logging.info(
            f"Total JAX model {model_name} on {fasta_name} predict time " \
            f"(includes compilation time, see --benchmark): {t_diff:.1f} s"
        )

        if benchmark:
            t_0 = time.time()
            model_runner.predict(
                processed_feature_dict, random_seed=model_random_seed
            )
            t_diff = time.time() - t_0
            timings[f"predict_benchmark_{model_name}"] = t_diff
            logging.info(
                f"Total JAX model {model_name} on {fasta_name} predict time " \
                f"(excludes compilation time):  {t_diff:.1f} s"
            )

        with open(temp_timings_output_path, "w") as f:
            f.write(json.dumps(timings, indent=4))


    if len(unrelaxed_pdbs) == 0:
        unrelaxed_pdbs = {name: protein.to_pdb(prot) for name, prot in unrelaxed_proteins.items()}

    # Rank by model confidence.
    ranked_order = [
        model_name for model_name, confidence in
        sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]

    # Relax predictions.
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
        use_gpu=use_gpu_relax)

    if models_to_relax == 'best':
        to_relax = [ranked_order[0]]
    elif models_to_relax == 'all':
        to_relax = ranked_order
    else:
        to_relax = []

    for model_name in to_relax:
        relaxed_output_path = os.path.join(
            output_dir, f'relaxed_{model_name}.pdb')
        if os.path.isfile(relaxed_output_path):
            logging.info(f'Found existing relaxed structure for: {relaxed_output_path}')
            continue
        t_0 = time.time()
        relaxed_pdb_str, _, violations = amber_relaxer.process(
            prot=unrelaxed_proteins[model_name])
        relax_metrics[model_name] = {
            'remaining_violations': violations,
            'remaining_violations_count': sum(violations)
        }
        timings[f'relax_{model_name}'] = time.time() - t_0

        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)

    # Write out relaxed PDBs in rank order.
    for idx, model_name in enumerate(ranked_order):
        ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
        with open(ranked_output_path, 'w') as f:
            if model_name in relaxed_pdbs:
                f.write(relaxed_pdbs[model_name])
            else:
                f.write(unrelaxed_pdbs[model_name])

    if (not os.path.exists(ranking_output_path) or 
        not allow_resume or 
        os.stat(ranking_output_path).st_size == 0 or
        START > 0): 
        # already exists if restored.
        with open(ranking_output_path, "w") as f:
            f.write(
                json.dumps(
                    {ranking_confidence_score_type: ranking_confidences,
                    "order": ranked_order}, indent=4
                )
            )
    nl = '\n'
    logging.info(f"Final timings for {fasta_name}: {nl.join([f'{k}: {v}' for k,v in timings.items()])}")
    timings_output_path = os.path.join(output_dir, "timings.json")
    with open(timings_output_path, "w") as f:
        f.write(json.dumps(timings, indent=4))
    if models_to_relax:
        relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
        with open(relax_metrics_path, 'w') as f:
            f.write(json.dumps(relax_metrics, indent=4))

    if os.path.exists(temp_timings_output_path):
    #should not happen at this stage but just in case
        try:
            os.remove(temp_timings_output_path)
        except OSError:
            pass


def run_prediction(model_runner, model_name,
                    processed_feature_dict, model_random_seed,
                    return_representations, callback, seqs, output_dir,
                    compress_results, remove_leading_feature_dimension):

    prediction_result, _ = model_runner.predict(
            processed_feature_dict, random_seed=model_random_seed,
            return_representations=return_representations,
            callback=callback
        )
        # update prediction_result with input seqs
    prediction_result.update({"seqs": seqs})

    unrelaxed_protein = write_prediction_output(output_dir, prediction_result, processed_feature_dict,
                            model_name, compress_results, not model_runner.multimer_mode, True)

    ranking_confidence_score_type = "iptm+ptm" if "iptm" in prediction_result else "plddts"

    return {model_name: unrelaxed_protein}, {model_name: prediction_result["ranking_confidence"].tolist()}, ranking_confidence_score_type

'''
def run_optimization(model_runner, model_name, max_iter, msa_params,
                    processed_feature_dict, model_random_seed,
                    return_representations, callback, seqs, output_dir,
                    compress_results, remove_leading_feature_dimension, learning_rate):

    msa_shape = processed_feature_dict['msa'].shape
    processed_feature_dict['msa_shape'] = msa_shape
    msa_params = jnp.zeros((config.model.embeddings_and_evoformer.num_msa, msa_shape[1], 23), dtype='float32') #23 classes for aa, gap, mask
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(msa_params)

    def loss_fn(msa_params, processed_features_dict):
        prediction_result, _ = model_runner.predict(
            processed_feature_dict, random_seed=model_random_seed,
            return_representations=return_representations,
            callback=callback, to_numpy = False,
        )
        return 1/prediction_result["ranking_confidence"], prediction_result

    def update(msa_params, opt_state, processed_feature_dict):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(msa_params, processed_feature_dict)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(msa_params, updates)
        return loss, aux, new_params, new_opt_state

    unrelaxed_proteins = {}
    ranking_confidences = {}

    for i in range(max_iter):
        if (np.array(metrics['ranking_confidence'])>confidence_threshold).sum()>=num_above_t:
            logging.info('Confidence threshold reached.')
            return unrelaxed_proteins

        loss, aux, msa_params, opt_state = update(msa_params, opt_state, processed_feature_dict)
        unrelaxed_proteins[model_name+f'_opt_{i}'] = write_prediction_output(output_dir,
                                                                            aux,
                                                                            processed_feature_dict
                                                                            model_name+f'_opt_{i}',
                                                                            compress_results,
                                                                            remove_leading_feature_dimension)
        ranking_confidence_score_type = "iptm+ptm" if "iptm" in aux else "plddts"


        ranking_confidences[model_name+f'_opt_{i}'] = {model_name: aux["ranking_confidence"].tolist()}

    return unrelaxed_proteins, ranking_confidences, ranking_confidence_score_type
'''

def write_prediction_output(output_dir, prediction_result, processed_feature_dict,
                            model_name, compress_results,
                            remove_leading_feature_dimension, save_all = False):


    result_output_path = os.path.join(output_dir,
                                    f"result_{model_name}.pkl")

    plddt_b_factors = np.repeat(
        prediction_result["plddt"][:, None], residue_constants.atom_type_num, axis=-1
    )
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=remove_leading_feature_dimension,
    )

    unrelaxed_pdb_path = os.path.join(output_dir,
                                    f"unrelaxed_{model_name}.pdb")
    with open(unrelaxed_pdb_path, "w") as f:
        f.write(protein.to_pdb(unrelaxed_protein))

    if save_all:
        if compress_results:
            out_file = bz2.BZ2File(result_output_path+'.bz2','w')
        else:
            out_file = open(result_output_path, 'wb')
        pickle.dump(prediction_result, out_file)

    return unrelaxed_protein