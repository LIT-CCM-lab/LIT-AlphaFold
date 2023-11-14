'''
Functions for predictions and results handling

* *get_score_from_result_pkl*, read confidence score
* *get_existing_model_info*, read model info
* *predict*, launch calculations
'''

#
# This script is
# based on run_alphafold.py by DeepMind from https://github.com/deepmind/alphafold
# and contains code copied from the script run_alphafold.py.
# #
import json
import os
import pickle
import time
import pdb
from absl import logging
import numpy as np

from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.relax import relax

from alphapulldown.utils import get_run_alphafold

run_af = get_run_alphafold()

RELAX_MAX_ITERATIONS = run_af.RELAX_MAX_ITERATIONS
RELAX_ENERGY_TOLERANCE = run_af.RELAX_ENERGY_TOLERANCE
RELAX_STIFFNESS = run_af.RELAX_STIFFNESS
RELAX_EXCLUDE_RESIDUES = run_af.RELAX_EXCLUDE_RESIDUES
RELAX_MAX_OUTER_ITERATIONS = run_af.RELAX_MAX_OUTER_ITERATIONS

ModelsToRelax = run_af.ModelsToRelax

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

    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
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

    for model_name, _ in model_runners.items():
        pdb_path = os.path.join(output_dir, f"unrelaxed_{model_name}.pdb")
        pkl_path = os.path.join(output_dir, f"result_{model_name}.pkl")

        if not (os.path.exists(pdb_path) and os.path.exists(pkl_path)):
            break

        try:
            with open(pkl_path, "rb") as f:
                result = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            break

        score_name, score = get_score_from_result_pkl(pkl_path)
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
    models_to_relax: ModelsToRelax,
    fasta_name,
    allow_resume=True,
    seqs=[],
    use_gpu_relax=True,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    save_recycles: bool = False,
    stop_at_score: float = 100,
    save_all: bool = False,
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
    unrelaxed_pdbs = {}
    relaxed_pdbs = {}
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
            if recycles == 0: result.pop("tol",None)
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
                final_atom_mask = result["structure_module"]["final_atom_mask"]
                b_factors = result["plddt"][:, None] * final_atom_mask
                unrelaxed_protein = protein.from_prediction(
                    features=input_features,
                    result=result,
                    b_factors=b_factors,
                    remove_leading_feature_dimension=(
                            "multimer" not in model_type)
                    )
                files.get("unrelaxed",
                        f"r{recycles}.pdb").write_text(
                                protein.to_pdb(unrelaxed_protein)
                                )
            
                if save_all:
                    with files.get("all",f"r{recycles}.pickle").open("wb") as handle:
                        pickle.dump(result, handle)
                del unrelaxed_protein

        return_representations = save_all or \
                                save_single_representations or \
                                save_pair_representations

        t_0 = time.time()
        prediction_result, _ = model_runner.predict(
            processed_feature_dict, random_seed=model_random_seed,
            return_representations=return_representations,
            callback=callback
        )

        ranking_confidence_score_type = "iptm+ptm" if "iptm" in prediction_result else "plddts"

        # update prediction_result with input seqs
        prediction_result.update({"seqs": seqs})

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

        plddt = prediction_result["plddt"]
        ranking_confidences[model_name] = prediction_result["ranking_confidence"].tolist()

        result_output_path = os.path.join(output_dir,
                                        f"result_{model_name}.pkl")
        with open(result_output_path, "wb") as f:
            pickle.dump(prediction_result, f, protocol=4)

        plddt_b_factors = np.repeat(
            plddt[:, None], residue_constants.atom_type_num, axis=-1
        )
        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode,
        )

        unrelaxed_proteins[model_name] = unrelaxed_protein
        unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(output_dir,
                                        f"unrelaxed_{model_name}.pdb")
        with open(unrelaxed_pdb_path, "w") as f:
            f.write(unrelaxed_pdbs[model_name])

        with open(temp_timings_output_path, "w") as f:
            f.write(json.dumps(timings, indent=4))


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

    if models_to_relax == ModelsToRelax.BEST:
        to_relax = [ranked_order[0]]
    elif models_to_relax == ModelsToRelax.ALL:
        to_relax = ranked_order
    elif models_to_relax == ModelsToRelax.NONE:
        to_relax = []

    for model_name in to_relax:
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
        relaxed_output_path = os.path.join(
            output_dir, f'relaxed_{model_name}.pdb')
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
        os.stat(ranking_output_path).st_size == 0): 
        # already exists if restored.
        with open(ranking_output_path, "w") as f:
            f.write(
                json.dumps(
                    {ranking_confidence_score_type: ranking_confidences,
                    "order": ranked_order}, indent=4
                )
            )

    logging.info("Final timings for %s: %s", fasta_name, timings)
    timings_output_path = os.path.join(output_dir, "timings.json")
    with open(timings_output_path, "w") as f:
        f.write(json.dumps(timings, indent=4))
    if models_to_relax != ModelsToRelax.NONE:
        relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
        with open(relax_metrics_path, 'w') as f:
            f.write(json.dumps(relax_metrics, indent=4))

    if os.path.exists(temp_timings_output_path):
    #should not happen at this stage but just in case
        try:
            os.remove(temp_timings_output_path)
        except OSError:
            pass