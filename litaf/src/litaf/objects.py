'''
Objects containing features for AlphaFold calculations

* *MonomericObject*, monomer
* *ChoppedObject*, chopped monomer
* *MultimerObject*, multimer

'''

#
# Original author Dingquan Yu
# Modified by Luca Chiesa
# scripts to create objects (e.g. monomeric object, multimeric objects)
#
# #
import logging
import tempfile
import os
import contextlib
import bz2
from pathlib import Path as plPath

try:
   import cPickle as pickle
except:
   import pickle

import numpy as np

from alphafold.data import (parsers,
                            pipeline,
                            pipeline_multimer,
                            msa_pairing,
                            feature_processing,
                            templates)
from alphafold.data.tools import hhsearch
from alphafold.data.feature_processing import MAX_TEMPLATES, MSA_CROP_SIZE
from alphafold.common import residue_constants

from colabfold.batch import (unserialize_msa,
                            get_msa_and_templates,
                            msa_to_str,
                            build_monomer_feature,)
from colabfold.utils import DEFAULT_API_SERVER

from alphapulldown.utils import mk_mock_template, check_empty_templates

from litaf.utils import remove_msa_for_template_aligned_regions
from litaf.filterpdb import (filter_template_hits,
                                filter_template_features,
                                generate_filter,)
from litaf.pipeline import make_msa_features, DataPipeline
from litaf.datatypes import FeatureDict
from litaf.rename import *

#Added for MSA clustering
from polyleven import levenshtein
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from litaf.utils import (encode_seqs,
                        consensusVoting,
                        plot_msa_landscape,
                        to_string_seq)

USER_AGENT = 'LIT-AlphaFold/v1.0 https://github.com/LIT-CCM-lab/LIT-AlphaFold'


@contextlib.contextmanager
def temp_fasta_file(sequence_str):
    """function that create temp file"""
    with tempfile.NamedTemporaryFile("w", suffix=".fasta") as fasta_file:
        fasta_file.write(sequence_str)
        fasta_file.seek(0)
        yield fasta_file.name

def load_monomer_objects(monomer_dir_dict, protein_name):
    """
    a function to load monomer an object from its pickle

    args
    monomer_dir_dict: a dictionary recording protein_name and its directory. created by make_dir_monomer_dictionary()
    """
    if monomer_dir_dict.get(protein_name, None) is not None:
        target_path = os.path.join(monomer_dir_dict[protein_name], f"{protein_name}.pkl")
        if os.path.isfile(target_path):
            monomer = pickle.load(open(target_path, "rb"))
        elif os.path.isfile(target_path+'.bz2') is not None:
            monomer = pickle.load(bz2.BZ2File(target_path+'.bz2', "rb"))
        else:
            raise OSError(f'Could not find {protein_name} in {monomer_dir_dict[protein_name]}')
    else:
        raise OSError(f'File for {protein_name} was not found')
    if isinstance(monomer, MultimericObject):
        return monomer
    if check_empty_templates(monomer.feature_dict):
        monomer.feature_dict = mk_mock_template(monomer.feature_dict)
    return monomer

def check_existing_objects(output_dir, pickle_name):
    """check whether the wanted monomer object already exists in the output_dir"""
    logging.info(f"checking if {os.path.join(output_dir, pickle_name)} already exists")
    return os.path.isfile(os.path.join(output_dir, pickle_name+'.pkl')) or os.path.isfile(os.path.join(output_dir, pickle_name+'.pkl.bz2'))


class MonomericObject:
    """Monomeric objects

    Parameters
    ----------
    description : str
        Description of the protein. By default is everything 
        after the ">" symbol in the fasta input file, the description is 
        updated automatically upon performing modification to the object

    sequence : str
        Amino acids sequence of the monomer

    Attributes
    ----------
    description : str
        Description of the protein. By default is everything 
        after the ">" symbol in the fasta input file, the description is 
        updated automatically upon performing modification to the object

    sequence : str
        Amino acids sequence of the monomer

    paired_msa : bool
        The monomer presents paired MSA information

    feature_dict : dict
        Information needed by AlphaFold to perform calculations on monomers

    _uniprot_runner : Jackhmmer
        Uniprot runner object

    template_hits : str
        Raw output of the search for template hits

    template_featuriser_mmseqs2 : None or HhsearchHitFeaturizer
        The template featuriser necessary when using mmseqs2 for MSA search.
        If mmseqs2 is not used it is set to None

    filter_pdb_results : list
        PDBIDs of the templates selected by the given query


    """

    def __init__(self, description, sequence) -> None:
        self.description = description
        self.sequence = sequence
        self.feature_dict = dict()
        self._uniprot_runner = None
        self.template_hits = None
        pass

    @property
    def uniprot_runner(self):
        return self._uniprot_runner

    @uniprot_runner.setter
    def uniprot_runner(self, uniprot_runner):
        self._uniprot_runner = uniprot_runner

    def all_seq_msa_features(
        self,
        input_fasta_path: str,
        uniprot_msa_runner,
        save_msa,
        output_dir=None,
        use_precomuted_msa=False,
    ) -> dict:
        """
        Get MSA features for unclustered uniprot, for pairing later on.

        Parameters
        ----------
        input_fasta_path: str
            Path of the fasta file containing the input sequence
        uniprot_msa_runner: 
            Runner object used to search the MSA in the uniprot database
        save_msa: bool
            Save the file generated from the MSA search
        output_dir: str, default=None
            Directory where to save the file generated by the MSA search
        use_precomputed_msa: bool, default=False
            If available read results from a previous MSA and template 
            search rather than run it again

        Retruns
        -------
        dict: the features from the unclustered uniprot search

        """
        if not use_precomuted_msa:
            if not save_msa:
                with tempfile.TemporaryDirectory() as tempdir:
                    logging.info("now going to run uniprot runner")
                    result = pipeline.run_msa_tool(
                        uniprot_msa_runner,
                        input_fasta_path,
                        f"{tempdir}/uniprot.sto",
                        "sto",
                        use_precomuted_msa,
                    )
            elif save_msa and (output_dir is not None):
                logging.info(
                    f"now going to run uniprot runner and save uniprot alignment in {output_dir}"
                )
                result = pipeline.run_msa_tool(
                    uniprot_msa_runner,
                    input_fasta_path,
                    f"{output_dir}/uniprot.sto",
                    "sto",
                    use_precomuted_msa,
                )
        else:
            result = pipeline.run_msa_tool(
                uniprot_msa_runner,
                input_fasta_path,
                f"{output_dir}/uniprot.sto",
                "sto",
                use_precomuted_msa,
            )
        msa = parsers.parse_stockholm(result["sto"])
        msa = msa.truncate(max_seqs=50000)
        all_seq_features = make_msa_features([msa])
        valid_feats = msa_pairing.MSA_FEATURES + (
            "msa_species_identifiers",
            "msa_uniprot_accession_identifiers",
        )
        feats = {
            f"{k}_all_seq": v for k, v in all_seq_features.items() if k in valid_feats
        }
        return feats

    def make_template_features(
        self,
        template_featuriser,
        filter_t: list = [{}, []],
        inplace: bool = False
        ) -> dict:
        '''
        Generate template features

        Parameters
        ----------
        template_featuriser:
            Method to generate template features from the raw template hits 
            results
        filter: dict
            Dictionary containing specific used to filter the templates based 
            on their PDBID or by annotation in online databases
        inplace: bool
            Update the object feature_dict with the newly obtained template features.

        Return
        ------
        dict: dictionary containing the updated template features

        Raises
        ------
        Exception
            If there are no template_hits results to be processed

        '''
        if self.template_hits is None:
            raise Exception('There is no information about the templates for the monomer')
        if filter_t is None:
            filtered_templates = self.template_hits
        else:
            filtered_templates = filter_template_hits(self.template_hits, filter_t)
            self.filter_pdb_results = generate_filter(filtered_templates)

        templates_result = template_featuriser.get_templates(
            query_sequence=self.sequence, hits=filtered_templates
        )

        if inplace:
            self.feature_dict.update(dict(templates_result.features))
            return self.feature_dict
        else:
            new_feature_dict = self.feature_dict.copy()
            return new_feature_dict.update(dict(templates_result.features))


    def make_features(
        self,
        pipeline : DataPipeline,
        output_dir=None,
        use_precomputed_msa=False,
        save_msa=True,
        paired_msa = True,
        filter_t=None,
    ) -> None:
        """A method that make msa and template features
        
        Parameters
        ----------
        pipeline: DataPipeline
            Object used for the search in the MSA and template databases
        save_msa: bool
            Save the file generated from the MSA search
        output_dir: str, default=None
            Directory where to save the file generated by the MSA search
        use_precomputed_msa: bool, default=False
            If available read results from a previous MSA and template 
            search rather than run it again
        paired_msa: bool, default=True
            Perform search in the Uniprot unclustered databses to obtain MSA 
            information used for paired MSA calculation for multimer 
            predictions. If the calculations are limited to monomers this 
            search can be omitted to save time.
        query: dict, default={}
            Query used to specify templates

        Returns
        -------
        None

        """
        self.paired_msa = paired_msa

        if not use_precomputed_msa:
            if not save_msa:
                """this means no msa files are going to be saved"""
                logging.info("You have chosen not to save msa output files")
                sequence_str = f">{self.description}\n{self.sequence}"
                with temp_fasta_file(
                    sequence_str
                ) as fasta_file, tempfile.TemporaryDirectory() as tmpdirname:
                    msa_feature_dict, self.template_hits = pipeline.process(
                        input_fasta_path=fasta_file, msa_output_dir=tmpdirname
                    )
                    self.feature_dict.update(msa_feature_dict)
                    self.make_template_features(pipeline.template_featuriser,
                                                filter_t, inplace = True)
                    if paired_msa:
                        pairing_results = self.all_seq_msa_features(
                            fasta_file, self._uniprot_runner, save_msa
                        )
                        self.feature_dict.update(pairing_results)

            else:
                """this means no precomputed msa available
                    and will save output msa files"""
                msa_output_dir = os.path.join(output_dir, self.description)
                sequence_str = f">{self.description}\n{self.sequence}"
                logging.info(f"will save msa files in :{msa_output_dir}")
                plPath(msa_output_dir).mkdir(parents=True, exist_ok=True)
                with temp_fasta_file(sequence_str) as fasta_file:
                    msa_feature_dict, self.template_hits = pipeline.process(
                        fasta_file, msa_output_dir)
                    self.make_template_features(
                                pipeline.template_featuriser,
                                filter_t,
                                inplace = True)
                    self.feature_dict.update(msa_feature_dict)
                    if paired_msa:
                        pairing_results = self.all_seq_msa_features(
                            fasta_file, self._uniprot_runner,
                            save_msa, msa_output_dir
                        )
                        self.feature_dict.update(pairing_results)
        else:
            """This means precomputed msa files are available"""
            msa_output_dir = os.path.join(output_dir, self.description)
            plPath(msa_output_dir).mkdir(parents=True, exist_ok=True)
            logging.info(
                "use precomputed msa. Searching for msa files in :{}".format(
                    msa_output_dir
                )
            )
            sequence_str = f">{self.description}\n{self.sequence}"
            with temp_fasta_file(sequence_str) as fasta_file:
                msa_feature_dict, self.template_hits = pipeline.process(
                    fasta_file, msa_output_dir)
                self.feature_dict.update(msa_feature_dict)
                self.make_template_features(
                    pipeline.template_featuriser, filter_t, inplace = True)
                if paired_msa:
                    pairing_results = self.all_seq_msa_features(
                        fasta_file,
                        self._uniprot_runner,
                        save_msa,
                        msa_output_dir,
                        use_precomuted_msa=True,
                    )
                    self.feature_dict.update(pairing_results)


    def remove_template_from_msa(self, inplace = False):
        '''Remove monomer MSA data from template aligned regions

        The residues in the MSA covered by template aligned regions are removed.
        It is done to model different conformational state of the receptor as shown by Heo and Feig [1]

        References
        ----------
        [1] `HEO, Lim; FEIG, Michael. Multi‐state modeling of G‐protein coupled receptors at experimental accuracy.
            Proteins: Structure, Function, and Bioinformatics, 2022, 90.11: 1873-1885.
            <https://onlinelibrary.wiley.com/doi/full/10.1002/prot.26382>`__

        '''

        if inplace:
            new_feature_dict = self.feature_dict
            self.description = rename_remove_template_from_msa(self.description)
        else:
            new_feature_dict = self.feature_dict.copy()

        logging.info('By using the tool "Template MSA removal" please cite:\n\
                Heo, L, Feig, M.\n\
                Multi-state modeling of G-protein coupled receptors at experimental accuracy.\n\
                Proteins. 2022; 90(11): 1873-1885. doi:10.1002/prot.26382')
        remove_msa_for_template_aligned_regions(new_feature_dict)
        
        return new_feature_dict


    def remove_msa_features(self, inplace = False):
        '''Remove monomer MSA

        The MSA features are removed, only the input sequence is left.

        '''

        if inplace:
            new_feature_dict = self.feature_dict
            self.description = rename_remove_msa_features(self.description)
        else:
            new_feature_dict = self.feature_dict.copy()

        a3m_lines = pipeline.parsers.parse_a3m(f'>{self.description}\n{self.sequence}')
        new_msa_feature = make_msa_features([a3m_lines])
        
        return new_feature_dict.update(new_msa_feature)


    def mutate_msa(self, pos_res, inplace = False, paired=False, unpaired=True):
        '''Mutate specific postions of the monomer MSA

        Mutate specific postions of the monomer MSA to the indicated aminoacid or gap. This is done to explore different conformation of the protein as in SPEACH_AF [2]

        References
        ----------
        [2] `STEIN, Richard A.; MCHAOURAB, Hassane S.
            SPEACH_AF: Sampling protein ensembles and conformational heterogeneity with Alphafold2.
            PLOS Computational Biology, 2022, 18.8: e1010483.
            <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010483>`__
        '''

        if inplace:
            new_feature_dict = self.feature_dict
            self.description = rename_mutate_msa(self.description, pos_res, paired, unpaired) 
        else:
            new_feature_dict = self.feature_dict.copy()
        
        mutated_msa = self.feature_dict['msa'].copy()
        mutated_pmsa = self.feature_dict['msa_all_seq'].copy()
        
        for i, mut in pos_res.items():
            int_mut = residue_constants.HHBLITS_AA_TO_ID[mut]
            if unpaired:
                mutated_msa[:,i] = [int_mut if r != 21 else 21 for r in mutated_msa[:,i]]
                logging.info(f'Mutating MSA in position {i} to residue {mut}')
            if paired:
                mutated_pmsa[:,i] = [int_mut if r != 21 else 21 for r in mutated_pmsa[:,i]]
                logging.info(f'Mutating pMSA in position {i} to residue {mut}')
        
        new_feature_dict['msa'] = mutated_msa
        new_feature_dict['msa_all_seq'] = mutated_pmsa
        return new_feature_dict

    def alanine_scanning(self, regions, inplace = False, paired=False, unpaired=True, window = 11):
        logging.info('By using the tool "Alanine scanning" please cite:\n\
                STEIN, Richard A.; MCHAOURAB, Hassane S.\n\
                SPEACH_AF: Sampling protein ensembles and conformational heterogeneity with Alphafold2.\n\
                PLOS Computational Biology, 2022, 18.8: e1010483. doi:10.1371/journal.pcbi.1010483')
        mutants = {}
        for idx_1, idx_2 in regions:
            for i in range(idx_1, idx_2, window):
                end = min(i+window, idx_2)
                mutations = {i: 'A' for i in range(i, end+1)}
                mutants[(i, end)] = self.mutate_msa(mutations,
                                                        inplace=True,
                                                        paired=paired,
                                                        unpaired=unpaired)

        return mutants


    def remove_templates(self, inplace = False):
        '''Remove template features
        '''
        if inplace:
            new_feature_dict = self.feature_dict
            self.description = rename_remove_templates(self.description)
        else:
            new_feature_dict = self.feature_dict.copy()

        new_feature_dict = mk_mock_template(new_feature_dict)
        
        return new_feature_dict


    def remove_msa_region(self, regions, inplace = False, paired=False, unpaired=True):
        '''Remove a user specified region of the monomer MSA
        '''
        #regions should be passed as iterator like [(first_residue, last_residue)]

        #p = 'p' if paired else ''
        #u = 'u' if unpaired else ''

        if not paired and not unpaired:
           raise ValueError("User must select at least one region of the MSA to mutate")

        if inplace:
            new_feature_dict = self.feature_dict
            self.description = rename_remove_msa_region(self.description, regions, p, u)
            #self.description += f'_removed_msa_region_{p}{u}_' + '_'.join([f'{idx1+1}-{idx2+1}' for idx1, idx2 in regions])
        else:
            new_feature_dict = self.feature_dict.copy()

        for idx_1, idx_2 in regions:
            if unpaired:
                new_feature_dict['deletion_matrix_int'][:,idx_1:idx_2] = 0
                new_feature_dict['msa'][:,idx_1:idx_2] = 21
            if paired:
                new_feature_dict['deletion_matrix_int_all_seq'][:,idx_1:idx_2] = 0
                new_feature_dict['msa_all_seq'][:,idx_1:idx_2] = 21


    def scan_cluster_msa(self, min_eps, max_eps, step_eps, min_samples):
        logging.info("Performing scanning before MSA clustering")
        n_clusters=[]
        eps_test_vals=np.arange(min_eps, max_eps+step_eps, step_eps)
        n_seqs = self.feature_dict['msa'].shape[0]
        for eps in eps_test_vals:
            test_idxs = np.random.choice(n_seqs, size = int(0.25 * n_seqs), replace = False)
            testset = encode_seqs(self.feature_dict['msa'][test_idxs])
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(testset)
            n_clust = len(set(clustering.labels_))
            n_not_clustered = len(clustering.labels_[np.where(clustering.labels_==-1)])
            logging.info(f"eps: {eps}; number of clusters: {n_clust}; unclustered sequences: {n_not_clustered}")
            n_clusters.append(n_clust)
            if eps>10 and n_clust==1:
                break

        return self.cluster_msa(eps_test_vals[np.argmax(n_clusters)], min_samples)


    def cluster_msa(self, eps_val, min_samples):
        '''Perform MSA clustering and return single objects
        UNTESTED FEATURE
        '''
        logging.info('By using the tool "MSA cluster" please cite:\n\
                Wayment-Steele H. K., et al.\n\
                Predicting multiple conformations via sequence clustering and AlphaFold2.\n\
                Nature. 2024; 625: 832-839. doi:10.1038/s41586-023-06832-9')
        
        ohe_seqs = encode_seqs(self.feature_dict['msa'])
        logging.info(f"Performing MSA clustering using eps {eps_val} and min_samples {min_samples}")
        clustering = DBSCAN(eps=eps_val, min_samples=min_samples).fit(ohe_seqs)

        self.msa_cluster_labels = clustering.labels_
        cluster_feature_dict = {}
        for clst in range(max(clustering.labels_)+1):
            out_dict = self.feature_dict.copy()
            num_alignments = np.count_nonzero(clustering.labels_ == clst)
            out_dict.update({'msa': np.concatenate([[self.feature_dict['msa'][0]],
                                                    self.feature_dict['msa'][clustering.labels_ == clst]]),
                            'deletion_matrix_int': np.concatenate([[self.feature_dict['deletion_matrix_int'][0]],
                                                                    self.feature_dict['deletion_matrix_int'][clustering.labels_ == clst]]),
                            'num_alignments': np.array([num_alignments for _ in self.feature_dict['num_alignments']])})
            cluster_feature_dict[clst] = out_dict

        if -1 in clustering.labels_:
            unclustered_seqs = self.feature_dict['msa'][clustering.labels_ == -1]
            #avg_dist_to_query = np.mean([1-levenshtein(to_string_seq(x),self.feature_dict['sequence'][0].decode("utf-8"))/L for x in unclustered_seqs])
            #logging.info(f"DBSCAN generated {len(unclustered_seqs)} unclustered sequences, with average distance from the query {avg_dist_to_query}")
            logging.info(f"DBSCAN generated {len(unclustered_seqs)} unclustered sequences")

        return cluster_feature_dict


    def plot_msa_proj(self, method = 'PCA'):
        '''Plot a 2D projection of the MSA based on distances between the sequences.
        The plot is colored based on the clustering of the MSA.
        '''
        proj_methods = {'PCA': self.get_msa_pca, 'TSNE': self.get_msa_tsne}
        embedding, query_embedding = proj_methods.get(method, self.get_msa_pca)()

        if hasattr(self, 'msa_cluster_labels'):
            labels = self.msa_cluster_labels
        else:
            labels = np.zeros(embedding.shape[0])

        if method == 'TSNE':
            ax_labels = ('', '')
        else:
            ax_labels = ('PCA 1', 'PCA 2')

        plot_msa_landscape(embedding[:,0], embedding[:,1],
                            query_embedding[:,0], query_embedding[:,1],
                            labels, ax_labels)


    def get_msa_pca(self):
        '''Get MSA projection in PCA
        '''
        ohe_seqs = encode_seqs(self.feature_dict['msa'])
        mdl = PCA(n_components=2)
        embedding = mdl.fit_transform(ohe_seqs)
        query_embedding = mdl.transform(residue_constants.sequence_to_onehot(
          sequence=self.feature_dict['sequence'][0].decode("utf-8"),
          mapping=residue_constants.HHBLITS_AA_TO_ID,
          map_unknown_to_x=True).reshape((1,-1)))

        return embedding, query_embedding


    def get_msa_tsne(self):
        '''Get MSA projection in TSNE
        '''
        ohe_seqs = encode_seqs(self.feature_dict['msa'])
        ohe_query = residue_constants.sequence_to_onehot(
                      sequence=self.feature_dict['sequence'][0].decode("utf-8"),
                      mapping=residue_constants.HHBLITS_AA_TO_ID,
                      map_unknown_to_x=True).reshape((1,-1))
        ohe_seqs_query = np.concatenate([ohe_query, ohe_seqs])
        mdl = TSNE(n_components=2)
        embedding = mdl.fit_transform(ohe_seqs_query)

        return embedding[1:], embedding[0].reshape((1,-1))


    def shuffle_templates(self, seed = 0, inplace = False):
        '''Shuffle tempaltes
        '''
        if inplace:
            new_feature_dict = self.feature_dict
            self.description = rename_shuffle_templates(self.description, seed)
        else:
            new_feature_dict = self.feature_dict.copy()

        logging.info('By using the "Templat shuffle" tool please cite:\n\
            SALA Davide; HILDEBRAND Peter W.; and MEILER Jens\n\
            Biasing AlphaFold2 to predict GPCRs and kinases with user-defined functional or structural properties.\n\
            Frontiers Molecular Biosciences 10:1121962. doi: 10.3389/fmolb.2023.1121962')        
        idxs = np.arange(new_feature_dict["template_sequence"].shape[0])
        rng = np.random.default_rng(seed = seed)
        rng.shuffle(idxs)
        idxs = idxs[:MAX_TEMPLATES]
        new_feature_dict["template_all_atom_positions"] = new_feature_dict["template_all_atom_positions"][idxs]
        new_feature_dict["template_all_atom_masks"] = new_feature_dict["template_all_atom_masks"][idxs]
        new_feature_dict["template_sequence"] = new_feature_dict["template_sequence"][idxs]
        new_feature_dict["template_aatype"] = new_feature_dict["template_aatype"][idxs]
        new_feature_dict["template_domain_names"] = new_feature_dict["template_domain_names"][idxs]
        new_feature_dict["template_sum_probs"] = new_feature_dict["template_sum_probs"][idxs]

        logging.info(f"New templates: {' '.join([n.decode('utf-8') for n in new_feature_dict['template_domain_names']])}")

        return new_feature_dict

    def filter_templates(self, query, query_name = 'query_filter', inplace = False):
        if inplace:
            new_feature_dict = self.feature_dict
            self.description += f'_{query_name}'
        else:
            new_feature_dict = self.feature_dict.copy()

        return filter_template_features(new_feature_dict, query)


class MonomericObjectMmseqs2(MonomericObject):
    def __init__(self, description, sequence) -> None:
        super().__init__(description, sequence)
        self.template_featuriser_mmseqs2 = None
        self.paired_msa = False
        pass

    def make_template_features(
        self,
        template_featuriser: None = None, #kept for consistency
        filter_t: dict = None,
        inplace: bool = False) -> None:
        '''Generates Template features when using mmseqs2

        Parameters
        ----------
        query: dict, default=None
            Query for the templates
        inplace: bool, default=False

        Raise
        -----
        Exception
            if the function is used without having previously generated mmseqs2 features

        '''
        if self.template_featuriser_mmseqs2 is None:
            raise Exception('No mmseqs2 data found')
        super().make_template_features(self.template_featuriser_mmseqs2,
            filter_t=filter_t, inplace = inplace)

    def make_features(
        self,
        DEFAULT_API_SERVER: str,
        output_dir: str ='',
        templates_path: str ='mmseqs2',
        filter_t : dict = None,
        max_template_date : str ='2100-01-01', 
        msa_mode : str = "MMseqs2 (UniRef+Environmental)",
        template_mmcif_dir : str = None,
        pdb70_database_path : str = None) -> None:
        '''Function to use the mmseqs2 server for MSA generation

        Parameters
        ----------
        DEFAULT_API_SERVER: str
            mmseqs2 server url
        output_dir: str
            Directory where to save the mmseqs2 search results
        templates_path: str, default:'mmseqs2'
            Location of the templates, it can be mmseqs2, uses the server, local, it uses the AlphaFold deafult template database, a path to a custom database of templates, None or False do not use templates
        query: dict
        max_template_date: str, default='2100-01-01'
            Maximum date for the used templates, the date must be in the 
            YYYY-MM-DD format
        msa_mode: str
            Method used for MSA calculation
        template_mmcif_dir: str
            Directory containing the template mmcif files
        pdb70_database_path: str
            File containing the pdb70 database
        '''

        use_templates = True if templates_path else False
        if templates_path == 'mmseqs2':
            mmcif_path = os.path.join(output_dir, f'{self.description}_all','templates_101')
            pdb70_path = os.path.join(output_dir, f'{self.description}_all','templates_101', 'pdb70')
        elif templates_path == 'local' and template_mmcif_dir and pdb70_database_path:
            mmcif_path = template_mmcif_dir
            pdb70_path = pdb70_database_path
        elif os.path.isdir(templates_path):
            mmcif_path = templates_path
            pdb70_path = os.path.join(templates_path, 'pdb70')
        else:
            logging.warning(f'Unrecognized template mode or non-existing folder for {templates_path}, tempalte search set to no templates')
            use_templates = False

        logging.info("You chose to calculate MSA with mmseq2")
        keep_existing_results=True
        result_dir = output_dir
        result_zip = os.path.join(result_dir,self.description,".result.zip")
        if keep_existing_results and plPath(result_zip).is_file():
            logging.info(f"Skipping {self.description} (result.zip)")

        input_path = os.path.join(result_dir,self.description+'.a3m')
        if os.path.isfile(input_path):
            logging.info(f"Found precomputed a3m at {input_path}")
            a3m_lines = [plPath(input_path).read_text()]
            (
                    unpaired_msa,
                    paired_msa,
                    query_seqs_unique,
                    query_seqs_cardinality,
                    template_features,
                ) = unserialize_msa(a3m_lines, self.sequence)
        
        else:
            (
                unpaired_msa,
                paired_msa,
                query_seqs_unique,
                query_seqs_cardinality,
                template_features,
            ) = get_msa_and_templates(
                jobname=self.description,
                query_sequences=self.sequence,
                a3m_lines=None,
                result_dir=plPath(result_dir),
                msa_mode=msa_mode,
                use_templates=templates_path == 'mmseqs2',
                custom_template_path=None,
                pair_mode="paired",
                host_url=DEFAULT_API_SERVER,
                user_agent=USER_AGENT,
            )
            msa = msa_to_str(
                unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality
            )
            plPath(os.path.join(result_dir,self.description + ".a3m")).write_text(msa)
            a3m_lines=[plPath(os.path.join(result_dir,self.description + ".a3m")).read_text()]

        self.feature_dict = build_monomer_feature(self.sequence,unpaired_msa[0],template_features[0])

        if use_templates:
            template_searcher = hhsearch.HHSearch(
                binary_path="hhsearch",
                databases=[pdb70_path]
                )
            self.template_featuriser_mmseqs2 = templates.HhsearchHitFeaturizer(
                mmcif_dir=mmcif_path,
                max_template_date=max_template_date,
                max_hits=20,
                kalign_binary_path="kalign",
                release_dates_path=None,
                obsolete_pdbs_path=None,
                )

            if self.template_hits is None:
                hhsearch_result = template_searcher.query(a3m_lines[0])
                self.template_hits = pipeline.parsers.parse_hhr(hhsearch_result)
            self.make_template_features(filter_t=filter_t, inplace = True)
            

        # update feature_dict with
        valid_feats = msa_pairing.MSA_FEATURES + (
            "msa_species_identifiers",
            "msa_uniprot_accession_identifiers",
        )
        feats = {
            f"{k}_all_seq": v for k, v in self.feature_dict.items() if k in valid_feats
        }
        self.feature_dict.update(feats)


class ChoppedObject(MonomericObject):
    """chopped monomeric objects
    
    Parameters
    ----------
    description: str
        description of the original monomer object
    sequence: str
        amino acids sequence of the original monomer
    feature_dict: dict
        Calculated features of the original monomer
    regions: list
        Regions of the monomer to be kept in the new chopped monomer

    Attributes:
    feature_dict: dict
    regions: list
    new_sequence: str
        amino acids sequence of the new chopped monomer
    new_feature_dict: dict
    description: str
        Description of the original object with added:
        '_chopped' and the selected regions i to j as '_i-j'



    """

    def __init__(
        self,
        description,
        sequence,
        feature_dict,
        regions
    ) -> None:
        super().__init__(description, sequence)
        self.feature_dict = feature_dict
        self.regions = regions
        self.new_sequence = ""
        self.new_feature_dict = dict()
        self.description = rename_chopped(description, regions)

    def prepare_new_msa_feature(
        self,
        msa_feature,
        start_point,
        end_point
    ):
        """
        prepare msa features

        Parameters
        ----------
        msa_feature:
        start_point: int
            First residue of the chopped object
        end_point: int
            Last residue of the chpped object

        Return
        ------
        dict, str: chopped msa features, chopped input sequence
        """
        start_point = start_point - 1
        length = end_point - start_point
        new_seq_length = np.array([length] * length)
        new_aa_type = msa_feature["aatype"][start_point:end_point, :]
        new_between_segment_residue = msa_feature["between_segment_residues"][
            start_point:end_point
        ]
        new_domain_name = msa_feature["domain_name"]
        new_residue_index = msa_feature["residue_index"][start_point:end_point]
        new_sequence = np.array([msa_feature["sequence"][0][start_point:end_point]])
        new_deletion_mtx = msa_feature["deletion_matrix_int"][:, start_point:end_point]
        new_deletion_mtx_all_seq = msa_feature["deletion_matrix_int_all_seq"][
            :, start_point:end_point
        ]
        new_msa = msa_feature["msa"][:, start_point:end_point]
        new_msa_all_seq = msa_feature["msa_all_seq"][:, start_point:end_point]
        new_num_alignments = np.array([msa_feature["msa"].shape[0]] * length)
        new_uniprot_species = msa_feature["msa_species_identifiers"]
        new_uniprot_species_all_seq = msa_feature["msa_species_identifiers_all_seq"]

        new_msa_feature = {
            "aatype": new_aa_type,
            "between_segment_residues": new_between_segment_residue,
            "domain_name": new_domain_name,
            "residue_index": new_residue_index,
            "seq_length": new_seq_length,
            "sequence": new_sequence,
            "deletion_matrix_int": new_deletion_mtx,
            "msa": new_msa,
            "num_alignments": new_num_alignments,
            "msa_species_identifiers": new_uniprot_species,
            "msa_all_seq": new_msa_all_seq,
            "deletion_matrix_int_all_seq": new_deletion_mtx_all_seq,
            "msa_species_identifiers_all_seq": new_uniprot_species_all_seq,
        }

        return new_msa_feature, new_sequence[0].decode("utf-8")

    def prepare_new_template_feature_dict(
        self,
        template_feature,
        start_point,
        end_point
    ):
        """
        prepare template  features

        Parameters
        ----------
        template_features: FeatrueDict
            Features of the original monomer
        start_point: int
            First residue of the new protein
        end_point: int
            Last residue of the new protein
        Return
        ------
        Features of the new object: FeatureDict
        """
        start_point = start_point - 1
        new_template_aatype = template_feature["template_aatype"][
            :, start_point:end_point, :
        ]
        new_template_all_atom_masks = template_feature["template_all_atom_masks"][
            :, start_point:end_point, :
        ]
        new_template_all_atom_positions = template_feature[
            "template_all_atom_positions"
        ][:, start_point:end_point, :, :]
        new_template_domain_names = template_feature["template_domain_names"]
        new_template_sequence = template_feature["template_sequence"]
        new_template_sum_probs = template_feature["template_sum_probs"]

        new_template_feature = {
            "template_aatype": new_template_aatype,
            "template_all_atom_masks": new_template_all_atom_masks,
            "template_all_atom_positions": new_template_all_atom_positions,
            "template_domain_names": new_template_domain_names,
            "template_sequence": new_template_sequence,
            "template_sum_probs": new_template_sum_probs,
        }
        return new_template_feature

    def prepare_individual_sliced_feature_dict(
        self,
        feature_dict,
        start_point,
        end_point
    ) -> FeatureDict:
        """
        combine prepare_new_template_feature_dict and prepare_new_template_feature_dict

        Parameters
        ----------
        template_features: FeatrueDict
            Features of the original monomer
        start_point: int
            First residue of the new protein
        end_point: int
            Last residue of the new protein

        Return
        ------
        Features of the new object: FeatureDict

        """
        new_msa_feature, new_sequence = self.prepare_new_msa_feature(
            feature_dict, start_point, end_point
        )
        sliced_feature_dict = {
            **self.prepare_new_template_feature_dict(
                feature_dict, start_point, end_point
            ),
            **new_msa_feature,
        }
        self.new_sequence = self.new_sequence + new_sequence
        return sliced_feature_dict

    def concatenate_sliced_feature_dict(
        self,
        feature_dicts: list
    ) -> FeatureDict:
        """concatenate different slices in one single monomer
        Parameters
        ----------
        feature_dicts: list of FeatureDict
            Features of the different slices
        Return
        Feature dictionary of the entire new monomer
        ------
        """
        fixed_features = {"template_domain_names", "template_sequence", "template_sum_probs", 'domain_name'}
        output_dict = feature_dicts[0]
        new_sequence_length = feature_dicts[0]["seq_length"][0]
        num_alignment = feature_dicts[0]["num_alignments"][0]
        for sub_dict in feature_dicts[1:]:
            new_sequence_length += sub_dict["seq_length"][0]
            for k in feature_dicts[0].keys():
                if k in fixed_features:
                    continue
                if sub_dict[k].ndim > 1:
                    if k == "aatype":
                        output_dict[k] = np.concatenate(
                            (output_dict[k], sub_dict[k]), axis=0
                        )
                    elif "msa_species_identifiers" in k:
                        pass
                    else:
                        output_dict[k] = np.concatenate(
                            (output_dict[k], sub_dict[k]), axis=1
                        )
                elif sub_dict[k].ndim == 1:
                    if "msa_species_identifiers" in k:
                        pass
                    else:
                        output_dict[k] = np.concatenate(
                            (output_dict[k], sub_dict[k]), axis=0
                        )

        update_dict = {
            "seq_length": np.array([len(self.new_sequence)] * len(self.new_sequence)),
            "num_alignments": np.array([num_alignment] * len(self.new_sequence)),
            "sequence": np.array([self.new_sequence.encode()]),
        }
        output_dict.update(update_dict)
        return output_dict

    def prepare_final_sliced_feature_dict(self):
        """prepare final features for the corresponding region
        Parameters
        ----------
        Return
        ------
        None
        """
        if len(self.regions) == 1:
            start_point = self.regions[0][0]
            end_point = self.regions[0][1]
            self.new_feature_dict = self.prepare_individual_sliced_feature_dict(
                self.feature_dict, start_point, end_point
            )
            self.sequence = self.new_sequence
            self.feature_dict = self.new_feature_dict
            self.new_feature_dict = dict()
        elif len(self.regions) > 1:
            temp_feature_dicts = []
            for sub_region in self.regions:
                start_point = sub_region[0]
                end_point = sub_region[1]
                curr_feature_dict = self.prepare_individual_sliced_feature_dict(
                    self.feature_dict, start_point, end_point
                )
                temp_feature_dicts.append(curr_feature_dict)
            self.sequence = self.new_sequence
            self.new_feature_dict = self.concatenate_sliced_feature_dict(
                temp_feature_dicts
            )
            self.feature_dict = self.new_feature_dict
            self.new_feature_dict = dict()
            del temp_feature_dicts


class MultimericObject:
    """
    multimeric objects with features for multimer structure prediction
    AF uses paired MSA information for multimer calculations.
    For homooligomers the paired MSA is generated autoamtically.
    For heteromers the paired MSA is generated using the results from the search in unclustered Uniprot.
    In case such data is not available in the monomers, because the Uniprot search was skipped or becauses
    the sequence data were obtained from the mmseqs2 webserver, the object uses the mmseqs2 server to 
    automatically generates the paired MSA, this beahviour can be eliminated by specifing to not use paired MSA.


    Parameters
    ----------
    interactors: list of MonomericObject
        Monomers to combine to form the final multimer
    pair_msa: bool, default = True
        Generate paired MSA

    Attributes
    ----------
    description: str
        Description of the multimer obtained by concatenating the descriptions of the monomers conntected by _and_.
        In case of a *N* homooligomer the description is the same as the monomer followed by _homo_*N*er.
    sequence: list
        Sequences of the monomers
    interactors: list of MonomericObject
        Individual monomers forming the final multimer
    pair_msa: bool
        The multimer presents paired MSA features in its feature dictionary
    chain_id_map: dict
        Identifies each chain to its specific sequence
    input_seqs: list
        Input sequences
    res_indexes: list
        Indexes of the first and last residues of each monomer

    """

    def __init__(self, interactors: list, pair_msa: bool = True) -> None:
        self.description = ""
        self.sequence = []
        self.interactors = interactors
        self.pair_msa = pair_msa
        self.chain_id_map = dict()
        self.input_seqs = []
        self.get_all_residue_index()
        self.create_output_name()

        if pair_msa and not all([inter.paired_msa for inter in interactors]):
            if all([isinstance(inter, MonomericObjectMmseqs2) for inter in interactors]):
                self.mmseqs2_pair_msa()
            else:
                missing_data = []
                for inter in interactors:
                    if not inter.paired_msa:
                        missing_data.append(inter.description)
                raise AttributeError(f"Missing paired MSA information for: {' '.join(missing_data)}")

        self.create_all_chain_features()
        pass

    def get_all_residue_index(self):
        """get all residue indexes from subunits
        Parameters
        ----------
        Return
        ------
        None
        """
        self.res_indexes=[]
        for i in self.interactors:
            curr_res_idx = i.feature_dict['residue_index']
            self.res_indexes.append([curr_res_idx[0],curr_res_idx[-1]])

    def create_output_name(self):
        """a method to create output name
        Parameters
        ----------
        Return
        ------
        None
        """
        for i in range(len(self.interactors)):
            if i == 0:
                self.description += f"{self.interactors[i].description}"
            else:
                self.description += f"_and_{self.interactors[i].description}"
            self.sequence.append(f"{self.interactors[i].sequence}")

    def create_chain_id_map(self):
        """a method to create chain id
        Parameters
        ----------
        Return
        ------
        None
        """
        multimer_sequence_str = ""
        for interactor in self.interactors:
            multimer_sequence_str = (
                multimer_sequence_str
                + f">{interactor.description}\n{interactor.sequence}\n"
            )
        self.input_seqs, input_descs = parsers.parse_fasta(multimer_sequence_str)
        self.chain_id_map = pipeline_multimer._make_chain_id_map(
            sequences=self.input_seqs, descriptions=input_descs
        )

    def remove_all_seq_features(self, np_chains_list):
        new_chain_list = []
        for chain in np_chains_list:
            for feature in chain.keys():
                if feature.endswith('all_seq'):
                    chain.update({feature: chain[feature][[0]]})

            new_chain_list.append(chain)

        return new_chain_list


    def pair_and_merge(self, all_chain_features):
        """merge all chain features
        Parameters
        ----------
        all_chain_features:
            Chain features of each monomer
        Return
            multimer features: FeatureDict
        ------
        """
        feature_processing.process_unmerged_features(all_chain_features)
        np_chains_list = list(all_chain_features.values())
        if not self.pair_msa:
            np_chains_list = self.remove_all_seq_features(np_chains_list)
        pair_msa_sequences = not feature_processing._is_homomer_or_monomer(np_chains_list)
        if pair_msa_sequences:
            np_chains_list = msa_pairing.create_paired_features(chains=np_chains_list)
            np_chains_list = msa_pairing.deduplicate_unpaired_sequences(np_chains_list)
            
        np_chains_list = feature_processing.crop_chains(
            np_chains_list,
            msa_crop_size=MSA_CROP_SIZE,
            pair_msa_sequences=pair_msa_sequences,
            max_templates=MAX_TEMPLATES,
            )
        np_example = msa_pairing.merge_chain_features(
            np_chains_list=np_chains_list,
            pair_msa_sequences=pair_msa_sequences,
            max_templates=MAX_TEMPLATES,
        )
        np_example = feature_processing.process_final(np_example)
        return np_example

    def create_all_chain_features(self):
        """
        concatenate and create all chain features

        Parameters
        ----------
        Return
        ------

        Args
        uniprot_runner: a jackhammer runner with path to the uniprot database
        msa_pairing: boolean pairs msas or not
        """
        self.create_chain_id_map()
        all_chain_features = {}
        sequence_features = {}
        count = 0  # keep the index of input_seqs
        for chain_id, fasta_chain in self.chain_id_map.items():
            chain_features = self.interactors[count].feature_dict
            chain_features = pipeline_multimer.convert_monomer_features(
                chain_features, chain_id
            )
            all_chain_features[chain_id] = chain_features
            sequence_features[fasta_chain.sequence] = chain_features
            count += 1
        self.all_chain_features = pipeline_multimer.add_assembly_features(
            all_chain_features
        )
        self.feature_dict = self.pair_and_merge(
            all_chain_features=self.all_chain_features
        )
        self.feature_dict = pipeline_multimer.pad_msa(self.feature_dict, 512)

    def mmseqs2_pair_msa(self):
        '''function calling the mmseqs2 webserver to retrive the multimer paired MSA

        The function updates the interactors feature dictionaries with the paired MSA 
        features obtained from the search.

        Parameters
        ----------
        Return
        ------
        None
        '''
        logging.info("MSA calculation with mmseq2")
        msa_mode = "mmseqs2_uniref"
        result_dir = 'mmseqs2_results'
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        input_path=os.path.join(result_dir,self.description+'_','pair.a3m')
        if os.path.isfile(input_path):
            logging.info(f"Found precomputed a3m at {input_path}")            
            paired_msa = [plPath(input_path).read_text()]
            query_seqs_unique = []
            for seq in self.sequences:
                if seq not in query_seqs_unique:
                    query_seqs_unique.append(seq)
        else:
            (unpaired_msa,
            paired_msa,
            query_seqs_unique,
            query_seqs_cardinality,
            template_features,
            ) = get_msa_and_templates(
                jobname=self.description,
                query_sequences=self.sequence,
                a3m_lines=None,
                result_dir=plPath(result_dir),
                msa_mode=msa_mode,
                use_templates=False,
                custom_template_path=None,
                pair_mode="paired",
                host_url=DEFAULT_API_SERVER,
                user_agent=USER_AGENT,
            )
            #msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
            #plPath(os.path.join(result_dir,self.description + ".a3m")).write_text(msa)

        valid_feats = msa_pairing.MSA_FEATURES + (
            "msa_species_identifiers",
            "msa_uniprot_accession_identifiers",
        )

        for p_msa, seq in zip(paired_msa, query_seqs_unique):
            for interactor in self.interactors:
                if interactor.sequence == seq:
                    parsed_paired_msa = pipeline.parsers.parse_a3m(p_msa)
                    all_seq_features = make_msa_features([parsed_paired_msa], duplicates = False)
                    up_dict = {f'{k}_all_seq': v for k, v in all_seq_features.items()
                                if k in valid_feats}
                    interactor.feature_dict.update(up_dict)