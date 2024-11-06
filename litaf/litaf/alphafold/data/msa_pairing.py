from alphafold.data.msa_pairing import (_pad_templates,
                                        _merge_homomers_dense_msa,
                                        _merge_features_from_multiple_chains,
                                        _concatenate_paired_and_unpaired_features,
                                        _correct_post_merged_feats)
from typing import List
from litaf.datatypes import FeatureDict

def merge_chain_features(np_chains_list: List[FeatureDict],
                         pair_msa_sequences: bool,
                         max_templates: int,
                         pair_homomers: bool = True) -> FeatureDict:
  """Merges features for multiple chains to single FeatureDict.
  Args:
    np_chains_list: List of FeatureDicts for each chain.
    pair_msa_sequences: Whether to merge paired MSAs.
    max_templates: The maximum number of templates to include.
  Returns:
    Single FeatureDict for entire complex.
  """
  np_chains_list = _pad_templates(
      np_chains_list, max_templates=max_templates)
  if pair_homomers:
    np_chains_list = _merge_homomers_dense_msa(np_chains_list)
  # Unpaired MSA features will be always block-diagonalised; paired MSA
  # features will be concatenated.
  np_example = _merge_features_from_multiple_chains(
      np_chains_list, pair_msa_sequences=False)
  if pair_msa_sequences:
    np_example = _concatenate_paired_and_unpaired_features(np_example)
  np_example = _correct_post_merged_feats(
      np_example=np_example,
      np_chains_list=np_chains_list,
      pair_msa_sequences=pair_msa_sequences)

  return np_example