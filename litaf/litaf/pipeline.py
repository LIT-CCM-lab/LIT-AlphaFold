'''
Modified AlpahFold code for data handling

* *make_msa_features*, generates MSA features
* *DataPipeline*, pipeline for MSA and template search

'''

import os

import numpy as np

from absl import logging

from alphafold.common import residue_constants
from alphafold.data import msa_identifiers, parsers, templates
from alphafold.data.tools import hhblits,jackhmmer
from alphafold.data.pipeline import run_msa_tool,make_sequence_features


from litaf.datatypes import FeatureDict, TemplateSearcher
from collections.abc import Sequence
from typing import Optional


# Internal import (7716).

def make_msa_features(msas: Sequence[parsers.Msa],
                      duplicates = True) -> FeatureDict:
  """
  Constructs a feature dict of MSA features.
  Modified version of the original alphafold function.
  The current version can be changed to remove or not previously seen sequences 
  to avoid duplicates.
  This is done to have compatibility with both the original alphafold and 
  colabfold implementations.
  """
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences and duplicates:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      species_ids.append(identifiers.species_id.encode('utf-8'))

  if all([s == b'' for s in species_ids]):
    species_ids = [f'{i}'.encode('utf-8') for i in range(len(species_ids)-1)]
    species_ids.insert(0, b'')

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features


class DataPipeline:
  """
  Runs the alignment tools and assembles the input features.
  Modified version of the original AlphaFold function with the same name.
  The current version does not generate template features at this step allowing 
  to filter the templates.
  """

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniref30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               use_precomputed_msas: bool = False,
               use_templates: bool = True):
    """Initializes the data pipeline."""
    self._use_small_bfd = use_small_bfd
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path)
    else:
      self.hhblits_bfd_uniref_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniref30_database_path])
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    self.template_searcher = template_searcher
    self.template_featuriser = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.use_precomputed_msas = use_precomputed_msas
    self.use_templates = use_templates
      

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    jackhmmer_uniref90_result = run_msa_tool(
        msa_runner=self.jackhmmer_uniref90_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=uniref90_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.uniref_max_hits)
    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    jackhmmer_mgnify_result = run_msa_tool(
        msa_runner=self.jackhmmer_mgnify_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=mgnify_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.mgnify_max_hits)

    msa_for_templates = jackhmmer_uniref90_result['sto']
    msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
        msa_for_templates)

    if self.template_searcher.input_format == 'sto':
      pdb_templates_result = self.template_searcher.query(msa_for_templates)
    elif self.template_searcher.input_format == 'a3m':
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
      pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
    else:
      raise ValueError('Unrecognized template input format: '
                       f'{self.template_searcher.input_format}')

    pdb_hits_out_path = os.path.join(
        msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
    with open(pdb_hits_out_path, 'w') as f:
      f.write(pdb_templates_result)

    uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
    mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])

    if self.use_templates:
      pdb_template_hits = self.template_searcher.get_template_hits(
          output_string=pdb_templates_result, input_sequence=input_sequence)
    else:
      pdb_template_hits = None

    if self._use_small_bfd:
      bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
      jackhmmer_small_bfd_result = run_msa_tool(
          msa_runner=self.jackhmmer_small_bfd_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=bfd_out_path,
          msa_format='sto',
          use_precomputed_msas=self.use_precomputed_msas)
      bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
    else:
      bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniref_hits.a3m')
      
      hhblits_bfd_uniref_result = run_msa_tool(
          msa_runner=self.hhblits_bfd_uniref_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=bfd_out_path,
          msa_format='a3m',
          use_precomputed_msas=self.use_precomputed_msas)
      bfd_msa = parsers.parse_a3m(hhblits_bfd_uniref_result['a3m'])
      

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))

    logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])

    return {**sequence_features, **msa_features}, pdb_template_hits