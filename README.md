# LIT-AlphaFold

## Description

LIT-AlphaFold is a modified version of AlphaFold developed in the [Laboratoire d'Innovation Thérapeutique](https://medchem.unistra.fr/) at the University of Strasbourg.

The module is based on [AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown) and [ColabFold](https://github.com/sokrypton/ColabFold). LIT-AlphaFold includes options to modify both the templates and the multiple sequence alignement (MSA) used by AlphaFold to predict different protein conformational states, with a focus on GPCRs.

## Pre-installation

Before it is adviced to download the models' weights and the genetic databases. To do so please follow the instructions in https://github.com/kalininalab/alphafold_non_docker.

LIT-AlphaFold does not require to download the genetic databases since it is configured to also use the *MMseqs2* for MSA generation, as in ColabFold. In case you opt out from using local genetic databases please follow the instruction as in [Tutorial n]().

It is mandatory to download the models' weights since LIT-AlphaFold has been developed to run calculations locally.

## Installation

Installation of LIT-AlphaFold requires Anaconda, if it is not present on your machine a guide to its installation can be found [here]().

The *conda* environment lit-af is created gathering the necessary dependencies:
1. Create the conda environment
```bash
(lit-af) $ conda create --name lit-af python==3.8
(lit-af) $ conda activate lit-af
```
1. Install the dependencies used by AlphaFold
```bash
(lit-af) $ conda install -y -c conda-forge openmm==7.5.1 cudatoolkit==11.2.2 pdbfixer
(lit-af) $ conda install -y -c bioconda hmmer hhsuite==3.3.0 kalign2
(lit-af) $ pip install absl-py==1.0.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.9 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.3.25 ml-collections==0.1.0 numpy==1.21.6 pandas==1.3.4 protobuf==3.20.1 scipy==1.7.0 tensorflow-cpu==2.9.0
```
1. Install the AlphaFold, ColabFold, and AlphaPulldown. AlphaPulldown is installed without its dependencies to avoid conflicts with the different versions, some modules might be missing, in case it is advised to install them manually via pip.
```bash
(lit-af) $ pip install --no-warn-conflicts "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold@v1.5.2"
(lit-af) $ pip install alphapulldown==0.30.7 --no-deps
(lit-af) $ pip install --upgrade --no-cache-dir jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

1. Move to the folder where you want to keep the *litaf* scripts and clone this repository
```bash
git clone https://github.com/LIT-CCM-lab/LIT-AlphaFold
```

## Usage

To learn how to use LIT-AlpahFold different tutorials are proposed covering different topics:


## How to reference this work ?
If you are using LIT-AF please cite:

- Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S, and Steinegger M. ColabFold: Making protein folding accessible to all. <br />
  Nature Methods (2022) doi: [10.1038/s41592-022-01488-1](https://www.nature.com/articles/s41592-022-01488-1)
- Yu D, Chojnowski G, Rosenthal M, and Kosinski J. AlphaPulldown—a python package for protein–protein interaction screens using AlphaFold-Multimer. <br />
  Bioinformatics (2023) doi: [10.1093/bioinformatics/btac749](https://academic.oup.com/bioinformatics/article/39/1/btac749/6839971)

- If you’re using **AlphaFold**, please also cite: <br />
  Jumper et al. "Highly accurate protein structure prediction with AlphaFold." <br />
  Nature (2021) doi: [10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- If you’re using **AlphaFold-multimer**, please also cite: <br />
  Evans et al. "Protein complex prediction with AlphaFold-Multimer." <br />
  biorxiv (2021) doi: [10.1101/2021.10.04.463034v1](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)
- If you are using *MMseqs2*, please also use the appropriate citation in: [MMseqs2](https://github.com/soedinglab/MMseqs2)

To cite specific methods from multistate structure prediction please use the reference in the appropriate tutorial or check the log file from the protein structure prediction.


## Roadmap
Future releases might include:
* User defined MSA for monomers
* User defined MSA for multimers
* MSA clustering