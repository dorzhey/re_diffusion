# Workplace for diffusion regulatory element design models

This repository contains code related to running diffusion models for regulatory element design on single cell data. This is a show repo of my work, original is private and belongs to mkarikomi.

Currently, the repo has 4 different models

All models reference and import original repos in the folder re_design located in the root folder of this repo

For reproduction, clone this repo and create re_design folder with the repo of interest.

The links and versions of original repos you can find below

## Directories
- [`benchmarks`](benchmarks/) ([readme](benchmarks/README.md)): scripts to run predictive and generative benchmarks on competing diffusion and supervised models on multiple single-cell datasets.
- [`classifier_diffusion`](classifier_diffusion/) : scripts run Classifier guided diffusion on single-cell data.
- [`data_preprocessing`](data_preprocessing/) : notebooks and scripts to create .csv files for training on ATAC or tCRE data
- [`dirichlet_diffusion`](dirichlet_diffusion/) ([readme](dirichlet_diffusion/README.md)): scripts to run Dirichlet diffusion models on single-cell data. 
- [`dna_diffusion`](dna_diffusion/) ([readme](dna_diffusion/README.md)): scripts to run DNA diffusion on single-cell data. 
- [`train_utils`](train_utils/) : scripts to preprocess csv files for training and validation
- [`universal_guidance`](universal_guidance/) : scripts to run Universal Guidance diffusion on single-cell data. Currently under construction


## Referenced repos
- [`Classifier Guided Diffusion`](https://github.com/openai/guided-diffusion) , commit 22e0df8 on Jul 15, 2022
- [`Dirichlet Diffusion Score Model`](https://github.com/jzhoulab/ddsm) , commit 4a58839 on  May 1, 2024
- [`DNA Diffusion`](https://github.com/pinellolab/DNA-Diffusion) , commit 74dc742 on Jun 6, 2024
- [`Universal Guidance for Diffusion Models`](https://github.com/arpitbansal297/Universal-Guided-Diffusion), commit ff82f88 on Jul 11, 2023
