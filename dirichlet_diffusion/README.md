# Dirichlet diffusion models for single-cell data

Scripts to run Dirichlet diffusion on single-cell data.

## Files
- `save_re_input_data.py`: filter the PBMC regulatory elements using cellranger annotations.
- `label_guided.py`: label-guided diffusion model on PBMC using Dirichlet diffusion (Avdeyev 2023).


## Usage

1. Filter the labels in 10X PBMC ATAC peaks:

    ```bash
    python dirichlet_diffusion/save_re_input_data.py
    ```

2. Train the model on PBMC guided by RNA labels (unsupervised hierarchical clustering):

    ```bash
    python dirichlet_diffusion/label_guided.py
    ```