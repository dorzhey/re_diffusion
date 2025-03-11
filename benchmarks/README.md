# Single-cell benchmarking with diffusion models

Scripts to run predictive and generative benchmarks on competing diffusion and supervised models on multiple single-cell datasets

## Key components:
- generative benchmarks for de-novo sequences
    - motif-level statistics: motif presentation within sets of endogenous and generated sequences
    - Cell-level statistics: cell similarity involving differential motif enrichment, TF expression

## Files
- `dirichlet_manova.ipynb`: run manova to compare the motif generating capacity of the label guided model to endogenous motifs in the training data.

