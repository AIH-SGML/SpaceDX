# SpaceDX

Code for the paper: "Joint localization and testing reveals spatial context of differential expression".

## Contents
- `spacedx/`: source code of the SpaceDX model (`spacedx.py`) and helpers for data preparation (`utils.py`)
- `notebooks/tutorial.ipynb`: demo notebook that runs SpaceDX on a single gene
- `data/demo_subset.h5ad`: small demo dataset (~9 MB), a subset of the preprocessed Alzheimer's disease data from Miyoshi et al., 2024 (https://doi.org/10.1038/s41588-024-01961-x), containing all spots but only the demo gene *Arrb2*
- `environment.yaml` / `setup.py`: environment and package definitions

## 1. System requirements

### Software dependencies
Install these before following the installation guide:
- conda (Miniconda, Miniforge or Anaconda) — tested with conda 26.7.1

All Python dependencies, with version numbers (Python 3.11, PyTorch 2.1.0, scanpy, anndata, etc.), are installed automatically from `environment.yaml` in the steps below.

### Operating systems
- macOS 26.5.2 
- Rocky Linux 10.2 

### Hardware requirements
- No non-standard hardware is required

## 2. Installation guide

### Instructions
```sh
conda env create --file ./environment.yaml
conda activate spacedx
pip install -e .
```

### Typical install time
About 3 minutes on a standard desktop computer.

## 3. Demo

### Instructions to run on data
The demo dataset is provided at `data/demo_subset.h5ad`.

```sh
conda activate spacedx
cd notebooks
jupyter lab tutorial.ipynb
```
Run all cells.

### Expected output
- `spacedx.iw_elbo`: test statistic (IW-ELBO) ≈ -22.82
- `spacedx.beta_posterior_m`: posterior mean of the effect size (beta) ≈ 1.63
- A spatial plot of the attention weights: 2 rows (one per group in `obs["group"]`) × 10 tissue sections each, with spots coloured by attention weight.

### Expected run time
About 30 seconds for the whole notebook.

## 4. Instructions for use

### How to run the software on your data
SpaceDX expects an `AnnData` object containing:
- `obs["sample"]`: sample (tissue section) annotations
- `obs["group"]`: integer group/condition labels
- `obs["biological_replicate"]`: biological replicate annotations
- `obsm[<latent>]`: latent embedding 
- `obsm["spatial"]`: spatial coordinates
- `uns["spatial"][<sample>]["scalefactors"]["spot_diameter_fullres"]`: spot diameter per sample

Minimal usage:
```python
import scanpy as sc
from spacedx import SpaceDX
from spacedx.utils import prepare_general_model_input, prepare_gene_input

ad = sc.read("your_data.h5ad")
y, F, Ls, R = prepare_general_model_input(ad, latent="your_latent_key")
xs = prepare_gene_input(ad, "YourGene")

model = SpaceDX(y, F, xs, Ls, R=R, n_samples=128)
model.optimize()

model.iw_elbo                 # test statistic
model.beta_posterior_m        # effect size
attention_weights, _ = model.get_attention_weights()
```

## License
This project is licensed under the terms of the MIT License. See the `LICENSE.txt` file for details.
