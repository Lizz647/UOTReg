# UOTReg

## Introduction

UOTReg is a trajectory inference method developed for temporal single-cell RNA-seq datasets. It comprises two key steps: (1) "Denoising" the distribution by estimating the expected distribution at time $t$ using Robust Local Fréchet Regression. (2) Inferring cellular trajectories by composing unbalanced optimal transport maps between consecutive time points. UOTReg outperforms existing methods, as demonstrated through both simulated and real data analyses.

## Installations

### Requirements

```bash
# ---- UOTReg core runtime requirements ----
# Python 3.10+ recommended
numpy>=1.24,<3
pandas>=2.0
scipy>=1.10
torch>=2.1,<2.4
scikit-learn>=1.3
tqdm>=4.66
matplotlib>=3.7
```

```bash
# ---- Additional packages for running the example notebooks/tutorials ----
-r requirements.txt     # include all core requirements above

scanpy>=1.9
anndata>=0.9
h5py>=3.10
umap-learn>=0.5
IPython>=8.0            # for display.clear_output
seaborn>=0.12
jupyter
ipykernel
```

### Install from GitHub

``````bash
# It's best to create a new conda environment to run the files
conda create -n uotreg-test python=3.10 -y
conda activate uotreg-test
# --- --- --- --- --- --- --- --- ---
git clone https://github.com/Lizz647/UOTReg.git
cd UOTReg # enter the downloaded folder
pip install -r requirements.txt   
pip install -r requirements_notebooks.txt
``````

## Usage

We provide a python notebook in the `tutorial` folder to guide you through the basic procedures of UOTReg using the Embryoid dataset, including:

- Estimating the expected distribution at time $t$ using Robust Local Fréchet Regression & visualizations.
- Inferring cellular trajectories from the estimated distributions & Visualization.

## Reproducibility

We have provided all necessary code in the `notebooks` folder to reproduce the figures and tables presented in the paper. 

Most notebooks demonstrate how to train the model from scratch. However, we also provide pre-trained models in the `results` folder, allowing you to reproduce the figures without training. Ensure the working directory is correctly set whenever saving or loading files.

The **Statefate** dataset is relatively large and therefore not included in this GitHub repository.
 To run the code that depends on this dataset, please download the folder **`scrna-statefate`** from our [Google Drive Link](https://drive.google.com/drive/folders/1tRTRKVKMqsjlB5PJU2l12YbtH5tXuDbs?usp=drive_link), and place it under **`data/timedata/`** (alongside the existing **`embryoid`** folder).

Specifically:

- `notebooks/simulations/outliers/`
  - `simu_pretrain_10.ipynb`: This notebook demonstrates how to reproduce the results shown in **Figure 1** (outlier simulation).

- `notebooks/simulations/batcheffect/`
  - `Simu_noise.ipynb`: How to generate simulated data with batch-effect-like noise (**Figure 2**) and estimate the underlying distribution using UOTReg.
  - `mioflow_noise_cc.ipynb`: How to infer trajectories using Mioflow in the simulation (**Figure 2**).
    - To run Mioflow correctly, refer to its GitHub repository to download the required environment.
  - `Learning-cell-trajectories.ipynb`: How to learn UOTReg and WOT-style trajectories, and visualize results for three methods (**Figure 2**).  
  
- `notebooks/realdata_analysis/benchmark/`
  - `benchmark_embryoid.ipynb` & `benchmark_embryoid.ipynb`: How to reproduce the leave-one-out benchmark experiment on the two datasets (**Table 1, Figure 3, Figure S2**).

- `notebooks/realdata_analysis/`
  - `embryoid_dist_est.ipynb` and `statefate_dist_est.ipynb`: How to reproduce **Figure S3** and **Figure S4** (estimated distributions for the two datasets).
  - `embryoid_visualization_trajs.ipynb` and `statefate_visualization_trajs.ipynb`: How to reproduce **Figure 4 (top panel)** and **Figure 6 (top panel)** (trajectory comparisons).
  - `embryoid_newanalysis.ipynb` and `statefate_newanalysis.ipynb`: How to reproduce **Figure 4 (bottom panel)**, **Figure 6 (bottom panel)**, and **Figure S5** (Clustering); How to reproduce **Figure 5, Figure S6, S7, S8, and S9** (Dynamic gene experissons).
  - `R` files: How we fit and save the LMM models in Figure 5, Figure S6, S7, S8, and S9.

---

**Datasets:**

- Embryoid: [Moon, Kevin R., *et al.* (2019)]((https://doi.org/10.1038/s41587-019-0336-3))
- Statefate: [Weinreb, Caleb, *et al.* (2020):](https://doi.org/10.1126/science.aaw3381)
  - Preprocessing of the **Statefate** dataset follows the approach described in [Bunne *et al.* (2023)](https://www.nature.com/articles/s41592-023-01969-x)

## Acknowledgement

## Citation

