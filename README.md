# pd-astrocyte-calcium-simulation

Supplementary code for Modeling Parkinson’s Disease in Astrocytes: A Simulation Model Characterizing Calcium Wave Propagation and Degradation in Atrophied Astrocytes

Created by Risha Guha and Dr. Kazutaka Takahashi


## Overview

This repository contains notebooks and scripts for:

- generating and processing astrocyte morphology data
- preparing simulation inputs from segmented images
- running calcium dynamics simulations for healthy and Parkinson’s disease-like astrocytes
- analyzing noise characterization and simulation outputs

The simulation code uses CuPy for GPU acceleration, so a CUDA-enabled Python environment is recommended.

## Prerequisites

Before starting, make sure you have:

- a working NVIDIA GPU driver installation
- a CUDA toolkit version compatible with your GPU and CuPy build
- Conda installed (Miniconda or Anaconda)

You can verify the GPU setup with:

```bash
nvidia-smi
nvcc --version
```

## Create and activate a Conda environment

```bash
conda create -n astro-sim python=3.10 -y
conda activate astro-sim
```

## Install dependencies

Install the core scientific Python packages and CuPy.

If you are using CUDA 12.x:

```bash
conda install -c conda-forge cupy cuda-version=12.1 numpy scipy matplotlib scikit-image jupyterlab ipykernel -y
```

If you are using CUDA 11.x instead:

```bash
conda install -c conda-forge cupy cuda-version=11.8 numpy scipy matplotlib scikit-image jupyterlab ipykernel -y
```

If your local CUDA setup differs, adjust the `cuda-version` to the version supported by your installation.

## Register the environment with Jupyter

```bash
python -m ipykernel install --user --name astro-sim --display-name "Python (astro-sim)"
```

## Verify the environment

Run the following checks to confirm that CuPy can see the GPU:

```bash
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.getDeviceCount())"
```

If the device count is greater than zero, the GPU backend is available.

## Running the repository

The main content is organized into two folders:

- `Morphology - Image Creation Sample Astrocyte/`: notebooks for preparing morphology and segmentation data
- `Sample Simulations - Healthy and PD versions/`: simulation notebooks and helper scripts
- `Sample Simulations - Noise Characterization/`: noise analysis scripts and notebooks

From the repository root, launch Jupyter:

```bash
jupyter lab
```

Then open the relevant notebook in the appropriate folder.

## Notes

- Some notebooks and scripts assume that the input `.npy` files are available in the working directory or in the expected project folders.
- If you encounter CuPy installation issues, make sure your NVIDIA driver and CUDA toolkit versions are compatible with the CuPy package you install.
- For GPU-heavy runs, it is often helpful to start from the sample notebooks in the simulation folders rather than the morphology preprocessing notebooks.
