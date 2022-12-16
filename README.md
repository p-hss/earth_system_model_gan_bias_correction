# Generative Adversarial Networks for Improving Earth System Model Precipitation


## Description
This repository contains the code for the training a cycle consistent generative adversarial network on Earth system model output data for bias correction. 



## Requirements
The dependencies are installed in a [Singularity](https://singularity-tutorial.github.io/) container that can be pulled from

```
singularity pull --arch amd64 library://phess/pytorch-stack/stack.sif:v3
```

## Data

- The W5E5 reanalysis data can be downloaded at [Link](https://data.isimip.org/10.48364/ISIMIP.342217).
- The CMIP6 data can be downloaded at [WCRP Coupled Model Intercomparison Project (Phase 6)](https://esgf-node.llnl.gov/projects/cmip6/).

## Usage

### Training:
1. Define the parameters and file paths in src/configuration.py
2. run:
```
 singularity run --nv --bind /path/to/current/directory /path/to/container/stack_v3.sif python main.py
```


### Evaluation:
To evaluate the results define parameters and paths in `src/configuration.py` and use the Jupyther notebooks:

- Evaluation of the GAN model checkpoints: `notebooks/summary-statistics.ipynb`
- Comparison of the GAN model and baselines: `notebooks/analysis-combined-results.ipynb`
- Evaluation of spectral densities: `notebooks/analysis-spectral-density.ipynb`
- Evaluation of fractals: `notebooks/analysis-fractal-dimension.ipynb`

To start Jupyter Lab run:

```
 singularity run --nv --bind /path/to/current/directory /path/to/container/stack_v3.sif jupyter-lab 
```
