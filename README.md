# Dabie-bandavirus (DBV) Protein Structure Analysis

## Project Overview

This repository contains computational tools and datasets for the structural analysis of Dabie bandavirus (DBV) proteins. The project integrates two advanced evolutionary variant effect analysis methods, EVE and EVEscape, to predict the impact of protein sequence variations, identify functionally critical regions, and predict potential immune escape sites.

## Theoretical Background

### EVE (Evolutionary model of Variant Effects)
EVE is a deep generative model-based method that predicts the effects of amino acid mutations by learning evolutionary patterns in protein families. The method uses a Bayesian variational autoencoder (VAE) to learn the distribution of amino acid sequences from multiple sequence alignment (MSA) data, calculates evolutionary indices for mutations, and uses Gaussian Mixture Models (GMM) to classify variants as benign or pathogenic.

### EVEscape
The EVEscape model predicts the likelihood of viral protein variants inducing immune escape based on three key components:

1. **Fitness**: Uses EVE to assess the impact of mutations on protein function
2. **Accessibility**: Calculates antibody accessibility based on protein structure
3. **Dissimilarity**: Measures changes in physicochemical properties between mutant and wild-type residues

## Analysis Workflow

The project's analysis workflow has been integrated into four main steps, to be executed in the following order:

### 1. Train VAE Model (Step1_train_VAE.sh)

This Shell script trains a variational autoencoder (VAE) model to learn the evolutionary patterns of DBV protein sequences.

```bash
cd scripts
bash Step1_train_VAE.sh
```

### 2. Calculate Evolutionary Indices (Step2_compute_evol_indices_all_singles.sh)

This script calculates evolutionary indices for all possible single amino acid mutations.

```bash
cd scripts
bash Step2_compute_evol_indices_all_singles.sh
```

### 3. Process Protein Data (Step3_process_protein_data.py)

This Python script processes protein structure data, calculating accessibility and other structural features.

```bash
cd scripts
python Step3_process_protein_data.py
```

### 4. Calculate EVEscape Scores (Step4_evescape_scores.py)

In the final step, this Python script integrates the results from previous steps to calculate the final EVEscape scores.

```bash
cd scripts
python Step4_evescape_scores.py
```

## Installation and Environment Setup

This project uses Conda to manage dependencies. You can create the runtime environment using the following commands:

conda env create -f protein_env.yml
conda activate protein_env

### Main Dependencies

- python=3.7
- pytorch=1.7
- cudatoolkit=11.0
- scikit-learn=0.24.1
- numpy=1.20.1
- pandas=1.2.4
- scipy=1.6.2
- matplotlib
- seaborn

## License

This project is licensed under the MIT License. See the LICENSE file for details.
