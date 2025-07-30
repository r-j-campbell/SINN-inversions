# SINN: Stokes Inversions based on Neural Networks

This repository accompanies our study on machine learning methods for solar spectropolarimetric inversion.

We explore neural networks as fast and accurate alternatives to traditional inversion codes, with a focus on **multi-layer perceptrons (MLPs)** and **Transformer architectures**. These models are trained on synthetic Stokes profiles and accompanying models from simulation snapshots, and learn to infer stratified atmospheric parameters as a function of optical depth.

---

## 🗂 Repository Structure

- `architectures.py`:  
  Neural network architecture definitions, including the MLP and Transformer models used for training and inference.

- `MLP_training_and_inference.py`:  
  Full training pipeline and evaluation for the MLP architecture.

- `Transformer_training.py`:  
  Full training pipeline for the Transformer-based model.

- `Transformer_inference.py`:  
  Script for running full-snapshot inference using a trained Transformer model.

- `LICENSE`:  
  MIT License (open-source).

- `README.md`:  
  This file.

---

## 📊 Model Outputs

All models predict the following six atmospheric quantities as a function of optical depth:

1. Temperature  
2. Magnetic field strength  
3. Line-of-sight velocity  
4. Magnetic inclination  
5. Azimuth angle (sin(2ϕ))  
6. Azimuth angle (cos(2ϕ))

Predictions can be saved in standard FITS format for scientific analysis or comparison with traditional inversion results.

---

## ⚙️ Requirements

- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- NumPy  
- Astropy  
- Matplotlib (optional, for plotting)  

Install dependencies via:
pip install torch numpy astropy matplotlib


To run inference on a trained Transformer model:
python Transformer_inference.py

To train a model from scratch, see the training scripts:
MLP_training_and_inference.py for the MLP baseline, which also runs inference
Transformer_training.py for the Transformer model


For questions or contributions, please contact:

Dr Ryan J. Campbell
rjcampbell.research@outlook.com

If you use this codebase in your work, please cite our accompanying paper (https://ui.adsabs.harvard.edu/abs/2025arXiv250616810C/abstract).