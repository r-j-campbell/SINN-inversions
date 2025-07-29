"""
MLP_training_and_inference.py

Train a multi-layer perceptron (MLP) to perform stratified spectropolarimetric inversion 
using synthetic Stokes profiles (produced from a forward synthesis or inversion code) and atmospheric parameters from a MANCHA simulation.

Key functionality:
- Loads MANCHA simulation data (Stokes vectors and model atmospheres) from FITS files.
- Randomly samples spatial positions to form a profile training set (user can control the number of training samples).
- Performs input and output standardisation using global statistics.
  * Inputs are flattened Stokes vectors, fed directly into the MLP.
  * Outputs include temperature, magnetic field strength, LOS velocity, and inclination
    (standardised per optical depth), plus sin(2φ) and cos(2φ) for azimuth.
- Splits into training and validation sets.
- Defines and trains an MLP using mixed precision and early stopping.
- Saves the best-performing model and associated scaling parameters.

The script is run with a job index specifying a particular hyperparameter combination. This was used in a SLURM script on QUB's HPC. This functionality may not be necessary for you.
This allows parallel execution across a hyperparameter grid.

Usage:
    python training_mlp.py --job_index <int>

Model architecture:
    The MLP implementation is defined in architectures.py, which must be in the same directory, or in your Python path.

Dependencies:
    - astropy
    - numpy
    - torch
    - sklearn

Expected input files:
    - MANCHA_ext_models.fits: Array of shape (11, N_tau, Ny, Nx)
        * 11 atmospheric parameters (e.g. T, B, v, inclination, azimuth, etc., in the same order as a SIR/DeSIRe model file)
        * N_tau: number of log(τ) depth layers
        * Ny, Nx: spatial dimensions of the snapshot

    - MANCHA_ext_profiles.fits: Array of shape (4, N_lambda, Ny, Nx)
        * 4 Stokes parameters (I, Q, U, V)
        * N_lambda: number of wavelength points
        * Ny, Nx: same spatial dimensions as models file
"""


from astropy.io import fits
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from itertools import product
from joblib import Parallel, delayed
import random
import os
from torch.cuda.amp import autocast, GradScaler
import time

from architectures import SINN_MLP

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

output_dir = '/outputs'
os.makedirs(output_dir, exist_ok=True)

snapshot_path = '/inputs/MANCHA_ext_models.fits'
stokes_path = '/inputs/MANCHA_ext_profiles.fits'

full_profiles = fits.getdata(stokes_path, memmap=False)  # (4, n_wave, ny, nx)
full_models = fits.getdata(snapshot_path, memmap=False)  # (11, n_tau, ny, nx)

ny, nx = full_profiles.shape[2], full_profiles.shape[3]
num_samples = 1000000  # or however many you want

all_indices = [(y, x) for y in range(ny) for x in range(nx)]
np.random.seed(42)
selected_indices = np.random.choice(len(all_indices), size=num_samples, replace=False)

selected_profiles = np.zeros((4, full_profiles.shape[1], num_samples), dtype=np.float32)
selected_models = np.zeros((11, full_models.shape[1], num_samples), dtype=np.float32)

for i, idx in enumerate(selected_indices):
    y, x = all_indices[idx]
    selected_profiles[:, :, i] = full_profiles[:, :, y, x]
    selected_models[:, :, i] = full_models[:, :, y, x]

profiles = selected_profiles
models = selected_models

print('data loaded')
print(profiles.shape,models.shape)
del full_profiles, full_models
del selected_profiles, selected_models  # Free up RAM

# Detect NaNs across each profile
nan_mask = np.isnan(profiles).any(axis=(0, 1))  # Shape: (N,)

num_bad_profiles = nan_mask.sum()
num_total_profiles = profiles.shape[2]

print(f"{num_bad_profiles} out of {num_total_profiles} profiles contain NaNs")

# Filter out any profiles containing NaNs
nan_mask = np.isnan(profiles).any(axis=(0, 1))  # shape: (N,)
print(f"Removing {nan_mask.sum()} profiles with NaNs")
profiles = profiles[:, :, ~nan_mask]
models = models[:, :, ~nan_mask]

# Reshape as for MLP
num_stokes, num_wavelengths, N = profiles.shape
num_params, num_optical_depths, _ = models.shape
# Determine number of outputs dynamically
num_outputs = num_optical_depths * 6
stokes_input_mlp = profiles.transpose(2, 0, 1).reshape(N, num_stokes * num_wavelengths)
print("Any NaNs before scaling?", np.isnan(stokes_input_mlp).any())

# Select output atmospheric parameters (T, B, v, incl, az)
def reshape_atmospheric_param(param_index):
    return models[param_index].T  # shape (N, num_optical_depths)

temperature_output_mlp = reshape_atmospheric_param(1)
magnetic_field_output_mlp = reshape_atmospheric_param(4)
velocity_output_mlp = reshape_atmospheric_param(5) / 100000.0  # cm/s → km/s
inclination_output_mlp = reshape_atmospheric_param(6)
azimuth_deg = reshape_atmospheric_param(7)
azimuth_rad = np.deg2rad(azimuth_deg)
sin_2azimuth = np.sin(2 * azimuth_rad)
cos_2azimuth = np.cos(2 * azimuth_rad)


print("Extracted atm parameters")

# Stack the outputs together: (4,000,000 pixels, 280 outputs)
target_output = np.stack((
    temperature_output_mlp,
    magnetic_field_output_mlp,
    velocity_output_mlp,
    inclination_output_mlp,
    sin_2azimuth,
    cos_2azimuth
), axis=-1)  # shape: (N, τ, 6)
print("stacked outputs")

# Compute global mean & std for standardization
global_mean_X = np.mean(stokes_input_mlp, axis=0)
global_std_X = np.std(stokes_input_mlp, axis=0)
T_mean, T_std = np.mean(temperature_output_mlp, axis=0), np.std(temperature_output_mlp, axis=0)
B_mean, B_std = np.mean(magnetic_field_output_mlp, axis=0), np.std(magnetic_field_output_mlp, axis=0)
V_mean, V_std = np.mean(velocity_output_mlp, axis=0), np.std(velocity_output_mlp, axis=0)
I_mean, I_std = np.mean(inclination_output_mlp, axis=0), np.std(inclination_output_mlp, axis=0)

# Prevent division by zero
global_std_X[global_std_X == 0] = 1.0
T_std[T_std == 0] = 1.0
B_std[B_std == 0] = 1.0
V_std[V_std == 0] = 1.0
I_std[I_std == 0] = 1.0

# Define custom scaling functions
def custom_standardize(data, mean, std):
    return (data - mean) / std

def custom_inverse_transform(data, mean, std):
    return (data * std) + mean

# Standardize inputs and outputs
stokes_input_scaled = custom_standardize(stokes_input_mlp, global_mean_X, global_std_X)
temperature_scaled = custom_standardize(target_output[:, :, 0], T_mean, T_std)
magnetic_scaled   = custom_standardize(target_output[:, :, 1], B_mean, B_std)
velocity_scaled    = custom_standardize(target_output[:, :, 2], V_mean, V_std)
inclination_scaled = custom_standardize(target_output[:, :, 3], I_mean, I_std)

# Rebuild scaled target
# Rebuild scaled target
target_output_scaled = np.stack((
    temperature_scaled,
    magnetic_scaled,
    velocity_scaled,
    inclination_scaled,
    target_output[:, :, 4],  # sin(2φ)
    target_output[:, :, 5],  # cos(2φ)
), axis=-1)
print("standardisation complete")

assert not np.isnan(stokes_input_scaled).any(), "Input contains NaNs!"
assert not np.isnan(target_output_scaled).any(), "Output contains NaNs!"


# Convert to PyTorch tensors
X_train, X_val, y_train, y_val = train_test_split(
    stokes_input_scaled, target_output_scaled.reshape(N, num_outputs), test_size=0.15, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


print("tensors created")
del stokes_input_mlp, temperature_output_mlp, magnetic_field_output_mlp
del velocity_output_mlp, inclination_output_mlp, azimuth_deg, azimuth_rad
del sin_2azimuth, cos_2azimuth
del profiles, models, target_output, target_output_scaled

torch.cuda.empty_cache()

# Parallel training function
def train_model(job_id, lr, batch_size, hidden_dim, num_layers, input_dim, output_dim):
    job_start = time.perf_counter()
    print(f'Starting Job {job_id} with lr={lr}, batch_size={batch_size}, hidden_dim={hidden_dim}')

    # Initialize model
    model = SINN_MLP(input_dim, hidden_dim, output_dim, num_layers).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, pin_memory=True)

    # Early stopping parameters
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    patience = 4
    max_epochs = 35
    train_losses = []

    for epoch in range(max_epochs):
        print(f"[Job {job_id}] Starting epoch {epoch+1}/{max_epochs}")
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(batch_X)
                loss = nn.MSELoss()(outputs, batch_y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            del batch_X, batch_y, outputs, loss

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"[Job {job_id}] Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad(), autocast():
            for val_X, _ in val_loader:
                val_X = val_X.to(device)
                val_out = model(val_X)
                val_preds.append(val_out.cpu())
        val_outputs = torch.cat(val_preds, dim=0)
        val_loss = nn.MSELoss()(val_outputs, y_val_tensor.cpu())
        print(f"[Job {job_id}] Validation Loss: {val_loss.item():.4f}")

        # Early stopping logic
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[Job {job_id}] Early stopping after {epoch+1} epochs.")
                break

        torch.cuda.empty_cache()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    job_end = time.perf_counter()
    print(f"[Job {job_id}] Training time: {job_end - job_start:.2f} seconds")
    return best_val_loss, model, lr, batch_size, hidden_dim, num_layers

# Run parallel jobs
learning_rates = [0.01, 1e-4, 3e-4]
batch_sizes = [48]
hidden_dims = [128, 256, 384, 512]
num_layers_list = [3, 4, 6, 8]  # or whatever range you want
hyperparams = list(product(learning_rates, batch_sizes, hidden_dims, num_layers_list))


results = Parallel(n_jobs=1)(
    delayed(train_model)(i, lr, batch_size, hidden_dim, num_layers, X_train_tensor.shape[1], num_outputs)
    for i, (lr, batch_size, hidden_dim, num_layers) in enumerate(hyperparams)
)


best_val_loss, best_model, best_lr, best_bs, best_hd, best_nl = min(results, key=lambda x: x[0])
print(f'Best Validation Loss: {best_val_loss:.4f}')

model_filename = os.path.join(
    output_dir,
    f"MLP_lr{best_lr}_bs{best_bs}_hd{best_hd}_nl{best_nl}_adamW_{num_samples}_{best_val_loss:.4f}.pt"
)

# Save the model, scaling parameters, and metadata
torch.save({
    'model_state_dict': best_model.state_dict(),
    'global_mean_X': global_mean_X,
    'global_std_X': global_std_X,
    'T_mean': T_mean, 'T_std': T_std,
    'B_mean': B_mean, 'B_std': B_std,
    'V_mean': V_mean, 'V_std': V_std,
    'I_mean': I_mean, 'I_std': I_std,
    'num_wavelengths': num_wavelengths,
    'num_optical_depths': num_optical_depths,
    'learning_rate': best_lr,
    'batch_size': best_bs,
    'hidden_dim': best_hd,
    'num_layers': best_nl,
    'num_training_profiles': num_samples
}, model_filename)

print(f"Saved model to: {model_filename}")


# --- INFERENCE ON GENERALISATION SNAPSHOT ---
print("Loading full snapshot for inference...")
snapshot_path = '/inputs/MANCHA9_ext_models.fits' #90 seconds after test snapshot
stokes_path = '/inputs/MANCHA9_ext_profiles.fits'
snapshot_data = fits.getdata(snapshot_path)  # (11, n_tau, ny, nx)
stokes_data = fits.getdata(stokes_path)      # (4, n_wave, ny, nx)

# Crop wavelength range if needed (match training shape)
stokes_data = stokes_data[:, :num_wavelengths, :, :]

ny, nx = stokes_data.shape[2], stokes_data.shape[3]

# Reshape to match training structure
X_full = stokes_data.reshape(4, num_wavelengths, -1).transpose(2, 0, 1).reshape(-1, 4 * num_wavelengths)

# Standardize
X_full_scaled = custom_standardize(X_full, global_mean_X, global_std_X)
X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32)
del X_full, X_full_scaled
torch.cuda.empty_cache()

# Run inference in batches
batch_size = 512
full_loader = DataLoader(X_full_tensor, batch_size=batch_size, pin_memory=True)

print("Running inference on full snapshot...")
all_preds = []
best_model.eval()
with torch.no_grad():
    for batch in full_loader:
        batch = batch.to(device)  # Move batch to GPU here
        out = best_model(batch)
        all_preds.append(out.cpu())  # Bring result back to CPU

full_predictions = torch.cat(all_preds, dim=0).numpy()  # shape (num_pixels, 5*n_tau)

# --- INVERSE TRANSFORM ---
def inverse(x, mean, std): return (x * std) + mean

# Reshape flat output to (N, τ, 6)
full_predictions = full_predictions.reshape(-1, num_optical_depths, 6)

predictions_dict = {
    "Temperature": inverse(full_predictions[:, :, 0], T_mean, T_std).reshape(ny, nx, num_optical_depths),
    "Magnetic Field Strength": inverse(full_predictions[:, :, 1], B_mean, B_std).reshape(ny, nx, num_optical_depths),
    "LOS Velocity": inverse(full_predictions[:, :, 2], V_mean, V_std).reshape(ny, nx, num_optical_depths),
    "Inclination": inverse(full_predictions[:, :, 3], I_mean, I_std).reshape(ny, nx, num_optical_depths),
}

# Reconstruct azimuth from sin(2φ) and cos(2φ)
sin2phi = full_predictions[:, :, 4]
cos2phi = full_predictions[:, :, 5]
azimuth_rad = 0.5 * np.arctan2(sin2phi, cos2phi)
azimuth_deg = np.rad2deg(azimuth_rad) % 180
predictions_dict["Azimuth"] = azimuth_deg.reshape(ny, nx, num_optical_depths)

# --- SAVE PREDICTIONS ---

fits_data = np.zeros((5, num_optical_depths, ny, nx), dtype=np.float32)
fits_data[0] = predictions_dict["Temperature"].transpose(2, 0, 1)
fits_data[1] = predictions_dict["Magnetic Field Strength"].transpose(2, 0, 1)
fits_data[2] = predictions_dict["LOS Velocity"].transpose(2, 0, 1)
fits_data[3] = predictions_dict["Inclination"].transpose(2, 0, 1)
fits_data[4] = predictions_dict["Azimuth"].transpose(2, 0, 1)

fits_path = os.path.join(output_dir, "predicted_atmosphere_maps_from_MLP.fits")
fits.writeto(fits_path, fits_data, overwrite=True)
print(f"Saved full snapshot inference to: {fits_path}")

