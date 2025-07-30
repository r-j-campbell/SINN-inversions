import os
import numpy as np
import torch
from astropy.io import fits
import torch.nn as nn

from architectures import SINN_Transformer_Sequence

print("Starting Transformer inference script...")
# === CONFIG ===
output_dir = '/outputs'
model_path = output_dir+'/model.pt'
snapshot_path = '/inputs/MANCHA9_ext_models.fits' #different snapshot than one used for training!
stokes_path = '/inputs/MANCHA9_ext_profiles.fits'
fits_output_path = os.path.join(output_dir, "predicted_atmosphere_maps.fits")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === LOAD MODEL CHECKPOINT ===
print("Loading model checkpoint...")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
global_mean_X = checkpoint['global_mean_X']
global_std_X = checkpoint['global_std_X']
T_mean, B_mean, V_mean, I_mean = checkpoint['T_mean'], checkpoint['B_mean'], checkpoint['V_mean'], checkpoint['I_mean']
T_std, B_std, V_std, I_std = checkpoint['T_std'], checkpoint['B_std'], checkpoint['V_std'], checkpoint['I_std']
num_wavelengths = checkpoint['num_wavelengths']
n_tau = checkpoint['num_optical_depths']
hidden_dim = checkpoint['hidden_dim']
num_layers = checkpoint['num_layers']
num_heads = checkpoint['num_heads']
print(f"Model trained on {checkpoint['num_samples']} samples")

print("Reading Stokes data from FITS...")
stokes = fits.getdata(stokes_path, memmap=False)  # Shape: (4, n_wave, ny, nx)

if stokes.shape[1] < num_wavelengths:
    raise ValueError(f"Stokes input has fewer wavelengths ({stokes.shape[1]}) than expected ({num_wavelengths})")

print("Reshaping input to flat profile list...")
ny, nx = stokes.shape[2], stokes.shape[3]
X = stokes.reshape(4, num_wavelengths, -1).transpose(2, 0, 1).reshape(-1, 4 * num_wavelengths)
print("Flattened input shape:", X.shape)

print("Converting to tensor first...")
X_native = X.byteswap().view(X.dtype.newbyteorder('='))
X_tensor = torch.tensor(X_native, dtype=torch.float32)  # keep on CPU
print("Tensor created:", X_tensor.shape)

print("Standardizing...")
mean_tensor = torch.tensor(global_mean_X, dtype=torch.float32)
std_tensor = torch.tensor(global_std_X, dtype=torch.float32)

def inplace_standardize(X_tensor, mean_tensor, std_tensor, batch_size=5000):
    for i in range(0, X_tensor.shape[0], batch_size):
        print(f"Standardizing batch {i // batch_size + 1} / {X_tensor.shape[0] // batch_size + 1}")
        X_tensor[i:i+batch_size] -= mean_tensor
        X_tensor[i:i+batch_size] /= std_tensor

inplace_standardize(X_tensor, mean_tensor, std_tensor)

print("Standardization done.")

# === INSTANTIATE MODEL ===
print("Instantiating model...")
model = SINN_Transformer_Sequence(
    seq_len=num_wavelengths,
    input_dim=4,
    n_tau=n_tau,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_heads=num_heads
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("Model loaded and ready.")

# === RUN INFERENCE ===
print("Running inference in batches...")
preds = []
batch_size = 1024  # Tune this down if virtual memory issues encountered on GPU
num_batches = int(np.ceil(X_tensor.shape[0] / batch_size))
print(f"Total batches: {num_batches}")

with torch.no_grad():
    for i in range(0, X_tensor.shape[0], batch_size):
        print(f"Inference batch {i // batch_size + 1}/{num_batches}")
        flat_batch = X_tensor[i:i+batch_size]  # [B, 4*num_wavelengths]
        reshaped_batch = flat_batch.reshape(-1, num_wavelengths, 4).to(device)
        out = model(reshaped_batch).cpu().numpy()
        preds.append(out)
        del flat_batch, reshaped_batch, out
        torch.cuda.empty_cache()


print("Finished inference.")
preds = np.concatenate(preds, axis=0)  # [N, n_tau, 6]

# === INVERSE TRANSFORM ===
def inverse(x, mean, std): return (x * std) + mean
pred_dict = {
    "Temperature": inverse(preds[:, :, 0], T_mean, T_std).reshape(ny, nx, n_tau),
    "Magnetic Field Strength": inverse(preds[:, :, 1], B_mean, B_std).reshape(ny, nx, n_tau),
    "LOS Velocity": inverse(preds[:, :, 2], V_mean, V_std).reshape(ny, nx, n_tau),
    "Inclination": inverse(preds[:, :, 3], I_mean, I_std).reshape(ny, nx, n_tau),
}
sin2phi = preds[:, :, 4]
cos2phi = preds[:, :, 5]
azimuth_rad = 0.5 * np.arctan2(sin2phi, cos2phi)
pred_dict["Azimuth"] = np.rad2deg(azimuth_rad % np.pi).reshape(ny, nx, n_tau)

print("Saving full predicted maps to FITS...")
fits_data = np.zeros((5, n_tau, ny, nx), dtype=np.float32)

fits_data[0] = pred_dict["Temperature"].transpose(2, 0, 1) # [ny, nx, n_tau] → [n_tau, ny, nx] for FITS
fits_data[1] = pred_dict["Magnetic Field Strength"].transpose(2, 0, 1)
fits_data[2] = pred_dict["LOS Velocity"].transpose(2, 0, 1)
fits_data[3] = pred_dict["Inclination"].transpose(2, 0, 1)
fits_data[4] = pred_dict["Azimuth"].transpose(2, 0, 1)

fits.writeto(fits_output_path, fits_data, overwrite=True)
print(f"Saved FITS file: {fits_output_path}")
