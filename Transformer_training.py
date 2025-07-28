from astropy.io import fits
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import os
from torch.cuda.amp import autocast, GradScaler
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {mem_gb:.1f} GB")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job_index', type=int, required=True)
args = parser.parse_args()
job_index = args.job_index


output_dir = '/users/3056857/sharedscratch/SINN/outputs_tuning'
os.makedirs(output_dir, exist_ok=True)

snapshot_path = '/users/3056857/sharedscratch/SINN/MANCHA_ext_models.fits'
stokes_path = '/users/3056857/sharedscratch/SINN/MANCHA_ext_profiles.fits'

full_profiles = fits.getdata(stokes_path, memmap=False)  # (4, n_wave, ny, nx)
full_models = fits.getdata(snapshot_path, memmap=False)  # (11, n_tau, ny, nx)

ny, nx = full_profiles.shape[2], full_profiles.shape[3]
num_samples = 1000000  # or however many you want
# ~ num_samples = ny*nx  # or however many you want

all_indices = [(y, x) for y in range(ny) for x in range(nx)]
np.random.seed(42)
selected_indices = np.random.choice(len(all_indices), size=num_samples, replace=False)

selected_profiles = np.zeros((4, full_profiles.shape[1], num_samples))
selected_models = np.zeros((11, full_models.shape[1], num_samples))

for i, idx in enumerate(selected_indices):
    y, x = all_indices[idx]
    selected_profiles[:, :, i] = full_profiles[:, :, y, x]
    selected_models[:, :, i] = full_models[:, :, y, x]

profiles = selected_profiles
models = selected_models

print('data loaded')
print(profiles.shape,models.shape)

# Detect NaNs across each profile
nan_mask = np.isnan(profiles).any(axis=(0, 1))  # Shape: (N,)

num_bad_profiles = nan_mask.sum()
num_total_profiles = profiles.shape[2]

print(f"{num_bad_profiles} out of {num_total_profiles} profiles contain NaNs")
print(f"Removing {nan_mask.sum()} profiles with NaNs")
profiles = profiles[:, :, ~nan_mask]
models = models[:, :, ~nan_mask]

num_stokes, num_wavelengths, N = profiles.shape
num_params, num_optical_depths, _ = models.shape

stokes_input_mlp = profiles.transpose(2, 0, 1).reshape(N, num_stokes * num_wavelengths)
print("Any NaNs before scaling?", np.isnan(stokes_input_mlp).any())

# Select output atmospheric parameters (T, B, v, incl, sin(az), cos(az))
def reshape_atmospheric_param(param_index):
    return models[param_index].T  # shape (N, num_optical_depths)

temperature_output_mlp = reshape_atmospheric_param(1)
magnetic_field_output_mlp = reshape_atmospheric_param(4)
velocity_output_mlp = reshape_atmospheric_param(5) / 100000.0  # cm/s to km/s
inclination_output_mlp = reshape_atmospheric_param(6)
azimuth_deg = reshape_atmospheric_param(7)  
azimuth_rad = np.deg2rad(azimuth_deg)
sin_2azimuth = np.sin(2 * azimuth_rad)
cos_2azimuth = np.cos(2 * azimuth_rad)


print("Extracted atm parameters")

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
temperature_output_scaled = custom_standardize(temperature_output_mlp, T_mean, T_std)
magnetic_field_output_scaled = custom_standardize(magnetic_field_output_mlp, B_mean, B_std)
velocity_output_scaled = custom_standardize(velocity_output_mlp, V_mean, V_std)
inclination_output_scaled = custom_standardize(inclination_output_mlp, I_mean, I_std)
print("Standardisation complete")


target_output_scaled = np.stack((
    temperature_output_scaled,
    magnetic_field_output_scaled,
    velocity_output_scaled,
    inclination_output_scaled,
    sin_2azimuth, #doesnt need scaled
    cos_2azimuth #doesnt need scaled
), axis=-1)

print("Stacked output shape:", target_output_scaled.shape)

# Reshape for transformer: (num_pixels, num_wavelengths, 4)
stokes_input_transformer = stokes_input_scaled.reshape(-1, num_wavelengths, 4)
print("Reshaped for transformer")

assert not np.isnan(stokes_input_scaled).any(), "Input contains NaNs!"
assert not np.isnan(target_output_scaled).any(), "Output contains NaNs!"


# Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    stokes_input_transformer, target_output_scaled, test_size=0.15, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
print("tensors created")

# Free memory from large original arrays - not an exhaustive list
del profiles, models, stokes_input_mlp
del temperature_output_mlp, magnetic_field_output_mlp, velocity_output_mlp
del inclination_output_mlp
del temperature_output_scaled, magnetic_field_output_scaled, velocity_output_scaled
del inclination_output_scaled
torch.cuda.empty_cache()

# Parallel training function
def train_model(job_id, lr, batch_size, hidden_dim, seq_len, input_dim, num_layers, num_heads):
    print(f"[Job {job_id}] lr={lr} bs={batch_size} hd={hidden_dim} nl={num_layers} nh={num_heads}")
    job_start = time.perf_counter()

    class SINN_Transformer_Sequence(nn.Module):
        def __init__(self, seq_len, input_dim, n_tau, hidden_dim, num_layers=4, num_heads=4):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_tau = n_tau
            self.seq_len = seq_len

            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

            self.query_content = nn.Parameter(torch.randn(n_tau, hidden_dim))     # learned content
            self.depth_pos_enc = nn.Parameter(torch.randn(n_tau, hidden_dim))     # learned positional encoding

            self.output_head = nn.Linear(hidden_dim, 6)

        def forward(self, x):
            # Encode the spectral sequence
            x = self.embedding(x) + self.positional_embedding  # (batch, seq_len, hidden_dim)
            memory = self.encoder(x)  # (batch, seq_len, hidden_dim)

            # Prepare query with content + depth encoding
            query = self.query_content + self.depth_pos_enc  # (n_tau, hidden_dim)
            query = query.unsqueeze(0).expand(x.size(0), -1, -1)  # (batch, n_tau, hidden_dim)

            # Cross-attend to encoder output
            out = self.decoder(tgt=query, memory=memory)  # (batch, n_tau, hidden_dim)

            return self.output_head(out)  # (batch, n_tau, 6)



    # Initialize model
    model = SINN_Transformer_Sequence(seq_len, input_dim, num_optical_depths, hidden_dim, num_layers=num_layers, num_heads=num_heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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
        print(f"[Job {job_id}] Starting epoch {epoch+1}/{max_epochs}", f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(batch_X)
                loss = nn.MSELoss()(outputs, batch_y)  # outputs and batch_y are both (batch, n_tau, 6)
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

    return best_val_loss, model, lr, batch_size, hidden_dim, num_layers, num_heads

# Select the hyperparameter set based on job_index
learning_rates = [0.001, 1e-4, 3e-4]
batch_sizes = [48]
hidden_dims = [128, 256, 384]
num_layers_list = [2, 3, 4]
num_heads_list = [2, 4]
hyperparams = list(product(learning_rates, batch_sizes, hidden_dims, num_layers_list, num_heads_list))

if job_index >= len(hyperparams):
    raise ValueError(f"Job index {job_index} exceeds number of hyperparameter combinations ({len(hyperparams)})")

lr, batch_size, hidden_dim, num_layers, num_heads = hyperparams[job_index]


best_val_loss, best_model, best_lr, best_bs, best_hd, best_nl, best_nh = train_model(
    job_index, lr, batch_size, hidden_dim,
    X_train_tensor.shape[1],  # seq_len
    X_train_tensor.shape[2],  # input_dim
    num_layers, num_heads
)


model_filename = os.path.join(
    output_dir,
    f"transformer_final_snapshot0_lr{best_lr}_bs{best_bs}_hd{best_hd}_nl{best_nl}_nh{best_nh}_val{best_val_loss:.4f}_{num_samples}.pt"
)


try:
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
    'num_heads': best_nh,
    'num_samples': num_samples
    }, model_filename)
    print(f"Saved model to: {model_filename}")
except Exception as e:
    print("Saving model failed!",e)