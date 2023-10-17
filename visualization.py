import torch
import numpy as np
import random
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from dataset import RespiratorySoundDataset
from model import Encoder, Decoder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

###################
## CONFIGURATION ##
###################

dataroot = "/home/lukas/thesis/anogan2d/dataset"
batch_size = 64
nc = 1
nz = 100
nf = 128
ndf = nf
ngf = nf
device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
checkpoint_path = 'runs/Oct11_09-37-57_code-server/saved_model.pth'

################
## EVALUATION ##
################

# Load the saved model
checkpoint = torch.load(checkpoint_path, map_location=device)
encoder = Encoder(nc, nz, ndf).to(device)
decoder = Decoder(nc, nz, ngf).to(device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

# Load dataset and create DataLoader for validation and test sets
dataset = RespiratorySoundDataset(root_dir=dataroot)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

validation_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - validation_size
actual_train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the ``BCELoss`` function
mse_loss = torch.nn.MSELoss(reduction='none')

import torch
import numpy as np
import random
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from dataset import RespiratorySoundDataset
from model import Encoder, Decoder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manualSeed = 6185
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

###################
## CONFIGURATION ##
###################

dataroot = "/home/lukas/thesis/anogan2d/dataset"
batch_size = 64
nc = 1
nz = 100
nf = 128
ndf = nf
ngf = nf
device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
checkpoint_path = 'runs/Oct11_09-37-57_code-server/saved_model.pth'

###################
## VISUALIZATION ##
###################

# Load the saved model
checkpoint = torch.load(checkpoint_path, map_location=device)
encoder = Encoder(nc, nz, ndf).to(device)
decoder = Decoder(nc, nz, ngf).to(device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

# Load dataset and create DataLoader for validation and test sets
dataset = RespiratorySoundDataset(root_dir=dataroot)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

validation_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - validation_size
actual_train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

mse_loss = torch.nn.MSELoss(reduction='none')

num_images = 64
real_images = []
reconstructed_images = []

# Generate real mel spectrogram images
real_batch = next(iter(test_loader))
for i in range(num_images):
    real_image = real_batch[0][i].cpu().detach().numpy()
    real_images.append(real_image)

# Generate reconstructed mel spectrogram images using VAE
with torch.no_grad():
    for i in range(num_images):
        data = real_batch[0][i].unsqueeze(0).to(device)
        mu, logvar = encoder(data)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        decoded = decoder(z)
        reconstructed_images.append(decoded.squeeze(0).cpu().numpy())

# Create a grid of 8x8 real images
fig_real, axs_real = plt.subplots(8, 8, figsize=(8, 8))
fig_real.subplots_adjust(hspace=0.5)

# Create a grid of 8x8 reconstructed images
fig_rec, axs_rec = plt.subplots(8, 8, figsize=(8, 8))
fig_rec.subplots_adjust(hspace=0.5)

# Create a grid of 8x8 difference images
fig_diff, axs_diff = plt.subplots(8, 8, figsize=(8, 8))
fig_diff.subplots_adjust(hspace=0.5)

# Loop through the images and plot them in the grids
for i in range(num_images):
    row, col = i // 8, i % 8
    
    # Plot real image and label
    ax_real = axs_real[row, col]
    real_image = real_images[i].squeeze()
    real_label = real_batch[1][i].item()
    ax_real.imshow(real_image, cmap='grey', origin='lower')
    ax_real.set_title(f"Real ({real_label})", fontsize=8)
    ax_real.axis('off')
    
    # Plot reconstructed image and label
    ax_rec = axs_rec[row, col]
    rec_image = reconstructed_images[i].squeeze()
    rec_label = real_batch[1][i].item()
    ax_rec.imshow(rec_image, cmap='grey', origin='lower')
    ax_rec.set_title(f"Rec ({rec_label})", fontsize=8)
    ax_rec.axis('off')

    # Calculate and plot the difference image
    ax_diff = axs_diff[row, col]
    diff_image = abs(real_image - rec_image)
    diff_label = real_batch[1][i].item()
    ax_diff.imshow(diff_image, cmap='hot', origin='lower')
    ax_diff.set_title(f"Diff ({diff_label})", fontsize=8)
    ax_diff.axis('off')

# Save the grids as separate files
fig_real.savefig("real_images_grid.png")
fig_rec.savefig("reconstructed_images_grid.png")
fig_diff.savefig("diff_images_grid.png")

# Close the figures
plt.close(fig_real)
plt.close(fig_rec)
plt.close(fig_diff)