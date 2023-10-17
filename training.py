import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from dataset import RespiratorySoundDataset
import model

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

parser = argparse.ArgumentParser(description="Training a VAE for respiratory sounds.")
parser.add_argument("--dataroot", default="/home/lukas/thesis/anogan2d/dataset", type=str, help="Location of the ICBHI dataset")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
parser.add_argument("--nz", default=100, type=int, help="Size of z latent vector.")
parser.add_argument("--nf", default=128, type=int, help="Size of feature maps.")
parser.add_argument("--num_epochs", default=1000, type=int, help="Number of training epochs.")
parser.add_argument("--lr", default=0.0002, type=float, help="Learning rate for optimizers.")
parser.add_argument("--beta1", default=0.5, type=float, help="Beta1 hyperparameter for Adam optimizers.")
parser.add_argument("--patience", default=10, type=int, help="Patience for early stopping.")
args = parser.parse_args()

###################
## CONFIGURATION ##
###################

# Root directory for dataset
dataroot = args.dataroot

# Batch size during training
batch_size = args.batch_size

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = args.nz

# Size of feature maps in generator
ngf = args.nf

# Size of feature maps in discriminator
ndf = args.nf

# Number of training epochs
num_epochs = args.num_epochs

# Learning rate for optimizers
lr = args.lr

# Beta1 hyperparameter for Adam optimizers
beta1 = args.beta1

# Patience for early stopping
patience = args.patience

# Device to run training on
device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Running on device: {device}")

##############
## TRAINING ##
##############

if device == torch.device('cuda'):
    # Free cache on cuda device
    torch.cuda.empty_cache()

# Initialize tensorboard
writer = SummaryWriter()

# Load dataset
dataset = RespiratorySoundDataset(root_dir=dataroot)

# Splitting dataset into train and test set
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Further splitting into training and validation set out of training set
validation_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - validation_size

actual_train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

# Filtering out samples with label 1 from training dataset
indices = [i for i in range(len(actual_train_dataset)) if actual_train_dataset[i][1] == 0]
filtered_train_dataset = Subset(actual_train_dataset, indices)

# Create DataLoaders
train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print(filtered_train_dataset.__len__())
# print(validation_dataset.__len__())
# print(test_dataset.__len__())

# Initialize the VAE and optimizers
encoder = model.Encoder(nc, nz, ndf).to(device)
decoder = model.Decoder(nc, nz, ngf).to(device)

encoder.apply(model.xavier_init)
decoder.apply(model.xavier_init)

optimizerE = optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(decoder.parameters(), lr=lr, betas=(beta1, 0.999))

# Initialize learning rate scheduler
schedulerE = optim.lr_scheduler.ReduceLROnPlateau(optimizerE, 'min', patience=5, factor=0.5, verbose=True)
schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, 'min', patience=5, factor=0.5, verbose=True)

mse_loss = nn.MSELoss(reduction='sum')

# Initialize counter and best loss storage for early stopping
best_loss = float('inf')
counter = 0

for epoch in tqdm(range(num_epochs)):
    # Training loop
    for i, (data, _) in enumerate(train_loader, 0):
        encoder.zero_grad()
        decoder.zero_grad()

        # Send data to device
        data = data.to(device)

        # Encode data
        mu, logvar = encoder(data)

        # Sample from latent space using reparametrization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # Decode the encoded data
        decoded = decoder(z)

        # Calculate loss
        reconstruction_loss = mse_loss(decoded, data)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        train_loss = reconstruction_loss + kl_divergence

        # Backward pass and optimize
        train_loss.backward()
        optimizerE.step()
        optimizerD.step()


        # Update tensorboard
        if i % 10 == 0:
            writer.add_scalar('Loss/Train', train_loss.item(), epoch*len(train_loader) + i)
            
            # Add images to compare to tensorboard
            if i % 100 == 0:
                with torch.no_grad():
                    img_grid_original = vutils.make_grid(data[:16], normalize=True)
                    img_grid_reconstructed = vutils.make_grid(decoded[:16], normalize=True)

                    writer.add_image('Image/Original', img_grid_original, epoch*len(train_loader) + i)
                    writer.add_image('Image/Reconstructed', img_grid_reconstructed, epoch*len(train_loader) + i)

    ##Validation loop
    val_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(validation_loader, 0):
            data = data.to(device)
            mu, logvar = encoder(data)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            decoded = decoder(z)
            
            reconstruction_loss_val = mse_loss(decoded, data)
            kl_divergence_val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            current_val_loss = reconstruction_loss_val + kl_divergence_val
            
            val_loss += current_val_loss.item()

    val_loss = val_loss / len(validation_loader.dataset)
    writer.add_scalar('Loss/Validation', val_loss, epoch)

    # Learning rate scheduler
    schedulerE.step(val_loss)
    schedulerD.step(val_loss)


    # Early stopping and state dict saving
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0

        save_path = os.path.join(writer.log_dir, 'saved_model.pth')
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizerE_state_dict': optimizerE.state,
            'optimiterD_state_dict': optimizerD.state_dict(),
            'config': vars(args)
        }, save_path)
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping after {patience} epochs without improvement.")
        break