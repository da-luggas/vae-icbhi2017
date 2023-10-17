import torch
import numpy as np
import random
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from dataset import RespiratorySoundDataset
from model import Encoder, Decoder
from torch.utils.data import DataLoader, random_split

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

mse_loss = torch.nn.MSELoss(reduction='none')

# Calculate losses on validation set
val_losses = []
val_labels = []

with torch.no_grad():
    for data, label in validation_loader:
        data = data.to(device)
        mu, logvar = encoder(data)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        decoded = decoder(z)

        reconstruction_loss_batch = torch.sum(mse_loss(decoded, data), axis=(1, 2, 3))
        kl_divergence_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=(1, 2, 3))
        loss_batch = reconstruction_loss_batch + kl_divergence_batch
        
        val_losses.extend(loss_batch.cpu().numpy())
        val_labels.extend(label.numpy())

# Find the threshold that gives the highest Balanced Accuracy on the validation set
unique_losses = np.unique(val_losses)
best_threshold = unique_losses[0]
best_bal_acc = 0
for threshold in unique_losses:
    predictions = [1 if loss > threshold else 0 for loss in val_losses]
    bal_acc = balanced_accuracy_score(val_labels, predictions)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_threshold = threshold
print(f"Best Balanced Accuracy on Validation Set: {best_bal_acc}")
print("-" * 20)

# Evaluate on the test set using the best threshold
test_losses = []
test_labels = []
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        mu, logvar = encoder(data)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        decoded = decoder(z)

        reconstruction_loss_test = torch.sum(mse_loss(decoded, data), axis=(1, 2, 3))
        kl_divergence_test = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=(1, 2, 3))
        loss = reconstruction_loss_test + kl_divergence_test
        test_losses.extend(loss.cpu().numpy())
        test_labels.extend(label.numpy())

# Calculate AUC score
auc_score = roc_auc_score(test_labels, test_losses)
print(f"AUC Score: {auc_score}")

# Calculate Balanced Accuracy using the best threshold
test_predictions = [1 if loss > best_threshold else 0 for loss in test_losses]
test_bal_acc = balanced_accuracy_score(test_labels, test_predictions)
print(f"Balanced Accuracy: {test_bal_acc}")