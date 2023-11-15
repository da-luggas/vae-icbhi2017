import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


import model
import utils

class RespiratorySoundDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def set_seeds(seed=999):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def get_optimal_threshold(losses, labels):
    accuracies = []
    for threshold in losses:
        y_pred = losses > threshold
        tpr = np.sum((y_pred == 1) & (labels == 1)) / np.sum(labels == 1)
        tnr = np.sum((y_pred == 0) & (labels == 0)) / np.sum(labels == 0)
        accuracy = 0.5 * (tpr + tnr)
        accuracies.append(accuracy)
    optimal_threshold = losses[np.argmax(accuracies)]
    return optimal_threshold


def split_data(dataset, prevent_leakage=False, seed=999):
    data = torch.load(dataset)
    
    recording_ids = data['recording_ids']
    cycles = data['cycles']
    labels = data['labels']

    if prevent_leakage:
        unique_recording_ids = torch.unique(recording_ids)
        train_ids, test_ids = train_test_split(unique_recording_ids, test_size=0.2, random_state=seed, stratify=labels)

        # Create masks for selecting data
        train_mask = torch.isin(recording_ids, train_ids)
        test_mask = torch.isin(recording_ids, test_ids)

        # Separate data based on recording_id
        X_train, y_train = cycles[train_mask], labels[train_mask]
        X_test, y_test = cycles[test_mask], labels[test_mask]
    else:
        # Random splitting 80/20
        X_train, X_test, y_train, y_test = train_test_split(cycles, labels, test_size=0.2, random_state=seed, stratify=labels)

    # Further splitting of train set into validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)

    X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]

    train_set = RespiratorySoundDataset(X_train, y_train)
    val_set = RespiratorySoundDataset(X_val, y_val)
    test_set = RespiratorySoundDataset(X_test, y_test)

    return train_set, val_set, test_set

def train_epoch(encoder_model, encoder_optimizer, decoder_model, decoder_optimizer, criterion, dataloader, args):
    train_loss = 0
    encoder_model.train()
    decoder_model.train()

    for idx, (data, _) in enumerate(dataloader, 0):
        data = data.to(args.device)
        data = data.unsqueeze(1)

        # Encode data
        mu, logvar = encoder_model(data)
        # Sample from latent space using reparametrization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # Decode the encoded data
        decoded = decoder_model(z)

        # Calculate loss
        reconstruction_loss = criterion(decoded, data)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_divergence

        encoder_model.zero_grad()
        decoder_model.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(dataloader)

    return train_loss, data[:16], decoded[:16]

def val_epoch(encoder_model, encoder_optimizer, decoder_model, decoder_optimizer, criterion, dataloader, args):
    schedulerE = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=5, factor=0.5, verbose=True)
    schedulerD = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=5, factor=0.5, verbose=True)

    val_loss = 0
    encoder_model.eval()
    decoder_model.eval()

    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader, 0):
            data = data.to(args.device)
            data = data.unsqueeze(1)
            mu, logvar = encoder_model(data)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            decoded = decoder_model(z)

            reconstruction_loss = criterion(decoded, data)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_divergence
            val_loss += loss.item()

    schedulerE.step(val_loss)
    schedulerD.step(val_loss)
    return val_loss

def test_model(encoder, decoder, val_dataloader, test_dataloader, encoder_state, decoder_state, args):
    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    encoder.eval()
    decoder.eval()

    criterion = nn.MSELoss(reduction='none')
    val_scores = []
    val_labels = []

    test_scores = []
    test_labels = []

    with torch.no_grad():
        for data, label in val_dataloader:
            data = data.to(args.device)
            data = data.unsqueeze(1)
            mu, logvar = encoder(data)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            decoded = decoder(z)

            reconstruction_loss_batch = torch.sum(criterion(decoded, data), axis=(1, 2, 3))
            kl_divergence_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=(1, 2, 3))
            loss_batch = reconstruction_loss_batch + kl_divergence_batch

            val_scores.extend(loss_batch.cpu().numpy())
            val_labels.extend(label.numpy())

        for data, label in test_dataloader:
            data = data.to(args.device)
            data = data.unsqueeze(1)
            mu, logvar = encoder(data)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            decoded = decoder(z)

            reconstruction_loss_batch = torch.sum(criterion(decoded, data), axis=(1, 2, 3))
            kl_divergence_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=(1, 2, 3))
            loss_batch = reconstruction_loss_batch + kl_divergence_batch

            test_scores.extend(loss_batch.cpu().numpy())
            test_labels.extend(label.numpy())

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)

    threshold = get_optimal_threshold(val_scores, val_labels)
    predictions = (test_scores > threshold)

    roc_auc = roc_auc_score(test_labels, test_scores)
    tpr = np.sum((predictions == 1) & (test_labels == 1)) / np.sum(test_labels == 1)
    tnr = np.sum((predictions == 0) & (test_labels == 0)) / np.sum(test_labels == 0)
    balanced_accuracy = 0.5 * (tpr + tnr)

    print("ROC-AUC Score: ", roc_auc.round(2))
    print("BALACC: ", balanced_accuracy.round(2))
    print("TPR: ", tpr.round(2))
    print("TNR: ", tnr.round(2))

    return roc_auc, balanced_accuracy, tpr, tnr