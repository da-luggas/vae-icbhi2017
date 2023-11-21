import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


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

def load_data(dataset, batch_size):
    saved_data = torch.load(dataset)

    X_train, X_val, X_test, y_train, y_val, y_test = saved_data['X_train'], saved_data['X_val'], saved_data['X_test'], saved_data['y_train'], saved_data['y_val'], saved_data['y_test']

    train_set = RespiratorySoundDataset(X_train, y_train)
    val_set = RespiratorySoundDataset(X_val, y_val)
    test_set = RespiratorySoundDataset(X_test, y_test)
    
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def get_optimal_threshold(losses, labels):
    unique_losses = np.unique(losses)
    best_threshold = unique_losses[0]
    best_bal_acc = 0

    for threshold in unique_losses:
        predictions = losses > threshold
        bal_acc = balanced_accuracy_score(labels, predictions)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = threshold

    return best_threshold

def train_epoch(encoder_model, encoder_optimizer, decoder_model, decoder_optimizer, dataloader, args):
    criterion = nn.MSELoss(reduction="sum")
    
    train_loss = 0
    encoder_model.train()
    decoder_model.train()

    for data, _ in dataloader:
        data = data.to(args.device)

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

def eval_epoch(encoder_model, encoder_scheduler, decoder_model, decoder_scheduler, dataloader, args):
    criterion = nn.MSELoss(reduction="sum")

    val_loss = 0
    encoder_model.eval()
    decoder_model.eval()

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(args.device)
            mu, logvar = encoder_model(data)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            decoded = decoder_model(z)

            reconstruction_loss = criterion(decoded, data)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_divergence
            val_loss += loss.item()

    val_loss = val_loss / len(dataloader)
    encoder_scheduler.step(val_loss)
    decoder_scheduler.step(val_loss)
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
            mu, logvar = encoder(data)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            decoded = decoder(z)

            reconstruction_loss_batch = torch.sum(criterion(decoded, data), axis=(1, 2))
            kl_divergence_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=(1))
            loss_batch = reconstruction_loss_batch + kl_divergence_batch

            val_scores.extend(loss_batch.cpu().numpy())
            val_labels.extend(label.numpy())

        for data, label in test_dataloader:
            data = data.to(args.device)
            mu, logvar = encoder(data)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            decoded = decoder(z)

            reconstruction_loss_batch = torch.sum(criterion(decoded, data), axis=(1, 2))
            kl_divergence_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=(1))
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
    balanced_accuracy = balanced_accuracy_score(test_labels, predictions)

    return roc_auc, balanced_accuracy