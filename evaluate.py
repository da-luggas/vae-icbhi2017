import torch
import numpy as np
import random
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from model import Encoder, Decoder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

import argparse

import utils

if __name__ == "__main__":
    # Set seeds for reproducibility
    utils.set_seeds()

    parser = argparse.ArgumentParser(description="Training a VAE for respiratory sounds.")
    parser.add_argument("--model", default=".runs/Oct11_09-37-57_code-server/saved_model.pth", type=str, help="Path to saved model from training",)
    parser.add_argument("--dataset", default="dataset.pt", type=str, help="Location of the ICBHI dataset",)
    parser.add_argument("--bs", default=64, type=int, help="Batch size during training.")
    parser.add_argument("--nz", default=100, type=int, help="Size of z latent vector.")
    parser.add_argument("--nf", default=128, type=int, help="Size of feature  maps.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of training epochs.")
    parser.add_argument("--lr", default=0.0002, type=float, help="Learning rate for optimizers.")
    parser.add_argument("--beta1", default=0.5, type=float, help="Beta1 hyperparameter for Adam optimizers.",)
    parser.add_argument("--patience", default=10, type=int, help="Patience for early stopping.")
    parser.add_argument("--device", default=torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")), help="Device to run training on",)
    args = parser.parse_args()
    model_dir = args.model

    # Load models state dict and args
    saved_model = torch.load(model_dir)
    encoder_state, decoder_state = saved_model['encoder_state_dict'], saved_model['decoder_state_dict']
    # args = saved_model['args']

    # Load data
    _, val_loader, test_loader = utils.load_data(args.dataset)

    # Initialize models and optimizers
    encoder = Encoder(1, args.nz, args.nf).to(args.device)
    decoder = Decoder(1, args.nz, args.nf).to(args.device)

    # Evaluation model generalization using test set
    roc_auc, balacc = utils.test_model(encoder, decoder, val_loader, test_loader, encoder_state, decoder_state, args)
    print('ROC-AUC:', roc_auc)
    print('BALACC:', balacc)