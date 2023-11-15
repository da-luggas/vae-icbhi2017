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
    parser.add_argument("--model", default="/home/lukas/thesis/anogan2d/dataset", type=str, help="Path to saved model from training",)
    model_dir = parser.parse_args().model

    # Load models state dict and args
    saved_model = torch.load(model_dir)
    encoder_state, decoder_state = saved_model['encoder_state_dict'], saved_model['decoder_state_dict']
    args = saved_model['args']

    # Split and load data
    _, val_set, test_set = utils.split_data(args.dataset)

    val_loader = DataLoader(val_set, batch_size=args.bs)
    test_loader = DataLoader(test_set, batch_size=args.bs)

    # Initialize models and optimizers
    encoder = Encoder(1, args.nz, args.nf).to(args.device)
    decoder = Decoder(1, args.nz, args.nf).to(args.device)

    # Evaluation model generalization using test set
    roc_auc, balacc, tpr, tnr = utils.test_model(encoder, decoder, val_loader, test_loader, encoder_state, decoder_state, args)
    print('ROC-AUC:', roc_auc)
    print('BALACC:', balacc)
    print('TPR:', tpr)
    print('TNR:', tnr)