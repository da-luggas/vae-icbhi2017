import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
import utils

if __name__ == "__main__":
    # Set seeds for reproducibility
    utils.set_seeds()

    parser = argparse.ArgumentParser(description="Training a VAE for respiratory sounds.")
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

    # Load data
    train_loader, val_loader, test_loader = utils.load_data(args.dataset, args.bs)

    # Initialize the mode and optimizer
    encoder = model.Encoder(13, args.nz, args.nf).to(args.device)
    decoder = model.Decoder(13, args.nz, args.nf).to(args.device)
    optimizerE = optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Weight initialization
    encoder.apply(model.xavier_init)
    decoder.apply(model.xavier_init)

    # Initialize scheduler
    schedulerE = optim.lr_scheduler.ReduceLROnPlateau(optimizerE, 'min', patience=args.patience // 2, factor=0.5)
    schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, 'min', patience=args.patience // 2, factor=0.5)

    # Initialize tensorboard
    writer = SummaryWriter()
    # Initialize learning rate scheduler

    # Initialize counter and best loss storage for early stopping
    best_val_loss = float("inf")
    waiting = 0

    for epoch in tqdm(range(args.epochs)):
        train_loss, source_example, recon_example = utils.train_epoch(
            encoder, optimizerE, decoder, optimizerD, train_loader, args
        )
        val_loss = utils.eval_epoch(
                encoder, schedulerE, decoder, schedulerD, val_loader, args
        )

        writer.add_scalar("Loss/Train", train_loss, global_step=epoch)
        writer.add_scalar("Loss/Validation", val_loss, global_step=epoch)

        # Add images to compare to tensorboard
        if epoch % 5 == 0:
            img_grid_original = vutils.make_grid(source_example.unsqueeze(1), normalize=True)
            img_grid_reconstructed = vutils.make_grid(recon_example.unsqueeze(1), normalize=True)

            writer.add_image("Image/Original", img_grid_original, global_step=epoch)
            writer.add_image(
                "Image/Reconstructed", img_grid_reconstructed, global_step=epoch
            )

        # Early stopping and state dict saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            waiting = 0

            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "args": args,
                },
                os.path.join(writer.log_dir, "model.pt"),
            )
        else:
            waiting += 1

        if waiting > args.patience:
            break

    writer.close()