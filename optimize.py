import argparse
import os

import optuna
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model as model
import utils

if __name__ == "__main__":
    # Set seeds for reproducibility
    utils.set_seeds()

    parser = argparse.ArgumentParser(description="Training a VAE for respiratory sounds.")
    parser.add_argument("--dataset", default="dataset.pt", type=str, help="Location of the ICBHI dataset",)
    parser.add_argument("--bs", default=64, type=int, help="Batch size during training.")
    parser.add_argument("--nz", default=100, type=int, help="Size of z latent vector.")
    parser.add_argument("--nf", default=128, type=int, help="Size of feature maps.")
    parser.add_argument("--epochs", default=500, type=int, help="Number of training epochs.")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for optimizers.")
    parser.add_argument("--beta1", default=0.5, type=float, help="Beta1 hyperparameter for Adam optimizers.",)
    parser.add_argument("--patience", default=10, type=int, help="Patience for early stopping.")
    parser.add_argument("--device", default=torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")), help="Device to run training on",)
    parser.add_argument("--alpha", default=0.5, type=float, help="Weighting parameter for MSE and KLD loss")
    args = parser.parse_args()

    def objective(trial):
        args.bs = trial.suggest_categorical('bs', [32, 64, 128])
        args.nz = trial.suggest_int('nz', 2, 1000)
        args.nf = trial.suggest_categorical('nf', [16, 32, 64, 128])
        args.alpha = trial.suggest_float('alpha', 0, 1)
        # args.patience = trial.suggest_int('patience', 10, 50)

        # Load data
        train_loader, val_loader, test_loader = utils.load_data(args.dataset, args.bs)

        # Initialize the model and optimizer
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
        last_epoch = 0

        for epoch in tqdm(range(args.epochs)):
            last_epoch += 1
            train_loss, source_example, recon_example = utils.train_epoch(
                encoder, optimizerE, decoder, optimizerD, train_loader, args
            )
            val_loss = utils.eval_epoch(
                encoder, schedulerE, decoder, schedulerD, val_loader, args
            )

            writer.add_scalar("Loss/Train", train_loss, global_step=epoch)
            writer.add_scalar("Loss/Validation", val_loss, global_step=epoch)

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

        saved_model = torch.load(os.path.join(writer.log_dir, "model.pt"))
        encoder_state, decoder_state = saved_model['encoder_state_dict'], saved_model['decoder_state_dict']
        roc_auc, balacc = utils.test_model(encoder, decoder, val_loader, val_loader, encoder_state, decoder_state, args)

        writer.close()
        return roc_auc, balacc, last_epoch
    
    study = optuna.create_study(directions=["maximize", "maximize", "minimize"], study_name="VAE_optimization", storage="sqlite:///optuna_vae.db", load_if_exists=True)
    study.optimize(objective, n_trials=50)