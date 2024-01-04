# VAE-ICBHI2017
An implementation of the Variational Autoencoder for the Respiratory Sound Database [1].

## How to Use
### 1. Install Dependencies
This installs the necessary dependencies using pip. If you want to use GPU accelerated learning (highly recommended), see the PyTorch Documentation on how to install the packages for your platform.

    pip install torch torchaudio scikit-learn numpy matplotlib tqdm
### 2. Clone Repository
Clone the repository locally and enter directory using:

    git clone https://github.com/da-luggas/vae-icbhi2017
    cd vae-icbhi2017

### 3. Download Dataset
You need to download the [Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) (caution: no SSL) and extract it into a folder within the repository. Once that is done, you can run the preprocessing:

    python preprocess.py --dataset path/to/dataset_folder --target path/to/output/dataset.pt
Make sure to include the *.pt suffix as the dataset will be saved as a pytorch object.
### 4. Run Training
Run the training with recommended hyperparameters:

    python train.py --dataset /path/to/preprocessed/dataset.pt
There is a multitude of parameters to tinker with. For now, refer to the `train.py` file to check which arguments you can pass.
You can view your training progress by using TensorBoard.
### 5. Evaluate
Once the training is done, it will have created a folder `runs` where you can find each training run you performed, including the saved `model.pt`. If you want to evaluate the generalization performance of a particular model on the test set, you can run:

    python evaluate.py --model /path/to/desired/model.pt

## Sources
[1] Rocha, Bruno M., et al. “An Open Access Database for the Evaluation of Respiratory Sound Classification Algorithms.” _Physiological Measurement_, vol. 40, no. 3, 22 Mar. 2019, p. 035001, pubmed.ncbi.nlm.nih.gov/30708353/, https://doi.org/10.1088/1361-6579/ab03ea.