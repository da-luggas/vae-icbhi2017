import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample

from sklearn.model_selection import train_test_split


def print_melspec(melspec):
    plt.imshow(melspec)
    plt.show()

def split_data(recording_ids, cycles, labels, prevent_leakage=False):
    if prevent_leakage:
        unique_recording_ids = torch.unique(recording_ids)
        train_ids, test_ids = train_test_split(unique_recording_ids, test_size=0.2, stratify=labels, random_state=21)

        # Create masks for selecting data
        train_mask = torch.isin(recording_ids, train_ids)
        test_mask = torch.isin(recording_ids, test_ids)

        # Separate data based on recording_id
        X_train, y_train = cycles[train_mask], labels[train_mask]
        X_test, y_test = cycles[test_mask], labels[test_mask]
    else:
        # Random splitting 80/20
        X_train, X_test, y_train, y_test = train_test_split(cycles, labels, test_size=0.2, stratify=labels, random_state=21)

    # Further splitting of train set into validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=999)

    X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]

    return X_train, X_val, X_test, y_train, y_val, y_test

def wrap_padding(tensor, target_length):
    # Truncate if the tensor is longer than the target length
    if len(tensor) > target_length:
        return tensor[:target_length]

    # If the tensor is shorter than the target length, apply wrap padding
    if len(tensor) < target_length:
        # Calculate how many times the tensor needs to be concatenated to exceed the target length
        num_repeats = (target_length + len(tensor) - 1) // len(tensor)

        # Tile the tensor
        tiled_tensor = tensor.repeat(num_repeats)

        # Trim the tensor to the target length
        return tiled_tensor[:target_length]

    return tensor

def zero_padding(tensor, target_length):
    tensor_length = len(tensor)

    # Truncate if the tensor is longer than the target length
    if tensor_length > target_length:
        return tensor[:target_length]

    # Zero padding if the tensor is shorter than the target length
    if tensor_length < target_length:
        padding_size = target_length - tensor_length
        zero_padding = torch.zeros(padding_size, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, zero_padding])

    return tensor

def process_cycle(waveform, sr):
    # Extract mel spectrogram
    mel_spec = MelSpectrogram(sample_rate=sr, hop_length=94, window_fn=torch.hamming_window)(waveform)

    # Convert to db scale
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    # Normalize mel spectrogram
    normalized_mel = (mel_spec_db - torch.mean(mel_spec_db)) / torch.std(mel_spec_db)

    return normalized_mel
    
def extract_cycles(dataset):
    wav_files = [f for f in os.listdir(dataset) if f.endswith('.wav')]
    cycles = []
    labels = []
    recording_ids = []

    for idx, wav_file in enumerate(wav_files):
        txt_file = wav_file.replace('.wav', '.txt')
        
        # Load audio file
        waveform, sr = torchaudio.load(os.path.join(dataset, wav_file))
        # Remove the channel dimension (mono sound)
        waveform = waveform.squeeze()

        # Read annotation
        with open(os.path.join(dataset, txt_file), 'r') as f:
            annotations = f.readlines()

        for line in annotations:
            start, end, crackles, wheezes = line.strip().split('\t')
            # Convert time to sample number
            start, end = float(start) * sr, float(end) * sr
            label = 1 if int(crackles) or int(wheezes) else 0
            
            # Extract and resample the cycle
            cycle = waveform[int(start):int(end)]
            resampler = Resample(orig_freq=sr, new_freq=4000)
            cycle = resampler(cycle)
            
            # Padding or truncating to 3 seconds
            excerpt_length = int(3 * 4000)

            cycle = wrap_padding(cycle, excerpt_length)
            cycle = process_cycle(cycle, 4000)

            recording_ids.append(idx)
            cycles.append(cycle)
            labels.append(label)

    return torch.tensor(recording_ids), torch.stack(cycles), torch.tensor(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing of ICBHI 2017 dataset for anomaly detection")
    parser.add_argument("--dataset", default="/Users/lukas/Documents/PARA/1 🚀 Projects/Bachelor Thesis/ICBHI_final_database", type=str, help="Directory where the original dataset is stored")
    parser.add_argument("--target", default="dataset.pt", type=str, help="Output path to store processed data")
    parser.add_argument("--recording_level", default=False, type=bool, help="Whether or not to split at recording level")
    args = parser.parse_args()

    recording_ids, cycles, labels = extract_cycles(args.dataset)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(recording_ids, cycles, labels, prevent_leakage=args.recording_level)

    torch.save({
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }, args.target)