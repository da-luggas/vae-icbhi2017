import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, Resample
import matplotlib.pyplot as plt 

def wrap_padding(tensor, target_length):
    """
    Adjusts the length of the tensor to the target length. It truncates the tensor if it's longer than the target length,
    and applies wrap padding if it's shorter.
    """
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
    """
    Adjusts the length of the tensor to the target length. It truncates the tensor if it's longer than the target length,
    and applies zero padding if it's shorter.
    """
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

class RespiratorySoundDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.wav_files = [f for f in os.listdir(root_dir) if f.endswith('.wav')]
        self.breathing_cycles = []
        self.labels = []
        
        for wav_file in self.wav_files:
            txt_file = wav_file.replace('.wav', '.txt')
            
            # Load audio file
            waveform, sr = torchaudio.load(os.path.join(self.root_dir, wav_file))
            waveform = waveform.squeeze()  # Removing the channel dimension

            # Read annotation
            with open(os.path.join(self.root_dir, txt_file), 'r') as f:
                annotations = f.readlines()
            
            for line in annotations:
                start, end, crackles, wheezes = line.strip().split('\t')
                start, end = float(start) * sr, float(end) * sr  # Convert time to sample number
                label = 1 if int(crackles) or int(wheezes) else 0
                
                # Extracting and resampling the cycle
                cycle = waveform[int(start):int(end)]
                resampler = Resample(orig_freq=sr, new_freq=4000)
                cycle = resampler(cycle)
                
                # Padding or truncating to 3 seconds
                excerpt_length = int(3 * 4000)

                cycle = wrap_padding(cycle, excerpt_length)
                
                self.breathing_cycles.append(cycle)
                self.labels.append(label)

    def __len__(self):
        return len(self.breathing_cycles)

    def __getitem__(self, idx):
        cycle = self.breathing_cycles[idx]
        
        # Extracting mel spectrogram
        mel_extractor = MelSpectrogram(sample_rate=4000, hop_length=94, window_fn=torch.hamming_window)
        mel_spec = mel_extractor(cycle)
        # Convert to db scale
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Normalize mel spectrogram using mean normalization
        normalized_mel = (mel_spec_db - torch.mean(mel_spec_db)) / torch.std(mel_spec_db)

        # Unsqueeze normalized mel to include one channel
        normalized_mel = normalized_mel.unsqueeze(0)
        
        return normalized_mel, self.labels[idx]