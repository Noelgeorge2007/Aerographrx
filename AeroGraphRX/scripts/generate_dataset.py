#!/usr/bin/env python3
"""
Generate synthetic signal dataset for AeroGraphRX.

This script:
- Loads configuration from configs/default.yaml
- Generates ALL synthetic signals for all modulation types
- Computes STFT features
- Saves to data/synthetic/ as .npz files
- Saves train/val/test splits
- Logs seed per trial
- Prints summary statistics
"""
import numpy as np
import os
import yaml
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def load_config(config_path="configs/default.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def generate_modulated_signal(modulation_type, fs, duration, snr_db, seed):
    """
    Generate synthetic modulated signal.

    Args:
        modulation_type: str, type of modulation (AM, FM, BPSK, etc.)
        fs: float, sampling frequency (Hz)
        duration: float, signal duration (seconds)
        snr_db: float, signal-to-noise ratio (dB)
        seed: int, random seed for reproducibility

    Returns:
        signal: np.ndarray, complex-valued signal
    """
    np.random.seed(seed)
    t = np.arange(0, duration, 1/fs)
    carrier_freq = 100e3  # 100 kHz carrier

    # Generate information signal
    info_rate = 10e3  # 10 kHz info rate
    info = np.sin(2 * np.pi * info_rate * t)

    # Apply modulation
    if modulation_type == "AM":
        signal = (1 + 0.5*info) * np.cos(2*np.pi*carrier_freq*t)
    elif modulation_type == "FM":
        phase = 2*np.pi*carrier_freq*t + 2*np.pi*5e3*np.cumsum(info)/fs
        signal = np.cos(phase)
    elif modulation_type == "BPSK":
        bits = np.random.randint(0, 2, size=int(fs*duration/1000))
        symbol_rep = np.repeat(bits, int(fs/1000))
        signal = (2*symbol_rep - 1) * np.cos(2*np.pi*carrier_freq*t)
    elif modulation_type == "QPSK":
        bits_i = np.random.randint(0, 2, size=int(fs*duration/2000))
        bits_q = np.random.randint(0, 2, size=int(fs*duration/2000))
        symbol_rep_i = np.repeat(2*bits_i - 1, int(fs/1000))
        symbol_rep_q = np.repeat(2*bits_q - 1, int(fs/1000))
        signal = (symbol_rep_i * np.cos(2*np.pi*carrier_freq*t) +
                 symbol_rep_q * np.sin(2*np.pi*carrier_freq*t))
    elif modulation_type in ["8PSK", "16QAM", "64QAM"]:
        if modulation_type == "8PSK":
            n_symbols = 8
            symbols = np.exp(1j * 2*np.pi*np.arange(n_symbols)/n_symbols)
        elif modulation_type == "16QAM":
            const = [-3, -1, 1, 3]
            symbols = np.array([i + 1j*q for i in const for q in const])
            symbols /= np.sqrt(np.mean(np.abs(symbols)**2))
        else:  # 64QAM
            const = np.arange(-8, 8, 2) / 10
            symbols = np.array([i + 1j*q for i in const for q in const])
            symbols /= np.sqrt(np.mean(np.abs(symbols)**2))

        bits = np.random.randint(0, len(symbols), size=int(fs*duration/1000))
        symbol_rep = np.repeat(symbols[bits], int(fs/1000))
        signal = symbol_rep * np.exp(1j*2*np.pi*carrier_freq*t)
    elif modulation_type == "OFDM":
        n_subcarriers = 64
        n_symbols = int(fs*duration/10000)
        subcarrier_data = np.random.randint(0, 4, size=(n_subcarriers, n_symbols))
        ofdm_symbols = np.fft.ifft(subcarrier_data, axis=0)
        ofdm_signal = ofdm_symbols.flatten()[:len(t)]
        signal = ofdm_signal * np.exp(1j*2*np.pi*carrier_freq*t[:len(ofdm_signal)])
        signal = np.pad(signal, (0, len(t)-len(signal)), mode='constant')
    elif modulation_type == "GFSK":
        bits = np.random.randint(0, 2, size=int(fs*duration/1000))
        symbol_rep = np.repeat(2*bits - 1, int(fs/1000))
        freq_dev = 25e3
        phase = 2*np.pi*carrier_freq*t + 2*np.pi*freq_dev*np.cumsum(symbol_rep)/fs
        signal = np.cos(phase)
    elif modulation_type == "GMSK":
        bits = np.random.randint(0, 2, size=int(fs*duration/1000))
        symbol_rep = np.repeat(2*bits - 1, int(fs/1000))
        phase = 2*np.pi*carrier_freq*t + 2*np.pi*12.5e3*np.cumsum(symbol_rep)/fs
        signal = np.cos(phase)
    else:
        raise ValueError(f"Unknown modulation: {modulation_type}")

    # Add AWGN
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) +
                                       1j*np.random.randn(len(signal)))
    signal = signal + noise

    return signal[:len(t)]


def compute_stft_features(signal, config):
    """
    Compute STFT features from signal.

    Args:
        signal: np.ndarray, complex signal
        config: dict, configuration

    Returns:
        stft_mag: np.ndarray, STFT magnitude spectrogram
    """
    from scipy import signal as sp_signal

    n_fft = config['stft']['n_fft']
    hop_len = config['stft']['hop_length']
    window = config['stft']['window']

    f, t, Zxx = sp_signal.stft(signal, nfft=n_fft, nperseg=n_fft,
                               noverlap=n_fft-hop_len, window=window)
    stft_mag = np.abs(Zxx)

    return stft_mag


def generate_dataset(config, output_dir="data/synthetic"):
    """
    Generate complete dataset with train/val/test splits.

    Args:
        config: dict, configuration
        output_dir: str, output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    modulation_types = config['dataset']['modulation_types']
    samples_per_class = config['dataset']['samples_per_class']
    train_ratio = config['dataset']['train_ratio']
    val_ratio = config['dataset']['val_ratio']
    test_ratio = config['dataset']['test_ratio']

    fs = config['stft']['fs']
    duration = 1.0  # 1 second per sample
    snr_values = np.arange(config['simulation']['snr_range_db'][0],
                          config['simulation']['snr_range_db'][1]+1,
                          config['simulation']['snr_step_db'])

    random_seed_base = config['simulation']['random_seed_base']

    print("\n" + "="*70)
    print("DATASET GENERATION")
    print("="*70)
    print(f"Modulation types: {modulation_types}")
    print(f"Samples per class: {samples_per_class}")
    print(f"SNR range: {snr_values[0]} to {snr_values[-1]} dB (step {config['simulation']['snr_step_db']})")
    print(f"Train/Val/Test split: {train_ratio}/{val_ratio}/{test_ratio}")

    # Generate signals for each modulation type and SNR
    all_data = []
    all_labels = []
    all_snrs = []
    seed_log = []

    for class_idx, mod_type in enumerate(tqdm(modulation_types, desc="Modulation types")):
        for snr_idx, snr_db in enumerate(snr_values):
            for sample_idx in range(samples_per_class):
                seed = random_seed_base + class_idx*10000 + snr_idx*100 + sample_idx

                # Generate signal
                signal = generate_modulated_signal(mod_type, fs, duration, snr_db, seed)

                # Compute STFT features
                stft_mag = compute_stft_features(signal, config)

                # Flatten to vector
                feature_vec = stft_mag.flatten()

                all_data.append(feature_vec)
                all_labels.append(class_idx)
                all_snrs.append(snr_db)
                seed_log.append(seed)

    # Convert to arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_snrs = np.array(all_snrs)

    print(f"\nTotal samples: {len(all_data)}")
    print(f"Feature vector shape: {all_data[0].shape}")

    # Create train/val/test splits
    n_total = len(all_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    indices = np.arange(n_total)
    np.random.seed(random_seed_base)
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    # Save datasets
    np.savez(f"{output_dir}/train.npz",
             data=all_data[train_idx], labels=all_labels[train_idx],
             snrs=all_snrs[train_idx])
    np.savez(f"{output_dir}/val.npz",
             data=all_data[val_idx], labels=all_labels[val_idx],
             snrs=all_snrs[val_idx])
    np.savez(f"{output_dir}/test.npz",
             data=all_data[test_idx], labels=all_labels[test_idx],
             snrs=all_snrs[test_idx])

    # Save seed log
    with open(f"{output_dir}/seed_log.txt", "w") as f:
        f.write("sample_id,seed,class,snr_db\n")
        for i, seed in enumerate(seed_log):
            f.write(f"{i},{seed},{all_labels[i]},{all_snrs[i]}\n")

    print(f"\nTrain samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"\nDataset saved to {output_dir}/")
    print(f"Seed log saved to {output_dir}/seed_log.txt")

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    for class_idx, mod_type in enumerate(modulation_types):
        mask = all_labels == class_idx
        print(f"{mod_type:8s}: {mask.sum():5d} samples, "
              f"SNR range [{all_snrs[mask].min():.0f}, {all_snrs[mask].max():.0f}] dB, "
              f"mean feature norm: {np.mean([np.linalg.norm(all_data[i]) for i in np.where(mask)[0]]):.3f}")


if __name__ == "__main__":
    config = load_config()
    generate_dataset(config)
    print("\nDataset generation complete!")
