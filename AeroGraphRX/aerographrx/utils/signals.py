"""
Signal synthesis and processing utilities for AeroGraphRX.

This module provides functions to generate various RF modulation schemes
(AM, FM, PSK, QAM, OFDM, etc.) with realistic modulation characteristics
and AWGN noise.

References:
    Usha A, Noel George. "AeroGraphRX: Graph Signal Processing for
    RF Signal Detection and Flight Tracking", 2024.
"""

import numpy as np
from scipy.signal import hilbert
from typing import Callable, Dict, Tuple


def add_awgn(
    signal: np.ndarray,
    snr_db: float
) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) to signal.

    Computes noise power from desired SNR and adds white Gaussian noise.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, shape (n_samples,).
    snr_db : float
        Signal-to-noise ratio in decibels.
        Relationship: SNR_db = 10 * log10(P_signal / P_noise)

    Returns
    -------
    noisy_signal : np.ndarray
        Signal with added AWGN, shape (n_samples,).

    Examples
    --------
    >>> s = np.ones(1000)
    >>> s_noisy = add_awgn(s, snr_db=10)
    >>> assert s_noisy.shape == (1000,)
    >>> # Check SNR
    >>> power_signal = np.mean(s ** 2)
    >>> power_noise = np.mean((s_noisy - s) ** 2)
    >>> snr_actual = 10 * np.log10(power_signal / power_noise)
    >>> assert abs(snr_actual - 10) < 0.5
    """
    signal = np.asarray(signal, dtype=np.complex128)

    # Compute signal power
    power_signal = np.mean(np.abs(signal) ** 2)

    # Compute noise power from SNR
    snr_linear = 10.0 ** (snr_db / 10.0)
    power_noise = power_signal / snr_linear

    # Generate white Gaussian noise
    sigma = np.sqrt(power_noise / 2.0)
    noise = sigma * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

    # Add noise to signal
    noisy_signal = signal + noise

    return noisy_signal


def generate_am(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6
) -> np.ndarray:
    """
    Generate AM (Amplitude Modulation) modulated signal.

    Message: low-frequency sinusoid at f_m = 1 kHz
    Carrier: sinusoid at f_c = fs/4

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    snr_db : float
        Signal-to-noise ratio in dB.
    fs : float, optional
        Sampling rate in Hz. Default: 2.4e6

    Returns
    -------
    signal : np.ndarray
        Complex baseband AM signal, shape (n_samples,).

    Examples
    --------
    >>> s = generate_am(1000, snr_db=15)
    >>> assert len(s) == 1000
    """
    t = np.arange(n_samples) / fs

    # Message signal: low frequency modulation
    f_m = 1000.0  # 1 kHz message frequency
    message = np.sin(2.0 * np.pi * f_m * t)

    # Carrier
    f_c = fs / 4.0  # Carrier at fs/4
    carrier = np.exp(1j * 2.0 * np.pi * f_c * t)

    # AM modulation: (1 + m(t)) * c(t)
    modulation_index = 0.8  # Typical value
    modulated = (1.0 + modulation_index * message) * carrier

    # Normalize
    modulated = modulated / np.max(np.abs(modulated))

    # Add noise
    signal = add_awgn(modulated, snr_db)

    return signal


def generate_fm(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6
) -> np.ndarray:
    """
    Generate FM (Frequency Modulation) modulated signal.

    Message: sinusoid at f_m = 5 kHz
    Carrier: at f_c = fs/4
    Frequency deviation: 50 kHz

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    snr_db : float
        Signal-to-noise ratio in dB.
    fs : float, optional
        Sampling rate in Hz. Default: 2.4e6

    Returns
    -------
    signal : np.ndarray
        Complex baseband FM signal, shape (n_samples,).
    """
    t = np.arange(n_samples) / fs

    # Message signal
    f_m = 5000.0  # 5 kHz
    message = np.sin(2.0 * np.pi * f_m * t)

    # FM: phase is modulated by message
    f_dev = 50000.0  # Frequency deviation
    phase = 2.0 * np.pi * f_dev * np.cumsum(message) / fs

    # Carrier at fs/4
    f_c = fs / 4.0
    carrier_phase = 2.0 * np.pi * f_c * t

    # FM signal
    modulated = np.exp(1j * (carrier_phase + phase))

    # Normalize
    modulated = modulated / np.max(np.abs(modulated))

    # Add noise
    signal = add_awgn(modulated, snr_db)

    return signal


def generate_bpsk(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6,
    baud_rate: float = 100000.0
) -> np.ndarray:
    """
    Generate BPSK (Binary Phase Shift Keying) modulated signal.

    Bits: random binary sequence
    Symbol rate: baud_rate
    Carrier: fs/4

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    snr_db : float
        Signal-to-noise ratio in dB.
    fs : float, optional
        Sampling rate in Hz. Default: 2.4e6
    baud_rate : float, optional
        Symbol rate in symbols/sec. Default: 100e3

    Returns
    -------
    signal : np.ndarray
        Complex BPSK signal, shape (n_samples,).
    """
    t = np.arange(n_samples) / fs

    # Generate random bits
    n_symbols = int(n_samples * baud_rate / fs)
    bits = np.random.randint(0, 2, n_symbols)

    # BPSK constellation: {+1, -1}
    symbols = 2.0 * bits - 1.0  # Convert 0/1 to -1/+1

    # Upsample to sample rate
    samples_per_symbol = int(fs / baud_rate)
    modulated = np.repeat(symbols, samples_per_symbol)[:n_samples]

    # Apply raised cosine pulse shaping (simplified as rectangular)
    # Modulate to carrier
    f_c = fs / 4.0
    carrier = np.exp(1j * 2.0 * np.pi * f_c * t)
    modulated_complex = modulated * carrier

    # Normalize
    modulated_complex = modulated_complex / np.max(np.abs(modulated_complex))

    # Add noise
    signal = add_awgn(modulated_complex, snr_db)

    return signal


def generate_qpsk(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6,
    baud_rate: float = 100000.0
) -> np.ndarray:
    """
    Generate QPSK (Quadrature PSK) modulated signal.

    QPSK constellation: {+1+1j, +1-1j, -1+1j, -1-1j} / sqrt(2)
    Symbol rate: baud_rate

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    snr_db : float
        Signal-to-noise ratio in dB.
    fs : float, optional
        Sampling rate in Hz. Default: 2.4e6
    baud_rate : float, optional
        Symbol rate. Default: 100e3

    Returns
    -------
    signal : np.ndarray
        Complex QPSK signal, shape (n_samples,).
    """
    t = np.arange(n_samples) / fs

    # Generate random symbols
    n_symbols = int(n_samples * baud_rate / fs)
    bits_i = np.random.randint(0, 2, n_symbols)
    bits_q = np.random.randint(0, 2, n_symbols)

    # QPSK constellation
    i_symbols = 2.0 * bits_i - 1.0
    q_symbols = 2.0 * bits_q - 1.0
    symbols = (i_symbols + 1j * q_symbols) / np.sqrt(2.0)

    # Upsample
    samples_per_symbol = int(fs / baud_rate)
    modulated = np.repeat(symbols, samples_per_symbol)[:n_samples]

    # Modulate to carrier
    f_c = fs / 4.0
    carrier = np.exp(1j * 2.0 * np.pi * f_c * t)
    modulated_complex = modulated * carrier

    # Normalize
    modulated_complex = modulated_complex / np.max(np.abs(modulated_complex))

    # Add noise
    signal = add_awgn(modulated_complex, snr_db)

    return signal


def generate_8psk(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6,
    baud_rate: float = 100000.0
) -> np.ndarray:
    """
    Generate 8-PSK modulated signal.

    8 equally-spaced constellation points on unit circle.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    snr_db : float
        SNR in dB.
    fs : float, optional
        Sampling rate. Default: 2.4e6
    baud_rate : float, optional
        Symbol rate. Default: 100e3

    Returns
    -------
    signal : np.ndarray
        Complex 8-PSK signal.
    """
    t = np.arange(n_samples) / fs

    # Generate random symbols
    n_symbols = int(n_samples * baud_rate / fs)
    symbol_indices = np.random.randint(0, 8, n_symbols)

    # 8-PSK constellation: angles 0, pi/4, pi/2, 3pi/4, pi, ...
    angles = 2.0 * np.pi * symbol_indices / 8.0
    symbols = np.exp(1j * angles)

    # Upsample
    samples_per_symbol = int(fs / baud_rate)
    modulated = np.repeat(symbols, samples_per_symbol)[:n_samples]

    # Modulate to carrier
    f_c = fs / 4.0
    carrier = np.exp(1j * 2.0 * np.pi * f_c * t)
    modulated_complex = modulated * carrier

    # Normalize
    modulated_complex = modulated_complex / np.max(np.abs(modulated_complex))

    # Add noise
    signal = add_awgn(modulated_complex, snr_db)

    return signal


def generate_16qam(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6,
    baud_rate: float = 100000.0
) -> np.ndarray:
    """
    Generate 16-QAM (Quadrature Amplitude Modulation) signal.

    4x4 grid constellation: I, Q in {-3, -1, +1, +3}.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    snr_db : float
        SNR in dB.
    fs : float, optional
        Sampling rate. Default: 2.4e6
    baud_rate : float, optional
        Symbol rate. Default: 100e3

    Returns
    -------
    signal : np.ndarray
        Complex 16-QAM signal.
    """
    t = np.arange(n_samples) / fs

    # Generate random symbols
    n_symbols = int(n_samples * baud_rate / fs)
    symbol_indices = np.random.randint(0, 16, n_symbols)

    # 16-QAM constellation
    constellation = []
    for i in range(4):
        for q in range(4):
            constellation.append((2.0 * i - 3.0) + 1j * (2.0 * q - 3.0))
    constellation = np.array(constellation) / np.sqrt(10.0)  # Normalize

    symbols = constellation[symbol_indices]

    # Upsample
    samples_per_symbol = int(fs / baud_rate)
    modulated = np.repeat(symbols, samples_per_symbol)[:n_samples]

    # Modulate to carrier
    f_c = fs / 4.0
    carrier = np.exp(1j * 2.0 * np.pi * f_c * t)
    modulated_complex = modulated * carrier

    # Normalize
    modulated_complex = modulated_complex / np.max(np.abs(modulated_complex))

    # Add noise
    signal = add_awgn(modulated_complex, snr_db)

    return signal


def generate_64qam(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6,
    baud_rate: float = 100000.0
) -> np.ndarray:
    """
    Generate 64-QAM signal.

    8x8 grid constellation.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    snr_db : float
        SNR in dB.
    fs : float, optional
        Sampling rate. Default: 2.4e6
    baud_rate : float, optional
        Symbol rate. Default: 100e3

    Returns
    -------
    signal : np.ndarray
        Complex 64-QAM signal.
    """
    t = np.arange(n_samples) / fs

    # Generate random symbols
    n_symbols = int(n_samples * baud_rate / fs)
    symbol_indices = np.random.randint(0, 64, n_symbols)

    # 64-QAM constellation
    constellation = []
    for i in range(8):
        for q in range(8):
            constellation.append((2.0 * i - 7.0) + 1j * (2.0 * q - 7.0))
    constellation = np.array(constellation) / np.sqrt(42.0)  # Normalize

    symbols = constellation[symbol_indices]

    # Upsample
    samples_per_symbol = int(fs / baud_rate)
    modulated = np.repeat(symbols, samples_per_symbol)[:n_samples]

    # Modulate to carrier
    f_c = fs / 4.0
    carrier = np.exp(1j * 2.0 * np.pi * f_c * t)
    modulated_complex = modulated * carrier

    # Normalize
    modulated_complex = modulated_complex / np.max(np.abs(modulated_complex))

    # Add noise
    signal = add_awgn(modulated_complex, snr_db)

    return signal


def generate_ofdm(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6,
    n_subcarriers: int = 64,
    cp_length: int = 16
) -> np.ndarray:
    """
    Generate OFDM (Orthogonal Frequency Division Multiplexing) signal.

    Multiple subcarriers with cyclic prefix.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    snr_db : float
        SNR in dB.
    fs : float, optional
        Sampling rate. Default: 2.4e6
    n_subcarriers : int, optional
        Number of subcarriers. Default: 64
    cp_length : int, optional
        Cyclic prefix length. Default: 16

    Returns
    -------
    signal : np.ndarray
        Complex OFDM signal.
    """
    # Generate random QAM symbols on subcarriers
    n_ofdm_symbols = (n_samples // (n_subcarriers + cp_length)) + 1
    data_symbols = (np.random.randn(n_subcarriers, n_ofdm_symbols) +
                    1j * np.random.randn(n_subcarriers, n_ofdm_symbols)) / np.sqrt(2.0)

    # IFFT to get time-domain OFDM signal
    ofdm_time = np.fft.ifft(data_symbols, axis=0)

    # Add cyclic prefix
    ofdm_with_cp = np.vstack([ofdm_time[-cp_length:, :], ofdm_time])

    # Flatten to 1D
    signal_time = ofdm_with_cp.flatten()[:n_samples]

    # Modulate to carrier
    t = np.arange(n_samples) / fs
    f_c = fs / 4.0
    carrier = np.exp(1j * 2.0 * np.pi * f_c * t)
    modulated = signal_time * carrier

    # Normalize
    modulated = modulated / np.max(np.abs(modulated))

    # Add noise
    signal = add_awgn(modulated, snr_db)

    return signal


def generate_gfsk(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6,
    baud_rate: float = 100000.0,
    bt: float = 0.3
) -> np.ndarray:
    """
    Generate GFSK (Gaussian Frequency Shift Keying) modulated signal.

    GFSK with Gaussian pulse shaping (BT product).

    Parameters
    ----------
    n_samples : int
        Number of samples.
    snr_db : float
        SNR in dB.
    fs : float, optional
        Sampling rate. Default: 2.4e6
    baud_rate : float, optional
        Baud rate. Default: 100e3
    bt : float, optional
        BT product (bandwidth-time product). Default: 0.3

    Returns
    -------
    signal : np.ndarray
        Complex GFSK signal.
    """
    t = np.arange(n_samples) / fs

    # Generate random bits
    n_symbols = int(n_samples * baud_rate / fs)
    bits = np.random.randint(0, 2, n_symbols)

    # GFSK: frequency deviation based on bits
    f_dev = baud_rate * bt / 2.0
    freq_dev = np.where(bits == 0, -f_dev, f_dev)

    # Repeat for samples
    samples_per_symbol = int(fs / baud_rate)
    freq_dev_samples = np.repeat(freq_dev, samples_per_symbol)[:n_samples]

    # Integrate frequency to get phase
    phase = 2.0 * np.pi * np.cumsum(freq_dev_samples) / fs

    # Modulate
    modulated = np.exp(1j * phase)

    # Normalize
    modulated = modulated / np.max(np.abs(modulated))

    # Add noise
    signal = add_awgn(modulated, snr_db)

    return signal


def generate_gmsk(
    n_samples: int,
    snr_db: float,
    fs: float = 2.4e6,
    baud_rate: float = 100000.0
) -> np.ndarray:
    """
    Generate GMSK (Gaussian Minimum Shift Keying) modulated signal.

    Special case of GFSK with BT = 0.3 (standard for GSM).

    Parameters
    ----------
    n_samples : int
        Number of samples.
    snr_db : float
        SNR in dB.
    fs : float, optional
        Sampling rate. Default: 2.4e6
    baud_rate : float, optional
        Baud rate. Default: 100e3

    Returns
    -------
    signal : np.ndarray
        Complex GMSK signal.
    """
    # GMSK is GFSK with BT = 0.3
    return generate_gfsk(n_samples, snr_db, fs, baud_rate, bt=0.3)


def compute_stft(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 1024,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT) of signal.

    Used for time-frequency feature extraction.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, shape (n_samples,).
    n_fft : int, optional
        FFT size. Default: 2048
    hop_length : int, optional
        Hop length between frames. Default: 1024
    window : str, optional
        Window function: 'hann', 'hamming', 'blackman'. Default: 'hann'

    Returns
    -------
    stft_matrix : np.ndarray
        STFT magnitude, shape (n_freqs, n_frames).
    freqs : np.ndarray
        Frequency bins, shape (n_freqs,).
    times : np.ndarray
        Time bins, shape (n_frames,).

    Examples
    --------
    >>> signal = np.random.randn(10000)
    >>> mag, freqs, times = compute_stft(signal)
    >>> assert mag.shape[0] == 1025  # n_fft // 2 + 1
    """
    signal = np.asarray(signal, dtype=np.complex128)

    # Window function
    if window == 'hann':
        win = np.hanning(n_fft)
    elif window == 'hamming':
        win = np.hamming(n_fft)
    elif window == 'blackman':
        win = np.blackman(n_fft)
    else:
        win = np.ones(n_fft)

    # Compute STFT
    n_frames = (len(signal) - n_fft) // hop_length + 1
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)

    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft] * win
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        stft_matrix[:, i] = np.fft.rfft(frame)

    # Frequency and time bins
    freqs = np.fft.rfftfreq(n_fft)
    times = np.arange(n_frames) * hop_length

    return np.abs(stft_matrix), freqs, times


# Dictionary mapping modulation names to generator functions
MODULATION_MAP: Dict[str, Callable] = {
    'AM': generate_am,
    'FM': generate_fm,
    'BPSK': generate_bpsk,
    'QPSK': generate_qpsk,
    '8PSK': generate_8psk,
    '16QAM': generate_16qam,
    '64QAM': generate_64qam,
    'OFDM': generate_ofdm,
    'GFSK': generate_gfsk,
    'GMSK': generate_gmsk,
}
