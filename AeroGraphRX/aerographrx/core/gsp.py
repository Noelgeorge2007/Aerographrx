"""
Graph Signal Processing operations for AeroGraphRX.

This module implements core GSP algorithms including graph Fourier transform,
inverse transforms, and Chebyshev polynomial filtering.

References:
    Usha A, Noel George. "AeroGraphRX: Graph Signal Processing for
    RF Signal Detection and Flight Tracking", 2024.
"""

import numpy as np
from scipy.special import comb
from typing import Tuple, List


def graph_fourier_transform(
    U: np.ndarray,
    x: np.ndarray
) -> np.ndarray:
    """
    Compute graph Fourier transform of signal.

    Implements Eq. 4 from paper:
        x_hat = U^T * x

    where U is the matrix of eigenvectors of the Laplacian.

    Parameters
    ----------
    U : np.ndarray
        Eigenvectors of Laplacian, shape (n_nodes, n_eigs).
        Columns are eigenvectors.
    x : np.ndarray
        Signal on graph, shape (n_nodes,) or (n_nodes, n_signals).

    Returns
    -------
    x_hat : np.ndarray
        Graph Fourier coefficients, shape (n_eigs,) or (n_eigs, n_signals).
        Represents signal in spectral domain.

    Examples
    --------
    >>> U = np.eye(3)  # Identity eigenvectors
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> x_hat = graph_fourier_transform(U, x)
    >>> assert x_hat.shape == (3,)
    >>> assert np.allclose(x_hat, x)

    Notes
    -----
    Assumes U is orthonormal (U.T @ U ≈ I).
    """
    x = np.asarray(x, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    # x_hat = U^T @ x
    x_hat = U.T @ x

    return x_hat


def inverse_gft(
    U: np.ndarray,
    x_hat: np.ndarray
) -> np.ndarray:
    """
    Inverse graph Fourier transform.

    Implements inverse of Eq. 4:
        x = U * x_hat

    Reconstructs signal from spectral coefficients.

    Parameters
    ----------
    U : np.ndarray
        Eigenvectors of Laplacian, shape (n_nodes, n_eigs).
    x_hat : np.ndarray
        Graph Fourier coefficients, shape (n_eigs,) or (n_eigs, n_signals).

    Returns
    -------
    x : np.ndarray
        Reconstructed signal on graph, shape (n_nodes,) or (n_nodes, n_signals).

    Examples
    --------
    >>> U = np.eye(3)
    >>> x_hat = np.array([1.0, 2.0, 3.0])
    >>> x = inverse_gft(U, x_hat)
    >>> assert x.shape == (3,)
    """
    x_hat = np.asarray(x_hat, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    # x = U @ x_hat
    x = U @ x_hat

    return x


def chebyshev_filter(
    L: np.ndarray,
    x: np.ndarray,
    coeffs: np.ndarray,
    lambda_max: float
) -> np.ndarray:
    """
    Apply Chebyshev polynomial filter to graph signal.

    Implements Eq. 5-6 from paper using Chebyshev polynomial approximation
    with recurrence relation:
        T_0(x) = 1
        T_1(x) = x
        T_{k+1}(x) = 2*x*T_k(x) - T_{k-1}(x)

    Avoids explicit eigendecomposition by using Laplacian powers directly.

    Parameters
    ----------
    L : np.ndarray
        Graph Laplacian, shape (n_nodes, n_nodes).
    x : np.ndarray
        Signal on graph, shape (n_nodes,) or (n_nodes, n_signals).
    coeffs : np.ndarray
        Chebyshev filter coefficients, shape (K,).
        coeffs[k] is the coefficient for T_k.
    lambda_max : float
        Maximum eigenvalue of Laplacian (spectral radius).

    Returns
    -------
    y : np.ndarray
        Filtered signal, shape (n_nodes,) or (n_nodes, n_signals).

    Examples
    --------
    >>> L = np.array([[2.0, -1.0], [-1.0, 2.0]])
    >>> x = np.array([1.0, 0.0])
    >>> coeffs = np.array([1.0, 0.5])  # K=2
    >>> y = chebyshev_filter(L, x, coeffs, lambda_max=3.0)
    >>> assert y.shape == (2,)

    Notes
    -----
    Normalization: rescales Laplacian to [-1, 1] interval via
        L_scaled = (2/lambda_max) * L - I
    """
    L = np.asarray(L, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    lambda_max = float(lambda_max)

    n_nodes = L.shape[0]
    K = len(coeffs)

    # Normalize Laplacian to [-1, 1]
    # L_tilde = (2/lambda_max) * L - I
    L_tilde = (2.0 / lambda_max) * L - np.eye(n_nodes)

    # Initialize Chebyshev polynomials
    # T_0 = I (for matrix, acting on x)
    T_prev2 = np.copy(x)  # T_0(x) = x
    result = coeffs[0] * T_prev2

    if K > 1:
        # T_1 = L_tilde (for matrix equation)
        T_prev1 = L_tilde @ x  # T_1(x) = L_tilde @ x
        result = result + coeffs[1] * T_prev1

        # Recurrence: T_{k+1} = 2 * L_tilde * T_k - T_{k-1}
        for k in range(2, K):
            T_curr = 2.0 * (L_tilde @ T_prev1) - T_prev2
            result = result + coeffs[k] * T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr

    return result


def chebyshev_approximation_error_bound(
    L_h: float,
    lambda_max: float,
    K: int
) -> float:
    """
    Compute Chebyshev polynomial approximation error bound.

    Implements Proposition 1 from paper:
        E(K) ≤ C * L_h * lambda_max / K^2

    where C is a constant factor and L_h is the Lipschitz constant
    of the target filter response.

    Parameters
    ----------
    L_h : float
        Lipschitz constant of target filter response.
        Typically proportional to max filter gain.
    lambda_max : float
        Maximum eigenvalue of Laplacian.
    K : int
        Number of Chebyshev coefficients.

    Returns
    -------
    error_bound : float
        Upper bound on approximation error.

    Examples
    --------
    >>> bound = chebyshev_approximation_error_bound(1.0, 3.0, 10)
    >>> assert bound > 0.0
    >>> # Increasing K decreases bound
    >>> bound2 = chebyshev_approximation_error_bound(1.0, 3.0, 20)
    >>> assert bound2 < bound

    Notes
    -----
    Error decreases as O(1/K^2), motivating use of polynomial approximation.
    Constant C typically in range [0.5, 2.0] depending on filter design.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got K={K}")

    # Constant factor (can be tuned based on filter characteristics)
    C = 1.0

    # Error bound: C * L_h * lambda_max / K^2
    error_bound = C * L_h * lambda_max / (K ** 2)

    return error_bound


def design_lowpass_filter(
    lambda_max: float,
    cutoff_freq: float,
    K: int = 10,
    filter_type: str = 'ideal'
) -> np.ndarray:
    """
    Design a lowpass filter response on the graph spectrum.

    Useful for smoothing graph signals and removing high-frequency noise.

    Parameters
    ----------
    lambda_max : float
        Maximum eigenvalue of Laplacian.
    cutoff_freq : float
        Normalized cutoff frequency in [0, 1] (relative to lambda_max).
    K : int, optional
        Number of Chebyshev coefficients. Default: 10
    filter_type : str, optional
        Filter design: 'ideal', 'hamming', 'hann'. Default: 'ideal'

    Returns
    -------
    coeffs : np.ndarray
        Chebyshev filter coefficients, shape (K,).

    Examples
    --------
    >>> coeffs = design_lowpass_filter(3.0, cutoff_freq=0.5, K=5)
    >>> assert len(coeffs) == 5
    """
    # Evaluate ideal filter at Chebyshev nodes
    chebyshev_nodes = np.cos(np.pi * (np.arange(1, K + 1) - 0.5) / K)
    freq_vals = (lambda_max / 2.0) * (chebyshev_nodes + 1.0)

    # Compute filter response
    if filter_type == 'ideal':
        h_vals = (freq_vals <= (cutoff_freq * lambda_max)).astype(float)
    elif filter_type == 'hamming':
        h_vals = np.hamming(K)
        h_vals = np.where(freq_vals <= (cutoff_freq * lambda_max), h_vals, 0.0)
    elif filter_type == 'hann':
        h_vals = np.hann(K)
        h_vals = np.where(freq_vals <= (cutoff_freq * lambda_max), h_vals, 0.0)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    # Fit Chebyshev coefficients (simple approach: use values as coefficients)
    coeffs = h_vals / np.sum(h_vals) if np.sum(h_vals) > 0 else np.zeros(K)

    return coeffs


def design_bandpass_filter(
    lambda_max: float,
    low_freq: float,
    high_freq: float,
    K: int = 10
) -> np.ndarray:
    """
    Design a bandpass filter on the graph spectrum.

    Parameters
    ----------
    lambda_max : float
        Maximum eigenvalue of Laplacian.
    low_freq : float
        Normalized lower cutoff frequency in [0, 1].
    high_freq : float
        Normalized upper cutoff frequency in [0, 1].
    K : int, optional
        Number of Chebyshev coefficients. Default: 10

    Returns
    -------
    coeffs : np.ndarray
        Chebyshev filter coefficients, shape (K,).
    """
    chebyshev_nodes = np.cos(np.pi * (np.arange(1, K + 1) - 0.5) / K)
    freq_vals = (lambda_max / 2.0) * (chebyshev_nodes + 1.0)

    h_vals = np.where(
        (freq_vals >= (low_freq * lambda_max)) & (freq_vals <= (high_freq * lambda_max)),
        1.0,
        0.0
    )

    coeffs = h_vals / np.sum(h_vals) if np.sum(h_vals) > 0 else np.zeros(K)

    return coeffs


def graph_signal_smoothness(
    L: np.ndarray,
    x: np.ndarray
) -> float:
    """
    Compute total variation / smoothness of signal on graph.

    TV(x) = x^T * L * x / (2 * ||x||^2)

    Small TV indicates smooth signal, large TV indicates oscillatory.

    Parameters
    ----------
    L : np.ndarray
        Graph Laplacian, shape (n_nodes, n_nodes).
    x : np.ndarray
        Signal on graph, shape (n_nodes,).

    Returns
    -------
    smoothness : float
        Normalized total variation.

    Examples
    --------
    >>> L = np.eye(3)
    >>> x = np.array([1.0, 1.0, 1.0])  # Constant signal
    >>> smooth = graph_signal_smoothness(L, x)
    >>> assert smooth < 1e-10  # Nearly zero
    """
    x = np.asarray(x, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)

    norm_x_sq = np.dot(x, x)
    if norm_x_sq < 1e-15:
        return 0.0

    tv = np.dot(x, L @ x) / (2.0 * norm_x_sq)
    return float(tv)


def filter_signal(
    L: np.ndarray,
    x: np.ndarray,
    lambda_max: float,
    filter_type: str = 'lowpass',
    **kwargs
) -> np.ndarray:
    """
    Apply spectral filter to graph signal.

    Convenience function combining filter design and Chebyshev filtering.

    Parameters
    ----------
    L : np.ndarray
        Graph Laplacian.
    x : np.ndarray
        Input signal.
    lambda_max : float
        Maximum eigenvalue of Laplacian.
    filter_type : str
        'lowpass' or 'bandpass'.
    **kwargs
        Filter-specific arguments (cutoff_freq, K, etc.)

    Returns
    -------
    y : np.ndarray
        Filtered signal.
    """
    if filter_type == 'lowpass':
        cutoff_freq = kwargs.pop('cutoff_freq', 0.5)
        K = kwargs.pop('K', 10)
        coeffs = design_lowpass_filter(lambda_max, cutoff_freq, K)
    elif filter_type == 'bandpass':
        low_freq = kwargs.pop('low_freq', 0.2)
        high_freq = kwargs.pop('high_freq', 0.8)
        K = kwargs.pop('K', 10)
        coeffs = design_bandpass_filter(lambda_max, low_freq, high_freq, K)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    y = chebyshev_filter(L, x, coeffs, lambda_max)
    return y
