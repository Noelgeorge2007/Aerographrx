"""
Time Difference of Arrival (TDOA) estimation for AeroGraphRX.

This module implements TDOA measurement simulation, maximum likelihood estimation,
Cramer-Rao lower bound computation, and localization performance metrics.

References:
    Usha A, Noel George. "AeroGraphRX: Graph Signal Processing for
    RF Signal Detection and Flight Tracking", 2024.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import inv
from typing import Tuple, Optional


def compute_tdoa(
    receiver_positions: np.ndarray,
    target_position: np.ndarray,
    noise_std: float = 1e-6,
    c: float = 3e8
) -> np.ndarray:
    """
    Simulate TDOA measurements from target to receivers.

    Computes time differences relative to a reference receiver with
    added Gaussian measurement noise.

    Parameters
    ----------
    receiver_positions : np.ndarray
        Receiver coordinates, shape (n_receivers, 3).
        Format: [x, y, z] in meters.
    target_position : np.ndarray
        Target position, shape (3,).
        Format: [x, y, z] in meters.
    noise_std : float, optional
        Standard deviation of TDOA measurement noise in seconds.
        Default: 1e-6
    c : float, optional
        Speed of light in m/s. Default: 3e8

    Returns
    -------
    tdoa : np.ndarray
        TDOA measurements, shape (n_receivers - 1,).
        TDOA[i] = (distance to receiver[i+1] - distance to receiver[0]) / c
        Relative to reference receiver (first one).

    Examples
    --------
    >>> receivers = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]])
    >>> target = np.array([500, 500, 0])
    >>> tdoa = compute_tdoa(receivers, target, noise_std=1e-9)
    >>> assert tdoa.shape == (2,)

    Notes
    -----
    TDOA is relative to reference receiver to avoid rank deficiency.
    Measurement noise is typically on order of 1-10 nanoseconds.
    """
    receiver_positions = np.asarray(receiver_positions, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)

    # Reference receiver is first one
    ref_pos = receiver_positions[0]

    # Compute distances from target to each receiver
    distances = np.linalg.norm(receiver_positions - target_position, axis=1)
    ref_distance = distances[0]

    # TDOA relative to reference receiver
    tdoa_clean = (distances[1:] - ref_distance) / c

    # Add measurement noise
    noise = np.random.randn(len(tdoa_clean)) * noise_std
    tdoa = tdoa_clean + noise

    return tdoa


def tdoa_residuals(
    target_pos: np.ndarray,
    receiver_positions: np.ndarray,
    tdoa_measurements: np.ndarray,
    c: float = 3e8
) -> np.ndarray:
    """
    Compute TDOA residuals for a candidate target position.

    Used internally by ML estimator.

    Parameters
    ----------
    target_pos : np.ndarray
        Candidate target position, shape (3,).
    receiver_positions : np.ndarray
        Receiver positions, shape (n_receivers, 3).
    tdoa_measurements : np.ndarray
        Measured TDOAs, shape (n_receivers - 1,).
    c : float, optional
        Speed of light. Default: 3e8

    Returns
    -------
    residuals : np.ndarray
        TDOA residuals, shape (n_receivers - 1,).
    """
    ref_pos = receiver_positions[0]
    target_pos = np.asarray(target_pos, dtype=np.float64).flatten()

    # Compute distances
    distances = np.linalg.norm(receiver_positions - target_pos, axis=1)
    ref_distance = distances[0]

    # TDOA predictions
    tdoa_pred = (distances[1:] - ref_distance) / c

    # Residuals
    residuals = tdoa_measurements - tdoa_pred

    return residuals


def tdoa_ml_estimate(
    tdoa_measurements: np.ndarray,
    receiver_positions: np.ndarray,
    noise_cov: np.ndarray,
    x0: Optional[np.ndarray] = None,
    c: float = 3e8
) -> Tuple[np.ndarray, float]:
    """
    Maximum likelihood estimation of target position from TDOA measurements.

    Solves the optimization problem (Eq. 13):
        x_hat = argmin_x ||H(x)||^2_Sigma
    where H(x) is the TDOA residual vector and Sigma is the noise covariance.

    Uses Gauss-Newton iterative refinement.

    Parameters
    ----------
    tdoa_measurements : np.ndarray
        TDOA measurements, shape (n_receivers - 1,).
    receiver_positions : np.ndarray
        Receiver positions, shape (n_receivers, 3).
    noise_cov : np.ndarray
        TDOA measurement noise covariance, shape (n_receivers - 1, n_receivers - 1).
    x0 : np.ndarray, optional
        Initial position estimate, shape (3,).
        If None, uses centroid of receivers.
    c : float, optional
        Speed of light. Default: 3e8

    Returns
    -------
    x_hat : np.ndarray
        Estimated target position, shape (3,).
    residual_norm : float
        Final weighted residual norm.

    Examples
    --------
    >>> receivers = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]])
    >>> target_true = np.array([500, 500, 0])
    >>> tdoa = compute_tdoa(receivers, target_true, noise_std=1e-9)
    >>> cov = np.eye(2) * (1e-9)**2
    >>> x_hat, resid = tdoa_ml_estimate(tdoa, receivers, cov)
    >>> error = np.linalg.norm(x_hat - target_true)
    >>> assert error < 100  # Should be close to true position

    Notes
    -----
    Uses scipy.optimize.least_squares for robustness.
    Noise covariance should be diagonal for i.i.d. measurements.
    """
    tdoa_measurements = np.asarray(tdoa_measurements, dtype=np.float64)
    receiver_positions = np.asarray(receiver_positions, dtype=np.float64)
    noise_cov = np.asarray(noise_cov, dtype=np.float64)

    # Default initial estimate: centroid of receivers
    if x0 is None:
        x0 = np.mean(receiver_positions, axis=0)
    else:
        x0 = np.asarray(x0, dtype=np.float64).flatten()

    # Compute Cholesky decomposition of covariance for weighting
    L = np.linalg.cholesky(noise_cov)
    L_inv = np.linalg.inv(L)

    # Weighted least squares with covariance weighting
    def weighted_residuals(x):
        res = tdoa_residuals(x, receiver_positions, tdoa_measurements, c=c)
        return L_inv @ res

    # Optimize
    result = least_squares(
        weighted_residuals,
        x0,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=1000
    )

    x_hat = result.x
    residual_norm = np.linalg.norm(result.fun)

    return x_hat, residual_norm


def compute_jacobian_tdoa(
    target_pos: np.ndarray,
    receiver_positions: np.ndarray,
    c: float = 3e8
) -> np.ndarray:
    """
    Compute Jacobian matrix of TDOA measurements.

    Used for Cramer-Rao bound computation.

    Parameters
    ----------
    target_pos : np.ndarray
        Target position, shape (3,).
    receiver_positions : np.ndarray
        Receiver positions, shape (n_receivers, 3).
    c : float, optional
        Speed of light. Default: 3e8

    Returns
    -------
    H : np.ndarray
        Jacobian matrix, shape (n_receivers - 1, 3).
        H[i, j] = d(TDOA_i) / d(target_j)
    """
    target_pos = np.asarray(target_pos, dtype=np.float64).flatten()
    receiver_positions = np.asarray(receiver_positions, dtype=np.float64)

    n_receivers = receiver_positions.shape[0]

    # Distances from target to receivers
    diffs = receiver_positions - target_pos  # (n_receivers, 3)
    distances = np.linalg.norm(diffs, axis=1)

    # Reference distance
    ref_distance = distances[0]

    # Jacobian: H[i, j] = (diffs[i+1, j] / dist[i+1] - diffs[0, j] / dist[0]) / c
    H = np.zeros((n_receivers - 1, 3), dtype=np.float64)
    for i in range(n_receivers - 1):
        idx = i + 1
        if distances[idx] > 0:
            H[i, :] = (diffs[idx] / distances[idx] - diffs[0] / ref_distance) / c

    return H


def compute_crlb(
    receiver_positions: np.ndarray,
    target_position: np.ndarray,
    noise_cov: np.ndarray,
    c: float = 3e8
) -> np.ndarray:
    """
    Compute Cramer-Rao Lower Bound on position estimation error.

    Implements Eq. 14-15 from paper:
        CRLB = (H^T * Sigma^{-1} * H)^{-1}
    where H is the Jacobian of TDOA measurements and Sigma is noise covariance.

    CRLB is a lower bound on the error covariance of any unbiased estimator.

    Parameters
    ----------
    receiver_positions : np.ndarray
        Receiver positions, shape (n_receivers, 3).
    target_position : np.ndarray
        Target position, shape (3,).
    noise_cov : np.ndarray
        TDOA measurement noise covariance, shape (n_receivers - 1, n_receivers - 1).
    c : float, optional
        Speed of light. Default: 3e8

    Returns
    -------
    crlb : np.ndarray
        Cramer-Rao lower bound on position error covariance, shape (3, 3).
        Diagonal contains bounds on variance of each coordinate.

    Examples
    --------
    >>> receivers = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]])
    >>> target = np.array([500, 500, 0])
    >>> cov = np.eye(2) * (1e-9)**2
    >>> crlb = compute_crlb(receivers, target, cov)
    >>> assert crlb.shape == (3, 3)
    >>> assert np.all(np.diag(crlb) >= 0)  # Positive variances

    Notes
    -----
    Fisher information matrix: F = H^T * Sigma^{-1} * H
    CRLB is the inverse of Fisher information.
    """
    receiver_positions = np.asarray(receiver_positions, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    noise_cov = np.asarray(noise_cov, dtype=np.float64)

    # Compute Jacobian
    H = compute_jacobian_tdoa(target_position, receiver_positions, c=c)

    # Compute Fisher information matrix
    # F = H^T * Sigma^{-1} * H
    noise_cov_inv = np.linalg.inv(noise_cov)
    F = H.T @ noise_cov_inv @ H

    # CRLB is inverse of Fisher information
    try:
        crlb = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        # Singular matrix, return pseudo-inverse
        crlb = np.linalg.pinv(F)

    return crlb


def compute_cep(
    position_errors: np.ndarray,
    percentile: float = 50.0
) -> float:
    """
    Compute Circular Error Probable (CEP) from position errors.

    CEP is the radius of a circle centered at the true position that
    contains a specified percentile of the error distribution.

    Parameters
    ----------
    position_errors : np.ndarray
        Position estimation errors, shape (n_trials,) or (n_trials, 3).
        If 2D, assumed to be [x_error, y_error, z_error] for each trial.
        If 1D, assumed to be already computed radial errors.
    percentile : float, optional
        Percentile level (default: 50, i.e., median).
        Common values: 50 (CEP), 68 (similar to 1-sigma), 95 (DRMS).

    Returns
    -------
    cep : float
        Circular error probable in same units as input positions.

    Examples
    --------
    >>> # Generate synthetic errors
    >>> errors = np.random.randn(1000, 3) * 10  # 10m std dev in each axis
    >>> cep = compute_cep(errors, percentile=50)
    >>> assert cep > 0
    >>> # CEP should increase with percentile
    >>> cep95 = compute_cep(errors, percentile=95)
    >>> assert cep95 > cep

    Notes
    -----
    For 2D errors, CEP ≈ 2.4477 * RMS for normal distribution.
    For 3D spherical errors, use spherical CEP formula.
    """
    position_errors = np.asarray(position_errors, dtype=np.float64)

    if position_errors.ndim == 2:
        # Compute radial error
        radial_errors = np.linalg.norm(position_errors, axis=1)
    else:
        # Already radial errors
        radial_errors = position_errors

    # Compute percentile
    cep = np.percentile(radial_errors, percentile)

    return float(cep)


def compute_localization_metrics(
    true_positions: np.ndarray,
    estimated_positions: np.ndarray
) -> dict:
    """
    Compute comprehensive localization performance metrics.

    Parameters
    ----------
    true_positions : np.ndarray
        True target positions, shape (n_targets, 3).
    estimated_positions : np.ndarray
        Estimated positions, shape (n_targets, 3).

    Returns
    -------
    metrics : dict
        Dictionary with keys:
            - 'mean_error': Mean Euclidean error
            - 'std_error': Standard deviation of errors
            - 'rmse': Root mean square error
            - 'max_error': Maximum error
            - 'cep50': 50th percentile (median) error
            - 'cep95': 95th percentile error
    """
    true_positions = np.asarray(true_positions, dtype=np.float64)
    estimated_positions = np.asarray(estimated_positions, dtype=np.float64)

    # Compute errors
    errors = estimated_positions - true_positions
    radial_errors = np.linalg.norm(errors, axis=1)

    metrics = {
        'mean_error': float(np.mean(radial_errors)),
        'std_error': float(np.std(radial_errors)),
        'rmse': float(np.sqrt(np.mean(radial_errors ** 2))),
        'max_error': float(np.max(radial_errors)),
        'cep50': float(np.percentile(radial_errors, 50)),
        'cep95': float(np.percentile(radial_errors, 95)),
    }

    return metrics
