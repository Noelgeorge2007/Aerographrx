"""
Graph construction and spectral analysis for AeroGraphRX.

This module implements core graph signal processing operations including
spatial, spectral, and temporal adjacency matrices, Laplacian computation,
and eigendecomposition.

References:
    Usha A, Noel George. "AeroGraphRX: Graph Signal Processing for
    RF Signal Detection and Flight Tracking", 2024.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional, Union


def build_spatial_adjacency(
    positions: np.ndarray,
    sigma_s: float
) -> np.ndarray:
    """
    Build spatial adjacency matrix using Gaussian RBF kernel.

    Implements Eq. 6 from paper:
        W_ij = exp(-||p_i - p_j||^2 / (2 * sigma_s^2))

    Parameters
    ----------
    positions : np.ndarray
        Receiver positions, shape (n_nodes, n_dims).
        For 3D: shape (n_nodes, 3) with (x, y, z) coordinates.
    sigma_s : float
        Spatial scale parameter. Controls kernel width.

    Returns
    -------
    W_spatial : np.ndarray
        Spatial adjacency matrix, shape (n_nodes, n_nodes).
        Symmetric, non-negative, diagonal = 1.0.

    Examples
    --------
    >>> positions = np.array([[0, 0], [1, 1], [2, 2]])
    >>> W = build_spatial_adjacency(positions, sigma_s=1.0)
    >>> assert W.shape == (3, 3)
    >>> assert np.allclose(W, W.T)  # Symmetry
    """
    # Compute pairwise squared Euclidean distances
    distances_sq = cdist(positions, positions, metric='sqeuclidean')

    # Apply Gaussian kernel
    W_spatial = np.exp(-distances_sq / (2.0 * sigma_s ** 2))

    return W_spatial


def build_spectral_adjacency(
    features: np.ndarray,
    sigma_f: float,
    epsilon: float = 1e-3
) -> np.ndarray:
    """
    Build spectral adjacency matrix using feature similarity.

    Implements Eq. 7 from paper. Computes Gaussian kernel on feature
    vectors with sparsity threshold epsilon.

    Parameters
    ----------
    features : np.ndarray
        Feature vectors, shape (n_nodes, n_features).
        Typically frequency-domain or time-frequency features.
    sigma_f : float
        Feature scale parameter. Controls kernel width.
    epsilon : float, optional
        Sparsity threshold. Adjacencies below epsilon are zeroed out.
        Default: 1e-3

    Returns
    -------
    W_spectral : np.ndarray
        Spectral adjacency matrix, shape (n_nodes, n_nodes).
        Sparse, symmetric, diagonal = 1.0 (before thresholding).

    Examples
    --------
    >>> features = np.random.randn(5, 10)
    >>> W = build_spectral_adjacency(features, sigma_f=1.0, epsilon=0.01)
    >>> assert W.shape == (5, 5)
    >>> assert np.allclose(W, W.T)
    """
    # Compute pairwise squared Euclidean distances in feature space
    distances_sq = cdist(features, features, metric='sqeuclidean')

    # Apply Gaussian kernel
    W_spectral = np.exp(-distances_sq / (2.0 * sigma_f ** 2))

    # Apply sparsity threshold
    W_spectral[W_spectral < epsilon] = 0.0

    return W_spectral


def build_temporal_adjacency(
    timestamps: np.ndarray,
    spectral_adj: np.ndarray,
    tau: float
) -> np.ndarray:
    """
    Build temporal adjacency matrix incorporating time differences.

    Implements Eq. 8 from paper:
        W_temporal_ij = exp(-|t_i - t_j| / tau) * W_spectral_ij

    Combines temporal proximity with spectral similarity.

    Parameters
    ----------
    timestamps : np.ndarray
        Measurement timestamps, shape (n_nodes,).
        Units arbitrary but should be consistent with tau.
    spectral_adj : np.ndarray
        Spectral adjacency matrix, shape (n_nodes, n_nodes).
    tau : float
        Temporal decay parameter. Larger tau = slower decay.

    Returns
    -------
    W_temporal : np.ndarray
        Temporal adjacency matrix, shape (n_nodes, n_nodes).
        Symmetric if spectral_adj is symmetric.

    Examples
    --------
    >>> times = np.array([0.0, 0.1, 0.2])
    >>> W_spec = np.eye(3)  # Identity for illustration
    >>> W_temp = build_temporal_adjacency(times, W_spec, tau=1.0)
    >>> assert W_temp.shape == (3, 3)
    """
    n_nodes = len(timestamps)

    # Compute pairwise time differences (absolute value)
    time_diffs = np.abs(timestamps[:, np.newaxis] - timestamps[np.newaxis, :])

    # Apply exponential temporal decay
    temporal_kernel = np.exp(-time_diffs / tau)

    # Modulate spectral adjacency by temporal kernel
    W_temporal = temporal_kernel * spectral_adj

    return W_temporal


def build_composite_adjacency(
    spatial: np.ndarray,
    spectral: np.ndarray,
    temporal: np.ndarray,
    alpha_s: float = 0.33,
    alpha_f: float = 0.33,
    alpha_t: float = 0.34
) -> np.ndarray:
    """
    Build composite adjacency from spatial, spectral, and temporal components.

    Implements Eq. 9 from paper:
        W_composite = alpha_s * W_spatial + alpha_f * W_spectral + alpha_t * W_temporal

    Weights should sum to 1.0 for normalized combination.

    Parameters
    ----------
    spatial : np.ndarray
        Spatial adjacency matrix, shape (n_nodes, n_nodes).
    spectral : np.ndarray
        Spectral adjacency matrix, shape (n_nodes, n_nodes).
    temporal : np.ndarray
        Temporal adjacency matrix, shape (n_nodes, n_nodes).
    alpha_s : float, optional
        Weight for spatial component. Default: 0.33
    alpha_f : float, optional
        Weight for spectral component. Default: 0.33
    alpha_t : float, optional
        Weight for temporal component. Default: 0.34

    Returns
    -------
    W_composite : np.ndarray
        Composite adjacency matrix, shape (n_nodes, n_nodes).
        Weighted combination of input matrices.

    Notes
    -----
    Recommend: alpha_s + alpha_f + alpha_t = 1.0

    Examples
    --------
    >>> W_s = np.eye(3)
    >>> W_f = np.eye(3) * 0.5
    >>> W_t = np.eye(3) * 0.2
    >>> W = build_composite_adjacency(W_s, W_f, W_t)
    >>> assert W.shape == (3, 3)
    """
    W_composite = (
        alpha_s * spatial +
        alpha_f * spectral +
        alpha_t * temporal
    )

    return W_composite


def compute_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Compute graph Laplacian matrix.

    Laplacian is defined as:
        L = D - W
    where D is the degree matrix (diagonal, D_ii = sum_j W_ij).

    Parameters
    ----------
    W : np.ndarray
        Adjacency matrix, shape (n_nodes, n_nodes).

    Returns
    -------
    L : np.ndarray
        Graph Laplacian matrix, shape (n_nodes, n_nodes).
        Symmetric, positive semidefinite.

    Examples
    --------
    >>> W = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> L = compute_laplacian(W)
    >>> assert L.shape == (2, 2)
    >>> assert np.allclose(L, L.T)
    """
    # Compute degree matrix
    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)

    # Compute Laplacian
    L = D - W

    return L


def compute_normalized_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Compute normalized graph Laplacian.

    Normalized Laplacian is defined as:
        L_norm = D^{-1/2} * L * D^{-1/2}
    where L = D - W.

    Properties: eigenvalues in [0, 2], useful for spectral clustering.

    Parameters
    ----------
    W : np.ndarray
        Adjacency matrix, shape (n_nodes, n_nodes).

    Returns
    -------
    L_norm : np.ndarray
        Normalized Laplacian, shape (n_nodes, n_nodes).
        Symmetric, eigenvalues in [0, 2].

    Examples
    --------
    >>> W = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> L_norm = compute_normalized_laplacian(W)
    >>> eigvals = np.linalg.eigvalsh(L_norm)
    >>> assert np.all(eigvals >= -1e-10)  # Non-negative (up to numerical error)
    >>> assert np.all(eigvals <= 2.0 + 1e-10)
    """
    # Compute degree matrix
    degrees = np.sum(W, axis=1)

    # Avoid division by zero
    D_inv_sqrt = np.zeros_like(degrees, dtype=float)
    nonzero_mask = degrees > 0
    D_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])

    # Build normalized Laplacian: D^{-1/2} (D - W) D^{-1/2}
    D_inv_sqrt_diag = np.diag(D_inv_sqrt)
    L = compute_laplacian(W)
    L_norm = D_inv_sqrt_diag @ L @ D_inv_sqrt_diag

    return L_norm


def eigendecompose(
    L: np.ndarray,
    n_eigs: Optional[int] = None,
    which: str = 'SM'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of Laplacian matrix.

    Returns eigenvalues and eigenvectors sorted in ascending order.
    Uses scipy.sparse.linalg.eigsh for efficiency with large sparse matrices.

    Parameters
    ----------
    L : np.ndarray
        Laplacian matrix, shape (n_nodes, n_nodes).
    n_eigs : int, optional
        Number of eigenvalues/eigenvectors to compute.
        If None, computes all (using dense eigsh).
        Default: None (all eigenvalues)
    which : str, optional
        Which eigenvalues: 'SM' (smallest magnitude), 'LA' (largest).
        Default: 'SM'

    Returns
    -------
    Lambda : np.ndarray
        Eigenvalues, shape (n_eigs,), sorted ascending.
    U : np.ndarray
        Eigenvectors, shape (n_nodes, n_eigs).
        Columns are eigenvectors corresponding to Lambda.

    Examples
    --------
    >>> L = np.array([[2.0, -1.0], [-1.0, 2.0]])
    >>> Lambda, U = eigendecompose(L)
    >>> assert Lambda.shape[0] <= 2
    >>> assert U.shape[1] == Lambda.shape[0]

    Notes
    -----
    For large sparse matrices, use sparse Laplacian (csr_matrix).
    """
    n_nodes = L.shape[0]

    # Convert to sparse if dense for efficiency
    if isinstance(L, np.ndarray):
        L_sparse = csr_matrix(L)
    else:
        L_sparse = L

    # Determine number of eigenvalues to compute
    if n_eigs is None or n_eigs >= n_nodes:
        # Compute all eigenvalues using dense method
        Lambda, U = np.linalg.eigh(L if isinstance(L, np.ndarray) else L.toarray())
        if n_eigs is not None and n_eigs < n_nodes:
            Lambda = Lambda[:n_eigs]
            U = U[:, :n_eigs]
    else:
        # Compute sparse eigendecomposition
        n_eigs = min(n_eigs, n_nodes - 2)  # eigsh needs n_eigs < n - 1
        Lambda, U = eigsh(L_sparse, k=n_eigs, which=which)
        # Sort by eigenvalue
        sort_idx = np.argsort(Lambda)
        Lambda = Lambda[sort_idx]
        U = U[:, sort_idx]

    return Lambda, U


class SignalGraph:
    """
    Graph signal processing object with sliding window updates.

    Manages graph construction, Laplacian computation, and spectral
    analysis for RF signal detection and flight tracking.

    Attributes
    ----------
    positions : np.ndarray
        Node positions, shape (n_nodes, n_dims).
    features : np.ndarray
        Node feature vectors, shape (n_nodes, n_features).
    timestamps : np.ndarray
        Measurement timestamps, shape (n_nodes,).
    W : np.ndarray
        Composite adjacency matrix, shape (n_nodes, n_nodes).
    L : np.ndarray
        Graph Laplacian, shape (n_nodes, n_nodes).
    L_norm : np.ndarray
        Normalized Laplacian, shape (n_nodes, n_nodes).
    Lambda : np.ndarray
        Eigenvalues of Laplacian, shape (n_eigs,).
    U : np.ndarray
        Eigenvectors of Laplacian, shape (n_nodes, n_eigs).
    lambda_max : float
        Maximum eigenvalue (spectral radius).
    """

    def __init__(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        timestamps: np.ndarray,
        sigma_s: float = 1.0,
        sigma_f: float = 1.0,
        tau: float = 1.0,
        alpha_s: float = 0.33,
        alpha_f: float = 0.33,
        alpha_t: float = 0.34,
        n_eigs: Optional[int] = None
    ):
        """
        Initialize SignalGraph.

        Parameters
        ----------
        positions : np.ndarray
            Receiver positions, shape (n_nodes, n_dims).
        features : np.ndarray
            Node features (e.g., frequency signatures), shape (n_nodes, n_features).
        timestamps : np.ndarray
            Measurement times, shape (n_nodes,).
        sigma_s : float, optional
            Spatial kernel bandwidth. Default: 1.0
        sigma_f : float, optional
            Feature kernel bandwidth. Default: 1.0
        tau : float, optional
            Temporal decay constant. Default: 1.0
        alpha_s : float, optional
            Spatial weight. Default: 0.33
        alpha_f : float, optional
            Feature weight. Default: 0.33
        alpha_t : float, optional
            Temporal weight. Default: 0.34
        n_eigs : int, optional
            Number of eigenvalues to compute. Default: None (all)
        """
        self.positions = np.asarray(positions, dtype=np.float64)
        self.features = np.asarray(features, dtype=np.float64)
        self.timestamps = np.asarray(timestamps, dtype=np.float64)

        self.sigma_s = float(sigma_s)
        self.sigma_f = float(sigma_f)
        self.tau = float(tau)
        self.alpha_s = float(alpha_s)
        self.alpha_f = float(alpha_f)
        self.alpha_t = float(alpha_t)
        self.n_eigs = n_eigs

        # Compute adjacency matrices
        self._build_adjacency()

        # Compute Laplacian and spectrum
        self._eigendecompose()

    def _build_adjacency(self) -> None:
        """Build spatial, spectral, temporal, and composite adjacency matrices."""
        self.W_spatial = build_spatial_adjacency(self.positions, self.sigma_s)
        self.W_spectral = build_spectral_adjacency(self.features, self.sigma_f)
        self.W_temporal = build_temporal_adjacency(
            self.timestamps, self.W_spectral, self.tau
        )
        self.W = build_composite_adjacency(
            self.W_spatial,
            self.W_spectral,
            self.W_temporal,
            self.alpha_s,
            self.alpha_f,
            self.alpha_t
        )

    def _eigendecompose(self) -> None:
        """Compute Laplacian and its eigendecomposition."""
        self.L = compute_laplacian(self.W)
        self.L_norm = compute_normalized_laplacian(self.W)
        self.Lambda, self.U = eigendecompose(self.L, n_eigs=self.n_eigs)
        self.lambda_max = float(np.max(self.Lambda))

    def update_window(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        timestamps: np.ndarray
    ) -> None:
        """
        Update graph with new sliding window of data.

        Parameters
        ----------
        positions : np.ndarray
            New node positions, shape (n_new, n_dims).
        features : np.ndarray
            New features, shape (n_new, n_features).
        timestamps : np.ndarray
            New timestamps, shape (n_new,).
        """
        self.positions = np.asarray(positions, dtype=np.float64)
        self.features = np.asarray(features, dtype=np.float64)
        self.timestamps = np.asarray(timestamps, dtype=np.float64)

        self._build_adjacency()
        self._eigendecompose()

    def get_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get graph spectrum.

        Returns
        -------
        Lambda : np.ndarray
            Eigenvalues.
        U : np.ndarray
            Eigenvectors (columns).
        """
        return self.Lambda, self.U

    def __repr__(self) -> str:
        """String representation."""
        n_nodes = self.positions.shape[0]
        n_eigs = self.Lambda.shape[0] if self.Lambda is not None else 0
        return (
            f"SignalGraph(n_nodes={n_nodes}, n_eigs={n_eigs}, "
            f"lambda_max={self.lambda_max:.4f})"
        )
