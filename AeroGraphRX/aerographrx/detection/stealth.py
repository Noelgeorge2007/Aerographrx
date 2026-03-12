"""Graph-Based Stealth Detection Algorithm (Algorithm 2).

Detects stealth anomalies in networked signals using spectral graph theory,
GFT analysis, and statistical hypothesis testing.
"""

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import DBSCAN


class StealthDetector:
    """Graph-based stealth detection for anomalous signals.

    Detects anomalies in graph-structured signal data using spectral
    analysis and local deviation metrics.

    Args:
        K0: Spectral cutoff frequency (default: 50).
        gamma: High-frequency energy weighting (default: 1.0).
    """

    def __init__(self, K0=50, gamma=1.0):
        """Initialize stealth detector.

        Args:
            K0: Spectral cutoff index.
            gamma: High-frequency energy weight.
        """
        self.K0 = K0
        self.gamma = gamma

    def compute_anomaly_score(self, x_i, neighbors_x, gft_coeffs, K0=None):
        """Compute anomaly score for a single node (Eq. 20).

        anomaly_score = local_deviation + gamma * high_freq_energy

        Args:
            x_i: Signal at node i of shape (T,).
            neighbors_x: Signals at neighboring nodes of shape (num_neighbors, T).
            gft_coeffs: Graph Fourier Transform coefficients of shape (K, T)
                       where K is number of spectral components.
            K0: Spectral cutoff (uses self.K0 if None).

        Returns:
            Scalar anomaly score.
        """
        if K0 is None:
            K0 = self.K0

        # Local deviation: ||x_i - mean(neighbors)||_2^2
        if neighbors_x.shape[0] > 0:
            neighbor_mean = np.mean(neighbors_x, axis=0)
            local_dev = np.sum((x_i - neighbor_mean) ** 2)
        else:
            local_dev = np.sum(x_i ** 2)

        # High-frequency energy: sum of coefficients beyond K0
        if gft_coeffs.shape[0] > K0:
            high_freq_energy = np.sum(gft_coeffs[K0:, :] ** 2)
        else:
            high_freq_energy = 0.0

        # Combined score
        score = local_dev + self.gamma * high_freq_energy
        return float(score)

    def compute_threshold(self, d, sigma_hat, pfa=0.05):
        """Compute detection threshold from theoretical false alarm rate (Theorem 1).

        Using chi-squared distribution:
            P_FA(tau) = 1 - F_chi2_d(tau / sigma_hat^2)

        Solves for tau such that P_FA(tau) = pfa.

        Args:
            d: Degrees of freedom (signal dimensionality).
            sigma_hat: Estimated noise standard deviation.
            pfa: Target false alarm probability (default: 0.05).

        Returns:
            Detection threshold tau (scalar).
        """
        # Find chi-squared quantile
        chi2_quantile = stats.chi2.ppf(1.0 - pfa, df=d)

        # Threshold
        tau = chi2_quantile * (sigma_hat ** 2)
        return float(tau)

    def detect(self, graph, signals, pfa=0.05):
        """Run full stealth detection algorithm (Algorithm 2).

        Steps:
            1. Compute graph Laplacian L = D - A
            2. Eigendecompose to get spectral components
            3. Compute Graph Fourier Transform
            4. For each node: compute anomaly score
            5. Threshold against computed tau
            6. Cluster detections via DBSCAN

        Args:
            graph: Graph represented as adjacency matrix of shape (N, N).
            signals: Signal matrix of shape (N, T) where N is number of nodes
                    and T is signal length.
            pfa: Target false alarm probability (default: 0.05).

        Returns:
            Dictionary containing:
                - 'detection_mask': Boolean array of shape (N,) indicating detections
                - 'cluster_labels': Cluster assignment for each node
                - 'scores': Anomaly scores for all nodes
                - 'threshold': Detection threshold used
        """
        N, T = signals.shape

        # Step 1: Compute Laplacian L = D - A
        degree = np.sum(graph, axis=1)
        D = np.diag(degree)
        laplacian = D - graph

        # Step 2: Eigendecompose Laplacian
        # Get smallest K eigenvalues/eigenvectors
        K = min(self.K0 + 10, N)
        try:
            # Use sparse eigendecomposition for large graphs
            if N > 100:
                laplacian_sparse = csr_matrix(laplacian)
                eigenvalues, eigenvectors = eigsh(
                    laplacian_sparse, k=K, which='SM', return_eigenvectors=True
                )
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                eigenvalues = eigenvalues[:K]
                eigenvectors = eigenvectors[:, :K]
        except Exception:
            # Fallback if eigendecomposition fails
            eigenvalues = np.ones(K)
            eigenvectors = np.eye(N, K)

        # Step 3: Compute Graph Fourier Transform
        # GFT coefficients: theta_k = V_k^T x for each signal
        gft_coeffs = eigenvectors.T @ signals  # (K, T)

        # Step 4: Compute anomaly scores for all nodes
        scores = np.zeros(N)
        for i in range(N):
            # Get neighbors
            neighbor_indices = np.where(graph[i] > 0)[0]
            if len(neighbor_indices) > 0:
                neighbors_x = signals[neighbor_indices, :]
            else:
                neighbors_x = np.empty((0, T))

            scores[i] = self.compute_anomaly_score(
                signals[i, :], neighbors_x, gft_coeffs, K0=self.K0
            )

        # Step 5: Compute threshold
        sigma_hat = np.std(scores)
        tau = self.compute_threshold(d=T, sigma_hat=sigma_hat, pfa=pfa)

        detection_mask = scores > tau

        # Step 6: Cluster detections via DBSCAN
        if np.sum(detection_mask) > 0:
            detected_indices = np.where(detection_mask)[0]
            detected_scores = scores[detected_indices].reshape(-1, 1)

            # Use DBSCAN on normalized scores
            dbscan = DBSCAN(eps=0.5, min_samples=1)
            cluster_labels_detected = dbscan.fit_predict(detected_scores)

            # Map back to full index space
            cluster_labels = np.full(N, -1, dtype=int)
            cluster_labels[detected_indices] = cluster_labels_detected
        else:
            cluster_labels = np.full(N, -1, dtype=int)

        # Compute cluster centroids (mean score per cluster)
        cluster_centroids = {}
        for cluster_id in np.unique(cluster_labels):
            if cluster_id >= 0:
                cluster_nodes = np.where(cluster_labels == cluster_id)[0]
                cluster_centroids[cluster_id] = float(np.mean(scores[cluster_nodes]))

        return {
            'detection_mask': detection_mask,
            'cluster_labels': cluster_labels,
            'scores': scores,
            'threshold': tau,
            'cluster_centroids': cluster_centroids
        }

    def calibrate_pfa(self, scores, designed_pfa_values):
        """Empirical PFA calibration check.

        Compares theoretical and empirical false alarm rates for various
        threshold levels.

        Args:
            scores: Anomaly scores of shape (N,).
            designed_pfa_values: List of target PFA values to test.

        Returns:
            Dictionary mapping pfa values to (theoretical_tau, empirical_pfa) tuples.
        """
        sigma_hat = np.std(scores)
        d = len(scores)

        calibration_results = {}

        for pfa in designed_pfa_values:
            # Theoretical threshold
            tau = self.compute_threshold(d=d, sigma_hat=sigma_hat, pfa=pfa)

            # Empirical false alarm rate
            num_detections = np.sum(scores > tau)
            empirical_pfa = num_detections / len(scores)

            calibration_results[pfa] = (tau, empirical_pfa)

        return calibration_results
