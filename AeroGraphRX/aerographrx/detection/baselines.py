"""Baseline anomaly detection algorithms for comparison.

Implements standard detection methods including Energy Detector, CFAR,
One-Class SVM, and a simple Autoencoder baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.svm import OneClassSVM


class EnergyDetector:
    """Energy-based anomaly detector.

    Simple threshold-based detector using signal energy/power.
    """

    def __init__(self):
        """Initialize energy detector."""
        pass

    def detect(self, signal, threshold):
        """Detect anomalies based on energy threshold.

        Args:
            signal: Input signal of shape (N,) or (T, N).
            threshold: Energy threshold for detection.

        Returns:
            Detection scores (energy values) of shape (N,).
        """
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        # Compute energy per signal
        if signal.ndim == 2:
            energy = np.sum(signal ** 2, axis=0)
        else:
            energy = np.sum(signal ** 2)

        return energy

    def fit_threshold(self, signals, labels, method='youden'):
        """Fit optimal threshold using labeled data.

        Args:
            signals: Training signals of shape (N, T) or (T,).
            labels: Binary labels (0=normal, 1=anomaly).
            method: Threshold method - 'youden' or 'roc' (default: 'youden').

        Returns:
            Optimal threshold (scalar).
        """
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        # Compute energy scores
        energy_scores = np.sum(signals ** 2, axis=1 if signals.ndim == 2 else 0)

        if method == 'youden':
            # Find threshold maximizing Youden index
            thresholds = np.linspace(energy_scores.min(), energy_scores.max(), 100)
            best_youden = -1
            best_threshold = thresholds[0]

            for thresh in thresholds:
                predicted = energy_scores > thresh
                tp = np.sum((predicted == 1) & (labels == 1))
                tn = np.sum((predicted == 0) & (labels == 0))
                fp = np.sum((predicted == 1) & (labels == 0))
                fn = np.sum((predicted == 0) & (labels == 1))

                if (tp + fn) > 0 and (fp + tn) > 0:
                    tpr = tp / (tp + fn)
                    tnr = tn / (fp + tn)
                    youden = tpr + tnr - 1
                    if youden > best_youden:
                        best_youden = youden
                        best_threshold = thresh

            return best_threshold
        else:
            # Default: use mean energy of anomalies
            anomaly_energy = energy_scores[labels == 1]
            normal_energy = energy_scores[labels == 0]
            if len(anomaly_energy) > 0 and len(normal_energy) > 0:
                return (np.max(normal_energy) + np.min(anomaly_energy)) / 2
            else:
                return np.median(energy_scores)


class CFARDetector:
    """Constant False Alarm Rate (CFAR) detector.

    Implements cell-averaging CFAR for adaptive thresholding.

    Args:
        guard_cells: Number of guard cells around test cell (default: 4).
        reference_cells: Number of reference cells on each side (default: 16).
        pfa: Target false alarm probability (default: 0.05).
    """

    def __init__(self, guard_cells=4, reference_cells=16, pfa=0.05):
        """Initialize CFAR detector.

        Args:
            guard_cells: Guard cell radius.
            reference_cells: Reference cell count per side.
            pfa: Target false alarm rate.
        """
        self.guard_cells = guard_cells
        self.reference_cells = reference_cells
        self.pfa = pfa

    def detect(self, signal):
        """Detect anomalies using cell-averaging CFAR.

        Args:
            signal: Input signal of shape (T,).

        Returns:
            Detection mask of shape (T,) (boolean).
        """
        T = len(signal)
        detection_mask = np.zeros(T, dtype=bool)

        for i in range(T):
            # Define guard and reference windows
            guard_start = max(0, i - self.guard_cells)
            guard_end = min(T, i + self.guard_cells + 1)

            ref_start = max(0, i - self.guard_cells - self.reference_cells)
            ref_end = min(T, i + self.guard_cells + self.reference_cells + 1)

            # Reference window (excluding guard region)
            ref_left = signal[ref_start:guard_start]
            ref_right = signal[guard_end:ref_end]
            reference = np.concatenate([ref_left, ref_right])

            if len(reference) > 0:
                # Estimate noise power
                noise_power = np.mean(reference ** 2)

                # Threshold computation (Gaussian approximation)
                # T = lambda * N0 where lambda depends on pfa and N0 is noise power
                lambda_param = -np.log(self.pfa) / self.reference_cells
                threshold = lambda_param * noise_power

                # Test cell
                test_power = signal[i] ** 2
                detection_mask[i] = test_power > threshold

        return detection_mask


class OneClassSVMDetector:
    """One-Class SVM for anomaly detection.

    Wrapper around scikit-learn OneClassSVM for unsupervised anomaly detection.

    Args:
        nu: Upper bound on fraction of training errors (default: 0.05).
        kernel: Kernel type - 'rbf', 'linear', 'poly' (default: 'rbf').
        gamma: Kernel coefficient (default: 'auto').
    """

    def __init__(self, nu=0.05, kernel='rbf', gamma='auto'):
        """Initialize One-Class SVM.

        Args:
            nu: Outlier fraction parameter.
            kernel: Kernel type.
            gamma: Kernel coefficient.
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(
            nu=nu, kernel=kernel, gamma=gamma
        )
        self.is_fitted = False

    def fit(self, X_train):
        """Fit One-Class SVM on training data.

        Args:
            X_train: Training features of shape (N, D).
        """
        self.model.fit(X_train)
        self.is_fitted = True

    def predict(self, X_test):
        """Predict anomaly labels.

        Args:
            X_test: Test features of shape (M, D).

        Returns:
            Predictions: 1 for inliers, -1 for outliers of shape (M,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X_test)

    def decision_function(self, X_test):
        """Compute decision function scores.

        Args:
            X_test: Test features of shape (M, D).

        Returns:
            Decision scores of shape (M,). Higher = more normal.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.decision_function(X_test)


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for anomaly detection.

    Non-VAE baseline using deterministic encoder-decoder architecture.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension (default: 128).
    """

    def __init__(self, input_dim, hidden_dim=128):
        """Initialize simple autoencoder.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden layer dimension.
        """
        super(SimpleAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        """Forward pass through autoencoder.

        Args:
            x: Input of shape (B, input_dim).

        Returns:
            Reconstructed output of shape (B, input_dim).
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector:
    """Autoencoder-based anomaly detector (baseline).

    Uses simple autoencoder for unsupervised anomaly detection via
    reconstruction error.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension (default: 128).
        device: Device to run on ('cpu' or 'cuda', default: 'cpu').
    """

    def __init__(self, input_dim, hidden_dim=128, device='cpu'):
        """Initialize autoencoder detector.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
            device: Torch device.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.model = SimpleAutoencoder(input_dim, hidden_dim).to(device)
        self.is_fitted = False

    def fit(self, X_train, epochs=50, batch_size=32, lr=0.001):
        """Fit autoencoder on training data.

        Args:
            X_train: Training features of shape (N, input_dim).
            epochs: Number of training epochs (default: 50).
            batch_size: Batch size (default: 32).
            lr: Learning rate (default: 0.001).
        """
        # Convert to tensor
        X_train_tensor = torch.from_numpy(X_train).float().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                batch = X_train_tensor[i:i + batch_size]

                # Forward pass
                x_recon = self.model(batch)
                loss = criterion(x_recon, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        self.is_fitted = True

    def reconstruction_error(self, X_test):
        """Compute reconstruction error for test data.

        Args:
            X_test: Test features of shape (M, input_dim).

        Returns:
            Per-sample reconstruction errors of shape (M,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_test_tensor = torch.from_numpy(X_test).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            x_recon = self.model(X_test_tensor)
            errors = torch.mean((x_recon - X_test_tensor) ** 2, dim=1)

        return errors.cpu().numpy()

    def predict(self, X_test, threshold):
        """Predict anomaly labels based on reconstruction error.

        Args:
            X_test: Test features of shape (M, input_dim).
            threshold: Reconstruction error threshold.

        Returns:
            Predictions: 0 for normal, 1 for anomaly of shape (M,).
        """
        errors = self.reconstruction_error(X_test)
        return (errors > threshold).astype(int)
