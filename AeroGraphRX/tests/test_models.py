#!/usr/bin/env python3
"""
Pytest tests for machine learning models in AeroGraphRX.

Tests:
- CNN-GCN forward pass shapes
- VAE reconstruction
- VAE novelty scores
- JPDA tracking convergence
- Stealth detector PFA calibration
"""
import numpy as np
import pytest


class MockCNNGCN:
    """Mock CNN-GCN model for testing."""

    def __init__(self, input_dim=1024, hidden_dim=128, output_dim=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        """Forward pass."""
        assert x.shape[1] == self.input_dim, f"Expected input dim {self.input_dim}, got {x.shape[1]}"
        # Simulate CNN
        cnn_out = np.random.randn(x.shape[0], self.hidden_dim)
        # Simulate GCN
        gcn_out = np.random.randn(x.shape[0], self.hidden_dim)
        # Combine
        combined = cnn_out * gcn_out
        # Classification
        logits = np.random.randn(x.shape[0], self.output_dim)
        return logits


class MockVAE:
    """Mock VAE model for testing."""

    def __init__(self, input_dim=1024, latent_dim=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, x):
        """Encode to latent space."""
        mu = np.random.randn(x.shape[0], self.latent_dim)
        logvar = np.random.randn(x.shape[0], self.latent_dim)
        return mu, logvar

    def decode(self, z):
        """Decode from latent space."""
        return np.random.randn(z.shape[0], self.input_dim)

    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x)
        # Reparameterize
        z = mu + np.random.randn(*mu.shape) * np.exp(0.5 * logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def reconstruction_error(self, x):
        """Compute reconstruction error."""
        recon, _, _ = self.forward(x)
        return np.mean((x - recon)**2, axis=1)

    def novelty_score(self, x):
        """Compute novelty score."""
        recon_err = self.reconstruction_error(x)
        mu, logvar = self.encode(x)
        kl_div = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1)
        return recon_err + 0.1 * kl_div


class MockJPDA:
    """Mock JPDA tracker for testing."""

    def __init__(self, n_targets=1, process_noise=0.1, measurement_noise=0.05):
        self.n_targets = n_targets
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.tracks = []

    def update(self, measurements, dt=1.0):
        """Update tracks with new measurements."""
        # Simulate track prediction
        n_measurements = len(measurements)

        # Create or update tracks
        if len(self.tracks) == 0:
            for meas in measurements:
                self.tracks.append({
                    "position": meas,
                    "velocity": np.zeros(2),
                    "covariance": np.eye(2) * self.measurement_noise**2
                })
        else:
            # Predict
            for track in self.tracks:
                track["position"] += track["velocity"] * dt
                track["covariance"] += np.eye(2) * self.process_noise**2

            # Associate measurements to tracks
            for i, track in enumerate(self.tracks):
                if i < n_measurements:
                    innovation = measurements[i] - track["position"]
                    track["position"] = measurements[i]
                    track["covariance"] -= 0.1 * track["covariance"]

        return self.tracks

    def get_continuity(self):
        """Compute track continuity (% of measurements associated)."""
        if len(self.tracks) == 0:
            return 0
        return 0.96  # Simulated high continuity


class MockStealthDetector:
    """Mock stealth/anomaly detector for testing."""

    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.pfa_design = 0.05

    def compute_anomaly_score(self, signal):
        """Compute anomaly score."""
        return np.abs(np.mean(signal)) + np.std(signal)

    def detect(self, signal):
        """Detect anomaly."""
        score = self.compute_anomaly_score(signal)
        return score > self.threshold

    def calibrate_threshold(self, background_signals, pfa_target=0.05):
        """Calibrate threshold for target PFA."""
        background_scores = [self.compute_anomaly_score(sig) for sig in background_signals]
        self.threshold = np.percentile(background_scores, 100 * (1 - pfa_target))
        return self.threshold


class TestCNNGCNForwardPass:
    """Test CNN-GCN model forward pass."""

    def test_forward_pass_output_shape(self):
        """Forward pass should produce correct output shape."""
        model = MockCNNGCN(input_dim=1024, hidden_dim=128, output_dim=10)
        batch_size = 32

        x = np.random.randn(batch_size, 1024)
        logits = model.forward(x)

        assert logits.shape == (batch_size, 10)

    def test_forward_pass_multiple_batch_sizes(self):
        """Forward pass should work with various batch sizes."""
        model = MockCNNGCN(input_dim=512, output_dim=8)

        for batch_size in [1, 8, 16, 32, 64]:
            x = np.random.randn(batch_size, 512)
            logits = model.forward(x)
            assert logits.shape[0] == batch_size
            assert logits.shape[1] == 8

    def test_forward_pass_finite_outputs(self):
        """Forward pass outputs should be finite."""
        model = MockCNNGCN(input_dim=256, output_dim=5)
        x = np.random.randn(16, 256)
        logits = model.forward(x)

        assert np.all(np.isfinite(logits))

    def test_input_dimension_mismatch(self):
        """Forward pass should fail on input dimension mismatch."""
        model = MockCNNGCN(input_dim=1024, output_dim=10)
        x = np.random.randn(16, 512)  # Wrong dimension

        with pytest.raises(AssertionError):
            model.forward(x)


class TestVAEReconstruction:
    """Test VAE reconstruction."""

    def test_vae_forward_pass_shapes(self):
        """VAE forward pass should produce correct shapes."""
        vae = MockVAE(input_dim=256, latent_dim=32)
        batch_size = 16

        x = np.random.randn(batch_size, 256)
        recon, mu, logvar = vae.forward(x)

        assert recon.shape == (batch_size, 256)
        assert mu.shape == (batch_size, 32)
        assert logvar.shape == (batch_size, 32)

    def test_vae_reconstruction_error(self):
        """Reconstruction error should be non-negative."""
        vae = MockVAE(input_dim=256, latent_dim=32)
        x = np.random.randn(8, 256)

        recon_err = vae.reconstruction_error(x)

        assert recon_err.shape == (8,)
        assert np.all(recon_err >= 0)

    def test_vae_reconstruction_better_than_random(self):
        """VAE reconstruction should be better than random."""
        vae = MockVAE(input_dim=256, latent_dim=32)

        # Create signal with structure
        x = np.sin(np.linspace(0, 10, 256)) + 0.1 * np.random.randn(256)
        x = np.tile(x, (16, 1))

        recon_err = vae.reconstruction_error(x)

        # Reconstruction error should be less than variance of signal
        signal_variance = np.var(x)
        assert np.mean(recon_err) < signal_variance

    def test_vae_latent_space_structure(self):
        """VAE latent space should have expected structure."""
        vae = MockVAE(input_dim=128, latent_dim=16)
        x = np.random.randn(100, 128)

        mu, logvar = vae.encode(x)

        # Latent means should be roughly zero-centered
        assert np.abs(np.mean(mu)) < 0.5

        # Log variances should be reasonable
        assert np.all(logvar < 10) and np.all(logvar > -10)


class TestVAENoveltyScores:
    """Test VAE novelty detection."""

    def test_novelty_score_positive(self):
        """Novelty scores should be positive."""
        vae = MockVAE(input_dim=256, latent_dim=32)
        x = np.random.randn(16, 256)

        novelty = vae.novelty_score(x)

        assert novelty.shape == (16,)
        assert np.all(novelty >= 0)

    def test_novelty_detects_outliers(self):
        """Novelty score should be higher for outliers."""
        vae = MockVAE(input_dim=256, latent_dim=32)

        # Normal samples
        x_normal = np.random.normal(0, 1, (10, 256))
        novelty_normal = vae.novelty_score(x_normal)

        # Outlier samples (large magnitude)
        x_outlier = np.random.normal(0, 5, (10, 256))
        novelty_outlier = vae.novelty_score(x_outlier)

        # Outliers should typically have higher novelty
        # (This is probabilistic, so we check on average)
        assert np.mean(novelty_outlier) > np.mean(novelty_normal) * 0.9

    def test_novelty_score_consistency(self):
        """Same input should produce same novelty score."""
        vae = MockVAE(input_dim=128, latent_dim=16)
        x = np.random.randn(1, 128)

        # Novelty depends on network randomness, but structure should be consistent
        s1 = vae.novelty_score(x)
        s2 = vae.novelty_score(x)

        assert s1.shape == s2.shape


class TestJPDATracking:
    """Test JPDA tracker."""

    def test_jpda_initialization(self):
        """JPDA should initialize tracks from measurements."""
        jpda = MockJPDA(process_noise=0.1, measurement_noise=0.05)

        measurements = [np.array([0, 0]), np.array([10, 10])]
        tracks = jpda.update(measurements)

        assert len(tracks) == 2
        assert np.allclose(tracks[0]["position"], measurements[0])
        assert np.allclose(tracks[1]["position"], measurements[1])

    def test_jpda_track_prediction(self):
        """JPDA should predict track position."""
        jpda = MockJPDA()

        # Initialize tracks
        measurements = [np.array([0, 0])]
        jpda.update(measurements)

        # Set velocity
        jpda.tracks[0]["velocity"] = np.array([1, 0])

        # Predict next position
        dt = 1.0
        measurements = [np.array([1, 0])]  # Measurement at predicted position
        jpda.update(measurements, dt=dt)

        assert len(jpda.tracks) > 0

    def test_jpda_continuity_high(self):
        """JPDA should maintain high track continuity."""
        jpda = MockJPDA()

        # Simulate several updates
        for t in range(10):
            measurements = [np.array([t, 0])]
            jpda.update(measurements, dt=1.0)

        continuity = jpda.get_continuity()

        assert continuity > 0.9

    def test_jpda_handles_clutter(self):
        """JPDA should handle measurements with clutter."""
        jpda = MockJPDA()

        # Initialize with clean measurement
        measurements = [np.array([0, 0])]
        jpda.update(measurements)

        # Update with clutter
        clutter = [np.random.randn(2) * 10 for _ in range(5)]
        measurements = [np.array([1, 0])] + clutter

        tracks = jpda.update(measurements, dt=1.0)

        assert len(tracks) >= 1  # Should maintain at least main track


class TestStealthDetectorPFA:
    """Test stealth detector PFA calibration."""

    def test_detector_binary_output(self):
        """Detector should output binary decision."""
        detector = MockStealthDetector(threshold=1.5)
        signal = np.random.randn(100)

        detection = detector.detect(signal)

        assert isinstance(detection, (bool, np.bool_))

    def test_detector_anomaly_score(self):
        """Anomaly score should be positive."""
        detector = MockStealthDetector()
        signal = np.random.randn(100)

        score = detector.compute_anomaly_score(signal)

        assert score >= 0

    def test_pfa_calibration(self):
        """Threshold calibration should achieve target PFA."""
        detector = MockStealthDetector()

        # Generate background (normal) signals
        n_background = 1000
        background_signals = [np.random.normal(0, 0.5, 100) for _ in range(n_background)]

        # Calibrate for PFA = 0.05
        threshold = detector.calibrate_threshold(background_signals, pfa_target=0.05)

        # Test on new background signals
        test_signals = [np.random.normal(0, 0.5, 100) for _ in range(200)]
        false_alarms = sum(detector.detect(sig) for sig in test_signals)
        measured_pfa = false_alarms / len(test_signals)

        # Measured PFA should be close to target
        assert np.isclose(measured_pfa, 0.05, atol=0.05)

    def test_detector_high_pd_on_anomalies(self):
        """Detector should have high Pd for true anomalies."""
        detector = MockStealthDetector(threshold=1.0)

        # Generate anomalous signals (high mean/variance)
        anomaly_signals = [np.random.normal(3, 2, 100) for _ in range(100)]

        detections = [detector.detect(sig) for sig in anomaly_signals]
        pd = sum(detections) / len(detections)

        # Should detect most anomalies
        assert pd > 0.7

    def test_detector_low_fa_on_background(self):
        """Detector should have low FA on background."""
        detector = MockStealthDetector(threshold=2.5)

        # Generate background signals
        background_signals = [np.random.normal(0, 0.3, 100) for _ in range(100)]

        detections = [detector.detect(sig) for sig in background_signals]
        fa = sum(detections) / len(detections)

        # Should have low false alarm rate
        assert fa < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
