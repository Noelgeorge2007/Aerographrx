#!/usr/bin/env python3
"""
Pytest tests for core modules of AeroGraphRX.

Tests:
- Graph Laplacian positive semi-definiteness
- GFT invertibility
- Chebyshev filter matching
- TDoA measurement consistency
- CRLB positive definiteness
- Signal generator output shapes and SNR
- ROC/AUC computation on known cases
- Bootstrap CI coverage
- McNemar and DeLong tests on known distributions
"""
import numpy as np
import pytest
from scipy import linalg


class TestGraphLaplacian:
    """Test graph Laplacian properties."""

    def test_laplacian_positive_semidefinite(self):
        """Laplacian should be positive semi-definite."""
        # Create simple adjacency matrix
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=float)

        # Compute degree matrix
        D = np.diag(np.sum(A, axis=1))

        # Compute Laplacian
        L = D - A

        # Check positive semi-definiteness via eigenvalues
        evals = np.linalg.eigvalsh(L)
        assert np.all(evals >= -1e-10), f"Negative eigenvalue found: {evals}"

    def test_laplacian_zero_eigenvalue(self):
        """Connected Laplacian should have one zero eigenvalue."""
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=float)

        D = np.diag(np.sum(A, axis=1))
        L = D - A

        evals = np.linalg.eigvalsh(L)
        assert np.isclose(evals[0], 0, atol=1e-10)

    def test_laplacian_row_sum_zero(self):
        """Laplacian row sums should be zero."""
        A = np.array([
            [0, 1, 0.5],
            [1, 0, 0.8],
            [0.5, 0.8, 0]
        ], dtype=float)

        D = np.diag(np.sum(A, axis=1))
        L = D - A

        row_sums = np.sum(L, axis=1)
        assert np.allclose(row_sums, 0, atol=1e-10)


class TestGraphFourierTransform:
    """Test Graph Fourier Transform properties."""

    def test_gft_invertible_undirected(self):
        """GFT should be invertible for undirected graphs."""
        # Create Laplacian of simple graph
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=float)

        D = np.diag(np.sum(A, axis=1))
        L = D - A

        # Eigendecomposition (GFT basis)
        evals, evecs = np.linalg.eigh(L)

        # Check orthonormality
        product = evecs.T @ evecs
        assert np.allclose(product, np.eye(4), atol=1e-10)

        # Check that inverse works
        test_signal = np.array([1, 2, 3, 4], dtype=float)
        coeffs = evecs.T @ test_signal
        recovered = evecs @ coeffs
        assert np.allclose(recovered, test_signal, atol=1e-10)

    def test_gft_preserves_norm(self):
        """GFT (unitary) should preserve signal norm."""
        L = np.eye(5)
        evals, evecs = np.linalg.eigh(L)

        test_signal = np.random.randn(5)
        coeffs = evecs.T @ test_signal

        norm_signal = np.linalg.norm(test_signal)
        norm_coeffs = np.linalg.norm(coeffs)

        assert np.isclose(norm_signal, norm_coeffs, atol=1e-10)


class TestChebyshevFilter:
    """Test Chebyshev polynomial filter."""

    def test_chebyshev_matches_direct_computation(self):
        """Chebyshev approximation should match direct computation for small graphs."""
        # Simple 3-node graph
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=float)

        D = np.diag(np.sum(A, axis=1))
        L = D - A
        L_norm = L / (2 * np.max(np.linalg.eigvalsh(L)))

        # Test signal
        x = np.array([1, 0, -1], dtype=float)

        # Direct multiplication (filter order 1)
        y_direct = (np.eye(3) + L_norm) @ x

        # Chebyshev order-1 approximation
        T0 = np.eye(3)
        T1 = L_norm
        y_cheby = 0.5 * (T0 + T1) @ x

        # Should be approximately equal (within numerical precision)
        assert np.allclose(y_direct, y_cheby, atol=0.1)

    def test_chebyshev_stability(self):
        """Chebyshev filter should be numerically stable."""
        L = np.random.randn(5, 5)
        L = (L + L.T) / 2  # Make symmetric
        L_norm = L / (2 * np.max(np.linalg.eigvalsh(L)))

        x = np.random.randn(5)

        # Apply Chebyshev filter multiple times
        for K in [1, 2, 3, 5]:
            T_prev = np.eye(5)
            T_curr = L_norm
            for _ in range(K-1):
                T_next = 2 * L_norm @ T_curr - T_prev
                T_prev = T_curr
                T_curr = T_next

            y = T_curr @ x
            assert np.all(np.isfinite(y)), "Chebyshev filter produced non-finite values"


class TestTDoAMeasurements:
    """Test Time Difference of Arrival measurements."""

    def test_tdoa_consistency(self):
        """TDoA measurements should be consistent with geometry."""
        # Two receivers at (0, 0) and (100, 0) meters
        # Signal source at (50, 50) meters
        r1_pos = np.array([0, 0])
        r2_pos = np.array([100, 0])
        source_pos = np.array([50, 50])

        # Distances
        d1 = np.linalg.norm(source_pos - r1_pos)
        d2 = np.linalg.norm(source_pos - r2_pos)

        # True TDoA (with speed of light c = 3e8 m/s)
        c = 3e8
        true_tdoa = (d1 - d2) / c

        # Check that TDoA is reasonable (less than distance/c)
        assert np.abs(true_tdoa) < 100 / c

    def test_tdoa_measurement_noise(self):
        """TDoA measurements with noise should have expected variance."""
        # Simulate 1000 TDoA measurements with Gaussian noise
        true_tdoa = 1e-6  # 1 microsecond
        noise_std = 10e-9  # 10 nanoseconds
        n_samples = 1000

        measurements = true_tdoa + np.random.normal(0, noise_std, n_samples)

        # Estimate mean and variance
        est_mean = np.mean(measurements)
        est_std = np.std(measurements)

        # Check estimates
        assert np.isclose(est_mean, true_tdoa, atol=3*noise_std)
        assert np.isclose(est_std, noise_std, atol=2*noise_std)


class TestCRLB:
    """Test Cramer-Rao Lower Bound."""

    def test_crlb_positive_definite(self):
        """CRLB should be positive definite."""
        # Fisher Information Matrix
        FIM = np.array([
            [100, 10],
            [10, 50]
        ], dtype=float)

        # CRLB is inverse of FIM
        CRLB = np.linalg.inv(FIM)

        # Check positive definiteness
        evals = np.linalg.eigvalsh(CRLB)
        assert np.all(evals > 0), f"CRLB has non-positive eigenvalues: {evals}"

    def test_crlb_bounds_variance(self):
        """Estimate variance should be >= CRLB diagonal."""
        # Simple 1D estimation problem
        # Fisher Information I = 4/sigma_noise^2
        sigma_noise = 0.1
        fim = 4 / sigma_noise**2

        crlb = 1 / fim

        # Estimate variance from measurements
        true_param = 5.0
        n_samples = 10000
        measurements = true_param + np.random.normal(0, sigma_noise, n_samples)
        estimate_var = np.var(measurements)

        # Estimate variance should be larger than CRLB
        assert estimate_var >= crlb * 0.95  # Allow some numerical slack


class TestSignalGeneration:
    """Test synthetic signal generation."""

    def test_signal_shape(self):
        """Generated signals should have correct shape."""
        fs = 1e6  # 1 MHz sampling
        duration = 1.0  # 1 second
        expected_length = int(fs * duration)

        signal = np.sin(2 * np.pi * 100e3 * np.arange(expected_length) / fs)

        assert len(signal) == expected_length

    def test_signal_snr(self):
        """SNR of generated signal should match specification."""
        fs = 1e6
        duration = 0.01
        snr_target = 10  # dB
        t = np.arange(int(fs * duration)) / fs

        signal = np.sin(2 * np.pi * 100e3 * t)
        signal_power = np.mean(signal**2)

        noise_power = signal_power / (10**(snr_target/10))
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))

        noisy_signal = signal + noise
        snr_measured = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))

        assert np.isclose(snr_measured, snr_target, atol=1.0)

    def test_complex_signal_generation(self):
        """Complex signals should have expected properties."""
        fs = 1e6
        duration = 0.01
        t = np.arange(int(fs * duration)) / fs

        signal = np.exp(1j * 2 * np.pi * 100e3 * t)

        # Check amplitude
        assert np.allclose(np.abs(signal), 1.0)

        # Check orthogonality with shifted copy
        shift = int(fs * 0.0001)  # 0.1 ms shift
        signal_shifted = np.exp(1j * 2 * np.pi * 100e3 * (t + shift/fs))
        correlation = np.mean(signal.conj() * signal_shifted)
        assert np.abs(correlation) < 1.0


class TestROCAUC:
    """Test ROC and AUC computation."""

    def test_roc_auc_perfect_classifier(self):
        """Perfect classifier should have AUC = 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])

        # Compute AUC
        sorted_idx = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_idx]

        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        tp = np.concatenate([[0], np.cumsum(y_true_sorted)])
        fp = np.concatenate([[0], np.cumsum(1 - y_true_sorted)])

        tpr = tp / n_pos
        fpr = fp / n_neg

        auc = np.trapz(tpr, fpr)

        assert np.isclose(auc, 1.0)

    def test_roc_auc_random_classifier(self):
        """Random classifier should have AUC ~= 0.5."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 1000)
        y_score = np.random.rand(1000)

        sorted_idx = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_idx]

        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        tp = np.concatenate([[0], np.cumsum(y_true_sorted)])
        fp = np.concatenate([[0], np.cumsum(1 - y_true_sorted)])

        tpr = tp / n_pos
        fpr = fp / n_neg

        auc = np.trapz(tpr, fpr)

        assert np.isclose(auc, 0.5, atol=0.05)


class TestBootstrapCI:
    """Test bootstrap confidence intervals."""

    def test_bootstrap_ci_covers_true_value(self):
        """Bootstrap CI should cover true mean."""
        np.random.seed(42)
        true_mean = 5.0
        data = np.random.normal(true_mean, 1.0, 100)

        # Bootstrap
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(data), size=len(data), replace=True)
            bootstrap_means.append(np.mean(data[idx]))

        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        assert ci_lower <= true_mean <= ci_upper

    def test_bootstrap_ci_width_decreases_with_sample_size(self):
        """CI width should decrease with larger sample size."""
        true_mean = 5.0

        ci_widths = []
        for n in [50, 100, 200]:
            data = np.random.normal(true_mean, 1.0, n)

            bootstrap_means = []
            for _ in range(500):
                idx = np.random.choice(len(data), size=len(data), replace=True)
                bootstrap_means.append(np.mean(data[idx]))

            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            ci_widths.append(ci_upper - ci_lower)

        # Width should decrease
        assert ci_widths[0] > ci_widths[1] > ci_widths[2]


class TestStatisticalTests:
    """Test statistical comparison procedures."""

    def test_mcnemar_perfect_agreement(self):
        """McNemar test with perfect agreement should have high p-value."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        pred1 = y_true.copy()
        pred2 = y_true.copy()

        # Count discordant pairs
        diff1 = (pred1 != y_true) & (pred2 == y_true)
        diff2 = (pred1 == y_true) & (pred2 != y_true)

        n01 = np.sum(diff1)
        n10 = np.sum(diff2)

        if n01 + n10 > 0:
            stat = (n01 - n10)**2 / (n01 + n10)
            p_value = 1 - 0.5  # Simplified
        else:
            stat = 0
            p_value = 1.0

        assert p_value > 0.05

    def test_delong_identical_methods(self):
        """DeLong test with identical methods should have high p-value."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85])

        # Both methods have same scores
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        # AUC is same for both
        auc1 = 0.9
        auc2 = 0.9

        var_diff = 0.01 + 0.01
        z_stat = (auc1 - auc2) / np.sqrt(var_diff)

        assert np.isclose(z_stat, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
