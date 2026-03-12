"""Variational Autoencoder (VAE) for Novelty Detection.

Implements a VAE for unsupervised anomaly and novelty detection in signal data,
with Doppler drift estimation and entity classification (Algorithm 3).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal


class VAEEncoder(nn.Module):
    """VAE encoder (inference network).

    Maps input signal to latent distribution parameters (mu, log_var).

    Args:
        input_dim: Input signal dimension.
        latent_dim: Latent space dimension (default: 32).
    """

    def __init__(self, input_dim, latent_dim=32):
        """Initialize VAE encoder.

        Args:
            input_dim: Input signal dimension.
            latent_dim: Latent space dimension.
        """
        super(VAEEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # MLP: input_dim -> 256 -> 128 -> latent_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)

    def forward(self, x):
        """Forward pass through encoder.

        Args:
            x: Input signal of shape (B, input_dim).

        Returns:
            Tuple of (mu, log_var) each of shape (B, latent_dim).
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    """VAE decoder (generative network).

    Maps latent code to reconstructed signal.

    Args:
        latent_dim: Latent space dimension (default: 32).
        output_dim: Output signal dimension.
    """

    def __init__(self, latent_dim, output_dim):
        """Initialize VAE decoder.

        Args:
            latent_dim: Latent space dimension.
            output_dim: Output signal dimension.
        """
        super(VAEDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # MLP: latent_dim -> 128 -> 256 -> output_dim
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, z):
        """Forward pass through decoder.

        Args:
            z: Latent code of shape (B, latent_dim).

        Returns:
            Reconstructed signal of shape (B, output_dim).
        """
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_recon = self.fc3(h)
        return x_recon


class SignalVAE(nn.Module):
    """Variational Autoencoder for signal novelty detection.

    Full VAE implementation with ELBO loss, novelty scoring, and entity classification.

    Args:
        input_dim: Input signal dimension.
        latent_dim: Latent space dimension (default: 32).
    """

    def __init__(self, input_dim, latent_dim=32):
        """Initialize Signal VAE.

        Args:
            input_dim: Input signal dimension.
            latent_dim: Latent space dimension.
        """
        super(SignalVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: sample z ~ N(mu, exp(log_var)).

        Args:
            mu: Mean of latent distribution of shape (B, latent_dim).
            log_var: Log variance of latent distribution of shape (B, latent_dim).

        Returns:
            Sampled latent code of shape (B, latent_dim).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """Forward pass through VAE.

        Args:
            x: Input signal of shape (B, input_dim).

        Returns:
            Tuple of (x_recon, mu, log_var) where:
                - x_recon: Reconstructed signal of shape (B, input_dim)
                - mu: Mean of posterior of shape (B, latent_dim)
                - log_var: Log variance of posterior of shape (B, latent_dim)
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def compute_elbo(self, x, x_recon, mu, log_var):
        """Compute Evidence Lower BOund (ELBO) loss (Eq. 21).

        ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

        Args:
            x: Original signal of shape (B, input_dim).
            x_recon: Reconstructed signal of shape (B, input_dim).
            mu: Mean of posterior of shape (B, latent_dim).
            log_var: Log variance of posterior of shape (B, latent_dim).

        Returns:
            Scalar ELBO loss.
        """
        # Reconstruction loss (Gaussian likelihood)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence loss: KL(N(mu, exp(log_var)) || N(0, I))
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )

        elbo = recon_loss + kl_loss
        return elbo

    def novelty_score(self, x, beta=0.1):
        """Compute novelty score for signal (Eq. 22).

        novelty_score = ||x - x_hat||^2 + beta * KL(q(z|x) || p(z))

        Args:
            x: Input signal of shape (B, input_dim).
            beta: KL weighting coefficient (default: 0.1).

        Returns:
            Novelty scores of shape (B,).
        """
        x_recon, mu, log_var = self.forward(x)

        # Reconstruction error
        recon_error = torch.sum((x - x_recon) ** 2, dim=1)  # (B,)

        # KL divergence
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        # Combined score
        score = recon_error + beta * kl
        return score

    def doppler_drift(self, signal_np, f0, dt):
        """Estimate Doppler drift rate from signal (Eq. 23).

        Computes df/dt by analyzing the rate of change of dominant frequency.

        Args:
            signal_np: Signal as numpy array of shape (T,).
            f0: Nominal frequency (reference).
            dt: Time step between samples.

        Returns:
            Estimated Doppler drift rate df/dt (scalar).
        """
        # Compute STFT
        f, t, Sxx = scipy_signal.spectrogram(
            signal_np,
            fs=1.0 / dt,
            nperseg=min(256, len(signal_np))
        )

        # Find dominant frequency at each time step
        dominant_freqs = f[np.argmax(Sxx, axis=0)]

        # Estimate df/dt via linear regression
        t_indices = np.arange(len(dominant_freqs))
        if len(t_indices) < 2:
            return 0.0

        # Fit line to dominant frequencies
        coeffs = np.polyfit(t_indices, dominant_freqs, 1)
        df_dt = coeffs[0]  # Slope

        return df_dt

    def classify(self, x, tau_recon, f_dot_min=0.1):
        """Classify entity type using novelty score and Doppler drift (Algorithm 3).

        Returns 'known', 'unknown', or 'et-candidate' based on reconstruction
        error threshold and Doppler drift characteristics.

        Args:
            x: Input signal, either torch tensor of shape (B, input_dim)
               or numpy array of shape (T,).
            tau_recon: Reconstruction error threshold.
            f_dot_min: Minimum Doppler drift for ET classification (default: 0.1).

        Returns:
            Classification string: 'known', 'unknown', or 'et-candidate'.
        """
        # Convert numpy to torch if needed
        if isinstance(x, np.ndarray):
            x_torch = torch.from_numpy(x).float()
            if x_torch.dim() == 1:
                x_torch = x_torch.unsqueeze(0)
        else:
            x_torch = x
            if x_torch.dim() == 1:
                x_torch = x_torch.unsqueeze(0)

        # Compute novelty score
        novelty_scores = self.novelty_score(x_torch)  # (B,)
        recon_error = novelty_scores.mean().item()

        # Check against threshold
        if recon_error < tau_recon:
            # Known signal
            return 'known'
        else:
            # Unknown: estimate Doppler drift
            if isinstance(x, np.ndarray):
                signal_np = x
            else:
                signal_np = x_torch[0].detach().cpu().numpy()

            # Estimate drift
            f0_est = 1.0  # Nominal frequency estimate
            dt = 1.0  # Time step
            f_dot = self.doppler_drift(signal_np, f0_est, dt)

            if abs(f_dot) > f_dot_min:
                return 'et-candidate'
            else:
                return 'unknown'
