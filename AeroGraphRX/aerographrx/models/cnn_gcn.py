"""CNN-GCN Hybrid Classifier for Modulation Type Detection.

Implements a hybrid architecture combining CNN for spectrogram feature extraction
and GCN for graph-based refinement using adjacency relationships.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """CNN-based encoder for spectrogram feature extraction.

    Extracts features from spectrograms using convolutional layers
    with progressive filter expansion and spatial reduction.

    Args:
        output_dim: Dimension of output feature embedding (default: 128).
    """

    def __init__(self, output_dim=128):
        """Initialize CNN encoder.

        Args:
            output_dim: Output feature embedding dimension.
        """
        super(CNNEncoder, self).__init__()

        # Stage 1: 1 channel -> 32 filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: 32 -> 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: 64 -> 128 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global average pooling followed by output projection
        self.output_dim = output_dim
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        """Forward pass through CNN encoder.

        Args:
            x: Input spectrogram of shape (B, 1, T, F) where
               B is batch size, T is time steps, F is frequency bins.

        Returns:
            Feature embedding of shape (B, output_dim).
        """
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 128)

        # Output projection
        x = self.fc(x)
        return x


class GCNLayer(nn.Module):
    """Graph Convolutional Network (GCN) layer.

    Implements the spectral GCN update rule:
        H^{l+1} = sigma(D^{-1/2} A_tilde D^{-1/2} H^l W^l)

    where A_tilde = A + I (adjacency matrix with self-loops).

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
    """

    def __init__(self, in_features, out_features):
        """Initialize GCN layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix W^l
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, h, adj_norm):
        """Forward pass through GCN layer.

        Args:
            h: Node features of shape (N, in_features).
            adj_norm: Normalized adjacency matrix D^{-1/2} A_tilde D^{-1/2}
                     of shape (N, N).

        Returns:
            Updated node features of shape (N, out_features).
        """
        # h^l: (N, in_features)
        # W^l: (in_features, out_features)
        support = torch.matmul(h, self.weight)  # (N, out_features)

        # Apply normalized adjacency
        output = torch.matmul(adj_norm, support)  # (N, out_features)

        # Add bias and apply activation
        output = output + self.bias
        return output


class CNNGCN(nn.Module):
    """CNN-GCN Hybrid Classifier for modulation type detection.

    Two-stage architecture:
        Stage 1: CNN for spectrogram feature extraction
        Stage 2: Graph convolution for refinement using adjacency relationships

    Args:
        num_classes: Number of modulation types to classify.
        latent_dim: CNN output / GCN input dimension (default: 128).
        num_gcn_layers: Number of GCN layers (default: 2).
    """

    def __init__(self, num_classes, latent_dim=128, num_gcn_layers=2):
        """Initialize CNN-GCN classifier.

        Args:
            num_classes: Number of modulation classes.
            latent_dim: Latent feature dimension.
            num_gcn_layers: Number of GCN layers.
        """
        super(CNNGCN, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_gcn_layers = num_gcn_layers

        # Stage 1: CNN encoder
        self.cnn_encoder = CNNEncoder(output_dim=latent_dim)

        # Stage 2: GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            in_feat = latent_dim
            out_feat = latent_dim
            self.gcn_layers.append(GCNLayer(in_feat, out_feat))

        # Classification head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, spectrograms, adjacency_matrix):
        """Forward pass through CNN-GCN classifier.

        Args:
            spectrograms: Batch of spectrograms of shape (B, 1, T, F).
            adjacency_matrix: Graph adjacency matrix of shape (B, B).

        Returns:
            Class logits of shape (B, num_classes).
        """
        # Stage 1: Extract features via CNN
        features = self.cnn_encoder(spectrograms)  # (B, latent_dim)

        # Normalize adjacency matrix: D^{-1/2} A_tilde D^{-1/2}
        adj_norm = self._normalize_adjacency(adjacency_matrix)

        # Stage 2: Refine features via GCN
        h = features
        for gcn_layer in self.gcn_layers:
            h = F.relu(gcn_layer(h, adj_norm))

        # Classification
        logits = self.classifier(h)
        return logits

    def _normalize_adjacency(self, adj):
        """Normalize adjacency matrix: D^{-1/2} A_tilde D^{-1/2}.

        Args:
            adj: Adjacency matrix of shape (N, N).

        Returns:
            Normalized adjacency matrix.
        """
        # Add self-loops: A_tilde = A + I
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=adj.device)

        # Compute degree matrix D
        degree = adj_with_self_loops.sum(dim=1)
        degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree + 1e-8))

        # D^{-1/2} A_tilde D^{-1/2}
        adj_norm = degree_sqrt_inv @ adj_with_self_loops @ degree_sqrt_inv
        return adj_norm

    def compute_loss(self, logits, labels, features, laplacian,
                     lambda_reg=0.001, mu_smooth=0.01):
        """Compute total loss: cross-entropy + L2 regularization + graph smoothness.

        Args:
            logits: Model predictions of shape (B, num_classes).
            labels: Ground truth labels of shape (B,).
            features: Learned features of shape (B, latent_dim).
            laplacian: Graph Laplacian matrix of shape (B, B).
            lambda_reg: L2 regularization coefficient (default: 0.001).
            mu_smooth: Graph smoothness coefficient (default: 0.01).

        Returns:
            Total loss (scalar).
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)

        # L2 regularization on weights
        l2_loss = lambda_reg * sum(
            torch.norm(p) ** 2 for p in self.parameters()
        )

        # Graph smoothness loss (Eq. 19)
        # smooth_loss = tr(H^T L H)
        smooth_loss = mu_smooth * torch.trace(
            torch.matmul(features.T, torch.matmul(laplacian, features))
        )

        total_loss = ce_loss + l2_loss + smooth_loss
        return total_loss
