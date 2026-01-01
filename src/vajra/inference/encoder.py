"""
Encoder models for mapping observations to latent representations.

The encoder z = enc(x) maps observable records to latent states.
This enables storing latent transition evidence:
    ẽ_t = (z(t), a(t), z(t+1)) with z(t) = enc(x(t))
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F

from vajra.core.latent_state import LatentState


class Encoder(ABC):
    """Abstract base class for encoders."""

    @abstractmethod
    def encode(self, x: NDArray) -> LatentState:
        """Encode observation to latent state."""
        pass

    @abstractmethod
    def encode_batch(self, x_batch: NDArray) -> NDArray:
        """Encode batch of observations."""
        pass


class MLPEncoder(Encoder, nn.Module):
    """
    MLP-based deterministic encoder.

    Maps observations to latent states via a feedforward network.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = [64, 64],
        activation: str = "relu"
    ):
        """
        Initialize MLP encoder.

        Args:
            obs_dim: Observation dimensionality
            latent_dim: Latent state dimensionality
            hidden_dims: Hidden layer dimensions
            activation: Activation function ("relu", "tanh", "elu")
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        # Build network
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def encode(self, x: NDArray) -> LatentState:
        """Encode single observation."""
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        with torch.no_grad():
            z = self.forward(x_tensor).squeeze(0).numpy()
        return LatentState(vector=z)

    def encode_batch(self, x_batch: NDArray) -> NDArray:
        """Encode batch of observations."""
        x_tensor = torch.FloatTensor(x_batch)
        with torch.no_grad():
            z = self.forward(x_tensor).numpy()
        return z


class VAEEncoder(Encoder, nn.Module):
    """
    Variational Autoencoder encoder.

    Maps observations to latent distributions, enabling uncertainty
    quantification in the latent space.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = [64, 64],
        min_std: float = 1e-3
    ):
        """
        Initialize VAE encoder.

        Args:
            obs_dim: Observation dimensionality
            latent_dim: Latent state dimensionality
            hidden_dims: Hidden layer dimensions
            min_std: Minimum standard deviation
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.min_std = min_std

        # Build shared network
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, latent_dim)
        self.logvar_head = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and log-variance."""
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from the encoder distribution using reparameterization.

        Returns:
            z: Sampled latent vectors
            mean: Distribution means
            logvar: Log-variances
        """
        mean, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=self.min_std)

        # Reparameterization trick
        if n_samples == 1:
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            eps = torch.randn(n_samples, *std.shape)
            z = mean.unsqueeze(0) + eps * std.unsqueeze(0)

        return z, mean, logvar

    def encode(self, x: NDArray) -> LatentState:
        """Encode single observation (returns mean)."""
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        with torch.no_grad():
            mean, logvar = self.forward(x_tensor)
            z = mean.squeeze(0).numpy()
            std = torch.exp(0.5 * logvar).squeeze(0).numpy()

        return LatentState(
            vector=z,
            metadata={"encoding_std": std.tolist()}
        )

    def encode_batch(self, x_batch: NDArray) -> NDArray:
        """Encode batch (returns means)."""
        x_tensor = torch.FloatTensor(x_batch)
        with torch.no_grad():
            mean, _ = self.forward(x_tensor)
        return mean.numpy()

    def kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from prior N(0,I).

        Used for VAE training (ELBO objective).
        """
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)


class Decoder(nn.Module):
    """
    Decoder network for VAE.

    Maps latent states back to observation space.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dims: list[int] = [64, 64],
        heteroscedastic: bool = False,
        min_std: float = 1e-3
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.heteroscedastic = heteroscedastic
        self.min_std = min_std

        # Build network
        layers = []
        input_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, obs_dim)

        if heteroscedastic:
            self.logvar_head = nn.Linear(input_dim, obs_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(obs_dim))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return reconstructed mean and std."""
        h = self.shared(z)
        mean = self.mean_head(h)

        if self.heteroscedastic:
            logvar = self.logvar_head(h)
            std = torch.exp(0.5 * logvar)
        else:
            std = torch.exp(self.log_std).expand_as(mean)

        std = torch.clamp(std, min=self.min_std)
        return mean, std

    def reconstruction_loss(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor
    ) -> torch.Tensor:
        """Negative log likelihood of reconstruction."""
        mean, std = self.forward(z)
        var = std ** 2

        nll = 0.5 * torch.log(2 * np.pi * var)
        nll += 0.5 * (x_target - mean) ** 2 / var
        return nll.sum(dim=-1).mean()


class VAE(nn.Module):
    """
    Complete Variational Autoencoder.

    Combines encoder and decoder with training functionality.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = [64, 64],
        beta: float = 1.0
    ):
        """
        Initialize VAE.

        Args:
            obs_dim: Observation dimensionality
            latent_dim: Latent state dimensionality
            hidden_dims: Hidden layer dimensions
            beta: KL divergence weight (beta-VAE)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = VAEEncoder(obs_dim, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, obs_dim, hidden_dims[::-1])

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Returns:
            x_recon: Reconstructed observations
            z: Latent samples
            mean: Encoder means
            logvar: Encoder log-variances
        """
        z, mean, logvar = self.encoder.sample(x)
        x_recon, _ = self.decoder(z)
        return x_recon, z, mean, logvar

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute ELBO loss.

        Returns:
            loss: Total loss
            metrics: Dictionary of component losses
        """
        z, mean, logvar = self.encoder.sample(x)

        recon_loss = self.decoder.reconstruction_loss(z, x)
        kl_loss = self.encoder.kl_divergence(mean, logvar).mean()

        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
        }

    def encode(self, x: NDArray) -> LatentState:
        """Encode observation."""
        return self.encoder.encode(x)

    def encode_batch(self, x_batch: NDArray) -> NDArray:
        """Encode batch."""
        return self.encoder.encode_batch(x_batch)

    def decode(self, z: NDArray) -> NDArray:
        """Decode latent state to observation."""
        z_tensor = torch.FloatTensor(z).unsqueeze(0)
        with torch.no_grad():
            x_recon, _ = self.decoder(z_tensor)
        return x_recon.squeeze(0).numpy()


class IdentityEncoder(Encoder):
    """
    Identity encoder (no transformation).

    Used when observations are directly usable as latent states.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, x: NDArray) -> LatentState:
        return LatentState(vector=x.copy())

    def encode_batch(self, x_batch: NDArray) -> NDArray:
        return x_batch.copy()
