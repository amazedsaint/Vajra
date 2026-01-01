"""
Emission models E(x|z) for mapping latent states to observations.

The observation model:
    x(t) ~ E(·|z(t))

This provides the likelihood of observations given latent states,
used for belief updates and inference.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable, Union
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from vajra.core.latent_state import LatentState


class EmissionModel(ABC):
    """
    Abstract base class for emission models E(x|z).

    Maps latent states z to observable records x.
    """

    @abstractmethod
    def sample(
        self,
        z: LatentState,
        rng: Optional[np.random.Generator] = None
    ) -> NDArray[np.float64]:
        """Sample x ~ E(·|z)."""
        pass

    @abstractmethod
    def log_prob(self, z: LatentState, x: NDArray) -> float:
        """Compute log E(x|z)."""
        pass

    def prob(self, z: LatentState, x: NDArray) -> float:
        """Compute E(x|z)."""
        return np.exp(self.log_prob(z, x))

    @abstractmethod
    def mean(self, z: LatentState) -> NDArray[np.float64]:
        """Compute E[x|z]."""
        pass


class GaussianEmission(EmissionModel):
    """
    Gaussian emission model with state-dependent mean.

    x ~ N(f(z), Σ)

    where f is an observation function and Σ is fixed covariance.
    """

    def __init__(
        self,
        obs_fn: Callable[[NDArray], NDArray],
        obs_dim: int,
        noise_std: Union[float, NDArray] = 0.1
    ):
        """
        Initialize Gaussian emission model.

        Args:
            obs_fn: Function z -> μ(z) mapping latent to observation mean
            obs_dim: Dimensionality of observations
            noise_std: Observation noise (scalar or vector)
        """
        self.obs_fn = obs_fn
        self.obs_dim = obs_dim

        if isinstance(noise_std, (int, float)):
            self.noise_std = np.full(obs_dim, noise_std)
        else:
            self.noise_std = np.asarray(noise_std)

        self.noise_var = self.noise_std ** 2

    def sample(
        self,
        z: LatentState,
        rng: Optional[np.random.Generator] = None
    ) -> NDArray[np.float64]:
        if rng is None:
            rng = np.random.default_rng()

        mean = self.obs_fn(z.vector)
        return rng.normal(mean, self.noise_std)

    def log_prob(self, z: LatentState, x: NDArray) -> float:
        mean = self.obs_fn(z.vector)
        diff = x - mean

        log_p = -0.5 * self.obs_dim * np.log(2 * np.pi)
        log_p -= np.sum(np.log(self.noise_std))
        log_p -= 0.5 * np.sum(diff ** 2 / self.noise_var)

        return float(log_p)

    def mean(self, z: LatentState) -> NDArray[np.float64]:
        return self.obs_fn(z.vector)


class DiscreteEmission(EmissionModel):
    """
    Discrete/deterministic emission for finite observation spaces.

    Used for the lumpability construction where x = φ(s).
    """

    def __init__(
        self,
        emission_map: Callable[[int], int],
        n_states: int,
        n_observations: int
    ):
        """
        Initialize discrete emission.

        Args:
            emission_map: Function s -> x mapping state index to observation
            n_states: Number of hidden states
            n_observations: Number of possible observations
        """
        self.emission_map = emission_map
        self.n_states = n_states
        self.n_observations = n_observations

        # Build observation classes
        self.obs_classes = {}
        for s in range(n_states):
            obs = emission_map(s)
            if obs not in self.obs_classes:
                self.obs_classes[obs] = []
            self.obs_classes[obs].append(s)

    def sample(
        self,
        z: LatentState,
        rng: Optional[np.random.Generator] = None
    ) -> NDArray[np.float64]:
        # Get state index
        if "discrete_index" in z.metadata:
            state_idx = z.metadata["discrete_index"]
        else:
            state_idx = int(np.argmax(z.vector))

        obs_idx = self.emission_map(state_idx)

        # Return one-hot observation
        obs = np.zeros(self.n_observations)
        obs[obs_idx] = 1.0
        return obs

    def log_prob(self, z: LatentState, x: NDArray) -> float:
        if "discrete_index" in z.metadata:
            state_idx = z.metadata["discrete_index"]
        else:
            state_idx = int(np.argmax(z.vector))

        obs_idx = int(np.argmax(x))
        expected_obs = self.emission_map(state_idx)

        if obs_idx == expected_obs:
            return 0.0  # log(1)
        return float('-inf')  # log(0)

    def mean(self, z: LatentState) -> NDArray[np.float64]:
        return self.sample(z)  # Deterministic

    def get_observation_classes(self) -> dict:
        """Return mapping from observations to state sets."""
        return {k: list(v) for k, v in self.obs_classes.items()}


class NeuralEmission(EmissionModel, nn.Module):
    """
    Neural network emission model.

    Uses a decoder network to map latent states to observation distributions.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dims: list[int] = [64, 64],
        heteroscedastic: bool = True,
        min_std: float = 1e-3
    ):
        """
        Initialize neural emission model.

        Args:
            latent_dim: Dimensionality of latent space
            obs_dim: Dimensionality of observations
            hidden_dims: Hidden layer dimensions
            heteroscedastic: If True, predict state-dependent variance
            min_std: Minimum standard deviation
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.heteroscedastic = heteroscedastic
        self.min_std = min_std

        # Build decoder network
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

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std."""
        h = self.shared(z)
        mean = self.mean_head(h)

        if self.heteroscedastic:
            logvar = self.logvar_head(h)
            std = torch.exp(0.5 * logvar)
        else:
            std = torch.exp(self.log_std).expand_as(mean)

        std = torch.clamp(std, min=self.min_std)
        return mean, std

    def sample(
        self,
        z: LatentState,
        rng: Optional[np.random.Generator] = None
    ) -> NDArray[np.float64]:
        if rng is None:
            rng = np.random.default_rng()

        z_tensor = torch.FloatTensor(z.vector).unsqueeze(0)

        with torch.no_grad():
            mean, std = self.forward(z_tensor)
            mean = mean.squeeze(0).numpy()
            std = std.squeeze(0).numpy()

        return rng.normal(mean, std)

    def log_prob(self, z: LatentState, x: NDArray) -> float:
        z_tensor = torch.FloatTensor(z.vector).unsqueeze(0)
        x_tensor = torch.FloatTensor(x).unsqueeze(0)

        with torch.no_grad():
            mean, std = self.forward(z_tensor)
            var = std ** 2

            log_p = -0.5 * torch.sum(torch.log(2 * np.pi * var))
            log_p -= 0.5 * torch.sum((x_tensor - mean) ** 2 / var)

        return float(log_p.item())

    def mean(self, z: LatentState) -> NDArray[np.float64]:
        z_tensor = torch.FloatTensor(z.vector).unsqueeze(0)

        with torch.no_grad():
            mean, _ = self.forward(z_tensor)

        return mean.squeeze(0).numpy()


class IdentityEmission(EmissionModel):
    """
    Identity emission model where observations equal latent states.

    Useful for testing and when the observation is a direct projection.
    """

    def __init__(self, dim: int, noise_std: float = 0.0):
        self.dim = dim
        self.noise_std = noise_std

    def sample(
        self,
        z: LatentState,
        rng: Optional[np.random.Generator] = None
    ) -> NDArray[np.float64]:
        if self.noise_std > 0:
            if rng is None:
                rng = np.random.default_rng()
            return z.vector + rng.normal(0, self.noise_std, size=self.dim)
        return z.vector.copy()

    def log_prob(self, z: LatentState, x: NDArray) -> float:
        if self.noise_std > 0:
            diff = x - z.vector
            log_p = -0.5 * self.dim * np.log(2 * np.pi)
            log_p -= self.dim * np.log(self.noise_std)
            log_p -= 0.5 * np.sum((diff / self.noise_std) ** 2)
            return float(log_p)
        else:
            if np.allclose(x, z.vector):
                return 0.0
            return float('-inf')

    def mean(self, z: LatentState) -> NDArray[np.float64]:
        return z.vector.copy()
