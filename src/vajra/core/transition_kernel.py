"""
Action-conditioned transition kernels K_a(z'|z).

The dynamics are:
    z(t+1) ~ K_{a(t)}(·|z(t))

This module provides various transition kernel implementations:
- Deterministic: z' = T_a(z)
- Gaussian: z' ~ N(μ_a(z), Σ_a)
- Heteroscedastic Gaussian: z' ~ N(μ_a(z), Σ_a(z))
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Union
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from vajra.core.latent_state import LatentState, LatentSpace


class TransitionKernel(ABC):
    """
    Abstract base class for action-conditioned transition kernels.

    A transition kernel K_a defines a probability distribution over
    next states z' given current state z and action a.
    """

    @abstractmethod
    def sample(
        self,
        z: LatentState,
        action: Any,
        rng: Optional[np.random.Generator] = None
    ) -> LatentState:
        """Sample z' ~ K_a(·|z)."""
        pass

    @abstractmethod
    def log_prob(
        self,
        z: LatentState,
        action: Any,
        z_next: LatentState
    ) -> float:
        """Compute log K_a(z'|z)."""
        pass

    def prob(self, z: LatentState, action: Any, z_next: LatentState) -> float:
        """Compute K_a(z'|z)."""
        return np.exp(self.log_prob(z, action, z_next))

    @abstractmethod
    def mean(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        """Compute E[z'|z, a]."""
        pass

    def variance(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        """Compute Var[z'|z, a] (diagonal)."""
        raise NotImplementedError("Variance not implemented for this kernel")


class DeterministicTransition(TransitionKernel):
    """
    Deterministic transition z' = T_a(z).

    Used for systems with deterministic dynamics or as a mean-function
    baseline. Also used for structural equivalence testing.
    """

    def __init__(
        self,
        transition_fn: Callable[[NDArray, Any], NDArray],
        latent_dim: int,
        noise_std: float = 0.0
    ):
        """
        Initialize deterministic transition.

        Args:
            transition_fn: Function (z, a) -> z' mapping state and action to next state
            latent_dim: Dimensionality of latent space
            noise_std: Optional noise standard deviation (for slightly stochastic version)
        """
        self.transition_fn = transition_fn
        self.latent_dim = latent_dim
        self.noise_std = noise_std

    def sample(
        self,
        z: LatentState,
        action: Any,
        rng: Optional[np.random.Generator] = None
    ) -> LatentState:
        if rng is None:
            rng = np.random.default_rng()

        z_next = self.transition_fn(z.vector, action)

        if self.noise_std > 0:
            noise = rng.normal(0, self.noise_std, size=self.latent_dim)
            z_next = z_next + noise

        return LatentState(
            vector=z_next,
            metadata={"action": action, "parent_id": z.state_id},
            timestamp=z.timestamp + 1 if z.timestamp is not None else None
        )

    def log_prob(self, z: LatentState, action: Any, z_next: LatentState) -> float:
        mean = self.transition_fn(z.vector, action)

        if self.noise_std > 0:
            # Gaussian log probability
            diff = z_next.vector - mean
            log_p = -0.5 * np.sum((diff / self.noise_std) ** 2)
            log_p -= 0.5 * self.latent_dim * np.log(2 * np.pi)
            log_p -= self.latent_dim * np.log(self.noise_std)
            return float(log_p)
        else:
            # Delta distribution
            if np.allclose(z_next.vector, mean):
                return 0.0  # log(1)
            return float('-inf')  # log(0)

    def mean(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        return self.transition_fn(z.vector, action)


class GaussianTransitionKernel(TransitionKernel):
    """
    Gaussian transition kernel with constant covariance.

    z' ~ N(μ_a(z), Σ)

    The mean function μ_a can be learned; the covariance is fixed.
    """

    def __init__(
        self,
        mean_fn: Callable[[NDArray, Any], NDArray],
        covariance: Union[float, NDArray],
        latent_dim: int
    ):
        """
        Initialize Gaussian transition kernel.

        Args:
            mean_fn: Function (z, a) -> μ computing the mean
            covariance: Covariance matrix or scalar (isotropic)
            latent_dim: Dimensionality of latent space
        """
        self.mean_fn = mean_fn
        self.latent_dim = latent_dim

        if isinstance(covariance, (int, float)):
            self.covariance = np.eye(latent_dim) * covariance
            self.std = np.sqrt(covariance)
            self.is_isotropic = True
        else:
            self.covariance = np.asarray(covariance)
            self.std = np.sqrt(np.diag(self.covariance))
            self.is_isotropic = False

        # Precompute for log_prob
        self.log_det_cov = np.log(np.linalg.det(self.covariance))
        self.cov_inv = np.linalg.inv(self.covariance)

    def sample(
        self,
        z: LatentState,
        action: Any,
        rng: Optional[np.random.Generator] = None
    ) -> LatentState:
        if rng is None:
            rng = np.random.default_rng()

        mean = self.mean_fn(z.vector, action)

        if self.is_isotropic:
            z_next = rng.normal(mean, self.std)
        else:
            z_next = rng.multivariate_normal(mean, self.covariance)

        return LatentState(
            vector=z_next,
            metadata={"action": action, "parent_id": z.state_id},
            timestamp=z.timestamp + 1 if z.timestamp is not None else None
        )

    def log_prob(self, z: LatentState, action: Any, z_next: LatentState) -> float:
        mean = self.mean_fn(z.vector, action)
        diff = z_next.vector - mean

        # log N(z'; μ, Σ)
        log_p = -0.5 * self.latent_dim * np.log(2 * np.pi)
        log_p -= 0.5 * self.log_det_cov
        log_p -= 0.5 * diff @ self.cov_inv @ diff

        return float(log_p)

    def mean(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        return self.mean_fn(z.vector, action)

    def variance(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        return np.diag(self.covariance)


class HeteroscedasticTransitionKernel(TransitionKernel):
    """
    Heteroscedastic Gaussian transition kernel.

    z' ~ N(μ_a(z), diag(σ²_a(z)))

    Both mean and variance depend on the current state and action.
    This is crucial for modeling conditional uncertainty (Section 7).
    """

    def __init__(
        self,
        mean_fn: Callable[[NDArray, Any], NDArray],
        var_fn: Callable[[NDArray, Any], NDArray],
        latent_dim: int,
        min_var: float = 1e-6
    ):
        """
        Initialize heteroscedastic kernel.

        Args:
            mean_fn: Function (z, a) -> μ(z, a)
            var_fn: Function (z, a) -> σ²(z, a) (diagonal variances)
            latent_dim: Dimensionality of latent space
            min_var: Minimum variance (numerical stability)
        """
        self.mean_fn = mean_fn
        self.var_fn = var_fn
        self.latent_dim = latent_dim
        self.min_var = min_var

    def sample(
        self,
        z: LatentState,
        action: Any,
        rng: Optional[np.random.Generator] = None
    ) -> LatentState:
        if rng is None:
            rng = np.random.default_rng()

        mean = self.mean_fn(z.vector, action)
        var = np.maximum(self.var_fn(z.vector, action), self.min_var)
        std = np.sqrt(var)

        z_next = rng.normal(mean, std)

        return LatentState(
            vector=z_next,
            metadata={
                "action": action,
                "parent_id": z.state_id,
                "transition_std": std.tolist()
            },
            timestamp=z.timestamp + 1 if z.timestamp is not None else None
        )

    def log_prob(self, z: LatentState, action: Any, z_next: LatentState) -> float:
        mean = self.mean_fn(z.vector, action)
        var = np.maximum(self.var_fn(z.vector, action), self.min_var)

        diff = z_next.vector - mean

        # log N(z'; μ, diag(σ²))
        log_p = -0.5 * self.latent_dim * np.log(2 * np.pi)
        log_p -= 0.5 * np.sum(np.log(var))  # log det
        log_p -= 0.5 * np.sum(diff ** 2 / var)

        return float(log_p)

    def mean(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        return self.mean_fn(z.vector, action)

    def variance(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        return np.maximum(self.var_fn(z.vector, action), self.min_var)

    def nll(self, z: LatentState, action: Any, z_next: LatentState) -> float:
        """Negative log likelihood (for training)."""
        return -self.log_prob(z, action, z_next)


class NeuralTransitionKernel(TransitionKernel, nn.Module):
    """
    Neural network-based heteroscedastic transition kernel.

    Uses a neural network to predict both mean and log-variance,
    following the Deep Kalman Filter architecture.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [64, 64],
        min_std: float = 1e-3,
        max_std: float = 10.0
    ):
        """
        Initialize neural transition kernel.

        Args:
            latent_dim: Dimensionality of latent space
            action_dim: Dimensionality of action space
            hidden_dims: Hidden layer dimensions
            min_std: Minimum standard deviation
            max_std: Maximum standard deviation
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        # Build network
        layers = []
        input_dim = latent_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, latent_dim)
        self.logvar_head = nn.Linear(input_dim, latent_dim)

        # Initialize with small outputs
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.logvar_head.bias)
        nn.init.normal_(self.mean_head.weight, std=0.01)
        nn.init.normal_(self.logvar_head.weight, std=0.01)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std."""
        x = torch.cat([z, action], dim=-1)
        h = self.shared(x)

        mean = self.mean_head(h)
        logvar = self.logvar_head(h)

        # Clamp std to reasonable range
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, self.min_std, self.max_std)

        return mean, std

    def _encode_action(self, action: Any) -> NDArray:
        """Convert action to vector."""
        if isinstance(action, (int, np.integer)):
            # One-hot encode discrete action
            vec = np.zeros(self.action_dim)
            vec[action] = 1.0
            return vec
        elif isinstance(action, np.ndarray):
            return action.astype(np.float64)
        else:
            return np.array([action], dtype=np.float64)

    def sample(
        self,
        z: LatentState,
        action: Any,
        rng: Optional[np.random.Generator] = None
    ) -> LatentState:
        if rng is None:
            rng = np.random.default_rng()

        z_tensor = torch.FloatTensor(z.vector).unsqueeze(0)
        a_tensor = torch.FloatTensor(self._encode_action(action)).unsqueeze(0)

        with torch.no_grad():
            mean, std = self.forward(z_tensor, a_tensor)
            mean = mean.squeeze(0).numpy()
            std = std.squeeze(0).numpy()

        z_next = rng.normal(mean, std)

        return LatentState(
            vector=z_next,
            metadata={
                "action": action,
                "parent_id": z.state_id,
                "transition_std": std.tolist()
            },
            timestamp=z.timestamp + 1 if z.timestamp is not None else None
        )

    def log_prob(self, z: LatentState, action: Any, z_next: LatentState) -> float:
        z_tensor = torch.FloatTensor(z.vector).unsqueeze(0)
        a_tensor = torch.FloatTensor(self._encode_action(action)).unsqueeze(0)
        z_next_tensor = torch.FloatTensor(z_next.vector).unsqueeze(0)

        with torch.no_grad():
            mean, std = self.forward(z_tensor, a_tensor)

            # Gaussian log probability
            var = std ** 2
            log_p = -0.5 * torch.sum(torch.log(2 * np.pi * var))
            log_p -= 0.5 * torch.sum((z_next_tensor - mean) ** 2 / var)

        return float(log_p.item())

    def mean(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        z_tensor = torch.FloatTensor(z.vector).unsqueeze(0)
        a_tensor = torch.FloatTensor(self._encode_action(action)).unsqueeze(0)

        with torch.no_grad():
            mean, _ = self.forward(z_tensor, a_tensor)

        return mean.squeeze(0).numpy()

    def variance(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        z_tensor = torch.FloatTensor(z.vector).unsqueeze(0)
        a_tensor = torch.FloatTensor(self._encode_action(action)).unsqueeze(0)

        with torch.no_grad():
            _, std = self.forward(z_tensor, a_tensor)

        return (std ** 2).squeeze(0).numpy()

    def nll_loss(
        self,
        z_batch: torch.Tensor,
        action_batch: torch.Tensor,
        z_next_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log likelihood loss for training.

        This implements the heteroscedastic NLL from Section 7:
        ℓ = 0.5 log(2π) + 0.5 log(σ²) + (y-μ)²/(2σ²)
        """
        mean, std = self.forward(z_batch, action_batch)
        var = std ** 2

        # NLL per sample
        nll = 0.5 * torch.log(2 * np.pi * var)
        nll += 0.5 * (z_next_batch - mean) ** 2 / var
        nll = nll.sum(dim=-1)  # Sum over dimensions

        return nll.mean()  # Mean over batch


class DiscreteTransitionMatrix(TransitionKernel):
    """
    Discrete transition kernel defined by transition matrices.

    Used for the lumpability construction (Section 3) and exact
    structural equivalence computations (Section 5).
    """

    def __init__(
        self,
        transition_matrices: Dict[Any, NDArray],
        n_states: int
    ):
        """
        Initialize discrete transition kernel.

        Args:
            transition_matrices: Dict mapping actions to transition matrices P_a
                                 where P_a[i,j] = P(s'=j | s=i, a)
            n_states: Number of discrete states
        """
        self.n_states = n_states
        self.transition_matrices = {}

        for action, P in transition_matrices.items():
            P = np.asarray(P, dtype=np.float64)
            assert P.shape == (n_states, n_states), f"Matrix shape mismatch for action {action}"
            # Normalize rows
            row_sums = P.sum(axis=1, keepdims=True)
            P = P / row_sums
            self.transition_matrices[action] = P

        self.actions = list(self.transition_matrices.keys())

    def sample(
        self,
        z: LatentState,
        action: Any,
        rng: Optional[np.random.Generator] = None
    ) -> LatentState:
        if rng is None:
            rng = np.random.default_rng()

        # Get current state index from one-hot or metadata
        if "discrete_index" in z.metadata:
            state_idx = z.metadata["discrete_index"]
        else:
            state_idx = int(np.argmax(z.vector))

        P = self.transition_matrices[action]
        next_idx = rng.choice(self.n_states, p=P[state_idx])

        # Create one-hot encoding
        z_next_vec = np.zeros(self.n_states)
        z_next_vec[next_idx] = 1.0

        return LatentState(
            vector=z_next_vec,
            metadata={"discrete_index": next_idx, "action": action},
            timestamp=z.timestamp + 1 if z.timestamp is not None else None
        )

    def log_prob(self, z: LatentState, action: Any, z_next: LatentState) -> float:
        if "discrete_index" in z.metadata:
            state_idx = z.metadata["discrete_index"]
        else:
            state_idx = int(np.argmax(z.vector))

        if "discrete_index" in z_next.metadata:
            next_idx = z_next.metadata["discrete_index"]
        else:
            next_idx = int(np.argmax(z_next.vector))

        P = self.transition_matrices[action]
        prob = P[state_idx, next_idx]

        return float(np.log(prob + 1e-10))

    def mean(self, z: LatentState, action: Any) -> NDArray[np.float64]:
        if "discrete_index" in z.metadata:
            state_idx = z.metadata["discrete_index"]
        else:
            state_idx = int(np.argmax(z.vector))

        P = self.transition_matrices[action]
        # Expected next state in one-hot space
        return P[state_idx]

    def get_matrix(self, action: Any) -> NDArray:
        """Get the transition matrix for a specific action."""
        return self.transition_matrices[action].copy()

    def get_aggregated_matrix(self, observation_map: Dict[int, int]) -> NDArray:
        """
        Compute the aggregated transition matrix under an observation mapping.

        Used for checking lumpability (Section 3).

        Args:
            observation_map: Maps state indices to observation indices
        """
        n_obs = len(set(observation_map.values()))

        # Aggregate over all actions (assuming single action for now)
        P = list(self.transition_matrices.values())[0]

        # Compute q(c'|s) for each state s and observation c'
        aggregated = np.zeros((self.n_states, n_obs))
        for s in range(self.n_states):
            for s_next in range(self.n_states):
                c_next = observation_map[s_next]
                aggregated[s, c_next] += P[s, s_next]

        return aggregated
