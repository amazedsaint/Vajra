"""
Posterior approximations for Bayesian operator inference.

Section 8 of the paper shows that operator ambiguity can be intrinsic,
so a simulation-capable context system should maintain uncertainty over K_a.

This module provides practical posterior approximations:
- Ensembles: Maintain M models with different initializations
- Particles: Weight particles by likelihood (Bayesian filtering in parameter space)
- Variational: Optimize q_λ(θ) to approximate the posterior

The predictive distribution integrates over parameter uncertainty:
    p(z'|z, a, D) = ∫ K_{a,θ}(z'|z) p(θ|D) dθ
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from copy import deepcopy

from vajra.core.latent_state import LatentState
from vajra.core.transition_kernel import TransitionKernel, NeuralTransitionKernel
from vajra.evidence.store import EvidenceStore, TransitionEvidence


class PosteriorApproximation(ABC):
    """
    Abstract base class for posterior approximations over dynamics parameters.
    """

    @abstractmethod
    def predict(
        self,
        z: LatentState,
        action: Any,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> List[LatentState]:
        """
        Sample from the predictive distribution.

        Integrates over parameter uncertainty.
        """
        pass

    @abstractmethod
    def predictive_distribution(
        self,
        z: LatentState,
        action: Any
    ) -> Tuple[NDArray, NDArray]:
        """
        Get predictive mean and variance.

        Returns:
            mean: E[z'|z, a, D]
            var: Var[z'|z, a, D] (including epistemic uncertainty)
        """
        pass

    @abstractmethod
    def update(
        self,
        evidence: TransitionEvidence
    ) -> None:
        """Update posterior with new evidence."""
        pass

    @abstractmethod
    def epistemic_uncertainty(
        self,
        z: LatentState,
        action: Any
    ) -> float:
        """Estimate epistemic uncertainty (reducible with more data)."""
        pass


class EnsemblePosterior(PosteriorApproximation):
    """
    Ensemble-based posterior approximation.

    Maintains M models trained with different initializations or
    bootstrap resampling. Epistemic uncertainty comes from model disagreement.

    p(z'|z, a, D) ≈ (1/M) Σ_m K_{a,θ^(m)}(z'|z)
    """

    def __init__(
        self,
        base_kernel_class: type,
        kernel_kwargs: Dict[str, Any],
        n_models: int = 5,
        bootstrap: bool = True
    ):
        """
        Initialize ensemble.

        Args:
            base_kernel_class: Class for individual kernels (e.g., NeuralTransitionKernel)
            kernel_kwargs: Arguments for kernel initialization
            n_models: Number of ensemble members
            bootstrap: Whether to use bootstrap resampling during training
        """
        self.n_models = n_models
        self.bootstrap = bootstrap
        self.kernel_kwargs = kernel_kwargs

        # Initialize ensemble members with different random seeds
        self.models: List[TransitionKernel] = []
        for i in range(n_models):
            torch.manual_seed(i * 42)
            model = base_kernel_class(**kernel_kwargs)
            self.models.append(model)

        self.trained = False

    def predict(
        self,
        z: LatentState,
        action: Any,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> List[LatentState]:
        """Sample from predictive by sampling models then sampling transitions."""
        if rng is None:
            rng = np.random.default_rng()

        samples = []
        for _ in range(n_samples):
            # Select random model
            model_idx = rng.integers(self.n_models)
            model = self.models[model_idx]

            # Sample from that model
            z_next = model.sample(z, action, rng)
            samples.append(z_next)

        return samples

    def predictive_distribution(
        self,
        z: LatentState,
        action: Any
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute predictive mean and variance.

        Variance decomposes into:
        - Aleatoric: E[Var[z'|z,a,θ]] (average variance within models)
        - Epistemic: Var[E[z'|z,a,θ]] (variance of means across models)
        """
        means = []
        variances = []

        for model in self.models:
            means.append(model.mean(z, action))
            if hasattr(model, 'variance'):
                variances.append(model.variance(z, action))

        means = np.array(means)

        # Predictive mean
        pred_mean = means.mean(axis=0)

        # Epistemic variance (model disagreement)
        epistemic_var = means.var(axis=0)

        # Aleatoric variance (if available)
        if variances:
            aleatoric_var = np.array(variances).mean(axis=0)
        else:
            aleatoric_var = np.zeros_like(pred_mean)

        # Total predictive variance
        pred_var = epistemic_var + aleatoric_var

        return pred_mean, pred_var

    def epistemic_uncertainty(self, z: LatentState, action: Any) -> float:
        """Epistemic uncertainty as variance of model means."""
        means = np.array([model.mean(z, action) for model in self.models])
        return float(means.var(axis=0).sum())

    def aleatoric_uncertainty(self, z: LatentState, action: Any) -> float:
        """Aleatoric uncertainty as average model variance."""
        variances = []
        for model in self.models:
            if hasattr(model, 'variance'):
                variances.append(model.variance(z, action).sum())
        return float(np.mean(variances)) if variances else 0.0

    def update(self, evidence: TransitionEvidence) -> None:
        """
        Online update is complex for neural models.
        Use fit() for batch training.
        """
        pass

    def fit(
        self,
        evidence_store: EvidenceStore,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = False
    ) -> Dict[str, List[float]]:
        """
        Train ensemble on evidence store.

        Args:
            evidence_store: Store containing transition evidence
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print training progress
        """
        z_data, a_data, z_next_data = evidence_store.get_training_data(use_latent=True)

        if len(z_data) == 0:
            # Try observable data
            z_data, a_data, z_next_data = evidence_store.get_training_data(use_latent=False)

        if len(z_data) == 0:
            raise ValueError("No training data in evidence store")

        n_samples = len(z_data)
        history = {"loss": []}

        for model_idx, model in enumerate(self.models):
            if not isinstance(model, nn.Module):
                continue

            # Bootstrap resample if enabled
            if self.bootstrap:
                rng = np.random.default_rng(model_idx)
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                z_train = z_data[indices]
                a_train = a_data[indices]
                z_next_train = z_next_data[indices]
            else:
                z_train = z_data
                a_train = a_data
                z_next_train = z_next_data

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            model_losses = []
            for epoch in range(epochs):
                # Shuffle
                perm = np.random.permutation(len(z_train))
                epoch_loss = 0.0

                for i in range(0, len(z_train), batch_size):
                    batch_idx = perm[i:i+batch_size]

                    z_batch = torch.FloatTensor(z_train[batch_idx])
                    a_batch = self._encode_actions(a_train[batch_idx], model)
                    z_next_batch = torch.FloatTensor(z_next_train[batch_idx])

                    optimizer.zero_grad()
                    loss = model.nll_loss(z_batch, a_batch, z_next_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * len(batch_idx)

                epoch_loss /= len(z_train)
                model_losses.append(epoch_loss)

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"Model {model_idx}, Epoch {epoch+1}: loss = {epoch_loss:.4f}")

            history[f"model_{model_idx}"] = model_losses

        self.trained = True
        history["loss"] = [
            np.mean([history[f"model_{i}"][e] for i in range(len(self.models))])
            for e in range(epochs)
        ]

        return history

    def _encode_actions(self, actions: NDArray, model: Any) -> torch.Tensor:
        """Encode actions for model input."""
        if hasattr(model, 'action_dim'):
            action_dim = model.action_dim
            if actions.ndim == 1:
                # Assume integer actions, one-hot encode
                encoded = np.zeros((len(actions), action_dim))
                for i, a in enumerate(actions):
                    if isinstance(a, (int, np.integer)) and a < action_dim:
                        encoded[i, a] = 1.0
                    else:
                        encoded[i, 0] = float(a)
                return torch.FloatTensor(encoded)
        return torch.FloatTensor(actions.reshape(-1, 1) if actions.ndim == 1 else actions)


class ParticlePosterior(PosteriorApproximation):
    """
    Particle-based posterior approximation.

    Maintains particles {θ^(m)} with weights {w^(m)} updated
    sequentially by likelihood (Bayesian filtering in parameter space).

    w^(m) ∝ w^(m) * K_{a,θ^(m)}(z'|z)
    """

    def __init__(
        self,
        base_kernel_class: type,
        kernel_kwargs: Dict[str, Any],
        n_particles: int = 50,
        resample_threshold: float = 0.5
    ):
        """
        Initialize particle posterior.

        Args:
            base_kernel_class: Class for kernels
            kernel_kwargs: Arguments for kernel initialization
            n_particles: Number of parameter particles
            resample_threshold: ESS threshold for resampling (fraction of n_particles)
        """
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.kernel_kwargs = kernel_kwargs
        self.base_kernel_class = base_kernel_class

        # Initialize particles with different random seeds
        self.particles: List[TransitionKernel] = []
        for i in range(n_particles):
            torch.manual_seed(i * 42 + 1000)
            particle = base_kernel_class(**kernel_kwargs)
            self.particles.append(particle)

        self.weights = np.ones(n_particles) / n_particles

    def predict(
        self,
        z: LatentState,
        action: Any,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> List[LatentState]:
        """Sample from predictive by sampling particles then transitions."""
        if rng is None:
            rng = np.random.default_rng()

        samples = []
        particle_indices = rng.choice(
            self.n_particles,
            size=n_samples,
            p=self.weights
        )

        for idx in particle_indices:
            z_next = self.particles[idx].sample(z, action, rng)
            samples.append(z_next)

        return samples

    def predictive_distribution(
        self,
        z: LatentState,
        action: Any
    ) -> Tuple[NDArray, NDArray]:
        """Weighted mean and variance over particles."""
        means = np.array([p.mean(z, action) for p in self.particles])

        # Weighted mean
        pred_mean = np.sum(means * self.weights[:, np.newaxis], axis=0)

        # Weighted variance
        centered = means - pred_mean
        pred_var = np.sum(
            self.weights[:, np.newaxis] * centered ** 2,
            axis=0
        )

        return pred_mean, pred_var

    def epistemic_uncertainty(self, z: LatentState, action: Any) -> float:
        """Epistemic uncertainty from particle spread."""
        _, var = self.predictive_distribution(z, action)
        return float(var.sum())

    def update(self, evidence: TransitionEvidence) -> None:
        """
        Update particle weights with new evidence.

        w^(m) ∝ w^(m) * K_{a,θ^(m)}(z'|z)
        """
        z = LatentState(
            vector=evidence.z_current if evidence.z_current is not None else evidence.x_current
        )
        z_next = LatentState(
            vector=evidence.z_next if evidence.z_next is not None else evidence.x_next
        )
        action = evidence.action

        # Compute log-likelihoods
        log_likes = np.array([
            p.log_prob(z, action, z_next) for p in self.particles
        ])

        # Update log-weights
        log_weights = np.log(self.weights + 1e-10) + log_likes

        # Normalize
        max_log_w = np.max(log_weights)
        self.weights = np.exp(log_weights - max_log_w)
        self.weights = self.weights / self.weights.sum()

        # Check for resampling
        if self.effective_sample_size() < self.resample_threshold * self.n_particles:
            self._resample()

    def effective_sample_size(self) -> float:
        """ESS = 1 / Σ w²."""
        return 1.0 / np.sum(self.weights ** 2)

    def _resample(self, rng: Optional[np.random.Generator] = None) -> None:
        """Resample particles to combat weight degeneracy."""
        if rng is None:
            rng = np.random.default_rng()

        indices = rng.choice(
            self.n_particles,
            size=self.n_particles,
            p=self.weights
        )

        # Deep copy selected particles
        new_particles = [deepcopy(self.particles[i]) for i in indices]
        self.particles = new_particles
        self.weights = np.ones(self.n_particles) / self.n_particles


class MeanFieldVariationalPosterior(PosteriorApproximation):
    """
    Mean-field variational approximation.

    Optimizes q_λ(θ) to minimize KL(q||p(θ|D)).
    Uses a diagonal Gaussian approximation over network weights.

    Note: This is a simplified implementation. Full variational inference
    over neural network weights is computationally intensive.
    """

    def __init__(
        self,
        base_kernel: NeuralTransitionKernel,
        prior_std: float = 1.0
    ):
        """
        Initialize variational posterior.

        Args:
            base_kernel: Base neural transition kernel
            prior_std: Prior standard deviation on weights
        """
        self.base_kernel = base_kernel
        self.prior_std = prior_std

        # Store mean and log-variance for each parameter
        self.param_means = {}
        self.param_logvars = {}

        for name, param in base_kernel.named_parameters():
            self.param_means[name] = param.data.clone()
            # Initialize with small variance
            self.param_logvars[name] = torch.full_like(param.data, -4.0)

    def sample_parameters(self) -> None:
        """Sample network parameters from variational distribution."""
        for name, param in self.base_kernel.named_parameters():
            mean = self.param_means[name]
            std = torch.exp(0.5 * self.param_logvars[name])
            param.data = mean + torch.randn_like(mean) * std

    def set_mean_parameters(self) -> None:
        """Set parameters to variational mean (for evaluation)."""
        for name, param in self.base_kernel.named_parameters():
            param.data = self.param_means[name].clone()

    def predict(
        self,
        z: LatentState,
        action: Any,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> List[LatentState]:
        """Sample by sampling parameters then transitions."""
        if rng is None:
            rng = np.random.default_rng()

        samples = []
        for _ in range(n_samples):
            self.sample_parameters()
            z_next = self.base_kernel.sample(z, action, rng)
            samples.append(z_next)

        return samples

    def predictive_distribution(
        self,
        z: LatentState,
        action: Any,
        n_mc_samples: int = 20
    ) -> Tuple[NDArray, NDArray]:
        """Monte Carlo estimate of predictive moments."""
        samples = []
        for _ in range(n_mc_samples):
            self.sample_parameters()
            mean = self.base_kernel.mean(z, action)
            samples.append(mean)

        samples = np.array(samples)
        return samples.mean(axis=0), samples.var(axis=0)

    def epistemic_uncertainty(
        self,
        z: LatentState,
        action: Any,
        n_mc_samples: int = 20
    ) -> float:
        """MC estimate of epistemic uncertainty."""
        _, var = self.predictive_distribution(z, action, n_mc_samples)
        return float(var.sum())

    def update(self, evidence: TransitionEvidence) -> None:
        """
        Online variational update is complex.
        Use fit() for batch optimization.
        """
        pass

    def kl_divergence(self) -> torch.Tensor:
        """KL divergence from prior."""
        kl = torch.tensor(0.0)
        for name in self.param_means:
            mean = self.param_means[name]
            logvar = self.param_logvars[name]
            var = torch.exp(logvar)

            # KL(q||N(0, prior_std²))
            kl += 0.5 * torch.sum(
                (mean ** 2 + var) / self.prior_std ** 2
                - 1 - logvar + 2 * np.log(self.prior_std)
            )

        return kl
