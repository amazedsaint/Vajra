"""
Belief state representations for POMDPs.

A belief state b_t(z) is a probability distribution over z(t) given
the observable history. This is the sufficient statistic for
decision-making under partial observability.

The predictive distribution for the next observation under action a:
    p(x(t+1) | history, a) = ∫∫ E(x(t+1)|z(t+1)) K_a(z(t+1)|z(t)) b_t(z(t)) dz(t) dz(t+1)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Any, Tuple
import numpy as np
from numpy.typing import NDArray

from vajra.core.latent_state import LatentState, LatentSpace
from vajra.core.transition_kernel import TransitionKernel
from vajra.core.emission_model import EmissionModel


class BeliefState(ABC):
    """
    Abstract base class for belief state representations.

    A belief state represents uncertainty over the latent state z(t)
    given all observations up to time t.
    """

    @abstractmethod
    def mean(self) -> NDArray[np.float64]:
        """Expected latent state E[z]."""
        pass

    @abstractmethod
    def covariance(self) -> NDArray[np.float64]:
        """Covariance Cov[z]."""
        pass

    @abstractmethod
    def sample(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> List[LatentState]:
        """Sample from the belief distribution."""
        pass

    @abstractmethod
    def update(
        self,
        observation: NDArray,
        emission: EmissionModel
    ) -> BeliefState:
        """Update belief given a new observation (Bayes rule)."""
        pass

    @abstractmethod
    def predict(
        self,
        action: Any,
        transition: TransitionKernel
    ) -> BeliefState:
        """Predict belief after taking an action."""
        pass

    def entropy(self) -> float:
        """Differential entropy of the belief (if computable)."""
        raise NotImplementedError

    @property
    def uncertainty(self) -> float:
        """Scalar measure of belief uncertainty (trace of covariance)."""
        return float(np.trace(self.covariance()))


@dataclass
class GaussianBeliefState(BeliefState):
    """
    Gaussian belief state: b(z) = N(μ, Σ).

    Efficient for linear-Gaussian systems (Kalman filtering).
    """
    mean_vec: NDArray[np.float64]
    cov_mat: NDArray[np.float64]

    def __post_init__(self):
        self.mean_vec = np.asarray(self.mean_vec, dtype=np.float64)
        self.cov_mat = np.asarray(self.cov_mat, dtype=np.float64)
        self.dim = len(self.mean_vec)

    def mean(self) -> NDArray[np.float64]:
        return self.mean_vec.copy()

    def covariance(self) -> NDArray[np.float64]:
        return self.cov_mat.copy()

    def sample(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> List[LatentState]:
        if rng is None:
            rng = np.random.default_rng()

        samples = rng.multivariate_normal(self.mean_vec, self.cov_mat, size=n)
        return [LatentState(vector=s) for s in samples]

    def update(
        self,
        observation: NDArray,
        emission: EmissionModel
    ) -> GaussianBeliefState:
        """
        Kalman-style update for Gaussian emission with linear observation.

        For general emissions, use particle-based belief.
        """
        # Simplified update assuming identity observation matrix
        # In practice, would need emission model Jacobian
        obs = np.asarray(observation)

        # Get observation mean and variance from emission
        z_sample = LatentState(vector=self.mean_vec)
        obs_mean = emission.mean(z_sample)

        if hasattr(emission, 'noise_var'):
            obs_var = emission.noise_var
        else:
            obs_var = np.ones(len(obs)) * 0.1

        # Kalman gain (simplified for diagonal observation noise)
        K = self.cov_mat @ np.linalg.inv(self.cov_mat + np.diag(obs_var))

        # Update
        innovation = obs - obs_mean
        new_mean = self.mean_vec + K @ innovation
        new_cov = (np.eye(self.dim) - K) @ self.cov_mat

        return GaussianBeliefState(mean_vec=new_mean, cov_mat=new_cov)

    def predict(
        self,
        action: Any,
        transition: TransitionKernel
    ) -> GaussianBeliefState:
        """
        Predict belief after action (propagate uncertainty).
        """
        # Get transition mean and variance
        z_sample = LatentState(vector=self.mean_vec)
        new_mean = transition.mean(z_sample, action)

        if hasattr(transition, 'variance'):
            trans_var = transition.variance(z_sample, action)
            # Simple propagation (linearized)
            new_cov = self.cov_mat + np.diag(trans_var)
        else:
            new_cov = self.cov_mat.copy()

        return GaussianBeliefState(mean_vec=new_mean, cov_mat=new_cov)

    def entropy(self) -> float:
        """Gaussian differential entropy."""
        det = np.linalg.det(self.cov_mat)
        return 0.5 * (self.dim * np.log(2 * np.pi * np.e) + np.log(det))


class ParticleBeliefState(BeliefState):
    """
    Particle-based belief state (Sequential Monte Carlo).

    Represents the belief as a weighted set of particles:
        b(z) ≈ Σᵢ wᵢ δ(z - zᵢ)

    This is the most flexible representation, handling arbitrary
    transitions and emissions.
    """

    def __init__(
        self,
        particles: List[LatentState],
        weights: Optional[NDArray] = None,
        n_particles: int = 100
    ):
        """
        Initialize particle belief.

        Args:
            particles: List of particle states
            weights: Particle weights (normalized). If None, uniform.
            n_particles: Target number of particles
        """
        self.particles = particles
        self.n_particles = len(particles)

        if weights is None:
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            self.weights = self.weights / self.weights.sum()

        self.dim = particles[0].dim if particles else 0

    def mean(self) -> NDArray[np.float64]:
        """Weighted mean of particles."""
        vectors = np.array([p.vector for p in self.particles])
        return np.sum(vectors * self.weights[:, np.newaxis], axis=0)

    def covariance(self) -> NDArray[np.float64]:
        """Weighted covariance of particles."""
        vectors = np.array([p.vector for p in self.particles])
        mean = self.mean()
        centered = vectors - mean

        # Weighted covariance
        cov = np.zeros((self.dim, self.dim))
        for i, (w, c) in enumerate(zip(self.weights, centered)):
            cov += w * np.outer(c, c)

        return cov

    def sample(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> List[LatentState]:
        """Sample particles according to weights."""
        if rng is None:
            rng = np.random.default_rng()

        indices = rng.choice(self.n_particles, size=n, p=self.weights)
        return [self.particles[i].copy() for i in indices]

    def update(
        self,
        observation: NDArray,
        emission: EmissionModel
    ) -> ParticleBeliefState:
        """
        Bayesian update: weight particles by observation likelihood.
        """
        obs = np.asarray(observation)

        # Compute likelihoods
        log_likelihoods = np.array([
            emission.log_prob(p, obs) for p in self.particles
        ])

        # Update weights (log-space for stability)
        log_weights = np.log(self.weights + 1e-10) + log_likelihoods

        # Normalize
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        weights = weights / weights.sum()

        return ParticleBeliefState(
            particles=self.particles,
            weights=weights,
            n_particles=self.n_particles
        )

    def predict(
        self,
        action: Any,
        transition: TransitionKernel,
        rng: Optional[np.random.Generator] = None
    ) -> ParticleBeliefState:
        """
        Predict belief by propagating particles through transition.
        """
        if rng is None:
            rng = np.random.default_rng()

        new_particles = [
            transition.sample(p, action, rng) for p in self.particles
        ]

        return ParticleBeliefState(
            particles=new_particles,
            weights=self.weights.copy(),
            n_particles=self.n_particles
        )

    def resample(
        self,
        rng: Optional[np.random.Generator] = None,
        method: str = "systematic"
    ) -> ParticleBeliefState:
        """
        Resample particles to combat weight degeneracy.

        Args:
            rng: Random generator
            method: Resampling method ("multinomial" or "systematic")
        """
        if rng is None:
            rng = np.random.default_rng()

        if method == "systematic":
            indices = self._systematic_resample(rng)
        else:
            indices = rng.choice(
                self.n_particles,
                size=self.n_particles,
                p=self.weights
            )

        new_particles = [self.particles[i].copy() for i in indices]

        return ParticleBeliefState(
            particles=new_particles,
            weights=None,  # Uniform after resampling
            n_particles=self.n_particles
        )

    def _systematic_resample(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Systematic resampling (low-variance)."""
        positions = (rng.random() + np.arange(self.n_particles)) / self.n_particles
        cumsum = np.cumsum(self.weights)

        indices = np.zeros(self.n_particles, dtype=np.int64)
        i, j = 0, 0
        while i < self.n_particles:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        return indices

    def effective_sample_size(self) -> float:
        """
        Effective sample size (ESS).

        ESS = 1 / Σᵢ wᵢ²

        Low ESS indicates weight degeneracy and need for resampling.
        """
        return 1.0 / np.sum(self.weights ** 2)

    def needs_resampling(self, threshold: float = 0.5) -> bool:
        """Check if resampling is needed based on ESS."""
        return self.effective_sample_size() < threshold * self.n_particles

    @classmethod
    def from_prior(
        cls,
        prior_space: LatentSpace,
        n_particles: int = 100,
        rng: Optional[np.random.Generator] = None
    ) -> ParticleBeliefState:
        """Initialize belief from prior samples."""
        particles = prior_space.sample_uniform(n_particles, rng)
        return cls(particles=particles, n_particles=n_particles)


@dataclass
class DiscreteBeliefState(BeliefState):
    """
    Belief state for discrete latent spaces.

    Used for exact computations in the lumpability examples.
    """
    probabilities: NDArray[np.float64]

    def __post_init__(self):
        self.probabilities = np.asarray(self.probabilities, dtype=np.float64)
        self.probabilities = self.probabilities / self.probabilities.sum()
        self.n_states = len(self.probabilities)

    def mean(self) -> NDArray[np.float64]:
        """Expected state index."""
        return np.array([np.sum(np.arange(self.n_states) * self.probabilities)])

    def covariance(self) -> NDArray[np.float64]:
        """Variance of state distribution."""
        mean = self.mean()[0]
        var = np.sum(self.probabilities * (np.arange(self.n_states) - mean) ** 2)
        return np.array([[var]])

    def sample(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> List[LatentState]:
        if rng is None:
            rng = np.random.default_rng()

        indices = rng.choice(self.n_states, size=n, p=self.probabilities)
        return [
            LatentState(
                vector=np.eye(self.n_states)[idx],
                metadata={"discrete_index": idx}
            )
            for idx in indices
        ]

    def update(
        self,
        observation: NDArray,
        emission: EmissionModel
    ) -> DiscreteBeliefState:
        """Update with observation likelihood."""
        likelihoods = np.array([
            emission.prob(
                LatentState(
                    vector=np.eye(self.n_states)[i],
                    metadata={"discrete_index": i}
                ),
                observation
            )
            for i in range(self.n_states)
        ])

        new_probs = self.probabilities * likelihoods
        new_probs = new_probs / (new_probs.sum() + 1e-10)

        return DiscreteBeliefState(probabilities=new_probs)

    def predict(
        self,
        action: Any,
        transition: TransitionKernel
    ) -> DiscreteBeliefState:
        """Predict via transition matrix multiplication."""
        if hasattr(transition, 'get_matrix'):
            P = transition.get_matrix(action)
            new_probs = self.probabilities @ P
        else:
            # Sample-based prediction
            new_probs = np.zeros(self.n_states)
            for i, p_i in enumerate(self.probabilities):
                if p_i > 0:
                    z_i = LatentState(
                        vector=np.eye(self.n_states)[i],
                        metadata={"discrete_index": i}
                    )
                    mean = transition.mean(z_i, action)
                    new_probs += p_i * mean

        return DiscreteBeliefState(probabilities=new_probs)

    def entropy(self) -> float:
        """Shannon entropy in bits."""
        p = self.probabilities[self.probabilities > 0]
        return -np.sum(p * np.log2(p))

    def most_likely_state(self) -> int:
        """Return index of most likely state."""
        return int(np.argmax(self.probabilities))

    def conditional_entropy_given_next(
        self,
        transition: TransitionKernel,
        action: Any
    ) -> float:
        """
        Compute H(z_t | z_{t+1}, a) for reversibility analysis.

        This quantifies information loss in the forward transition.
        """
        if not hasattr(transition, 'get_matrix'):
            raise NotImplementedError("Requires discrete transition matrix")

        P = transition.get_matrix(action)
        prior = self.probabilities

        # Joint distribution P(z_t, z_{t+1}) = P(z_t) * P(z_{t+1}|z_t)
        joint = prior[:, np.newaxis] * P

        # Marginal P(z_{t+1})
        marginal_next = joint.sum(axis=0)

        # Conditional entropy H(z_t | z_{t+1})
        H = 0.0
        for j in range(self.n_states):
            if marginal_next[j] > 1e-10:
                # P(z_t | z_{t+1} = j)
                posterior = joint[:, j] / marginal_next[j]
                for i in range(self.n_states):
                    if posterior[i] > 1e-10:
                        H -= joint[i, j] * np.log2(posterior[i])

        return float(H)
