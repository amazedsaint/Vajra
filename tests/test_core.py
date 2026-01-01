"""
Tests for core Vajra components.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from vajra.core.latent_state import LatentState, LatentSpace, DiscreteLatentState
from vajra.core.transition_kernel import (
    DeterministicTransition,
    GaussianTransitionKernel,
    HeteroscedasticTransitionKernel,
    DiscreteTransitionMatrix,
)
from vajra.core.emission_model import GaussianEmission, DiscreteEmission, IdentityEmission
from vajra.core.belief_state import (
    GaussianBeliefState,
    ParticleBeliefState,
    DiscreteBeliefState,
)


class TestLatentState:
    """Tests for LatentState."""

    def test_creation(self):
        z = LatentState(vector=np.array([1.0, 2.0, 3.0]))
        assert z.dim == 3
        assert_array_almost_equal(z.vector, [1.0, 2.0, 3.0])

    def test_distance(self):
        z1 = LatentState(vector=np.array([0.0, 0.0]))
        z2 = LatentState(vector=np.array([3.0, 4.0]))
        assert z1.distance(z2) == 5.0

    def test_copy(self):
        z1 = LatentState(vector=np.array([1.0, 2.0]))
        z2 = z1.copy()
        z2.vector[0] = 99.0
        assert z1.vector[0] == 1.0  # Original unchanged


class TestLatentSpace:
    """Tests for LatentSpace."""

    def test_bounded_sampling(self):
        space = LatentSpace(
            dim=2,
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        )
        samples = space.sample_uniform(100, rng=np.random.default_rng(42))
        for s in samples:
            assert np.all(s.vector >= 0.0)
            assert np.all(s.vector <= 1.0)

    def test_gaussian_sampling(self):
        space = LatentSpace(dim=3)
        samples = space.sample_gaussian(
            mean=np.array([1.0, 2.0, 3.0]),
            std=0.1,
            n=1000,
            rng=np.random.default_rng(42)
        )
        vectors = np.array([s.vector for s in samples])
        assert_allclose(vectors.mean(axis=0), [1.0, 2.0, 3.0], atol=0.05)

    def test_projection(self):
        space = LatentSpace(
            dim=2,
            bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        )
        z = LatentState(vector=np.array([2.0, -1.0]))
        z_proj = space.project(z)
        assert_array_almost_equal(z_proj.vector, [1.0, 0.0])


class TestTransitionKernels:
    """Tests for transition kernels."""

    def test_deterministic(self):
        def transition_fn(z, a):
            return z + np.array([a, 0])

        kernel = DeterministicTransition(
            transition_fn=transition_fn,
            latent_dim=2,
            noise_std=0.0
        )

        z = LatentState(vector=np.array([1.0, 1.0]))
        z_next = kernel.sample(z, action=0.5)

        assert_array_almost_equal(z_next.vector, [1.5, 1.0])

    def test_gaussian_kernel(self):
        kernel = GaussianTransitionKernel(
            mean_fn=lambda z, a: z + a,
            covariance=0.01,
            latent_dim=2
        )

        z = LatentState(vector=np.array([0.0, 0.0]))
        rng = np.random.default_rng(42)

        samples = [kernel.sample(z, action=np.array([1.0, 0.0]), rng=rng) for _ in range(1000)]
        means = np.mean([s.vector for s in samples], axis=0)

        assert_allclose(means, [1.0, 0.0], atol=0.05)

    def test_heteroscedastic_kernel(self):
        def mean_fn(z, a):
            return z + a

        def var_fn(z, a):
            return np.abs(z) + 0.01  # Variance depends on state

        kernel = HeteroscedasticTransitionKernel(
            mean_fn=mean_fn,
            var_fn=var_fn,
            latent_dim=2
        )

        z = LatentState(vector=np.array([1.0, 0.0]))
        var = kernel.variance(z, action=np.array([0.0, 0.0]))

        # Variance should be higher for dimension with larger |z|
        assert var[0] > var[1]

    def test_discrete_matrix(self):
        P = np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        kernel = DiscreteTransitionMatrix(
            transition_matrices={0: P},
            n_states=2
        )

        # Check probabilities sum to 1
        z = LatentState(vector=np.array([1.0, 0.0]), metadata={"discrete_index": 0})
        assert np.isclose(sum(kernel.prob(z, 0, LatentState(
            vector=np.eye(2)[j], metadata={"discrete_index": j}
        )) for j in range(2)), 1.0)


class TestEmissionModels:
    """Tests for emission models."""

    def test_gaussian_emission(self):
        emission = GaussianEmission(
            obs_fn=lambda z: z * 2,
            obs_dim=2,
            noise_std=0.1
        )

        z = LatentState(vector=np.array([1.0, 2.0]))
        mean = emission.mean(z)
        assert_array_almost_equal(mean, [2.0, 4.0])

    def test_identity_emission(self):
        emission = IdentityEmission(dim=3, noise_std=0.0)
        z = LatentState(vector=np.array([1.0, 2.0, 3.0]))
        x = emission.sample(z)
        assert_array_almost_equal(x, z.vector)


class TestBeliefStates:
    """Tests for belief state representations."""

    def test_gaussian_belief(self):
        belief = GaussianBeliefState(
            mean_vec=np.array([1.0, 2.0]),
            cov_mat=np.eye(2) * 0.5
        )

        assert_array_almost_equal(belief.mean(), [1.0, 2.0])
        assert belief.uncertainty == 1.0  # trace of covariance

    def test_particle_belief(self):
        particles = [
            LatentState(vector=np.array([0.0, 0.0])),
            LatentState(vector=np.array([2.0, 0.0])),
        ]
        weights = np.array([0.5, 0.5])

        belief = ParticleBeliefState(particles=particles, weights=weights)

        assert_array_almost_equal(belief.mean(), [1.0, 0.0])
        assert belief.effective_sample_size() == 2.0

    def test_discrete_belief(self):
        belief = DiscreteBeliefState(probabilities=np.array([0.3, 0.7]))

        assert belief.most_likely_state() == 1
        assert np.isclose(belief.entropy(), -0.3 * np.log2(0.3) - 0.7 * np.log2(0.7))


class TestParticleFilter:
    """Tests for particle filter operations."""

    def test_resampling(self):
        # Create degenerate weights
        particles = [LatentState(vector=np.array([float(i)])) for i in range(10)]
        weights = np.zeros(10)
        weights[0] = 1.0  # All weight on first particle

        belief = ParticleBeliefState(particles=particles, weights=weights)
        assert belief.effective_sample_size() == 1.0

        # Resample
        resampled = belief.resample(rng=np.random.default_rng(42))

        # All particles should now be copies of the first
        for p in resampled.particles:
            assert p.vector[0] == 0.0

        # Weights should be uniform
        assert_allclose(resampled.weights, np.ones(10) / 10)
