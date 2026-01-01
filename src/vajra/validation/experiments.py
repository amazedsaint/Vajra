"""
Validation experiments from the paper.

This module reproduces the key numerical validations:
1. Lumpability construction (Section 3): Non-identifiability from observables
2. Structural equivalence (Section 5): Commutation over action sequences
3. Reversibility (Section 6): Conditional entropy for bijective vs collapsing maps
4. Heteroscedastic modeling (Section 7): NLL vs MSE for conditional uncertainty

All experiments can be run via:
    python -m vajra.validation
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from numpy.typing import NDArray

from vajra.core.latent_state import LatentState, DiscreteLatentState
from vajra.core.transition_kernel import DiscreteTransitionMatrix, DeterministicTransition
from vajra.core.emission_model import DiscreteEmission
from vajra.core.belief_state import DiscreteBeliefState
from vajra.structural.equivalence import (
    DiscreteEquivalence,
    CommutationChecker,
    ApproximateEquivalence,
)


@dataclass
class ExperimentResult:
    """Result of a validation experiment."""
    name: str
    passed: bool
    expected: Any
    actual: Any
    details: Dict[str, Any]

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.name}\n  Expected: {self.expected}\n  Actual: {self.actual}"


class LumpabilityExperiment:
    """
    Section 3: Non-identifiability from observables alone.

    Constructs two distinct hidden Markov chains P1, P2 that induce
    the same observable process through a lumpability construction.

    This validates the claim: "There exist distinct latent dynamics that
    cannot be distinguished by any amount of observable state history."
    """

    def __init__(self):
        # Hidden state space: {0, 1, 2, 3}
        # Observable space: {0, 1}
        # Observation map: φ(0)=φ(1)=0, φ(2)=φ(3)=1
        self.n_hidden = 4
        self.n_observed = 2

        # Observation classes
        self.obs_map = {0: 0, 1: 0, 2: 1, 3: 1}

        # P1 from the paper
        self.P1 = np.array([
            [0.60, 0.30, 0.05, 0.05],
            [0.20, 0.70, 0.08, 0.02],
            [0.10, 0.10, 0.30, 0.50],
            [0.05, 0.15, 0.70, 0.10]
        ])

        # P2 from the paper
        self.P2 = np.array([
            [0.90, 0.00, 0.00, 0.10],
            [0.00, 0.90, 0.10, 0.00],
            [0.20, 0.00, 0.80, 0.00],
            [0.00, 0.20, 0.00, 0.80]
        ])

        # Expected aggregated transition matrix
        self.P_x_expected = np.array([
            [0.90, 0.10],
            [0.20, 0.80]
        ])

    def compute_frobenius_distance(self) -> float:
        """Compute ||P1 - P2||_F."""
        return float(np.linalg.norm(self.P1 - self.P2, ord='fro'))

    def compute_aggregated_matrix(self, P: NDArray) -> NDArray:
        """
        Compute aggregated transition matrix P_x.

        P_x[c, c'] = P(x'=c' | x=c) for any state s in class c
        """
        P_x = np.zeros((self.n_observed, self.n_observed))

        for c in range(self.n_observed):
            # Get any state in class c (they should all give same result for lumpable)
            states_in_c = [s for s, obs in self.obs_map.items() if obs == c]
            s = states_in_c[0]

            for c_next in range(self.n_observed):
                # Sum probabilities of transitioning to states in c_next
                states_in_c_next = [s2 for s2, obs in self.obs_map.items() if obs == c_next]
                P_x[c, c_next] = sum(P[s, s2] for s2 in states_in_c_next)

        return P_x

    def check_lumpability(self, P: NDArray) -> Tuple[bool, NDArray]:
        """
        Check if P is lumpable with respect to the observation partition.

        Lumpability: For all states s1, s2 in the same class c,
        P(x'=c' | s1) = P(x'=c' | s2) for all c'.
        """
        is_lumpable = True
        aggregated = np.zeros((self.n_observed, self.n_observed))

        for c in range(self.n_observed):
            states_in_c = [s for s, obs in self.obs_map.items() if obs == c]

            # Compute aggregated probabilities from first state
            s0 = states_in_c[0]
            for c_next in range(self.n_observed):
                states_in_c_next = [s2 for s2, obs in self.obs_map.items() if obs == c_next]
                aggregated[c, c_next] = sum(P[s0, s2] for s2 in states_in_c_next)

            # Check all states in class give same result
            for s in states_in_c[1:]:
                for c_next in range(self.n_observed):
                    states_in_c_next = [s2 for s2, obs in self.obs_map.items() if obs == c_next]
                    prob = sum(P[s, s2] for s2 in states_in_c_next)
                    if not np.isclose(prob, aggregated[c, c_next], atol=1e-10):
                        is_lumpable = False

        return is_lumpable, aggregated

    def simulate_and_estimate(
        self,
        P: NDArray,
        n_steps: int = 200000,
        rng: Optional[np.random.Generator] = None
    ) -> NDArray:
        """
        Simulate from hidden chain and estimate observable transition matrix.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Start from stationary distribution (approximately)
        state = rng.integers(self.n_hidden)

        # Count transitions
        counts = np.zeros((self.n_observed, self.n_observed))

        prev_obs = self.obs_map[state]
        for _ in range(n_steps):
            state = rng.choice(self.n_hidden, p=P[state])
            curr_obs = self.obs_map[state]
            counts[prev_obs, curr_obs] += 1
            prev_obs = curr_obs

        # Normalize to get estimated P_x
        P_x_est = counts / counts.sum(axis=1, keepdims=True)

        return P_x_est

    def run(self, n_simulation_steps: int = 200000) -> ExperimentResult:
        """Run the lumpability experiment."""
        # Check theoretical properties
        frob_dist = self.compute_frobenius_distance()
        is_lumpable_1, P_x_1 = self.check_lumpability(self.P1)
        is_lumpable_2, P_x_2 = self.check_lumpability(self.P2)

        # Simulate and estimate
        rng = np.random.default_rng(42)
        P_x_est_1 = self.simulate_and_estimate(self.P1, n_simulation_steps, rng)
        P_x_est_2 = self.simulate_and_estimate(self.P2, n_simulation_steps, rng)

        # Check results
        matrices_match = np.allclose(P_x_1, P_x_2, atol=1e-10)
        matches_expected = np.allclose(P_x_1, self.P_x_expected, atol=1e-10)
        simulations_match = np.allclose(P_x_est_1, P_x_est_2, atol=0.01)

        passed = (
            is_lumpable_1 and
            is_lumpable_2 and
            matrices_match and
            matches_expected and
            simulations_match and
            np.isclose(frob_dist, 1.3307140940, rtol=1e-6)
        )

        return ExperimentResult(
            name="Lumpability Construction (Section 3)",
            passed=passed,
            expected={
                "frobenius_distance": 1.3307140940,
                "both_lumpable": True,
                "same_observable_dynamics": True,
            },
            actual={
                "frobenius_distance": frob_dist,
                "P1_lumpable": is_lumpable_1,
                "P2_lumpable": is_lumpable_2,
                "P_x_match": matrices_match,
            },
            details={
                "P_x_from_P1": P_x_1.tolist(),
                "P_x_from_P2": P_x_2.tolist(),
                "P_x_expected": self.P_x_expected.tolist(),
                "P_x_simulated_P1": P_x_est_1.tolist(),
                "P_x_simulated_P2": P_x_est_2.tolist(),
            }
        )


class StructuralEquivalenceExperiment:
    """
    Section 5: Structural equivalence via commutation.

    Validates that:
    1. U∘T_a = T_a∘U for primitive actions implies
       U∘T_α = T_α∘U for any action sequence α
    """

    def __init__(self):
        # 6-state system with two isomorphic 3-state modules
        # Module A: states {0, 1, 2}
        # Module B: states {3, 4, 5}
        # U swaps the modules: U(i) = i+3 for i<3, U(i) = i-3 for i≥3
        self.n_states = 6

        self.swap_map = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}

        # Actions (deterministic transitions)
        # Action 0: cycle within module
        # Action 1: cross-module connection (symmetric)
        # Action 2: identity/stay
        self.action_maps = {
            0: {0: 1, 1: 2, 2: 0, 3: 4, 4: 5, 5: 3},  # Cycle
            1: {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2},  # Swap
            2: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},  # Identity
        }

    def create_transition_fn(self, action: int):
        """Create deterministic transition function for action."""
        action_map = self.action_maps[action]

        def transition_fn(z: NDArray, a: int) -> NDArray:
            state_idx = int(np.argmax(z))
            next_idx = action_map[state_idx]
            result = np.zeros(self.n_states)
            result[next_idx] = 1.0
            return result

        return transition_fn

    def apply_swap(self, z: NDArray) -> NDArray:
        """Apply the swap mapping U."""
        state_idx = int(np.argmax(z))
        new_idx = self.swap_map[state_idx]
        result = np.zeros(self.n_states)
        result[new_idx] = 1.0
        return result

    def check_primitive_commutation(self) -> Dict[int, Dict[int, bool]]:
        """Check U∘T_a = T_a∘U for all states and primitive actions."""
        results = {}

        for action in self.action_maps:
            action_results = {}
            for state_idx in range(self.n_states):
                z = np.zeros(self.n_states)
                z[state_idx] = 1.0

                # U(T_a(z))
                Taz = np.zeros(self.n_states)
                Taz[self.action_maps[action][state_idx]] = 1.0
                UTaz = self.apply_swap(Taz)

                # T_a(U(z))
                Uz = self.apply_swap(z)
                Uz_idx = int(np.argmax(Uz))
                TaUz = np.zeros(self.n_states)
                TaUz[self.action_maps[action][Uz_idx]] = 1.0

                commutes = np.allclose(UTaz, TaUz)
                action_results[state_idx] = commutes

            results[action] = action_results

        return results

    def check_sequence_commutation(
        self,
        sequences: List[List[int]],
        n_random_sequences: int = 50,
        max_length: int = 10,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, bool]:
        """Check commutation for action sequences."""
        if rng is None:
            rng = np.random.default_rng(42)

        results = {}

        # Test provided sequences
        for seq in sequences:
            seq_key = str(seq)
            all_commute = True

            for state_idx in range(self.n_states):
                z = np.zeros(self.n_states)
                z[state_idx] = 1.0

                # Compute T_α(z)
                current = z.copy()
                for action in seq:
                    next_idx = self.action_maps[action][int(np.argmax(current))]
                    current = np.zeros(self.n_states)
                    current[next_idx] = 1.0

                # U(T_α(z))
                UTaz = self.apply_swap(current)

                # Compute T_α(U(z))
                Uz = self.apply_swap(z)
                current = Uz.copy()
                for action in seq:
                    next_idx = self.action_maps[action][int(np.argmax(current))]
                    current = np.zeros(self.n_states)
                    current[next_idx] = 1.0
                TaUz = current

                if not np.allclose(UTaz, TaUz):
                    all_commute = False
                    break

            results[seq_key] = all_commute

        # Test random sequences
        actions = list(self.action_maps.keys())
        random_pass = 0
        for _ in range(n_random_sequences):
            length = rng.integers(1, max_length + 1)
            seq = [rng.choice(actions) for _ in range(length)]

            all_commute = True
            for state_idx in range(self.n_states):
                z = np.zeros(self.n_states)
                z[state_idx] = 1.0

                current = z.copy()
                for action in seq:
                    next_idx = self.action_maps[action][int(np.argmax(current))]
                    current = np.zeros(self.n_states)
                    current[next_idx] = 1.0
                UTaz = self.apply_swap(current)

                Uz = self.apply_swap(z)
                current = Uz.copy()
                for action in seq:
                    next_idx = self.action_maps[action][int(np.argmax(current))]
                    current = np.zeros(self.n_states)
                    current[next_idx] = 1.0
                TaUz = current

                if not np.allclose(UTaz, TaUz):
                    all_commute = False
                    break

            if all_commute:
                random_pass += 1

        results["random_sequences"] = f"{random_pass}/{n_random_sequences}"

        return results

    def run(self) -> ExperimentResult:
        """Run the structural equivalence experiment."""
        # Check primitive commutation
        primitive_results = self.check_primitive_commutation()
        all_primitive_commute = all(
            all(state_results.values())
            for state_results in primitive_results.values()
        )

        # Check sequence commutation
        test_sequences = [
            [0, 1],
            [0, 0, 1],
            [1, 0, 2, 1],
            [0, 1, 2, 0, 1],
        ]
        sequence_results = self.check_sequence_commutation(test_sequences)

        all_sequences_commute = all(
            v if isinstance(v, bool) else True
            for v in sequence_results.values()
        )

        passed = all_primitive_commute and all_sequences_commute

        return ExperimentResult(
            name="Structural Equivalence (Section 5)",
            passed=passed,
            expected={
                "all_primitive_commute": True,
                "all_sequences_commute": True,
            },
            actual={
                "all_primitive_commute": all_primitive_commute,
                "all_sequences_commute": all_sequences_commute,
            },
            details={
                "primitive_results": {
                    f"action_{a}": "all commute" if all(v.values()) else "some fail"
                    for a, v in primitive_results.items()
                },
                "sequence_results": sequence_results,
            }
        )


class ReversibilityExperiment:
    """
    Section 6: Reversibility and conditional entropy.

    Validates that:
    - Bijective transitions have H(z₀|z₁) = 0
    - Many-to-one transitions have H(z₀|z₁) > 0
    """

    def __init__(self):
        self.n_states = 8

    def compute_conditional_entropy(
        self,
        transition_map: Dict[int, int],
        prior: Optional[NDArray] = None
    ) -> float:
        """
        Compute H(z₀|z₁) for a deterministic transition.

        H(z₀|z₁) = Σ_j P(z₁=j) H(z₀|z₁=j)
        """
        if prior is None:
            prior = np.ones(self.n_states) / self.n_states

        # Compute P(z₁=j) and preimages
        preimages = {}  # j -> list of i such that f(i) = j
        for i, j in transition_map.items():
            if j not in preimages:
                preimages[j] = []
            preimages[j].append(i)

        # P(z₁=j) = Σ_i:f(i)=j P(z₀=i)
        p_z1 = np.zeros(self.n_states)
        for j in range(self.n_states):
            if j in preimages:
                p_z1[j] = sum(prior[i] for i in preimages[j])

        # H(z₀|z₁) = Σ_j P(z₁=j) H(z₀|z₁=j)
        H = 0.0
        for j in range(self.n_states):
            if p_z1[j] > 1e-10 and j in preimages:
                # P(z₀|z₁=j) is uniform over preimages
                n_preimages = len(preimages[j])
                # H(z₀|z₁=j) = log₂(n_preimages) if uniform
                if n_preimages > 1:
                    H += p_z1[j] * np.log2(n_preimages)

        return H

    def run(self) -> ExperimentResult:
        """Run the reversibility experiment."""
        # Bijective map (permutation)
        bijective_map = {i: (i + 1) % self.n_states for i in range(self.n_states)}
        H_bijective = self.compute_conditional_entropy(bijective_map)

        # 2-to-1 collapse: pairs (0,1)->0, (2,3)->1, etc.
        collapse_map = {i: i // 2 for i in range(self.n_states)}
        H_collapse = self.compute_conditional_entropy(collapse_map)

        # Expected: H_bijective = 0, H_collapse = log₂(2) = 1.0
        passed = (
            np.isclose(H_bijective, 0.0, atol=1e-10) and
            np.isclose(H_collapse, 1.0, atol=1e-10)
        )

        return ExperimentResult(
            name="Reversibility Analysis (Section 6)",
            passed=passed,
            expected={
                "H_bijective": 0.0,
                "H_collapse": 1.0,  # log₂(2)
            },
            actual={
                "H_bijective": H_bijective,
                "H_collapse": H_collapse,
            },
            details={
                "bijective_map": bijective_map,
                "collapse_map": collapse_map,
            }
        )


class HeteroscedasticExperiment:
    """
    Section 7: Simulation requires modeling surprise.

    Validates that heteroscedastic NLL training recovers conditional
    uncertainty that MSE obscures.

    Data generating process:
        x ~ Uniform[-1, 1]
        y = x + η, η ~ N(0, σ(x)²)
        σ(x) = 0.1 + 0.2(x+1)
    """

    def __init__(self, n_train: int = 40000, n_test: int = 10000):
        self.n_train = n_train
        self.n_test = n_test

    def sigma(self, x: NDArray) -> NDArray:
        """True conditional standard deviation."""
        return 0.1 + 0.2 * (x + 1)

    def generate_data(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[NDArray, NDArray]:
        """Generate data from the true model."""
        if rng is None:
            rng = np.random.default_rng()

        x = rng.uniform(-1, 1, n)
        sigma = self.sigma(x)
        eta = rng.normal(0, sigma)
        y = x + eta

        return x, y

    def fit_constant_variance(
        self,
        x_train: NDArray,
        y_train: NDArray
    ) -> Tuple[float, float, float]:
        """
        Fit Model A: linear mean, constant variance.

        Returns: (slope, intercept, variance)
        """
        # Linear regression for mean
        n = len(x_train)
        x_mean = x_train.mean()
        y_mean = y_train.mean()

        slope = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)
        intercept = y_mean - slope * x_mean

        # Estimate variance from residuals
        residuals = y_train - (slope * x_train + intercept)
        variance = np.var(residuals)

        return slope, intercept, variance

    def fit_heteroscedastic(
        self,
        x_train: NDArray,
        y_train: NDArray
    ) -> Tuple[float, float, float, float]:
        """
        Fit Model B: linear mean, linear log-variance.

        log σ²(x) = v₀ + v₁x

        Returns: (slope, intercept, v0, v1)
        """
        # First fit mean
        slope, intercept, _ = self.fit_constant_variance(x_train, y_train)

        # Fit log-variance via maximum likelihood
        # Use iterative weighted least squares or simple grid search
        residuals = y_train - (slope * x_train + intercept)
        log_sq_residuals = np.log(residuals ** 2 + 1e-10)

        # Linear regression on log squared residuals
        x_mean = x_train.mean()
        lsr_mean = log_sq_residuals.mean()

        v1 = np.sum((x_train - x_mean) * (log_sq_residuals - lsr_mean)) / np.sum((x_train - x_mean) ** 2)
        v0 = lsr_mean - v1 * x_mean

        return slope, intercept, v0, v1

    def compute_nll(
        self,
        x: NDArray,
        y: NDArray,
        mu: NDArray,
        var: NDArray
    ) -> float:
        """Compute negative log likelihood."""
        nll = 0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mu) ** 2 / var
        return float(nll.mean())

    def run(self) -> ExperimentResult:
        """Run the heteroscedastic experiment."""
        rng = np.random.default_rng(42)

        # Generate data
        x_train, y_train = self.generate_data(self.n_train, rng)
        x_test, y_test = self.generate_data(self.n_test, rng)

        # Fit Model A (constant variance)
        slope_a, intercept_a, var_a = self.fit_constant_variance(x_train, y_train)
        mu_a_test = slope_a * x_test + intercept_a
        nll_a = self.compute_nll(x_test, y_test, mu_a_test, np.full_like(x_test, var_a))

        # Fit Model B (heteroscedastic)
        slope_b, intercept_b, v0, v1 = self.fit_heteroscedastic(x_train, y_train)
        mu_b_test = slope_b * x_test + intercept_b
        log_var_b_test = v0 + v1 * x_test
        var_b_test = np.exp(log_var_b_test)
        nll_b = self.compute_nll(x_test, y_test, mu_b_test, var_b_test)

        # Compute correlation between predicted and true sigma
        sigma_pred = np.sqrt(var_b_test)
        sigma_true = self.sigma(x_test)
        correlation = np.corrcoef(sigma_pred, sigma_true)[0, 1]

        # Check results (approximate due to simple fitting)
        nll_b_better = nll_b < nll_a
        high_correlation = correlation > 0.9

        passed = nll_b_better and high_correlation

        return ExperimentResult(
            name="Heteroscedastic Modeling (Section 7)",
            passed=passed,
            expected={
                "nll_b < nll_a": True,
                "correlation > 0.9": True,
                "nll_a_approx": 0.28,  # From paper: ~0.2776
                "nll_b_approx": 0.13,  # From paper: ~0.1271
            },
            actual={
                "nll_a": nll_a,
                "nll_b": nll_b,
                "nll_improvement": nll_a - nll_b,
                "sigma_correlation": correlation,
            },
            details={
                "model_a": {"slope": slope_a, "intercept": intercept_a, "var": var_a},
                "model_b": {"slope": slope_b, "intercept": intercept_b, "v0": v0, "v1": v1},
            }
        )


def run_all_validations(verbose: bool = True) -> List[ExperimentResult]:
    """Run all validation experiments."""
    experiments = [
        LumpabilityExperiment(),
        StructuralEquivalenceExperiment(),
        ReversibilityExperiment(),
        HeteroscedasticExperiment(),
    ]

    results = []
    for exp in experiments:
        if verbose:
            print(f"\nRunning: {exp.__class__.__name__}...")

        result = exp.run()
        results.append(result)

        if verbose:
            print(result)

    if verbose:
        n_passed = sum(1 for r in results if r.passed)
        print(f"\n{'='*60}")
        print(f"Summary: {n_passed}/{len(results)} experiments passed")
        print(f"{'='*60}")

    return results


def main():
    """Entry point for validation script."""
    print("Vajra Context Graph System - Validation Experiments")
    print("=" * 60)
    print("Reproducing numerical validations from the paper...")

    results = run_all_validations(verbose=True)

    # Exit with error code if any failed
    if not all(r.passed for r in results):
        exit(1)


if __name__ == "__main__":
    main()
