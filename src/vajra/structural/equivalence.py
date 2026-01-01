"""
Structural equivalence via action commutation.

Section 5 of the paper defines structural equivalence operationally:
two regions are structurally equivalent when there exists a mapping U such that
for all actions a and states z:

    U(T_a(z)) = T_a(U(z))

This commutation relation means that acting and then mapping is identical
to mapping and then acting. It implies indistinguishability under interventions.

This module provides:
- Exact commutation checking for discrete systems
- ε-approximate equivalence with error bounds
- Compositional equivalence over action sequences
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray

from vajra.core.latent_state import LatentState, DiscreteLatentState
from vajra.core.transition_kernel import TransitionKernel, DeterministicTransition


@dataclass
class CommutationResult:
    """Result of a commutation check."""
    commutes: bool
    defect: float  # ||U(T_a(z)) - T_a(U(z))||
    z: NDArray
    action: Any
    Uz: NDArray
    TaUz: NDArray
    UTaz: NDArray

    def __str__(self):
        status = "✓ Commutes" if self.commutes else "✗ Does not commute"
        return f"{status} (defect={self.defect:.6f})"


class StructuralEquivalence(ABC):
    """
    Abstract base class for structural equivalence mappings.

    A structural equivalence U satisfies U∘T_a = T_a∘U for all actions.
    """

    @abstractmethod
    def apply(self, z: LatentState) -> LatentState:
        """Apply the equivalence mapping U(z)."""
        pass

    @abstractmethod
    def inverse(self, z: LatentState) -> LatentState:
        """Apply the inverse mapping U⁻¹(z)."""
        pass

    def check_commutation(
        self,
        z: LatentState,
        action: Any,
        transition: TransitionKernel,
        tolerance: float = 1e-10
    ) -> CommutationResult:
        """
        Check if U∘T_a = T_a∘U for a specific state and action.

        Computes:
            - U(T_a(z))  [act then map]
            - T_a(U(z))  [map then act]
        """
        # Compute T_a(z)
        Taz = transition.sample(z, action)

        # Compute U(T_a(z))
        UTaz = self.apply(Taz)

        # Compute U(z)
        Uz = self.apply(z)

        # Compute T_a(U(z))
        TaUz = transition.sample(Uz, action)

        # Compute defect
        defect = np.linalg.norm(UTaz.vector - TaUz.vector)
        commutes = defect <= tolerance

        return CommutationResult(
            commutes=commutes,
            defect=defect,
            z=z.vector,
            action=action,
            Uz=Uz.vector,
            TaUz=TaUz.vector,
            UTaz=UTaz.vector,
        )


class LinearEquivalence(StructuralEquivalence):
    """
    Linear structural equivalence: U(z) = Az.

    Common case: permutation matrices that swap isomorphic subsystems.
    """

    def __init__(self, matrix: NDArray):
        """
        Initialize with transformation matrix.

        Args:
            matrix: Linear transformation matrix A
        """
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.dim = self.matrix.shape[0]

        # Compute inverse if possible
        try:
            self.inv_matrix = np.linalg.inv(self.matrix)
            self.is_invertible = True
        except np.linalg.LinAlgError:
            self.inv_matrix = None
            self.is_invertible = False

    def apply(self, z: LatentState) -> LatentState:
        """Apply U(z) = Az."""
        return LatentState(
            vector=self.matrix @ z.vector,
            metadata={**z.metadata, "transformed_by": "linear"},
        )

    def inverse(self, z: LatentState) -> LatentState:
        """Apply U⁻¹(z) = A⁻¹z."""
        if not self.is_invertible:
            raise ValueError("Transformation is not invertible")
        return LatentState(
            vector=self.inv_matrix @ z.vector,
            metadata={**z.metadata, "transformed_by": "linear_inv"},
        )

    @classmethod
    def permutation(cls, permutation: List[int]) -> LinearEquivalence:
        """
        Create a permutation equivalence.

        Args:
            permutation: List where permutation[i] = j means position i maps to position j
        """
        n = len(permutation)
        matrix = np.zeros((n, n))
        for i, j in enumerate(permutation):
            matrix[j, i] = 1.0
        return cls(matrix)

    @classmethod
    def swap_blocks(cls, dim: int, block1: Tuple[int, int], block2: Tuple[int, int]) -> LinearEquivalence:
        """
        Create a block-swap equivalence.

        Swaps two contiguous blocks of dimensions.
        """
        matrix = np.eye(dim)
        start1, end1 = block1
        start2, end2 = block2

        assert end1 - start1 == end2 - start2, "Blocks must be same size"
        block_size = end1 - start1

        # Swap blocks
        matrix[start1:end1, start1:end1] = 0
        matrix[start2:end2, start2:end2] = 0
        matrix[start1:end1, start2:end2] = np.eye(block_size)
        matrix[start2:end2, start1:end1] = np.eye(block_size)

        return cls(matrix)


class DiscreteEquivalence(StructuralEquivalence):
    """
    Discrete state equivalence via permutation.

    Maps state indices according to a permutation.
    """

    def __init__(self, state_map: Dict[int, int], n_states: int):
        """
        Initialize with state mapping.

        Args:
            state_map: Maps state index to equivalent state index
            n_states: Number of states
        """
        self.state_map = state_map
        self.n_states = n_states

        # Compute inverse
        self.inv_map = {v: k for k, v in state_map.items()}

    def apply(self, z: LatentState) -> LatentState:
        """Apply discrete equivalence."""
        if "discrete_index" in z.metadata:
            old_idx = z.metadata["discrete_index"]
        else:
            old_idx = int(np.argmax(z.vector))

        new_idx = self.state_map.get(old_idx, old_idx)

        new_vec = np.zeros(self.n_states)
        new_vec[new_idx] = 1.0

        return LatentState(
            vector=new_vec,
            metadata={"discrete_index": new_idx, "original_index": old_idx},
        )

    def inverse(self, z: LatentState) -> LatentState:
        """Apply inverse mapping."""
        if "discrete_index" in z.metadata:
            old_idx = z.metadata["discrete_index"]
        else:
            old_idx = int(np.argmax(z.vector))

        new_idx = self.inv_map.get(old_idx, old_idx)

        new_vec = np.zeros(self.n_states)
        new_vec[new_idx] = 1.0

        return LatentState(
            vector=new_vec,
            metadata={"discrete_index": new_idx},
        )


class CommutationChecker:
    """
    Comprehensive commutation checking for structural equivalence.

    Validates that U∘T_a = T_a∘U for:
    - All primitive actions
    - Composed action sequences
    - Approximate (ε) equivalence
    """

    def __init__(
        self,
        equivalence: StructuralEquivalence,
        transition: TransitionKernel,
        actions: List[Any],
        tolerance: float = 1e-10
    ):
        """
        Initialize checker.

        Args:
            equivalence: The equivalence mapping U
            transition: Transition kernel
            actions: List of primitive actions
            tolerance: Tolerance for exact commutation
        """
        self.equivalence = equivalence
        self.transition = transition
        self.actions = actions
        self.tolerance = tolerance

    def check_all_actions(
        self,
        z: LatentState
    ) -> Dict[Any, CommutationResult]:
        """Check commutation for all primitive actions at state z."""
        results = {}
        for action in self.actions:
            results[action] = self.equivalence.check_commutation(
                z, action, self.transition, self.tolerance
            )
        return results

    def check_exhaustive(
        self,
        states: List[LatentState]
    ) -> Dict[str, Any]:
        """
        Exhaustively check commutation over all states and actions.

        Returns summary statistics.
        """
        total_checks = 0
        failures = []
        max_defect = 0.0
        total_defect = 0.0

        for z in states:
            for action in self.actions:
                result = self.equivalence.check_commutation(
                    z, action, self.transition, self.tolerance
                )
                total_checks += 1
                total_defect += result.defect
                max_defect = max(max_defect, result.defect)

                if not result.commutes:
                    failures.append((z, action, result))

        return {
            "total_checks": total_checks,
            "n_failures": len(failures),
            "all_commute": len(failures) == 0,
            "max_defect": max_defect,
            "mean_defect": total_defect / total_checks if total_checks > 0 else 0.0,
            "failures": failures,
        }

    def check_sequence(
        self,
        z: LatentState,
        action_sequence: List[Any]
    ) -> CommutationResult:
        """
        Check commutation for a composed action sequence.

        Proposition from Section 5: If U∘T_a = T_a∘U for all primitive actions a,
        then U∘T_α = T_α∘U for any sequence α.
        """
        # Compute T_α(z)
        z_current = z
        for action in action_sequence:
            z_current = self.transition.sample(z_current, action)
        Talpha_z = z_current

        # Compute U(T_α(z))
        UTalpha_z = self.equivalence.apply(Talpha_z)

        # Compute U(z)
        Uz = self.equivalence.apply(z)

        # Compute T_α(U(z))
        z_current = Uz
        for action in action_sequence:
            z_current = self.transition.sample(z_current, action)
        Talpha_Uz = z_current

        defect = np.linalg.norm(UTalpha_z.vector - Talpha_Uz.vector)
        commutes = defect <= self.tolerance

        return CommutationResult(
            commutes=commutes,
            defect=defect,
            z=z.vector,
            action=action_sequence,
            Uz=Uz.vector,
            TaUz=Talpha_Uz.vector,
            UTaz=UTalpha_z.vector,
        )


class ApproximateEquivalence:
    """
    ε-structural equivalence with error accumulation bounds.

    Section 5.3 defines ε-equivalence: U defines ε-structural equivalence
    over a set S if sup_{z∈S, a∈A} Δ_a(z) ≤ ε

    Error bound: For L-Lipschitz transitions and sequence length n,
        Δ_α(z) ≤ ε Σ_{k=0}^{n-1} L^k

    If L ≥ 1: Δ_α(z) ≤ n ε L^{n-1}
    """

    def __init__(
        self,
        equivalence: StructuralEquivalence,
        transition: TransitionKernel,
        actions: List[Any],
        epsilon: float,
        lipschitz_constant: float = 1.0
    ):
        """
        Initialize approximate equivalence checker.

        Args:
            equivalence: The equivalence mapping
            transition: Transition kernel
            actions: List of primitive actions
            epsilon: Maximum commutation defect for primitives
            lipschitz_constant: Lipschitz constant L of transition maps
        """
        self.equivalence = equivalence
        self.transition = transition
        self.actions = actions
        self.epsilon = epsilon
        self.L = lipschitz_constant

    def compute_epsilon(self, states: List[LatentState]) -> float:
        """
        Compute the actual ε from samples.

        ε = sup_{z∈S, a∈A} ||U(T_a(z)) - T_a(U(z))||
        """
        max_defect = 0.0

        for z in states:
            for action in self.actions:
                result = self.equivalence.check_commutation(
                    z, action, self.transition, tolerance=float('inf')
                )
                max_defect = max(max_defect, result.defect)

        return max_defect

    def error_bound(self, sequence_length: int) -> float:
        """
        Compute the error accumulation bound for a sequence of given length.

        Δ_α(z) ≤ ε Σ_{k=0}^{n-1} L^k
        """
        n = sequence_length

        if self.L == 1.0:
            # Geometric sum simplifies
            return self.epsilon * n

        # General case
        if abs(self.L - 1.0) < 1e-10:
            return self.epsilon * n

        geometric_sum = (self.L ** n - 1) / (self.L - 1)
        return self.epsilon * geometric_sum

    def tight_bound(self, sequence_length: int) -> float:
        """
        Tighter bound when L ≥ 1.

        Δ_α(z) ≤ n ε L^{n-1}
        """
        n = sequence_length
        return n * self.epsilon * (self.L ** (n - 1))

    def is_epsilon_equivalent(
        self,
        states: List[LatentState],
        target_epsilon: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Check if equivalence holds to within ε.

        Returns:
            is_equivalent: Whether max defect ≤ target_epsilon
            actual_epsilon: The actual maximum defect
        """
        actual_epsilon = self.compute_epsilon(states)

        if target_epsilon is None:
            target_epsilon = self.epsilon

        return actual_epsilon <= target_epsilon, actual_epsilon

    def stability_analysis(self, max_sequence_length: int = 20) -> Dict[str, List[float]]:
        """
        Analyze error accumulation over sequence lengths.

        Returns bounds and (for contractive L<1) shows stability.
        """
        lengths = list(range(1, max_sequence_length + 1))
        bounds = [self.error_bound(n) for n in lengths]
        tight_bounds = [self.tight_bound(n) for n in lengths]

        return {
            "lengths": lengths,
            "error_bounds": bounds,
            "tight_bounds": tight_bounds,
            "is_contractive": self.L < 1.0,
            "asymptotic_bound": self.epsilon / (1 - self.L) if self.L < 1.0 else float('inf'),
        }


def estimate_lipschitz_constant(
    transition: TransitionKernel,
    action: Any,
    states: List[LatentState],
    n_pairs: int = 100,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Estimate Lipschitz constant of transition map from samples.

    L = max ||T_a(z) - T_a(z')|| / ||z - z'||
    """
    if rng is None:
        rng = np.random.default_rng()

    max_ratio = 0.0
    n_states = len(states)

    for _ in range(n_pairs):
        i, j = rng.choice(n_states, size=2, replace=False)
        z1, z2 = states[i], states[j]

        dist_in = np.linalg.norm(z1.vector - z2.vector)
        if dist_in < 1e-10:
            continue

        z1_next = transition.sample(z1, action, rng)
        z2_next = transition.sample(z2, action, rng)

        dist_out = np.linalg.norm(z1_next.vector - z2_next.vector)

        ratio = dist_out / dist_in
        max_ratio = max(max_ratio, ratio)

    return max_ratio
