"""
Latent State representation for POMDP dynamics.

The latent state z(t) contains sufficient information for dynamics,
including unrecorded constraints and hidden dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import numpy as np
from numpy.typing import NDArray


@dataclass
class LatentState:
    """
    Represents a latent state z(t) in the POMDP.

    The latent state is a continuous vector representation that captures
    the hidden organizational state including unrecorded constraints
    and dependencies.

    Attributes:
        vector: The continuous latent representation as a numpy array
        metadata: Optional dictionary of additional state information
        timestamp: Optional temporal index
        state_id: Unique identifier for this state instance
    """
    vector: NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    state_id: Optional[str] = None

    def __post_init__(self):
        self.vector = np.asarray(self.vector, dtype=np.float64)
        if self.state_id is None:
            self.state_id = f"z_{id(self)}"

    @property
    def dim(self) -> int:
        """Dimensionality of the latent state."""
        return len(self.vector)

    def __array__(self) -> NDArray[np.float64]:
        """Allow numpy operations on the state."""
        return self.vector

    def __len__(self) -> int:
        return self.dim

    def distance(self, other: LatentState) -> float:
        """Euclidean distance to another latent state."""
        return float(np.linalg.norm(self.vector - other.vector))

    def copy(self) -> LatentState:
        """Create a copy of this state."""
        return LatentState(
            vector=self.vector.copy(),
            metadata=self.metadata.copy(),
            timestamp=self.timestamp,
            state_id=None  # Generate new ID
        )


class LatentSpace:
    """
    Defines the structure and constraints of the latent state space.

    This class manages the geometry of the latent space, including
    dimensionality, bounds, and sampling procedures.
    """

    def __init__(
        self,
        dim: int,
        bounds: Optional[tuple[NDArray, NDArray]] = None,
        name: str = "latent_space"
    ):
        """
        Initialize the latent space.

        Args:
            dim: Dimensionality of the latent space
            bounds: Optional (lower, upper) bounds for each dimension
            name: Identifier for this space
        """
        self.dim = dim
        self.name = name

        if bounds is not None:
            self.lower_bound = np.asarray(bounds[0], dtype=np.float64)
            self.upper_bound = np.asarray(bounds[1], dtype=np.float64)
            assert len(self.lower_bound) == dim
            assert len(self.upper_bound) == dim
        else:
            self.lower_bound = np.full(dim, -np.inf)
            self.upper_bound = np.full(dim, np.inf)

    def sample_uniform(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> list[LatentState]:
        """
        Sample uniformly from the latent space (within bounds).

        For unbounded dimensions, samples from standard normal.
        """
        if rng is None:
            rng = np.random.default_rng()

        states = []
        for _ in range(n):
            vector = np.zeros(self.dim)
            for d in range(self.dim):
                if np.isfinite(self.lower_bound[d]) and np.isfinite(self.upper_bound[d]):
                    vector[d] = rng.uniform(self.lower_bound[d], self.upper_bound[d])
                else:
                    vector[d] = rng.standard_normal()
            states.append(LatentState(vector=vector))

        return states if n > 1 else states

    def sample_gaussian(
        self,
        mean: Optional[NDArray] = None,
        std: Optional[Union[float, NDArray]] = None,
        n: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> list[LatentState]:
        """Sample from a Gaussian in the latent space."""
        if rng is None:
            rng = np.random.default_rng()

        if mean is None:
            mean = np.zeros(self.dim)
        if std is None:
            std = 1.0

        mean = np.asarray(mean)
        if isinstance(std, (int, float)):
            std = np.full(self.dim, std)
        std = np.asarray(std)

        states = []
        for _ in range(n):
            vector = rng.normal(mean, std)
            # Clip to bounds
            vector = np.clip(vector, self.lower_bound, self.upper_bound)
            states.append(LatentState(vector=vector))

        return states

    def contains(self, state: LatentState) -> bool:
        """Check if a state is within the space bounds."""
        return (
            np.all(state.vector >= self.lower_bound) and
            np.all(state.vector <= self.upper_bound)
        )

    def project(self, state: LatentState) -> LatentState:
        """Project a state onto the bounded space."""
        projected = np.clip(state.vector, self.lower_bound, self.upper_bound)
        return LatentState(
            vector=projected,
            metadata=state.metadata.copy(),
            timestamp=state.timestamp
        )


class DiscreteLatentState:
    """
    Discrete latent state for systems with finite state spaces.

    Used for exact computations and validation (e.g., the lumpability
    construction in Section 3 of the paper).
    """

    def __init__(self, state_index: int, n_states: int, label: Optional[str] = None):
        """
        Initialize a discrete latent state.

        Args:
            state_index: Integer index of the state (0 to n_states-1)
            n_states: Total number of states in the space
            label: Optional human-readable label
        """
        assert 0 <= state_index < n_states
        self.state_index = state_index
        self.n_states = n_states
        self.label = label or f"s_{state_index}"

    def to_onehot(self) -> NDArray[np.float64]:
        """Convert to one-hot encoding."""
        vec = np.zeros(self.n_states, dtype=np.float64)
        vec[self.state_index] = 1.0
        return vec

    def to_latent_state(self) -> LatentState:
        """Convert to continuous LatentState (one-hot encoding)."""
        return LatentState(
            vector=self.to_onehot(),
            metadata={"discrete_index": self.state_index, "label": self.label}
        )

    @classmethod
    def from_index(cls, index: int, n_states: int) -> DiscreteLatentState:
        """Create from index."""
        return cls(state_index=index, n_states=n_states)

    def __eq__(self, other):
        if isinstance(other, DiscreteLatentState):
            return self.state_index == other.state_index
        return False

    def __hash__(self):
        return hash(self.state_index)

    def __repr__(self):
        return f"DiscreteLatentState({self.label})"
