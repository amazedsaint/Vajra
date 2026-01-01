"""
Append-only evidence store for transition triples.

A context graph is an append-only evidence substrate that constrains
the transition operators K_a. Each executed action generates evidence
that can be summarized as a transition observation:

    e_t = (x(t), a(t), x(t+1))     [observable]
    ẽ_t = (z(t), a(t), z(t+1))     [latent]

The central design choice is that the store is append-only.
Overwriting destroys evidence about dynamics, which weakens posterior inference.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Iterator, Tuple
from datetime import datetime
import uuid
import json
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from vajra.core.latent_state import LatentState


@dataclass
class TransitionEvidence:
    """
    A single transition observation (evidence triple).

    Contains both observable and latent representations of a transition.
    """
    # Observable transition
    x_current: NDArray[np.float64]
    action: Any
    x_next: NDArray[np.float64]

    # Latent transition (if encoder is available)
    z_current: Optional[NDArray[np.float64]] = None
    z_next: Optional[NDArray[np.float64]] = None

    # Metadata
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    source_system: Optional[str] = None
    case_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.x_current = np.asarray(self.x_current, dtype=np.float64)
        self.x_next = np.asarray(self.x_next, dtype=np.float64)
        if self.z_current is not None:
            self.z_current = np.asarray(self.z_current, dtype=np.float64)
        if self.z_next is not None:
            self.z_next = np.asarray(self.z_next, dtype=np.float64)

    @property
    def has_latent(self) -> bool:
        """Check if latent representation is available."""
        return self.z_current is not None and self.z_next is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "x_current": self.x_current.tolist(),
            "action": self.action if isinstance(self.action, (int, float, str)) else str(self.action),
            "x_next": self.x_next.tolist(),
            "z_current": self.z_current.tolist() if self.z_current is not None else None,
            "z_next": self.z_next.tolist() if self.z_next is not None else None,
            "timestamp": self.timestamp.isoformat(),
            "source_system": self.source_system,
            "case_id": self.case_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TransitionEvidence:
        """Deserialize from dictionary."""
        return cls(
            evidence_id=data["evidence_id"],
            x_current=np.array(data["x_current"]),
            action=data["action"],
            x_next=np.array(data["x_next"]),
            z_current=np.array(data["z_current"]) if data.get("z_current") else None,
            z_next=np.array(data["z_next"]) if data.get("z_next") else None,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_system=data.get("source_system"),
            case_id=data.get("case_id"),
            metadata=data.get("metadata", {}),
        )


class EvidenceStore:
    """
    Append-only store for transition evidence.

    This is the core data structure for the context graph.
    Evidence accumulates over time and constrains operator inference.
    """

    def __init__(self, name: str = "default"):
        """
        Initialize empty evidence store.

        Args:
            name: Identifier for this store
        """
        self.name = name
        self._evidence: List[TransitionEvidence] = []
        self._action_index: Dict[Any, List[int]] = {}
        self._case_index: Dict[str, List[int]] = {}
        self._created_at = datetime.now()

    def append(self, evidence: TransitionEvidence) -> None:
        """
        Append a new transition observation.

        This is the only write operation - no updates or deletes.
        """
        idx = len(self._evidence)
        self._evidence.append(evidence)

        # Index by action
        action = evidence.action
        if action not in self._action_index:
            self._action_index[action] = []
        self._action_index[action].append(idx)

        # Index by case
        if evidence.case_id:
            if evidence.case_id not in self._case_index:
                self._case_index[evidence.case_id] = []
            self._case_index[evidence.case_id].append(idx)

    def append_transition(
        self,
        x_current: NDArray,
        action: Any,
        x_next: NDArray,
        z_current: Optional[NDArray] = None,
        z_next: Optional[NDArray] = None,
        case_id: Optional[str] = None,
        source_system: Optional[str] = None,
        **metadata
    ) -> TransitionEvidence:
        """
        Convenience method to create and append evidence.
        """
        evidence = TransitionEvidence(
            x_current=x_current,
            action=action,
            x_next=x_next,
            z_current=z_current,
            z_next=z_next,
            case_id=case_id,
            source_system=source_system,
            metadata=metadata,
        )
        self.append(evidence)
        return evidence

    def __len__(self) -> int:
        return len(self._evidence)

    def __iter__(self) -> Iterator[TransitionEvidence]:
        return iter(self._evidence)

    def __getitem__(self, idx: int) -> TransitionEvidence:
        return self._evidence[idx]

    @property
    def actions(self) -> List[Any]:
        """List of unique actions in the store."""
        return list(self._action_index.keys())

    @property
    def case_ids(self) -> List[str]:
        """List of unique case IDs."""
        return list(self._case_index.keys())

    def get_by_action(self, action: Any) -> List[TransitionEvidence]:
        """Get all evidence for a specific action."""
        indices = self._action_index.get(action, [])
        return [self._evidence[i] for i in indices]

    def get_by_case(self, case_id: str) -> List[TransitionEvidence]:
        """Get all evidence for a specific case."""
        indices = self._case_index.get(case_id, [])
        return [self._evidence[i] for i in indices]

    def get_training_data(
        self,
        action: Optional[Any] = None,
        use_latent: bool = True
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Get training data as numpy arrays.

        Returns:
            z_batch: Current states (n_samples, latent_dim)
            a_batch: Actions (n_samples, action_dim) or (n_samples,)
            z_next_batch: Next states (n_samples, latent_dim)
        """
        if action is not None:
            evidence_list = self.get_by_action(action)
        else:
            evidence_list = self._evidence

        if use_latent:
            evidence_list = [e for e in evidence_list if e.has_latent]
            z_batch = np.array([e.z_current for e in evidence_list])
            z_next_batch = np.array([e.z_next for e in evidence_list])
        else:
            z_batch = np.array([e.x_current for e in evidence_list])
            z_next_batch = np.array([e.x_next for e in evidence_list])

        a_batch = np.array([e.action for e in evidence_list])

        return z_batch, a_batch, z_next_batch

    def get_action_statistics(self) -> Dict[Any, Dict[str, Any]]:
        """Get statistics for each action type."""
        stats = {}
        for action, indices in self._action_index.items():
            evidence_list = [self._evidence[i] for i in indices]
            stats[action] = {
                "count": len(evidence_list),
                "first_seen": min(e.timestamp for e in evidence_list),
                "last_seen": max(e.timestamp for e in evidence_list),
            }
        return stats

    def sample(
        self,
        n: int,
        action: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None
    ) -> List[TransitionEvidence]:
        """Sample evidence uniformly at random."""
        if rng is None:
            rng = np.random.default_rng()

        if action is not None:
            pool = self.get_by_action(action)
        else:
            pool = self._evidence

        if n >= len(pool):
            return list(pool)

        indices = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in indices]

    def save(self, path: Path) -> None:
        """Save evidence store to JSON file."""
        data = {
            "name": self.name,
            "created_at": self._created_at.isoformat(),
            "evidence": [e.to_dict() for e in self._evidence],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> EvidenceStore:
        """Load evidence store from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        store = cls(name=data["name"])
        store._created_at = datetime.fromisoformat(data["created_at"])

        for e_data in data["evidence"]:
            store.append(TransitionEvidence.from_dict(e_data))

        return store

    def compute_fisher_information(
        self,
        transition_kernel,
        action: Any,
        n_samples: int = 100
    ) -> NDArray:
        """
        Estimate Fisher information matrix for parameter identifiability.

        Section 9 notes: parameter identifiability depends on whether
        collected trajectories are informative. The Fisher information
        matrix having full rank is a local condition for identifiability.
        """
        evidence = self.get_by_action(action)
        if len(evidence) < n_samples:
            evidence = evidence
        else:
            rng = np.random.default_rng()
            indices = rng.choice(len(evidence), size=n_samples, replace=False)
            evidence = [evidence[i] for i in indices]

        if not evidence:
            return None

        # Numerical gradient estimation
        # This is a simplified version - full implementation would
        # compute gradients w.r.t. model parameters
        dim = len(evidence[0].z_current) if evidence[0].has_latent else len(evidence[0].x_current)

        # Placeholder - actual implementation depends on parametric form
        return np.eye(dim)


class CaseTrajectory:
    """
    A sequence of transitions belonging to a single case.

    Provides a process-oriented view of the evidence.
    """

    def __init__(self, case_id: str, evidence: List[TransitionEvidence]):
        self.case_id = case_id
        self.evidence = sorted(evidence, key=lambda e: e.timestamp)

    @property
    def length(self) -> int:
        return len(self.evidence)

    @property
    def actions(self) -> List[Any]:
        return [e.action for e in self.evidence]

    @property
    def duration(self) -> Optional[float]:
        if len(self.evidence) < 2:
            return None
        return (self.evidence[-1].timestamp - self.evidence[0].timestamp).total_seconds()

    def get_state_sequence(self, use_latent: bool = True) -> List[NDArray]:
        """Get sequence of states (latent or observable)."""
        if not self.evidence:
            return []

        if use_latent and all(e.has_latent for e in self.evidence):
            states = [self.evidence[0].z_current]
            states.extend(e.z_next for e in self.evidence)
        else:
            states = [self.evidence[0].x_current]
            states.extend(e.x_next for e in self.evidence)

        return states


class EvidenceBuffer:
    """
    Bounded buffer for recent evidence (for online learning).

    Maintains a sliding window of most recent evidence while
    preserving the append-only semantics for the main store.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: List[TransitionEvidence] = []
        self._pointer = 0
        self._is_full = False

    def append(self, evidence: TransitionEvidence) -> None:
        if not self._is_full:
            self._buffer.append(evidence)
            if len(self._buffer) >= self.max_size:
                self._is_full = True
        else:
            self._buffer[self._pointer] = evidence
            self._pointer = (self._pointer + 1) % self.max_size

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None
    ) -> List[TransitionEvidence]:
        if rng is None:
            rng = np.random.default_rng()

        if n >= len(self._buffer):
            return list(self._buffer)

        indices = rng.choice(len(self._buffer), size=n, replace=False)
        return [self._buffer[i] for i in indices]

    def to_store(self, name: str = "from_buffer") -> EvidenceStore:
        """Convert buffer to permanent store."""
        store = EvidenceStore(name=name)
        for e in self._buffer:
            store.append(e)
        return store
