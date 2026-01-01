"""
Context Graph: Main interface for the CGS system.

A context graph is an append-only evidence substrate that constrains
latent transition operators in partially observed decision processes.

This class provides the main interface for:
- Accumulating transition evidence
- Learning transition operators
- Maintaining beliefs about latent state
- Simulating counterfactuals
- Querying structural equivalences
"""

from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple, Union
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from vajra.core.latent_state import LatentState, LatentSpace
from vajra.core.transition_kernel import (
    TransitionKernel,
    NeuralTransitionKernel,
    HeteroscedasticTransitionKernel,
)
from vajra.core.emission_model import EmissionModel, GaussianEmission, IdentityEmission
from vajra.core.belief_state import BeliefState, ParticleBeliefState, GaussianBeliefState
from vajra.evidence.store import EvidenceStore, TransitionEvidence
from vajra.inference.encoder import Encoder, MLPEncoder, IdentityEncoder
from vajra.inference.posterior import PosteriorApproximation, EnsemblePosterior
from vajra.simulation.simulator import Simulator, CounterfactualSimulator, SimulationResult
from vajra.structural.equivalence import StructuralEquivalence, CommutationChecker


class ContextGraph:
    """
    Main interface for the Context Graph System.

    The context graph maintains:
    - An append-only evidence store of transitions
    - An encoder for mapping observations to latent space
    - Action-conditioned transition kernels
    - A posterior over dynamics parameters
    - Current belief state

    Example usage:
        # Create context graph
        cg = ContextGraph(obs_dim=10, latent_dim=4, n_actions=3)

        # Record evidence
        cg.record_transition(x_current, action, x_next)

        # Train on accumulated evidence
        cg.train(epochs=100)

        # Update belief with new observation
        cg.update_belief(observation)

        # Simulate outcomes under candidate action
        result = cg.simulate(action, n_trajectories=100)

        # Query counterfactual
        cf_result = cg.counterfactual(
            factual_action=0,
            counterfactual_action=1
        )
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        n_actions: int,
        encoder_type: str = "mlp",
        hidden_dims: List[int] = [64, 64],
        n_ensemble: int = 5,
        use_identity_encoder: bool = False,
        name: str = "context_graph"
    ):
        """
        Initialize Context Graph.

        Args:
            obs_dim: Dimensionality of observations
            latent_dim: Dimensionality of latent space
            n_actions: Number of discrete actions
            encoder_type: Type of encoder ("mlp", "vae", "identity")
            hidden_dims: Hidden layer dimensions for networks
            n_ensemble: Number of ensemble members for posterior
            use_identity_encoder: If True, use observations directly as latent states
            name: Name for this context graph
        """
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.name = name

        # Latent space
        self.latent_space = LatentSpace(dim=latent_dim, name=f"{name}_latent")

        # Evidence store
        self.evidence_store = EvidenceStore(name=f"{name}_evidence")

        # Encoder
        if use_identity_encoder or encoder_type == "identity":
            assert obs_dim == latent_dim, "Identity encoder requires obs_dim == latent_dim"
            self.encoder = IdentityEncoder(obs_dim)
        elif encoder_type == "mlp":
            self.encoder = MLPEncoder(obs_dim, latent_dim, hidden_dims)
        else:
            self.encoder = MLPEncoder(obs_dim, latent_dim, hidden_dims)

        # Emission model (default: identity with noise)
        self._emission: Optional[EmissionModel] = None

        # Transition kernel posterior (ensemble)
        self.posterior = EnsemblePosterior(
            base_kernel_class=NeuralTransitionKernel,
            kernel_kwargs={
                "latent_dim": latent_dim,
                "action_dim": n_actions,
                "hidden_dims": hidden_dims,
            },
            n_models=n_ensemble,
            bootstrap=True
        )

        # Current belief state
        self._belief: Optional[BeliefState] = None

        # Simulator
        self._simulator: Optional[Simulator] = None
        self._cf_simulator: Optional[CounterfactualSimulator] = None

        # Training state
        self.is_trained = False

    @property
    def emission(self) -> EmissionModel:
        """Get or create emission model."""
        if self._emission is None:
            self._emission = IdentityEmission(self.latent_dim, noise_std=0.1)
        return self._emission

    @emission.setter
    def emission(self, model: EmissionModel):
        """Set emission model."""
        self._emission = model

    @property
    def belief(self) -> BeliefState:
        """Get current belief state, initializing if needed."""
        if self._belief is None:
            self.reset_belief()
        return self._belief

    def reset_belief(self, prior_std: float = 1.0) -> None:
        """Reset belief to prior."""
        self._belief = GaussianBeliefState(
            mean_vec=np.zeros(self.latent_dim),
            cov_mat=np.eye(self.latent_dim) * prior_std ** 2
        )

    def record_transition(
        self,
        x_current: NDArray,
        action: int,
        x_next: NDArray,
        case_id: Optional[str] = None,
        **metadata
    ) -> TransitionEvidence:
        """
        Record a transition observation.

        This is the primary way to accumulate evidence.

        Args:
            x_current: Current observation
            action: Action taken
            x_next: Next observation
            case_id: Optional case identifier for trajectory grouping
            **metadata: Additional metadata to store
        """
        # Encode to latent
        z_current = self.encoder.encode(np.asarray(x_current))
        z_next = self.encoder.encode(np.asarray(x_next))

        evidence = self.evidence_store.append_transition(
            x_current=x_current,
            action=action,
            x_next=x_next,
            z_current=z_current.vector,
            z_next=z_next.vector,
            case_id=case_id,
            **metadata
        )

        return evidence

    def record_trajectory(
        self,
        observations: List[NDArray],
        actions: List[int],
        case_id: Optional[str] = None
    ) -> List[TransitionEvidence]:
        """
        Record a complete trajectory.

        Args:
            observations: List of observations [x_0, x_1, ..., x_T]
            actions: List of actions [a_0, a_1, ..., a_{T-1}]
            case_id: Optional case identifier
        """
        assert len(actions) == len(observations) - 1

        evidence_list = []
        for t, action in enumerate(actions):
            evidence = self.record_transition(
                x_current=observations[t],
                action=action,
                x_next=observations[t + 1],
                case_id=case_id,
                time_step=t
            )
            evidence_list.append(evidence)

        return evidence_list

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train transition models on accumulated evidence.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress
        """
        if len(self.evidence_store) == 0:
            raise ValueError("No evidence to train on")

        history = self.posterior.fit(
            self.evidence_store,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose
        )

        self.is_trained = True

        # Initialize simulators
        self._init_simulators()

        return history

    def _init_simulators(self) -> None:
        """Initialize simulation components."""
        # Use mean kernel from ensemble for simulation
        mean_kernel = self.posterior.models[0]

        self._simulator = Simulator(
            transition=mean_kernel,
            emission=self.emission,
            posterior=self.posterior
        )

        self._cf_simulator = CounterfactualSimulator(
            transition=mean_kernel,
            emission=self.emission,
            posterior=self.posterior
        )

    def update_belief(self, observation: NDArray) -> BeliefState:
        """
        Update belief state with new observation.

        Implements Bayesian belief update:
            b'(z) ∝ E(x|z) b(z)
        """
        self._belief = self.belief.update(observation, self.emission)
        return self._belief

    def predict_belief(self, action: int) -> BeliefState:
        """
        Predict belief after taking action.

        Implements belief prediction:
            b'(z') = ∫ K_a(z'|z) b(z) dz
        """
        # Use mean kernel from posterior
        if not self.posterior.models:
            raise ValueError("No trained models available")

        transition = self.posterior.models[0]
        self._belief = self.belief.predict(action, transition)
        return self._belief

    def simulate(
        self,
        action_sequence: Union[int, List[int]],
        n_trajectories: int = 100,
        from_belief: Optional[BeliefState] = None,
        rng: Optional[np.random.Generator] = None
    ) -> SimulationResult:
        """
        Simulate trajectories under action sequence.

        Args:
            action_sequence: Single action or sequence of actions
            n_trajectories: Number of Monte Carlo samples
            from_belief: Starting belief (default: current belief)
            rng: Random generator
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before simulation")

        if isinstance(action_sequence, int):
            action_sequence = [action_sequence]

        belief = from_belief or self.belief

        return self._simulator.simulate_from_belief(
            belief,
            action_sequence,
            n_trajectories,
            rng
        )

    def predict(
        self,
        action: int,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, NDArray]:
        """
        Predict distribution over next state under action.

        Returns mean, variance, and quantiles.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        state_mean, state_var, obs_mean, obs_var = self._simulator.predict_distribution(
            self.belief, action, n_samples, rng
        )

        return {
            "state_mean": state_mean,
            "state_var": state_var,
            "state_std": np.sqrt(state_var),
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "obs_std": np.sqrt(obs_var),
        }

    def compare_actions(
        self,
        actions: Optional[List[int]] = None,
        n_samples: int = 500,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compare outcome distributions for different actions.

        Useful for decision support.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before comparison")

        if actions is None:
            actions = list(range(self.n_actions))

        return self._simulator.action_comparison(
            self.belief, actions, n_samples, rng
        )

    def counterfactual(
        self,
        observation: NDArray,
        factual_action: int,
        counterfactual_action: int,
        n_samples: int = 500,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[SimulationResult, SimulationResult]:
        """
        Counterfactual reasoning: what if a different action had been taken?

        Args:
            observation: Current observation
            factual_action: Action that was actually taken
            counterfactual_action: Alternative action to consider
            n_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (factual_result, counterfactual_result)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before counterfactual reasoning")

        return self._cf_simulator.counterfactual_from_observation(
            observation,
            factual_action,
            counterfactual_action,
            self.belief,
            n_samples,
            rng
        )

    def epistemic_uncertainty(self, state: Optional[LatentState] = None, action: int = 0) -> float:
        """
        Get epistemic uncertainty at a state-action pair.

        High epistemic uncertainty indicates need for more data.
        """
        if state is None:
            # Use belief mean
            state = LatentState(vector=self.belief.mean())

        return self.posterior.epistemic_uncertainty(state, action)

    def save(self, path: Union[str, Path]) -> None:
        """Save context graph to directory."""
        import torch
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save evidence
        self.evidence_store.save(path / "evidence.json")

        # Save models
        for i, model in enumerate(self.posterior.models):
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), path / f"model_{i}.pt")

        # Save config
        import json
        config = {
            "obs_dim": self.obs_dim,
            "latent_dim": self.latent_dim,
            "n_actions": self.n_actions,
            "n_ensemble": len(self.posterior.models),
            "name": self.name,
            "is_trained": self.is_trained,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ContextGraph:
        """Load context graph from directory."""
        import torch
        path = Path(path)

        # Load config
        import json
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # Create instance
        cg = cls(
            obs_dim=config["obs_dim"],
            latent_dim=config["latent_dim"],
            n_actions=config["n_actions"],
            n_ensemble=config["n_ensemble"],
            name=config["name"]
        )

        # Load evidence
        cg.evidence_store = EvidenceStore.load(path / "evidence.json")

        # Load models
        for i, model in enumerate(cg.posterior.models):
            model_path = path / f"model_{i}.pt"
            if model_path.exists() and hasattr(model, 'load_state_dict'):
                model.load_state_dict(torch.load(model_path))

        cg.is_trained = config["is_trained"]
        if cg.is_trained:
            cg._init_simulators()

        return cg

    def summary(self) -> str:
        """Get summary of context graph state."""
        lines = [
            f"Context Graph: {self.name}",
            f"  Dimensions: obs={self.obs_dim}, latent={self.latent_dim}",
            f"  Actions: {self.n_actions}",
            f"  Evidence: {len(self.evidence_store)} transitions",
            f"  Trained: {self.is_trained}",
            f"  Ensemble size: {len(self.posterior.models)}",
        ]

        if len(self.evidence_store) > 0:
            stats = self.evidence_store.get_action_statistics()
            lines.append("  Actions:")
            for action, info in stats.items():
                lines.append(f"    {action}: {info['count']} observations")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ContextGraph(name='{self.name}', obs_dim={self.obs_dim}, "
            f"latent_dim={self.latent_dim}, n_actions={self.n_actions}, "
            f"n_evidence={len(self.evidence_store)}, trained={self.is_trained})"
        )


def create_simple_context_graph(
    obs_dim: int,
    n_actions: int,
    latent_dim: Optional[int] = None,
    name: str = "simple_cg"
) -> ContextGraph:
    """
    Create a simple context graph with sensible defaults.

    Args:
        obs_dim: Observation dimensionality
        n_actions: Number of discrete actions
        latent_dim: Latent dimensionality (default: same as obs_dim)
        name: Name for the context graph
    """
    if latent_dim is None:
        latent_dim = obs_dim

    return ContextGraph(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        n_actions=n_actions,
        use_identity_encoder=(obs_dim == latent_dim),
        hidden_dims=[32, 32],
        n_ensemble=3,
        name=name
    )
