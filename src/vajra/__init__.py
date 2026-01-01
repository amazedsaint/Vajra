"""
Vajra: Context Graph System for POMDP-based Organizational Dynamics

This package implements a first-principles formalization of context graphs as an
append-only evidence substrate that constrains latent transition operators in
partially observed decision processes.

Key components:
- LatentState: Representation of hidden organizational state
- TransitionKernel: Action-conditioned stochastic transition operators
- EvidenceStore: Append-only storage for transition triples
- BeliefState: Probability distribution over latent states
- ContextGraph: Main interface for the CGS system

Based on the theoretical framework:
    z(t+1) ~ K_a(·|z(t))  [latent dynamics]
    x(t) ~ E(·|z(t))      [emission model]
"""

__version__ = "0.1.0"

from vajra.core.latent_state import LatentState, LatentSpace
from vajra.core.transition_kernel import (
    TransitionKernel,
    DeterministicTransition,
    GaussianTransitionKernel,
    HeteroscedasticTransitionKernel,
)
from vajra.core.emission_model import EmissionModel, GaussianEmission
from vajra.core.belief_state import BeliefState, ParticleBeliefState
from vajra.evidence.store import EvidenceStore, TransitionEvidence
from vajra.inference.encoder import Encoder, MLPEncoder
from vajra.inference.posterior import (
    PosteriorApproximation,
    EnsemblePosterior,
    ParticlePosterior,
)
from vajra.structural.equivalence import (
    StructuralEquivalence,
    CommutationChecker,
    ApproximateEquivalence,
)
from vajra.simulation.simulator import Simulator, CounterfactualSimulator
from vajra.context_graph import ContextGraph

__all__ = [
    # Core
    "LatentState",
    "LatentSpace",
    "TransitionKernel",
    "DeterministicTransition",
    "GaussianTransitionKernel",
    "HeteroscedasticTransitionKernel",
    "EmissionModel",
    "GaussianEmission",
    "BeliefState",
    "ParticleBeliefState",
    # Evidence
    "EvidenceStore",
    "TransitionEvidence",
    # Inference
    "Encoder",
    "MLPEncoder",
    "PosteriorApproximation",
    "EnsemblePosterior",
    "ParticlePosterior",
    # Structural
    "StructuralEquivalence",
    "CommutationChecker",
    "ApproximateEquivalence",
    # Simulation
    "Simulator",
    "CounterfactualSimulator",
    # Main interface
    "ContextGraph",
]
