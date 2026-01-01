"""
Simulation and counterfactual reasoning.

At query time, simulation is inference under the constrained model.
For a candidate action a, the system evaluates a distribution over
next states and outcomes by integrating both latent state uncertainty
and operator uncertainty.

The predictive distribution (from Section 1):
    p(x(t+1) | history, a) = ∫∫ E(x(t+1)|z(t+1)) K_a(z(t+1)|z(t)) b_t(z(t)) dz(t) dz(t+1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from vajra.core.latent_state import LatentState
from vajra.core.transition_kernel import TransitionKernel
from vajra.core.emission_model import EmissionModel
from vajra.core.belief_state import BeliefState, ParticleBeliefState
from vajra.inference.posterior import PosteriorApproximation


@dataclass
class Trajectory:
    """A simulated trajectory of states, actions, and observations."""
    states: List[LatentState]
    actions: List[Any]
    observations: List[NDArray]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.actions)

    def get_state_sequence(self) -> NDArray:
        """Get state vectors as numpy array."""
        return np.array([s.vector for s in self.states])

    def get_observation_sequence(self) -> NDArray:
        """Get observations as numpy array."""
        return np.array(self.observations)


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation."""
    trajectories: List[Trajectory]
    action_sequence: List[Any]
    initial_belief: BeliefState

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)

    def final_state_distribution(self) -> Tuple[NDArray, NDArray]:
        """Get mean and variance of final states."""
        final_states = np.array([t.states[-1].vector for t in self.trajectories])
        return final_states.mean(axis=0), final_states.var(axis=0)

    def observation_distribution(self, time_step: int) -> Tuple[NDArray, NDArray]:
        """Get mean and variance of observations at a time step."""
        obs = np.array([t.observations[time_step] for t in self.trajectories])
        return obs.mean(axis=0), obs.var(axis=0)

    def quantiles(self, time_step: int, q: List[float] = [0.1, 0.5, 0.9]) -> Dict[str, NDArray]:
        """Get quantiles of observations at a time step."""
        obs = np.array([t.observations[time_step] for t in self.trajectories])
        result = {}
        for qi in q:
            result[f"q{int(qi*100)}"] = np.quantile(obs, qi, axis=0)
        return result


class Simulator:
    """
    Monte Carlo simulator for latent dynamics.

    Simulates trajectories by:
    1. Sampling from the current belief state
    2. Applying action-conditioned transitions
    3. Generating observations through the emission model
    """

    def __init__(
        self,
        transition: TransitionKernel,
        emission: EmissionModel,
        posterior: Optional[PosteriorApproximation] = None
    ):
        """
        Initialize simulator.

        Args:
            transition: Action-conditioned transition kernel
            emission: Observation emission model
            posterior: Optional posterior over dynamics (for epistemic uncertainty)
        """
        self.transition = transition
        self.emission = emission
        self.posterior = posterior

    def simulate_trajectory(
        self,
        initial_state: LatentState,
        action_sequence: List[Any],
        rng: Optional[np.random.Generator] = None
    ) -> Trajectory:
        """
        Simulate a single trajectory.

        Args:
            initial_state: Starting latent state
            action_sequence: Sequence of actions to take
            rng: Random generator
        """
        if rng is None:
            rng = np.random.default_rng()

        states = [initial_state]
        observations = [self.emission.sample(initial_state, rng)]
        actions = []

        z = initial_state
        for action in action_sequence:
            # Transition
            if self.posterior is not None:
                # Sample from posterior predictive
                z_next = self.posterior.predict(z, action, n_samples=1, rng=rng)[0]
            else:
                z_next = self.transition.sample(z, action, rng)

            # Emit observation
            x = self.emission.sample(z_next, rng)

            states.append(z_next)
            observations.append(x)
            actions.append(action)

            z = z_next

        return Trajectory(
            states=states,
            actions=actions,
            observations=observations,
        )

    def simulate_from_belief(
        self,
        belief: BeliefState,
        action_sequence: List[Any],
        n_trajectories: int = 100,
        rng: Optional[np.random.Generator] = None,
        show_progress: bool = False
    ) -> SimulationResult:
        """
        Monte Carlo simulation from a belief state.

        Samples initial states from belief, then simulates trajectories.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample initial states
        initial_states = belief.sample(n_trajectories, rng)

        trajectories = []
        iterator = initial_states
        if show_progress:
            iterator = tqdm(iterator, desc="Simulating")

        for z0 in iterator:
            traj = self.simulate_trajectory(z0, action_sequence, rng)
            trajectories.append(traj)

        return SimulationResult(
            trajectories=trajectories,
            action_sequence=action_sequence,
            initial_belief=belief,
        )

    def predict_distribution(
        self,
        belief: BeliefState,
        action: Any,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Predict distribution over next state and observation.

        Returns:
            state_mean: E[z']
            state_var: Var[z']
            obs_mean: E[x']
            obs_var: Var[x']
        """
        if rng is None:
            rng = np.random.default_rng()

        result = self.simulate_from_belief(
            belief, [action], n_samples, rng
        )

        state_mean, state_var = result.final_state_distribution()
        obs_mean, obs_var = result.observation_distribution(1)

        return state_mean, state_var, obs_mean, obs_var

    def action_comparison(
        self,
        belief: BeliefState,
        actions: List[Any],
        n_samples: int = 500,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Compare outcome distributions for different actions.

        Useful for decision support.
        """
        if rng is None:
            rng = np.random.default_rng()

        results = {}
        for action in actions:
            state_mean, state_var, obs_mean, obs_var = self.predict_distribution(
                belief, action, n_samples, rng
            )
            results[action] = {
                "state_mean": state_mean,
                "state_var": state_var,
                "state_uncertainty": float(state_var.sum()),
                "obs_mean": obs_mean,
                "obs_var": obs_var,
                "obs_uncertainty": float(obs_var.sum()),
            }

        return results


class CounterfactualSimulator:
    """
    Counterfactual reasoning via simulation.

    Answers questions of the form:
    "What would have happened if action a' had been taken instead of a?"

    This requires backward inference (reconstructing the prior state)
    followed by forward simulation under the counterfactual action.
    """

    def __init__(
        self,
        transition: TransitionKernel,
        emission: EmissionModel,
        posterior: Optional[PosteriorApproximation] = None
    ):
        self.transition = transition
        self.emission = emission
        self.posterior = posterior
        self.simulator = Simulator(transition, emission, posterior)

    def counterfactual_from_observation(
        self,
        observation: NDArray,
        factual_action: Any,
        counterfactual_action: Any,
        prior_belief: BeliefState,
        n_samples: int = 500,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[SimulationResult, SimulationResult]:
        """
        Compute counterfactual outcome.

        Args:
            observation: Current observation x(t)
            factual_action: Action that was actually taken
            counterfactual_action: Alternative action to consider
            prior_belief: Belief state before observation
            n_samples: Number of Monte Carlo samples

        Returns:
            factual_result: Simulation under factual action
            counterfactual_result: Simulation under counterfactual action
        """
        if rng is None:
            rng = np.random.default_rng()

        # Update belief with observation
        posterior_belief = prior_belief.update(observation, self.emission)

        # Simulate factual
        factual_result = self.simulator.simulate_from_belief(
            posterior_belief,
            [factual_action],
            n_samples,
            rng
        )

        # Simulate counterfactual
        counterfactual_result = self.simulator.simulate_from_belief(
            posterior_belief,
            [counterfactual_action],
            n_samples,
            rng
        )

        return factual_result, counterfactual_result

    def counterfactual_trajectory(
        self,
        observed_trajectory: Trajectory,
        intervention_time: int,
        counterfactual_actions: List[Any],
        prior_belief: BeliefState,
        n_samples: int = 500,
        rng: Optional[np.random.Generator] = None
    ) -> SimulationResult:
        """
        What if different actions had been taken from a certain point?

        Args:
            observed_trajectory: The factual trajectory that occurred
            intervention_time: Time step at which to intervene
            counterfactual_actions: Actions to take after intervention
            prior_belief: Prior belief at start of trajectory
            n_samples: Number of Monte Carlo samples
        """
        if rng is None:
            rng = np.random.default_rng()

        # Propagate belief up to intervention time
        belief = prior_belief
        for t in range(intervention_time):
            belief = belief.update(observed_trajectory.observations[t], self.emission)
            if t < len(observed_trajectory.actions):
                belief = belief.predict(observed_trajectory.actions[t], self.transition)

        # Update with observation at intervention time
        belief = belief.update(observed_trajectory.observations[intervention_time], self.emission)

        # Simulate counterfactual from this belief
        return self.simulator.simulate_from_belief(
            belief,
            counterfactual_actions,
            n_samples,
            rng
        )

    def causal_effect(
        self,
        belief: BeliefState,
        treatment_action: Any,
        control_action: Any,
        outcome_fn: callable,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, float]:
        """
        Estimate average causal effect of treatment vs control.

        Args:
            belief: Current belief state
            treatment_action: Treatment action
            control_action: Control action
            outcome_fn: Function mapping observation to scalar outcome
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary with ATE, treatment mean, control mean, and confidence interval
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample initial states
        initial_states = belief.sample(n_samples, rng)

        treatment_outcomes = []
        control_outcomes = []

        for z0 in initial_states:
            # Treatment trajectory
            traj_t = self.simulator.simulate_trajectory(z0, [treatment_action], rng)
            treatment_outcomes.append(outcome_fn(traj_t.observations[-1]))

            # Control trajectory (same initial state for pairing)
            traj_c = self.simulator.simulate_trajectory(z0, [control_action], rng)
            control_outcomes.append(outcome_fn(traj_c.observations[-1]))

        treatment_outcomes = np.array(treatment_outcomes)
        control_outcomes = np.array(control_outcomes)

        # Paired difference
        effects = treatment_outcomes - control_outcomes
        ate = effects.mean()
        se = effects.std() / np.sqrt(n_samples)

        return {
            "ate": float(ate),
            "se": float(se),
            "ci_lower": float(ate - 1.96 * se),
            "ci_upper": float(ate + 1.96 * se),
            "treatment_mean": float(treatment_outcomes.mean()),
            "control_mean": float(control_outcomes.mean()),
        }


class InformationMetrics:
    """
    Information-theoretic metrics for reversibility analysis.

    Section 6 of the paper discusses how counterfactual reasoning
    requires information preservation, quantified by conditional
    entropy and mutual information.
    """

    @staticmethod
    def conditional_entropy_mc(
        transition: TransitionKernel,
        action: Any,
        prior_samples: List[LatentState],
        n_posterior_samples: int = 100,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Monte Carlo estimate of H(z_t | z_{t+1}, a).

        This quantifies information loss in the forward transition.
        Large values indicate poor reversibility.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample z_{t+1} from prior predictive
        n_prior = len(prior_samples)
        posterior_log_probs = []

        for z0 in prior_samples:
            z1 = transition.sample(z0, action, rng)

            # Estimate posterior p(z0 | z1, a) using importance sampling
            log_probs = []
            for z0_sample in prior_samples:
                log_p = transition.log_prob(z0_sample, action, z1)
                log_probs.append(log_p)

            # Normalize to get posterior
            log_probs = np.array(log_probs)
            log_probs = log_probs - np.max(log_probs)
            probs = np.exp(log_probs)
            probs = probs / probs.sum()

            # Entropy of posterior
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            posterior_log_probs.append(entropy)

        return float(np.mean(posterior_log_probs))

    @staticmethod
    def mutual_information_mc(
        transition: TransitionKernel,
        action: Any,
        prior_samples: List[LatentState],
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Monte Carlo estimate of I(z_t; z_{t+1} | a).

        I(z_t; z_{t+1} | a) = H(z_{t+1} | a) - H(z_{t+1} | z_t, a)

        High mutual information indicates good information preservation.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample transitions
        next_states = []
        for z0 in prior_samples:
            z1 = transition.sample(z0, action, rng)
            next_states.append(z1.vector)

        next_states = np.array(next_states)

        # Estimate H(z_{t+1} | a) using kernel density or sample entropy
        # Simple proxy: use trace of covariance (Gaussian assumption)
        cov = np.cov(next_states.T)
        if next_states.shape[1] == 1:
            cov = np.array([[cov]])

        try:
            sign, logdet = np.linalg.slogdet(cov)
            if sign > 0:
                # Gaussian entropy
                d = next_states.shape[1]
                H_marginal = 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)
            else:
                H_marginal = 0.0
        except np.linalg.LinAlgError:
            H_marginal = 0.0

        # H(z_{t+1} | z_t, a) is the conditional entropy (from model)
        if hasattr(transition, 'variance'):
            # Average conditional entropy (Gaussian)
            d = next_states.shape[1]
            var_sum = 0
            for z0 in prior_samples:
                var = transition.variance(z0, action)
                var_sum += np.log(var + 1e-10).sum()
            H_conditional = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * var_sum / len(prior_samples)
        else:
            # Assume small noise
            H_conditional = 0.0

        return float(max(0, H_marginal - H_conditional))
