"""
Demo script for the Vajra Context Graph System.

This script demonstrates the key capabilities of the CGS:
1. Recording transition evidence
2. Training action-conditioned dynamics
3. Belief state tracking
4. Simulation and prediction
5. Counterfactual reasoning
6. Epistemic uncertainty quantification

Run with:
    python -m vajra.demo
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from vajra import ContextGraph, LatentState
from vajra.core.transition_kernel import HeteroscedasticTransitionKernel
from vajra.core.belief_state import ParticleBeliefState


def create_synthetic_environment(
    n_states: int = 4,
    n_actions: int = 3,
    noise_std: float = 0.1,
    rng: Optional[np.random.Generator] = None
) -> tuple:
    """
    Create a synthetic environment for demonstration.

    The environment has:
    - Continuous state space (2D for visualization)
    - Discrete action space
    - Stochastic transitions with action-dependent dynamics
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Define action effects (deterministic component)
    action_effects = {
        0: np.array([0.5, 0.0]),   # Move right
        1: np.array([0.0, 0.5]),   # Move up
        2: np.array([-0.3, -0.3]), # Move diagonally down-left
    }

    # Add some rotation to make it interesting
    theta = np.pi / 6  # 30 degrees
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    def transition(state: np.ndarray, action: int) -> np.ndarray:
        """Deterministic component of transition."""
        # Apply rotation and action effect
        effect = action_effects[action]
        # Slight state-dependent rotation
        scale = 0.9 + 0.1 * np.tanh(np.linalg.norm(state))
        next_state = scale * (rotation @ state) + effect
        return next_state

    def sample_transition(state: np.ndarray, action: int) -> np.ndarray:
        """Sample from stochastic transition."""
        mean = transition(state, action)
        # State-dependent noise (heteroscedastic)
        noise_scale = noise_std * (1 + 0.5 * np.abs(state).sum())
        noise = rng.normal(0, noise_scale, size=2)
        return mean + noise

    return transition, sample_transition, action_effects


def generate_trajectories(
    sample_transition,
    n_trajectories: int = 50,
    trajectory_length: int = 20,
    n_actions: int = 3,
    rng: Optional[np.random.Generator] = None
) -> list:
    """Generate synthetic trajectories."""
    if rng is None:
        rng = np.random.default_rng(42)

    trajectories = []
    for _ in range(n_trajectories):
        # Random initial state
        state = rng.uniform(-1, 1, size=2)
        observations = [state.copy()]
        actions = []

        for _ in range(trajectory_length):
            action = rng.integers(n_actions)
            next_state = sample_transition(state, action)
            observations.append(next_state.copy())
            actions.append(action)
            state = next_state

        trajectories.append((observations, actions))

    return trajectories


def demo_basic_usage():
    """Demonstrate basic context graph usage."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Context Graph Usage")
    print("=" * 60)

    # Create context graph
    print("\n1. Creating context graph...")
    cg = ContextGraph(
        obs_dim=2,
        latent_dim=2,
        n_actions=3,
        use_identity_encoder=True,
        hidden_dims=[32, 32],
        n_ensemble=3,
        name="demo_cg"
    )
    print(f"   {cg}")

    # Create synthetic environment
    print("\n2. Creating synthetic environment...")
    _, sample_transition, _ = create_synthetic_environment()

    # Generate and record trajectories
    print("\n3. Generating and recording trajectories...")
    rng = np.random.default_rng(42)
    trajectories = generate_trajectories(sample_transition, n_trajectories=30, rng=rng)

    for i, (obs, acts) in enumerate(trajectories):
        cg.record_trajectory(
            observations=obs,
            actions=acts,
            case_id=f"traj_{i}"
        )

    print(f"   Recorded {len(cg.evidence_store)} transitions")
    print(f"   Actions: {cg.evidence_store.actions}")

    # Train
    print("\n4. Training ensemble models...")
    history = cg.train(epochs=50, batch_size=32, lr=1e-3, verbose=False)
    print(f"   Final loss: {history['loss'][-1]:.4f}")

    # Make predictions
    print("\n5. Making predictions...")
    cg.reset_belief(prior_std=0.5)

    # Update with observation
    observation = np.array([0.0, 0.0])
    cg.update_belief(observation)

    # Compare actions
    comparison = cg.compare_actions([0, 1, 2], n_samples=200)
    print("   Action comparison:")
    for action, stats in comparison.items():
        print(f"     Action {action}: mean={stats['obs_mean']}, uncertainty={stats['obs_uncertainty']:.4f}")

    # Epistemic uncertainty
    print("\n6. Epistemic uncertainty:")
    for action in range(3):
        eps_unc = cg.epistemic_uncertainty(action=action)
        print(f"   Action {action}: {eps_unc:.4f}")

    return cg


def demo_simulation():
    """Demonstrate simulation capabilities."""
    print("\n" + "=" * 60)
    print("DEMO 2: Simulation and Counterfactuals")
    print("=" * 60)

    # Create and train context graph
    cg = ContextGraph(
        obs_dim=2,
        latent_dim=2,
        n_actions=3,
        use_identity_encoder=True,
        hidden_dims=[32, 32],
        n_ensemble=3
    )

    _, sample_transition, _ = create_synthetic_environment()
    rng = np.random.default_rng(123)
    trajectories = generate_trajectories(sample_transition, n_trajectories=50, rng=rng)

    for i, (obs, acts) in enumerate(trajectories):
        cg.record_trajectory(obs, acts, case_id=f"traj_{i}")

    cg.train(epochs=30, verbose=False)

    # Simulation
    print("\n1. Simulating trajectories...")
    cg.reset_belief(prior_std=0.3)
    cg.update_belief(np.array([0.0, 0.0]))

    # Simulate under different action sequences
    for action_seq in [[0], [1], [2], [0, 1, 2]]:
        result = cg.simulate(action_seq, n_trajectories=100)
        final_mean, final_var = result.final_state_distribution()
        print(f"   Actions {action_seq}: final_mean={final_mean.round(3)}, uncertainty={final_var.sum():.4f}")

    # Counterfactual
    print("\n2. Counterfactual reasoning...")
    observation = np.array([0.5, 0.5])
    cg.update_belief(observation)

    factual, counterfactual = cg.counterfactual(
        observation=observation,
        factual_action=0,
        counterfactual_action=1,
        n_samples=200
    )

    f_mean, f_var = factual.final_state_distribution()
    cf_mean, cf_var = counterfactual.final_state_distribution()

    print(f"   Factual (action 0): mean={f_mean.round(3)}")
    print(f"   Counterfactual (action 1): mean={cf_mean.round(3)}")
    print(f"   Difference: {(cf_mean - f_mean).round(3)}")

    return cg


def demo_structural_equivalence():
    """Demonstrate structural equivalence checking."""
    print("\n" + "=" * 60)
    print("DEMO 3: Structural Equivalence")
    print("=" * 60)

    from vajra.structural.equivalence import (
        LinearEquivalence,
        CommutationChecker,
        ApproximateEquivalence,
    )
    from vajra.core.transition_kernel import DeterministicTransition
    from vajra.core.latent_state import LatentState

    # Create a system with swap symmetry
    print("\n1. Creating symmetric system...")

    # 4D state: two 2D modules [x1, y1, x2, y2]
    # Swap symmetry: (x1,y1) <-> (x2,y2)
    swap_matrix = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)

    equivalence = LinearEquivalence(swap_matrix)

    # Define symmetric transition
    def symmetric_transition(z: np.ndarray, action: int) -> np.ndarray:
        """Transition that respects the symmetry."""
        if action == 0:
            # Rotate each module
            theta = 0.1
            rot = np.array([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, np.cos(theta), -np.sin(theta)],
                [0, 0, np.sin(theta), np.cos(theta)]
            ])
            return rot @ z
        else:
            # Scale
            return 0.95 * z

    transition = DeterministicTransition(
        transition_fn=symmetric_transition,
        latent_dim=4,
        noise_std=0.0
    )

    # Check commutation
    print("\n2. Checking commutation...")
    checker = CommutationChecker(
        equivalence=equivalence,
        transition=transition,
        actions=[0, 1],
        tolerance=1e-10
    )

    # Test on sample states
    test_states = [
        LatentState(vector=np.array([1.0, 0.0, 0.5, 0.5])),
        LatentState(vector=np.array([0.0, 1.0, -0.5, 0.3])),
        LatentState(vector=np.array([0.3, -0.2, 0.1, 0.8])),
    ]

    for state in test_states:
        results = checker.check_all_actions(state)
        for action, result in results.items():
            print(f"   State {state.vector.round(2)}, Action {action}: {result}")

    # Check sequences
    print("\n3. Checking action sequences...")
    sequences = [[0, 1], [0, 0, 1], [1, 0, 0, 1, 0]]
    for seq in sequences:
        result = checker.check_sequence(test_states[0], seq)
        print(f"   Sequence {seq}: {result}")


def demo_belief_tracking():
    """Demonstrate belief state tracking."""
    print("\n" + "=" * 60)
    print("DEMO 4: Belief State Tracking")
    print("=" * 60)

    from vajra.core.belief_state import ParticleBeliefState, GaussianBeliefState
    from vajra.core.latent_state import LatentSpace

    # Create latent space
    latent_space = LatentSpace(dim=2, bounds=(np.array([-5, -5]), np.array([5, 5])))

    print("\n1. Particle filter belief...")

    # Initialize from prior
    rng = np.random.default_rng(42)
    belief = ParticleBeliefState.from_prior(latent_space, n_particles=1000, rng=rng)

    print(f"   Initial: mean={belief.mean().round(3)}, uncertainty={belief.uncertainty:.4f}")
    print(f"   ESS: {belief.effective_sample_size():.1f}")

    # Simulate updates
    from vajra.core.emission_model import GaussianEmission
    from vajra.core.transition_kernel import GaussianTransitionKernel

    emission = GaussianEmission(
        obs_fn=lambda z: z,  # Identity
        obs_dim=2,
        noise_std=0.5
    )

    transition = GaussianTransitionKernel(
        mean_fn=lambda z, a: z + np.array([0.2, 0.1]) * a,
        covariance=0.1,
        latent_dim=2
    )

    # Observe and update
    true_state = np.array([1.0, 0.5])
    observation = true_state + rng.normal(0, 0.5, size=2)

    belief = belief.update(observation, emission)
    print(f"\n   After observation {observation.round(3)}:")
    print(f"   Belief mean={belief.mean().round(3)}, uncertainty={belief.uncertainty:.4f}")

    if belief.needs_resampling():
        belief = belief.resample(rng)
        print(f"   (Resampled, ESS was low)")

    # Predict under action
    belief = belief.predict(action=1, transition=transition, rng=rng)
    print(f"\n   After action 1:")
    print(f"   Belief mean={belief.mean().round(3)}, uncertainty={belief.uncertainty:.4f}")


def demo_full_workflow():
    """Demonstrate complete workflow with visualization."""
    print("\n" + "=" * 60)
    print("DEMO 5: Complete Workflow")
    print("=" * 60)

    # 1. Create environment and context graph
    print("\n1. Setup...")
    cg = ContextGraph(
        obs_dim=2,
        latent_dim=2,
        n_actions=3,
        use_identity_encoder=True,
        n_ensemble=5
    )

    _, sample_transition, action_effects = create_synthetic_environment(noise_std=0.15)
    rng = np.random.default_rng(42)

    # 2. Collect data
    print("\n2. Collecting training data...")
    trajectories = generate_trajectories(
        sample_transition,
        n_trajectories=100,
        trajectory_length=15,
        rng=rng
    )

    for i, (obs, acts) in enumerate(trajectories):
        cg.record_trajectory(obs, acts, case_id=f"case_{i}")

    print(f"   Total transitions: {len(cg.evidence_store)}")

    # 3. Train
    print("\n3. Training...")
    history = cg.train(epochs=80, batch_size=64, verbose=False)
    print(f"   Training complete. Final loss: {history['loss'][-1]:.4f}")

    # 4. Evaluate
    print("\n4. Evaluating predictions...")

    # Generate test trajectories
    test_trajs = generate_trajectories(
        sample_transition,
        n_trajectories=20,
        trajectory_length=5,
        rng=np.random.default_rng(999)
    )

    prediction_errors = {0: [], 1: [], 2: []}
    for obs, acts in test_trajs:
        cg.reset_belief(prior_std=0.5)
        cg.update_belief(obs[0])

        for t, action in enumerate(acts[:3]):  # First 3 steps
            # Predict
            pred = cg.predict(action, n_samples=100)
            pred_mean = pred['obs_mean']

            # Actual next state
            actual = obs[t + 1]

            # Record error
            error = np.linalg.norm(pred_mean - actual)
            prediction_errors[t].append(error)

            # Update belief
            cg.update_belief(actual)
            cg.predict_belief(action)

    print("   Prediction errors by step:")
    for step, errors in prediction_errors.items():
        print(f"     Step {step}: mean={np.mean(errors):.4f}, std={np.std(errors):.4f}")

    # 5. Decision support example
    print("\n5. Decision support example...")
    cg.reset_belief(prior_std=0.5)
    cg.update_belief(np.array([0.0, 0.0]))

    print("   Current state: [0, 0]")
    print("   Comparing actions for next step:")

    comparison = cg.compare_actions(n_samples=500)
    for action, stats in comparison.items():
        print(f"     Action {action}:")
        print(f"       Expected position: {stats['obs_mean'].round(3)}")
        print(f"       Uncertainty: {stats['obs_uncertainty']:.4f}")
        print(f"       Epistemic uncertainty: {cg.epistemic_uncertainty(action=action):.4f}")

    print("\n" + "=" * 60)
    print(cg.summary())

    return cg


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("VAJRA CONTEXT GRAPH SYSTEM - DEMO")
    print("=" * 60)
    print("\nThis demo showcases the key capabilities of the")
    print("Context Graph System for POMDP-based dynamics modeling.\n")

    # Run demos
    demo_basic_usage()
    demo_simulation()
    demo_structural_equivalence()
    demo_belief_tracking()
    cg = demo_full_workflow()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nThe Context Graph System provides:")
    print("  - Append-only evidence storage")
    print("  - Action-conditioned dynamics learning")
    print("  - Heteroscedastic uncertainty modeling")
    print("  - Bayesian posterior over dynamics (ensembles)")
    print("  - Belief state tracking")
    print("  - Monte Carlo simulation")
    print("  - Counterfactual reasoning")
    print("  - Structural equivalence checking")
    print("\nSee the documentation for more details.")

    return cg


if __name__ == "__main__":
    main()
