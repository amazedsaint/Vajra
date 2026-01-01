# Vajra

**Context Graph System for POMDP-based Organizational Dynamics**

Vajra is a first-principles implementation of context graphs as append-only evidence substrates that constrain latent transition operators in partially observed decision processes (POMDPs).

## Overview

This system provides:

- **POMDP Foundation**: Models organizational behavior as partially observed controlled dynamical systems with latent states, observable records, and actions
- **Append-Only Evidence Store**: Accumulates transition triples `(x(t), a(t), x(t+1))` without overwriting, preserving information for posterior inference
- **Action-Conditioned Dynamics**: Learns heteroscedastic transition kernels `K_a(z'|z)` that capture both mean predictions and conditional uncertainty
- **Bayesian Operator Inference**: Maintains uncertainty over dynamics parameters using ensemble methods
- **Structural Equivalence**: Defines and tests operational equivalence via commutation relations `U∘T_a = T_a∘U`
- **Counterfactual Simulation**: Answers "what if" questions by propagating uncertainty through learned dynamics

## Installation

```bash
# Clone the repository
git clone https://github.com/amazedsaint/Vajra.git
cd Vajra

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
from vajra import ContextGraph
import numpy as np

# Create context graph
cg = ContextGraph(
    obs_dim=4,           # Observation dimensionality
    latent_dim=4,        # Latent space dimensionality
    n_actions=3,         # Number of discrete actions
    n_ensemble=5,        # Ensemble size for uncertainty
)

# Record transition evidence
cg.record_transition(
    x_current=np.array([1.0, 2.0, 3.0, 4.0]),
    action=0,
    x_next=np.array([1.1, 2.1, 3.0, 4.1]),
    case_id="case_001"
)

# Record complete trajectories
cg.record_trajectory(
    observations=[obs_0, obs_1, obs_2, obs_3],
    actions=[action_0, action_1, action_2],
    case_id="case_002"
)

# Train on accumulated evidence
history = cg.train(epochs=100, verbose=True)

# Make predictions
cg.update_belief(current_observation)
prediction = cg.predict(action=1, n_samples=1000)
print(f"Predicted mean: {prediction['obs_mean']}")
print(f"Uncertainty: {prediction['obs_std']}")

# Compare actions
comparison = cg.compare_actions([0, 1, 2])
for action, stats in comparison.items():
    print(f"Action {action}: uncertainty={stats['obs_uncertainty']:.4f}")

# Counterfactual reasoning
factual, counterfactual = cg.counterfactual(
    observation=current_obs,
    factual_action=0,
    counterfactual_action=1
)
```

## Key Concepts

### 1. Latent State Dynamics

The system models hidden organizational state `z(t)` evolving according to:

```
z(t+1) ~ K_{a(t)}(·|z(t))    [transition]
x(t) ~ E(·|z(t))             [emission]
```

where `K_a` is an action-conditioned transition kernel and `E` is an emission model.

### 2. Evidence Accumulation

Each executed action generates evidence stored as transition triples:

```
e_t = (x(t), a(t), x(t+1))     [observable]
ẽ_t = (z(t), a(t), z(t+1))     [latent, via encoder]
```

The store is **append-only** - overwriting destroys evidence and weakens posterior inference.

### 3. Heteroscedastic Uncertainty

Transition kernels model both mean and variance:

```
z' ~ N(μ_a(z), Σ_a(z))
```

This enables proper uncertainty quantification beyond point estimates.

### 4. Bayesian Posterior

Uncertainty over dynamics parameters is maintained via ensembles:

```
p(z'|z, a, D) ≈ (1/M) Σ_m K_{a,θ^(m)}(z'|z)
```

where `M` models are trained with different initializations.

## Running Demos

```bash
# Run the demonstration
python -m vajra.demo

# Run validation experiments from the paper
python -m vajra.validation
```

## Validation Experiments

The system includes reproducible experiments from the theoretical paper:

1. **Lumpability Construction** (Section 3): Demonstrates non-identifiability of latent dynamics from observables
2. **Structural Equivalence** (Section 5): Validates commutation over action sequences
3. **Reversibility Analysis** (Section 6): Shows conditional entropy for bijective vs collapsing maps
4. **Heteroscedastic Modeling** (Section 7): Compares NLL vs MSE for learning conditional uncertainty

## Architecture

```
vajra/
├── core/                   # Core POMDP components
│   ├── latent_state.py    # Latent state representation
│   ├── transition_kernel.py # Action-conditioned transitions
│   ├── emission_model.py   # Observation models
│   └── belief_state.py     # Belief representations
├── evidence/               # Evidence storage
│   └── store.py           # Append-only evidence store
├── inference/              # Inference components
│   ├── encoder.py         # Observation encoders
│   └── posterior.py       # Posterior approximations
├── structural/             # Structural equivalence
│   └── equivalence.py     # Commutation checking
├── simulation/             # Simulation and counterfactuals
│   └── simulator.py       # Monte Carlo simulation
├── validation/             # Paper experiments
│   └── experiments.py     # Reproducible validations
├── context_graph.py        # Main interface
└── demo.py                # Demonstration script
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT

## Citation

If you use this work, please cite the theoretical paper:

```
Context Graphs as Evidence Constrained Latent Dynamics
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
