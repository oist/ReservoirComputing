# RC 

## Install

```bash
uv sync or pip install requirements.txt or pip install -e .
```

## Usage

```python
from rc import ESN, ESNConfig

config = ESNConfig(
    N=500,                      # reservoir size
    input_dim=3,                # input dimensionality
    spectral_radius=0.95,       # reservoir weight scaling
    alpha=1e-6,                 # ridge regularization
    sparsity=0.9,               # fraction of zero weights
    input_scaling=0.5,          # input weight scaling
    bias_scaling=0.1,           # bias scaling
    seed=42,                    # random seed
    mode="leaky",               # "standard", "leaky", "leakyrand", "es2n", "es2nrand"
    leaky_rate=0.3,             # leak rate (leaky mode)
    beta=0.5,                   # mixing param (es2n mode)
    scale=0.1,                  # scale parameter (es2nrand and leakyrand mode); range = leaky_rate/beta ± scale
    weights_generation_strategy="Gaussian", # "Gaussian", "Uniform", "Bernoulli", "Small-World", "Scale-Free"
    bias_generation_strategy="Uniform", # "Gaussian", "Uniform", "Bernoulli"
    input_generation_strategy="Uniform", # "Gaussian", "Uniform", "Bernoulli"
    self_connections=False,     # allow self-connections
    dtype=np.float64,           # data type
)

esn = ESN(config)
esn.train(data, washout=500)
predictions, states = esn.predict(warmup, steps=1000)

# save/load
esn.save("model.npz")
esn = ESN.load("model.npz")
```

## Hyperparameter Optimization

```python
from rc import ESNSearchSpaceBuilder, EvaluationConfig, optimize_esn

space = (ESNSearchSpaceBuilder()
    .optimize("spectral_radius")
    .optimize("alpha")
    .fix(N=500, mode="leaky")
    .build())

config = EvaluationConfig(
    washout=2000,               # initial transient to discard
    warmup_steps=1500,          # steps to drive reservoir before prediction
    predict_steps=6500,         # autonomous prediction length
    n_predictions=2,            # predictions per instance
    n_instances=6,              # ESN instances (different seeds)
    n_jobs=-1,                  # parallel jobs (-1 = all cores)
    metrics=['wasserstein'],    # ['wasserstein', 'vpt']
    wasserstein_projections=100,# sliced Wasserstein projections
    vpt_threshold=0.4,          # divergence threshold for VPT
    clip_wasserstein=1.0,       # clip large wasserstein values
    dt=0.01,                    # time step for VPT
    constrain_cle=True,         # reject unstable ESNs
    cle_threshold=-2.0,         # max conditional Lyapunov exponent
)

best_params, ax = optimize_esn(data, space, config, n_trials=30)

# use best params
final_config = space.build_config(best_params, input_dim=data.shape[0])
esn = ESN(final_config)
```

## Examples

`examples/` — Lorenz, double pendulum, C. elegans

