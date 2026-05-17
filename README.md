# pyreservoir

[![PyPI](https://img.shields.io/pypi/v/pyreservoir.svg)](https://pypi.org/project/pyreservoir/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-informational)](https://oist.github.io/ReservoirComputing/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/managed%20by-uv-de5fe9)](https://github.com/astral-sh/uv)

Echo State Network library for reservoir computing.

> Installed and imported:
>
> ```bash
> pip install pyreservoir
> ```
>
> ```python
> from rc import ESN
> ```

## Why pyreservoir?

`pyreservoir` is built for **reconstructing ergodic properties from data**:

- **Five reservoir variants with tunable memory.** Choose among
  `standard` (memoryless), `leaky` / `leakyrand` (leaky integrator with
  uniform or per-neuron leak rate), and `es2n` / `es2nrand` (ES²N with
  orthogonal mixing).

- **Sliced Wasserstein loss for hyperparameter tuning.**

- **Echo State Property as a hard constraint.**

- **Conditional and autonomous Lyapunov spectra.**

- **Bayesian optimization with Ax, including outcome constraints.**

- **Five weight-initialization strategies.**

## Install

From PyPI:

```bash
pip install pyreservoir
```

From source with uv:

```bash
git clone https://github.com/oist/ReservoirComputing.git
cd ReservoirComputing
uv sync                 # install runtime deps
uv sync --extra dev     # add jupyter/matplotlib/plotly for the notebooks
```

Or dev install with pip:

```bash
pip install -e .
```

## Quick Start

### Training

```python
from rc import ESN

esn = ESN(N=500, input_dim=3, spectral_radius=0.95)
esn.train(data, washout=500)  # data: (input_dim, T)
predictions, states = esn.predict(warmup, steps=1000)
```

### Optimization

```python
from rc import ESNSearchSpaceBuilder, EvaluationConfig, optimize_esn

space = ESNSearchSpaceBuilder().optimize("spectral_radius").optimize("alpha").fix(N=500).build()
config = EvaluationConfig(washout=2000, predict_steps=5000, n_predictions=4)
best_params, best_observed, ax, comparison = optimize_esn(data, space, config, n_trials=20)
final_config = space.build_config(best_params, input_dim=data.shape[0])
esn = ESN(final_config)
esn.train(data, washout=2000)
predictions, states = esn.predict(warmup, steps=1000)
```

## Full Configuration

```python
from rc import ESN, ESNConfig
import numpy as np

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
    scale=0.1,                  # randomness scale (leakyrand/es2nrand)
    weights_generation_strategy="Gaussian",  # "Gaussian", "Uniform", "Bernoulli", "Small-World", "Scale-Free"
    bias_generation_strategy="Uniform",      # "Gaussian", "Uniform", "Bernoulli"
    input_generation_strategy="Uniform",     # "Gaussian", "Uniform", "Bernoulli"
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

## Full Optimization Configuration

```python
from rc import ESNSearchSpaceBuilder, EvaluationConfig, optimize_esn

space = (ESNSearchSpaceBuilder()
    .optimize("spectral_radius")
    .optimize("alpha")
    .optimize("input_scaling")
    .fix(N=500, mode="leaky", leaky_rate=0.3)
    .build())

config = EvaluationConfig(
    washout=2000,               # initial transient to discard
    warmup_steps=1500,          # steps to drive reservoir before prediction
    predict_steps=6500,         # autonomous prediction length
    n_predictions=4,            # predictions per instance
    n_instances=5,              # ESN instances (different seeds)
    n_jobs=-1,                  # parallel jobs (-1 = all cores)
    metrics=['wasserstein'],    # ['wasserstein', 'vpt', 'max_cle']
    wasserstein_projections=100,# sliced Wasserstein projections
    vpt_threshold=0.4,          # divergence threshold for VPT
    dt=0.01,                    # time step for VPT
    constrain_cle=True,         # reject unstable ESNs
    cle_threshold=0.0,          # max conditional Lyapunov exponent
)

best_params, best_observed, ax, comparison = optimize_esn(data, space, config, n_trials=30)

# use best params
final_config = space.build_config(best_params, input_dim=data.shape[0])
esn = ESN(final_config)
```

## Examples

See `examples/` for notebooks: Lorenz attractor, double pendulum, C. elegans dynamics.
See 'figure/' for notebooks and scripts that reproduce the figures from the paper.

## Folder Structure

- `rc/`: core library code
  -esn.py: ESN implementation
  -optimization.py: hyperparameter optimization code
  -metrics.py: evaluation metrics
  -dynamics.py: ESN dynamics
  -analysis.py: tools for analyzing ESN behavior
- `examples/`: Jupyter notebooks demonstrating usage
- `figure/`: code to reproduce paper figures

## Citation

If you use this library in academic work, please cite the accompanying paper:

```bibtex
@article{kawano2026optimizing,
  title   = {Optimizing Reservoir Computing for Reconstructing Ergodic Properties},
  author  = {Kawano, Akira and Soroka, Ilia and Stephens, Greg J},
  journal = {arXiv preprint arXiv:2605.01439},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

## Contact

Bugs, feature requests, and questions: please open an issue at
[github.com/oist/ReservoirComputing/issues](https://github.com/oist/ReservoirComputing/issues).
