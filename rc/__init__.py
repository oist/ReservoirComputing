"""
Reservoir Computing Library
===========================

A Python library for Echo State Networks (ESN) and reservoir computing.

Quick Start
-----------
>>> from rc import ESN
>>> esn = ESN(N=500, input_dim=3, spectral_radius=0.95)
>>> esn.train(data, washout=500)
>>> predictions, states = esn.predict(warmup, steps=1000)

Core Classes
------------
ESN : Echo State Network model
ESNConfig : Configuration dataclass for ESN hyperparameters

Dynamics
--------
StandardDynamics : Standard ESN dynamics (tanh activation)
LeakyDynamics : Leaky integrator ESN dynamics  
ES2NDynamics : ES2N dynamics with orthogonal mixing
create_dynamics : Factory function for creating dynamics

Optimization
------------
ESNSearchSpaceBuilder : Builder for hyperparameter search spaces
optimize_esn : Main optimization function
EvaluationConfig : Evaluation configuration
"""

from rc.esn import (
    ESN,
    ESNConfig,
    StandardDynamics,
    LeakyDynamics,
    ES2NDynamics,
    create_dynamics,
    participation_ratio,
    analyse_dynamics,
    logger,
)
from rc.metrics import valid_prediction_time
from rc.optimisation import (
    ESNSearchSpace,
    ESNSearchSpaceBuilder,
    ESNObjective,
    EvaluationConfig,
    SearchParam,
    optimize_esn,
    calculate_max_conditional_lyapunov_exponent,
)

__version__ = "0.1.0"

__all__ = [
    "ESN",
    "ESNConfig",
    "StandardDynamics",
    "LeakyDynamics",
    "ES2NDynamics",
    "create_dynamics",
    "ESNSearchSpace",
    "ESNSearchSpaceBuilder",
    "ESNObjective",
    "EvaluationConfig",
    "SearchParam",
    "optimize_esn",
    "calculate_max_conditional_lyapunov_exponent",
    "participation_ratio",
    "analyse_dynamics",
    "valid_prediction_time",
    "logger",
    "__version__",
]
