from dataclasses import dataclass, field
from itertools import product
from typing import Any, ClassVar, Literal
import numpy as np
from joblib import Parallel, delayed
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import ot
from numpy.typing import NDArray
from esn import ESN, ESNConfig, logger
from metrics import valid_prediction_time
from ax.service.utils.instantiation import ObjectiveProperties

def calculate_max_conditional_lyapunov_exponent(esn: ESN, data: NDArray) -> float:
    """calculate the max conditional Lyapunov exponent of the ESN.
    
    Parameters
    ----------
    esn : ESN
    data : NDArray
        Data.
        
    Returns
    -------
    float
        Max conditional Lyapunov exponent.
    """
    return esn.conditional_lyapunov_spectrum(data[:,:10000], num_lyaps=1, norm_time=5, dt=1/16)["exponents"][0]
@dataclass
class SearchParam:
    """search parameter for ESN hyperparameters.
    
    Attributes
    ----------
    bounds : tuple[float, float]
        Bounds for the parameter.
    log_scale : bool, default=False
        Whether to log-scale the parameter.
    param_type : str, default="float"
        Type of the parameter. 
        
    Examples
    --------
    >>> spectral_radius = SearchParam(bounds=(0.5, 1.2))
    >>> alpha = SearchParam(bounds=(1e-8, 1e-2), log_scale=True)
    """
    bounds: tuple  
    log_scale: bool = False
    param_type: Literal["float", "categorical", "integer"] = "float"


@dataclass
class ESNSearchSpace:
    """search space for ESN hyperparameters.
    
    Attributes
    ----------
    optimize : dict[str, SearchParam]
        Optimizable parameters.
    fixed : dict[str, Any]
        Fixed parameters.
    _valid_optimizable : set[str]
        Valid optimizable parameters.
    _valid_fixed : set[str]
        Valid fixed parameters.
        
    Examples
    --------
    Create directly (prefer using ESNSearchSpaceBuilder instead):
    
    >>> space = ESNSearchSpace(
    ...     optimize={"spectral_radius": SearchParam((0.5, 1.2))},
    ...     fixed={"N": 500, "mode": "leaky"}
    ... )
    
    Use the builder for a cleaner API:
    
    >>> space = (ESNSearchSpaceBuilder()
    ...     .optimize("spectral_radius")
    ...     .optimize("alpha")
    ...     .fix(N=500, mode="leaky")
    ...     .build())
    """
    optimize: dict[str, SearchParam]
    fixed: dict[str, Any]
    
    _valid_optimizable: ClassVar[set[str]] = {"N", "spectral_radius", "alpha", "input_scaling", "sparsity", "bias_scaling", "leaky_rate", "beta", "scale", "weights_generation_strategy", "bias_generation_strategy", "input_generation_strategy", "self_connections"}
    _valid_fixed: ClassVar[set[str]] = _valid_optimizable | {"mode", "seed", "input_dim"}
    
    def __post_init__(self):
        unknown_opt = set(self.optimize) - self._valid_optimizable
        if unknown_opt: raise ValueError(f"unknown optimize: {unknown_opt}")
        unknown_fix = set(self.fixed) - self._valid_fixed
        if unknown_fix: raise ValueError(f"unknown fixed: {unknown_fix}")
        overlap = set(self.optimize) & set(self.fixed)
        if overlap: raise ValueError(f"overlap: {overlap}")
    
    def to_ax_params(self) -> list[dict]:
        """convert search space to Ax parameters.
        
        Returns
        -------
        list[dict]
            Ax parameters.
        """
        params = []
        for k, v in self.optimize.items():
            if v.param_type == "categorical": params.append({"name": k, "type": "choice", "values": list(v.bounds), "is_ordered": False, "sort_values": False })
            else: params.append({"name": k, "type": "range", "value_type": v.param_type, "bounds": list(v.bounds), "log_scale": v.log_scale})
        return params

    def build_config(self, sampled: dict[str, Any], input_dim: int) -> ESNConfig:
        """build ESN configuration from sampled parameters.
        
        Parameters
        ----------
        sampled : dict[str, Any]
            Sampled parameters.
        input_dim : int
            Input dimension.
            
        Returns
        -------
        ESNConfig
            ESN configuration.
        """
        return ESNConfig(**{**self.fixed, **sampled, "input_dim": input_dim})
    

    def create_ax_client(
        self, 
        objective_name: str = "wasserstein", 
        minimize: bool = True, 
        seed: int | None = None,
        constrain_cle: bool = False,
        cle_threshold: float = 0.0,
    ) -> AxClient:
        """Create Ax client for optimization.
        
        Parameters
        ----------
        objective_name : str, default="wasserstein"
            Name of the objective function.
        minimize : bool, default=True
            Whether to minimize the objective function.
        seed : int | None, default=None
            Random seed for reproducibility.
        constrain_cle : bool, default=False
            Whether to constrain max CLE < threshold.
        cle_threshold : float, default=0.0
            Upper bound for max CLE (typically 0 for echo state property).
            
        Returns
        -------
        AxClient
            Ax client.
        """
        ax = AxClient(random_seed=seed, verbose_logging=False)
        
        # Build outcome constraints
        outcome_constraints = []
        if constrain_cle:
            # Format: "metric_name <= bound"
            outcome_constraints.append(f"max_cle <= {cle_threshold}")
        
        ax.create_experiment(
            parameters=self.to_ax_params(), 
            objectives={objective_name: ObjectiveProperties(minimize=minimize)},
            outcome_constraints=outcome_constraints,
        )
        
        logger.debug(f"Created Ax client with constraints: {outcome_constraints}")
        return ax

class ESNSearchSpaceBuilder:
    """builder for ESN search space.
    
    Attributes
    ----------
    DEFAULTS : dict[str, SearchParam]
        default search parameters.
        
    Examples
    --------
    Build a search space with default bounds:
    
    >>> space = (ESNSearchSpaceBuilder()
    ...     .optimize("spectral_radius")
    ...     .optimize("alpha")
    ...     .fix(N=500, mode="standard")
    ...     .build())
    
    Override default bounds:
    
    >>> space = (ESNSearchSpaceBuilder()
    ...     .optimize("spectral_radius", bounds=(0.8, 1.5))
    ...     .optimize("input_scaling", bounds=(0.5, 3.0))
    ...     .fix(N=1000, mode="leaky", leaky_rate=0.3)
    ...     .build())
    """
    DEFAULTS: ClassVar[dict[str, SearchParam]] = {
        "spectral_radius": SearchParam((0.5, 1.2)),
        "alpha": SearchParam((1e-8, 1e-2), log_scale=True),
        "input_scaling": SearchParam((0.1, 2.0)),
        "bias_scaling": SearchParam((0.0, 1.0)),
        "leaky_rate": SearchParam((0.05, 0.5)),
        "beta": SearchParam((0.1, 0.9)),
        "sparsity": SearchParam((0.9, 0.995)),
        "scale": SearchParam((0.01, 0.3)),
        "weights_generation_strategy": SearchParam(("Gaussian", "Uniform","Bernoulli", "Small-World", "Scale-Free"), param_type="categorical"),
        "bias_generation_strategy": SearchParam(("Gaussian", "Uniform", "Bernoulli"), param_type="categorical"),
        "input_generation_strategy": SearchParam(("Gaussian", "Uniform", "Bernoulli"), param_type="categorical"),
        "self_connections": SearchParam((True, False), param_type="categorical"),
    }

    def __init__(self):
        self._optimize: dict[str, SearchParam] = {}
        self._fixed: dict[str, Any] = {}

    def optimize(self, name: str, bounds: tuple | None = None, log_scale: bool | None = None) -> "ESNSearchSpaceBuilder":
        """add parameter to optimization.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        bounds : tuple[float, float], optional
            Bounds for the parameter.
        log_scale : bool, optional
            Whether to log-scale the parameter.
            
        Returns
        -------
        ESNSearchSpaceBuilder
            Builder instance.
        """
        if name not in self.DEFAULTS and bounds is None: raise ValueError(f"no default for {name}")
    
        if bounds is None:
            base = self.DEFAULTS[name]
            param = SearchParam(
                bounds=base.bounds,
                log_scale=log_scale if log_scale is not None else base.log_scale,
                param_type=base.param_type
            )
        else:
            base = self.DEFAULTS.get(name)
            param_type = base.param_type if base else "float"
            param = SearchParam(bounds, log_scale or False, param_type)
        
        self._optimize[name] = param
        return self



    def fix(self, **kwargs: Any) -> "ESNSearchSpaceBuilder":
        """fix parameters to specific values.
        
        Parameters
        ----------
        **kwargs : dict[str, Any]
            Parameters to fix.
            
        Returns
        -------
        ESNSearchSpaceBuilder
            Builder instance.
        """
        self._fixed.update(kwargs)
        return self

    def build(self) -> ESNSearchSpace:
        """build search space.
        
        Returns
        -------
        ESNSearchSpace
            Search space.
        """
        if not self._optimize: raise ValueError("nothing to optimize")
        logger.debug(f"Built search space with {len(self._optimize)} optimizable parameters and {len(self._fixed)} fixed parameters")
        return ESNSearchSpace(optimize=self._optimize.copy(), fixed=self._fixed.copy())


@dataclass
class EvaluationConfig:
    """evaluation configuration.
    
    Attributes
    ----------
    washout : int, default=5000
        Washout steps.
    warmup_steps : int, default=1000
        Warmup steps.
    predict_steps : int, default=4500
        Prediction steps.
    n_predictions : int, default=8
        Number of predictions.
    n_instances : int, default=5
        Number of instances.
    n_jobs : int, default=-1
        Number of jobs.
    metrics : list[str], default=['wasserstein']
        Metrics to evaluate.
    wasserstein_projections : int, default=100
        Number of projections for Wasserstein distance.
    vpt_threshold : float, default=0.4
        Threshold for valid prediction time.
    dt : float, default=0.01
        Time step.
        
    Examples
    --------
    Default configuration:
    
    >>> config = EvaluationConfig()
    
    Custom configuration with both metrics:
    
    >>> config = EvaluationConfig(
    ...     washout=3000,
    ...     predict_steps=5000,
    ...     n_predictions=10,
    ...     metrics=['wasserstein', 'vpt']
    ... )
    """
    washout: int = 5000
    warmup_steps: int = 1000
    predict_steps: int = 4500
    n_predictions: int = 8
    n_instances: int = 5
    n_jobs: int = -1
    metrics: list[str] = field(default_factory=lambda: ['wasserstein'])
    wasserstein_projections: int = 100
    vpt_threshold: float = 0.4
    clip_wasserstein: float = 3.0
    dt: float = 0.01
    
    # cle constraint
    constrain_cle: bool = False
    cle_threshold: float = 0.0


class ESNObjective:
    """ESN objective function.
    
    Attributes
    ----------
    data : ndarray of shape (T, D)
        Data.
    space : ESNSearchSpace
        Search space.
    config : EvaluationConfig, default=EvaluationConfig()
        Evaluation configuration.
        
    Examples
    --------
    >>> space = (ESNSearchSpaceBuilder()
    ...     .optimize("spectral_radius")
    ...     .fix(N=500)
    ...     .build())
    >>> objective = ESNObjective(data, space)
    >>> result = objective({"spectral_radius": 0.9})
    >>> print(result)  # {'wasserstein': (median, std)}
    """
    def __init__(self, data: NDArray, space: ESNSearchSpace, config: EvaluationConfig = EvaluationConfig()):
        self.config = config
        self.space = space
        self.data = data.T if data.shape[0] > data.shape[1] else data
        self.D, self.T = self.data.shape
        
        max_start = self.T - config.warmup_steps - config.predict_steps
        n_total = config.n_predictions * config.n_instances
        self.starts = np.random.randint(config.washout, max_start, size=n_total)
    
    def _eval_instance(self, params: dict, idx: int) -> dict:
        """evaluate ESN on a single instance.
        
        Parameters
        ----------
        params : dict
            Parameters.
        idx : int
            Instance index.
            
        Returns
        -------
        dict[str, float]
            Results with median values for each metric.
        """
        cfg = self.space.build_config(params, self.D)
        cfg.seed = idx
        
        esn = ESN(cfg)
        esn.train(self.data, washout=self.config.washout)
        
        results = {m: [] for m in self.config.metrics}
        if 'max_cle' in self.config.metrics or self.config.constrain_cle:
            try:
                max_cle = calculate_max_conditional_lyapunov_exponent(esn, self.data)
                results['max_cle'] = [max_cle]  # Keep as list for consistency
            except Exception as e:
                logger.warning(f"CLE computation failed: {e}")
                results['max_cle'] = [1.0]  # Penalty value (violates constraint)       
                
        for j in range(self.config.n_predictions):
            start = self.starts[idx * self.config.n_predictions + j]
            warmup = self.data[:, start:start + self.config.warmup_steps]
            pred, _ = esn.predict(warmup, self.config.predict_steps)
            gt = self.data[:, start + self.config.warmup_steps:start + self.config.warmup_steps + self.config.predict_steps]
            
            if 'wasserstein' in results:
                sw = ot.max_sliced_wasserstein_distance(gt.T, pred.T, n_projections=self.config.wasserstein_projections)
                results['wasserstein'].append(np.clip(sw, 0, self.config.clip_wasserstein))
            
            if 'vpt' in results:
                results['vpt'].append(valid_prediction_time(gt, pred, self.config.vpt_threshold, self.config.dt))
            
        logger.debug(f"Evaluated ESN with parameters {params} on instance {idx} with results {results}")
        return {k: np.median(v) for k, v in results.items() if v}
    
    def __call__(self, params: dict) -> dict:
        """evaluate ESN on multiple instances.
        
        Parameters
        ----------
        params : dict
            Parameters.
            
        Returns
        -------
        dict[str, tuple[float, float]]
            Results with median and standard deviation for each metric.
        """
        results = Parallel(n_jobs=self.config.n_jobs)(delayed(self._eval_instance)(params, i) for i in range(self.config.n_instances))
        
        valid = [r for r in results if r]
        if not valid:
            logger.warning(f"No valid results for {params}")
            output = {k: (10.0, 0.0) for k in self.config.metrics}
            if self.config.constrain_cle:
                output['max_cle'] = (1.0, 0.0)  
            return output
        output = {}
        for k in valid[0]:
            values = [r[k] for r in valid]
            mean = np.median(values)
            sem = np.std(values) / np.sqrt(len(values))
            output[k] = (mean, sem)
        
        return output
    

def _generate_grid(space: ESNSearchSpace, points: dict[str, int | list[float]] | int) -> list[dict]:
    """generate parameter grid from search space.
    
    Parameters
    ----------
    space : ESNSearchSpace
        Search space.
    points : dict or int
        Grid resolution.
        
    Returns
    -------
    list[dict]
        Parameter grid.
    """
    if isinstance(points, int): points = {k: points for k in space.optimize}
    
    param_values = {}
    for name, search_param in space.optimize.items():
        spec = points.get(name, 5)
        if isinstance(spec, list): param_values[name] = spec
        elif search_param.param_type == "categorical": param_values[name] = list(search_param.bounds)
        else:
            low, high = search_param.bounds
            if search_param.log_scale: param_values[name] = np.geomspace(low, high, spec).tolist()
            else: param_values[name] = np.linspace(low, high, spec).tolist()
    
    keys = list(param_values.keys())
    return [dict(zip(keys, combo)) for combo in product(*param_values.values())]

def optimize_esn(
    data: NDArray,
    space: ESNSearchSpace,
    config: EvaluationConfig,
    n_trials: int = 30,
    method: Literal["bayesian", "grid"] = "bayesian",
    grid_points: dict[str, int | list[float]] | int = 5,
) -> tuple[dict, AxClient]:
    """Optimize ESN hyperparameters.
    
    Parameters
    ----------
    data : NDArray of shape (T, D)
        Data.
    space : ESNSearchSpace
        Search space.
    config : EvaluationConfig
        Evaluation configuration.
    n_trials : int, default=30
        Number of trials (Bayesian only).
    method : {"bayesian", "grid"}, default="bayesian"
        Optimization method.
    grid_points : dict or int, default=5
        Grid resolution (grid only).
    
    Returns
    -------
    best_params : dict
        Best parameters found.
    ax : AxClient
        Ax client with all trials.
        
    Examples
    --------
    Bayesian optimization:
    
    >>> space = (ESNSearchSpaceBuilder()
    ...     .optimize("spectral_radius")
    ...     .optimize("alpha")
    ...     .fix(N=500, mode="standard")
    ...     .build())
    >>> config = EvaluationConfig(n_predictions=5, n_instances=3)
    >>> best_params, ax = optimize_esn(data, space, config, n_trials=30)
    
    Grid search:
    
    >>> best_params, ax = optimize_esn(
    ...     data, space, config,
    ...     method="grid",
    ...     grid_points={"spectral_radius": 10, "alpha": 5}
    ... )
    """
    objective = ESNObjective(data, space, config)
    ax = space.create_ax_client(
        constrain_cle=config.constrain_cle,
        cle_threshold=config.cle_threshold,
    )
    if method == "grid":
        grid = _generate_grid(space, grid_points)
        logger.debug(f"Grid search: {len(grid)} combinations")
        
        best_val = float('inf')
        for i, params in enumerate(grid):
            _, idx = ax.attach_trial(params)
            result = objective(params)
            ax.complete_trial(idx, raw_data=result)
            
            val = result["wasserstein"][0]
            if val < best_val:
                best_val = val
            
            if (i + 1) % 10 == 0 or i == len(grid) - 1:
                logger.debug(f"[{i+1}/{len(grid)}] best={best_val:.4f}")
    
    else:  # bayesian
        best_val = float('inf')
        for i in range(n_trials):
            params, idx = ax.get_next_trial()
            print(params)
            result = objective(params)
            print(result)
            ax.complete_trial(idx, raw_data=result)
            
            val = result["wasserstein"][0]
            if val < best_val:
                best_val = val
            logger.debug(f"trial {i+1}: {val:.4f} (best={best_val:.4f})")
    
    best_params, _ = ax.get_best_parameters()
    return best_params, ax
