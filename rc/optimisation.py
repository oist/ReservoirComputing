from dataclasses import dataclass, field, fields
from itertools import product
from typing import Any, ClassVar, Literal
import numpy as np
from joblib import Parallel, delayed
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from numpy.typing import NDArray
from scipy.sparse.linalg import ArpackNoConvergence
from rc.esn import ESN, ESNConfig, logger
from rc.metrics import valid_prediction_time, calculate_max_conditional_lyapunov_exponent
import time


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
    >>> from rc.optimisation import SearchParam
    >>> spectral_radius = SearchParam(bounds=(0.5, 1.2))
    >>> alpha = SearchParam(bounds=(1e-8, 1e-2), log_scale=True)
    """
    bounds: tuple  
    log_scale: bool = False
    param_type: Literal["float", "categorical", "integer"] = "float"
    
    def __post_init__(self):
        # validate bounds
        if not isinstance(self.bounds, tuple):
            raise ValueError(f"bounds must be a tuple, got {type(self.bounds).__name__}")
        
        if self.param_type == "categorical":
            if len(self.bounds) < 2:
                raise ValueError(f"categorical bounds must have at least 2 choices, got {len(self.bounds)}")
        else:
            if len(self.bounds) != 2:
                raise ValueError(f"bounds must be a tuple of 2 values (low, high), got {len(self.bounds)} values")
            low, high = self.bounds
            if low >= high:
                raise ValueError(f"bounds low ({low}) must be less than high ({high})")
            if self.log_scale and low <= 0:
                raise ValueError(f"log_scale requires positive bounds, got low={low}")
        
        # validate param_type
        valid_types = {"float", "categorical", "integer"}
        if self.param_type not in valid_types:
            raise ValueError(f"param_type must be one of {valid_types}, got '{self.param_type}'")


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
    
    >>> from rc.optimisation import ESNSearchSpace, SearchParam
    >>> space = ESNSearchSpace(
    ...     optimize={"spectral_radius": SearchParam((0.5, 1.2))},
    ...     fixed={"N": 500, "mode": "leaky"}
    ... )
    
    Use the builder for a cleaner API:
    
    >>> from rc.optimisation import ESNSearchSpaceBuilder
    >>> space = (ESNSearchSpaceBuilder()
    ...     .optimize("spectral_radius")
    ...     .optimize("alpha")
    ...     .fix(N=500, mode="leaky")
    ...     .build())
    """
    optimize: dict[str, SearchParam]
    fixed: dict[str, Any]
    
    _valid_optimizable = {f.name for f in fields(ESNConfig)}
    _valid_fixed = _valid_optimizable | {"mode", "seed", "input_dim"}
    
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
        num_trials: int | None = None,
        num_initialization_trials: int | None = None,
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
        
        outcome_constraints = []
        if constrain_cle:
            outcome_constraints.append(f"max_cle <= {cle_threshold}")
        
        ax.create_experiment(
            parameters=self.to_ax_params(), 
            objectives={objective_name: ObjectiveProperties(minimize=minimize)},
            outcome_constraints=outcome_constraints,
            choose_generation_strategy_kwargs={"num_trials": num_trials, "num_initialization_trials": num_initialization_trials, "min_sobol_trials_observed": int(num_initialization_trials / 2) if num_initialization_trials else None}
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
    
    >>> from rc.optimisation import ESNSearchSpaceBuilder
    >>> space = (ESNSearchSpaceBuilder()
    ...     .optimize("spectral_radius")
    ...     .optimize("alpha")
    ...     .fix(N=500, mode="standard")
    ...     .build())
    
    Override default bounds:
    
    >>> from rc.optimisation import ESNSearchSpaceBuilder
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
    
    >>> from rc.optimisation import EvaluationConfig
    >>> config = EvaluationConfig()
    
    Custom configuration with both metrics:
    
    >>> from rc.optimisation import EvaluationConfig
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
    length: int = 10000
    compare_metrics: list[str] | None = None
    def __post_init__(self):
        # validate integer parameters
        if not isinstance(self.washout, int) or self.washout < 0:
            raise ValueError(f"washout must be a non-negative integer, got {self.washout}")
        if not isinstance(self.warmup_steps, int) or self.warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be a positive integer, got {self.warmup_steps}")
        if not isinstance(self.predict_steps, int) or self.predict_steps <= 0:
            raise ValueError(f"predict_steps must be a positive integer, got {self.predict_steps}")
        if not isinstance(self.n_predictions, int) or self.n_predictions <= 0:
            raise ValueError(f"n_predictions must be a positive integer, got {self.n_predictions}")
        if not isinstance(self.n_instances, int) or self.n_instances <= 0:
            raise ValueError(f"n_instances must be a positive integer, got {self.n_instances}")
        if not isinstance(self.n_jobs, int):
            raise ValueError(f"n_jobs must be an integer, got {type(self.n_jobs).__name__}")
        if not isinstance(self.wasserstein_projections, int) or self.wasserstein_projections <= 0:
            raise ValueError(f"wasserstein_projections must be a positive integer, got {self.wasserstein_projections}")
        if not isinstance(self.length, int) or self.length <= 0:
            raise ValueError(f"length must be a positive integer, got {self.length}")
        
        # validate float parameters
        if not isinstance(self.vpt_threshold, (int, float)) or self.vpt_threshold <= 0:
            raise ValueError(f"vpt_threshold must be positive, got {self.vpt_threshold}")
        if not isinstance(self.clip_wasserstein, (int, float)) or self.clip_wasserstein <= 0:
            raise ValueError(f"clip_wasserstein must be positive, got {self.clip_wasserstein}")
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        
        # validate metrics
        if not isinstance(self.metrics, list) or len(self.metrics) == 0:
            raise ValueError("metrics must be a non-empty list")
        valid_metrics = {'wasserstein', 'max_wasserstein', 'vpt', 'max_cle'}
        invalid = set(self.metrics) - valid_metrics
        if invalid:
            raise ValueError(f"invalid metrics: {invalid}. Valid metrics are: {valid_metrics}")


class ESNObjective:
    """ESN objective function.
    
    Attributes
    ----------
    data : ndarray of shape (input_dim, T)
        Data.
    space : ESNSearchSpace
        Search space.
    config : EvaluationConfig, default=EvaluationConfig()
        Evaluation configuration.

    Examples
    --------
    >>> import numpy as np
    >>> from rc.optimisation import ESNSearchSpaceBuilder, ESNObjective
    >>> data = np.random.randn(3, 50000)  # (input_dim, T)
    >>> space = (ESNSearchSpaceBuilder()
    ...     .optimize("spectral_radius")
    ...     .fix(N=500)
    ...     .build())
    >>> objective = ESNObjective(data, space)
    >>> main, compare = objective({"spectral_radius": 0.9})
    >>> main  # {'wasserstein': (median, stderr), ...}
    """
    def __init__(self, data: NDArray, space: ESNSearchSpace, config: EvaluationConfig = EvaluationConfig(), seed: int = 42):
        # validate inputs
        if not isinstance(space, ESNSearchSpace):
            raise TypeError(f"space must be an ESNSearchSpace instance, got {type(space).__name__}")
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be an EvaluationConfig instance, got {type(config).__name__}")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed}")
        
        # validate and process data
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"data must be 2D array, got shape {data.shape}")
        if data.size == 0:
            raise ValueError("data cannot be empty")
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or infinite values")
        
        self.config = config
        self.space = space
        self.seed = seed
        if data.shape[0] > data.shape[1]:
            logger.warning(
                f"data shape {data.shape} has more features than timesteps; "
                f"auto-transposing to (input_dim, T). Pass (input_dim, T) explicitly to silence this."
            )
            self.data = data.T
        else:
            self.data = data
        self.D, self.T = self.data.shape
        
        # validate data length is sufficient
        min_length = config.washout + config.warmup_steps + config.predict_steps
        if self.T < min_length:
            raise ValueError(
                f"data length ({self.T}) is too short. Need at least {min_length} timesteps "
                f"(washout={config.washout} + warmup={config.warmup_steps} + predict={config.predict_steps})"
            )
        
        # validate high > low for stratified starts
        high = self.T - config.warmup_steps - config.predict_steps
        if high <= config.washout:
            raise ValueError(
                f"data length ({self.T}) is too short to create {config.n_predictions} prediction windows. "
                f"Reduce washout, warmup_steps, predict_steps, or provide more data."
            )
        
        self._needs_wasserstein = (
            'wasserstein' in config.metrics 
            or 'max_wasserstein' in config.metrics 
            or (config.compare_metrics and (
                'wasserstein' in config.compare_metrics 
                or 'max_wasserstein' in config.compare_metrics
            ))
        )
        time_start = time.time()
        self.starts = self._get_stratified_starts(
            n_total=config.n_predictions,
            low=config.washout,
            high=high,
            seed=seed
        )
        time_end = time.time()
        logger.debug(f"Time taken to get stratified starts: {time_end - time_start:.2f} seconds")
        
        time_start = time.time()
        if self._needs_wasserstein:
            self.projections = self._get_projections(
                self.D, 
                config.wasserstein_projections, 
                seed=seed,
                method='orthogonal'
            )
        else:
            self.projections = None
        time_end = time.time()
        logger.debug(f"Time taken to get projections: {time_end - time_start:.2f} seconds")
        
        time_start = time.time()
        self._precompute_windows()
        time_end = time.time()
        logger.debug(f"Time taken to precompute windows: {time_end - time_start:.2f} seconds")
    
    def _get_stratified_starts(self, n_total: int, low: int, high: int, seed: int) -> NDArray:
        rng = np.random.default_rng(seed)
        edges = np.linspace(low, high, n_total + 1)
        starts = np.array([rng.integers(int(edges[i]), int(edges[i + 1])) for i in range(n_total)])
        rng.shuffle(starts)
        return starts
    
    def _get_projections(self, d: int, n: int, seed: int, method: str = 'orthogonal') -> NDArray:
        rng = np.random.default_rng(seed)
        
        if method == 'orthogonal':
            blocks = []
            for _ in range(0, n, d):
                Z = rng.standard_normal((d, d))
                Q, _ = np.linalg.qr(Z)
                blocks.append(Q)
            return np.hstack(blocks)[:, :n].astype(self.data.dtype)
        
        else: 
            raise ValueError(f"unknown method: {method}")
    
    def _precompute_windows(self):
        cfg = self.config
        n_windows = len(self.starts)
        
        self.warmup_windows = np.empty((n_windows, self.D, cfg.warmup_steps), dtype=self.data.dtype)
        self.gt_windows = np.empty((n_windows, self.D, cfg.predict_steps), dtype=self.data.dtype)
        
        for i, start in enumerate(self.starts):
            self.warmup_windows[i] = self.data[:, start:start + cfg.warmup_steps]
            self.gt_windows[i] = self.data[:, start + cfg.warmup_steps:start + cfg.warmup_steps + cfg.predict_steps]
        
        if self._needs_wasserstein:
            self.gt_projected_sorted = np.empty(
                (n_windows, cfg.predict_steps, cfg.wasserstein_projections),
                dtype=self.data.dtype
            )
            for i in range(n_windows):
                gt_proj = self.gt_windows[i].T @ self.projections  
                self.gt_projected_sorted[i] = np.sort(gt_proj, axis=0)
        else:
            self.gt_projected_sorted = None
        
        logger.debug(f"Precomputed {n_windows} windows")
    
    def _compute_wasserstein(self, pred: NDArray, window_idx: int, p: int = 2) -> tuple[float, float]:
        time_start = time.time()
        pred_proj = pred.T @ self.projections  
        pred_proj_sorted = np.sort(pred_proj, axis=0)
        time_end = time.time()
        logger.debug(f"Time taken to project and sort predictions: {time_end - time_start:.2f} seconds")
        
        time_start = time.time()
        diff_p = np.abs(self.gt_projected_sorted[window_idx] - pred_proj_sorted) ** p
        wasserstein_1d_p = np.mean(diff_p, axis=0) 
        time_end = time.time()
        logger.debug(f"Time taken to compute Wasserstein distance: {time_end - time_start:.2f} seconds")
        
        time_start = time.time()
        max_sw = np.max(wasserstein_1d_p) ** (1.0 / p)
        mean_sw = np.mean(wasserstein_1d_p) ** (1.0 / p)
        time_end = time.time()
        logger.debug(f"Time taken to compute max and mean Wasserstein distance: {time_end - time_start:.2f} seconds")
        return max_sw, mean_sw
    
    def _eval_instance(self, params: dict, idx: int) -> dict:
        """Evaluate ESN on a single instance."""
        try:
            cfg = self.space.build_config(params, self.D)
            cfg.seed = idx
            time_start = time.time()
            esn = ESN(cfg)
            esn.train(self.data, washout=self.config.washout)
            time_end = time.time()
            logger.debug(f"Time taken to train ESN: {time_end - time_start:.2f} seconds")
        except (ArpackNoConvergence, np.linalg.LinAlgError) as e:
            logger.warning(f"ESN initialization failed for seed {idx}: {e}")
            return {}  
        
        results = {m: [] for m in self.config.metrics}
        compare_results = {m: [] for m in (self.config.compare_metrics or [])}
        
        time_start = time.time()
        if 'max_cle' in self.config.metrics or self.config.constrain_cle:
            try:
                max_cle = calculate_max_conditional_lyapunov_exponent(esn, self.data, self.config.dt, self.config.length)
                results['max_cle'] = [max_cle]
            except Exception as e:
                logger.warning(f"CLE computation failed: {e}")
                results['max_cle'] = [1.0]
        time_end = time.time()
        logger.debug(f"Time taken to compute max CLE: {time_end - time_start:.2f} seconds")
        
        time_start = time.time()
        for j in range(self.config.n_predictions):
            try:
                warmup = self.warmup_windows[j]
                pred, _ = esn.predict(warmup, self.config.predict_steps)
                
                if self._needs_wasserstein:
                    sw_max, sw_mean = self._compute_wasserstein(pred, j)
                    if 'max_wasserstein' in results:
                        results['max_wasserstein'].append(np.clip(sw_max, 0, self.config.clip_wasserstein))
                    if 'wasserstein' in results:
                        results['wasserstein'].append(np.clip(sw_mean, 0, self.config.clip_wasserstein))
                    if 'max_wasserstein' in compare_results:
                        compare_results['max_wasserstein'].append(np.clip(sw_max, 0, self.config.clip_wasserstein))
                    if 'wasserstein' in compare_results:
                        compare_results['wasserstein'].append(np.clip(sw_mean, 0, self.config.clip_wasserstein))
                if 'vpt' in results:
                    gt = self.gt_windows[j]
                    results['vpt'].append(valid_prediction_time(gt, pred, self.config.vpt_threshold, self.config.dt))
            except Exception as e:
                logger.warning(f"Prediction {j} failed: {e}")
                continue
        time_end = time.time()
        logger.debug(f"Time taken to evaluate instance {idx}: {time_end - time_start:.2f} seconds")
        return {k: np.median(v) for k, v in results.items() if v}, {k: np.median(v) for k, v in compare_results.items() if v}
    
    def __call__(self, params: dict) -> tuple[dict, dict]:
        """Evaluate ESN on multiple instances. Returns (main_output, compare_output)."""
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._eval_instance)(params, i) 
            for i in range(self.config.n_instances)
        )
        
        # Separate main results and compare results
        main_results = [r[0] for r in results if r]
        compare_results = [r[1] for r in results if r and len(r) > 1]
        
        if not main_results:
            logger.warning(f"No valid results for {params}")
            output = {k: (10.0, 0.0) for k in self.config.metrics}
            if self.config.constrain_cle:
                output['max_cle'] = (1.0, 0.0)
            return output, {}

        output = {}
        for k in main_results[0]:
            values = [r[k] for r in main_results]
            output[k] = (np.median(values), np.std(values) / np.sqrt(len(values)))

        output_compare = {}
        if compare_results:
            for k in compare_results[0]:
                values = [r[k] for r in compare_results]
                output_compare[k] = (np.median(values), np.std(values) / np.sqrt(len(values)))

        return output, output_compare

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
    if not isinstance(space, ESNSearchSpace):
        raise TypeError(f"space must be an ESNSearchSpace instance, got {type(space).__name__}")
    
    if isinstance(points, int):
        if points <= 0:
            raise ValueError(f"points must be a positive integer, got {points}")
        points = {k: points for k in space.optimize}
    elif isinstance(points, dict):
        for k, v in points.items():
            if k not in space.optimize:
                raise ValueError(f"unknown parameter in points: '{k}'. Valid parameters: {list(space.optimize.keys())}")
            if isinstance(v, int) and v <= 0:
                raise ValueError(f"grid points for '{k}' must be positive, got {v}")
    else:
        raise TypeError(f"points must be an int or dict, got {type(points).__name__}")
    
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
    num_initialization_trials: int | None = None,
) -> tuple[dict, dict, AxClient, NDArray]:
    """Optimize ESN hyperparameters.
    
    Parameters
    ----------
    data : NDArray of shape (input_dim, T)
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
    num_initialization_trials : int | None, default=None
        Number of initialization trials (Bayesian only).
    
    Returns
    -------
    best_params_ax : dict
        Best parameters according to Ax model.
    best_params_observed : dict
        Best parameters from actual observations.
    ax : AxClient
        Ax client with all trials.
    comparison : NDArray of shape (len(compare_metrics) + len(metrics), n_trials)
        Per-trial values for compare_metrics followed by metrics.
        
    Examples
    --------
    Bayesian optimization:
    
    >>> import numpy as np
    >>> from rc.optimisation import ESNSearchSpaceBuilder, EvaluationConfig, optimize_esn
    >>> data = np.random.randn(3, 50000)  # (input_dim, T)
    >>> space = (ESNSearchSpaceBuilder()
    ...     .optimize("spectral_radius")
    ...     .optimize("alpha")
    ...     .fix(N=500, mode="standard")
    ...     .build())
    >>> config = EvaluationConfig(n_predictions=5, n_instances=3)
    >>> best_ax, best_obs, ax, comparison = optimize_esn(data, space, config, n_trials=30, num_initialization_trials=10)

    Grid search:

    >>> best_ax, best_obs, ax, comparison = optimize_esn(
    ...     data, space, config,
    ...     method="grid",
    ...     grid_points={"spectral_radius": 10, "alpha": 5},
    ...     num_initialization_trials=10
    ... )
    """
    # validate inputs
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D array, got shape {data.shape}")
    if data.size == 0:
        raise ValueError("data cannot be empty")
    if not np.isfinite(data).all():
        raise ValueError("data contains NaN or infinite values")
    
    if not isinstance(space, ESNSearchSpace):
        raise TypeError(f"space must be an ESNSearchSpace instance, got {type(space).__name__}")
    if not isinstance(config, EvaluationConfig):
        raise TypeError(f"config must be an EvaluationConfig instance, got {type(config).__name__}")
    
    # validate n_trials
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials}")
    
    # validate method
    valid_methods = {"bayesian", "grid"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
    
    objective = ESNObjective(data, space, config)
    if 'max_wasserstein' in config.metrics:
        objective_name = 'max_wasserstein'
    elif 'wasserstein' in config.metrics:
        objective_name = 'wasserstein'
    elif 'vpt' in config.metrics:
        objective_name = 'vpt'
    else:
        objective_name = config.metrics[0]
    minimize = objective_name in {'wasserstein', 'max_wasserstein', 'max_cle'}
    ax = space.create_ax_client(
        objective_name=objective_name,
        minimize=minimize,
        constrain_cle=config.constrain_cle,
        cle_threshold=config.cle_threshold,
        num_trials=n_trials,
        num_initialization_trials=num_initialization_trials,
    )
    
    best_val = float('inf')
    best_params_observed = None
    compare_metrics = config.compare_metrics or []
    comparison = np.zeros((len(compare_metrics) + len(config.metrics), n_trials))
    def _passes_cle_constraint(result, config):
        if not config.constrain_cle:
            return True
        cle_entry = result.get('max_cle')
        if cle_entry is None:
            return False
        cle_val = cle_entry[0] if isinstance(cle_entry, tuple) else cle_entry
        return cle_val <= config.cle_threshold

    if method == "grid":
        grid = _generate_grid(space, grid_points)
        logger.debug(f"Grid search: {len(grid)} combinations")
        
        for i, params in enumerate(grid):
            _, idx = ax.attach_trial(params)
            result, compare_result = objective(params)
            comparison[:, i] = [compare_result[m][0] for m in compare_metrics] + [result[m][0] for m in config.metrics]
            ax.complete_trial(idx, raw_data=result)
            
            val = result[config.metrics[0]][0]
            if _passes_cle_constraint(result, config) and val < best_val:
                best_val = val
                best_params_observed = params.copy()
            
            if (i + 1) % 10 == 0 or i == len(grid) - 1:
                logger.debug(f"[{i+1}/{len(grid)}] best={best_val:.4f}")
    
    else:
        for i in range(n_trials):
            params, idx = ax.get_next_trial()
            result, compare_result = objective(params)
            comparison[:, i] = [compare_result[m][0] for m in compare_metrics] + [result[m][0] for m in config.metrics]
            ax.complete_trial(idx, raw_data=result)
            
            val = result[config.metrics[0]][0]
            if _passes_cle_constraint(result, config) and val < best_val:
                best_val = val
                best_params_observed = params.copy()
            logger.debug(f"trial {i+1}: {val:.4f} (best={best_val:.4f})")
    
    best_params_ax, _ = ax.get_best_parameters()
    return best_params_ax, best_params_observed, ax, comparison
