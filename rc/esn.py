from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import random as sparse_random, csr_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge
from dataclasses import dataclass, field
from typing import Protocol, Callable, Literal, runtime_checkable
from scipy.stats import ortho_group
import ot
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

  

def participation_ratio(explained_variance: NDArray) -> float:
    """calculate the participation ratio of the pca scores
    
    Parameters
    ----------
    explained_variance : array-like
        Explained variance of the PCA components.
        
    Returns
    -------
    participation_ratio : float
        Participation ratio of the PCA components.
    """
    explained_variance = np.asarray(explained_variance)
    if explained_variance.ndim != 1:
        raise ValueError(f"explained_variance must be 1D array, got shape {explained_variance.shape}")
    if len(explained_variance) == 0:
        raise ValueError("explained_variance cannot be empty")
    if np.any(explained_variance < 0):
        raise ValueError("explained_variance values must be non-negative")
    
    sum_sq = np.sum(explained_variance**2)
    if sum_sq == 0:
        raise ValueError("explained_variance cannot be all zeros")
    
    return np.sum(explained_variance)**2 / sum_sq

def analyse_dynamics(rc_trajectory: NDArray, pca_components: int = 0.95) -> dict:
    """analyse the dynamics of the reservoir trajectory
    
    Parameters
    ----------
    rc_trajectory : array-like
        Reservoir trajectory of shape (N, T) where N is reservoir size and T is timesteps.
        
    Returns
    -------
    dict with keys: 'effective_dim', 'explained_variance', 'pca_scores'
        'effective_dim' : float
            Effective dimension (Participation Ratio) of the reservoir trajectory.
        'explained_variance' : array-like
            Explained variance of the PCA components.
        'pca_scores' : array-like
            PCA scores of the reservoir trajectory.
    """
    rc_trajectory = np.asarray(rc_trajectory)
    if rc_trajectory.ndim != 2:
        raise ValueError(f"rc_trajectory must be 2D array of shape (N, T), got shape {rc_trajectory.shape}")
    if rc_trajectory.shape[0] == 0 or rc_trajectory.shape[1] == 0:
        raise ValueError(f"rc_trajectory cannot have zero-length dimensions, got shape {rc_trajectory.shape}")
    if rc_trajectory.shape[1] < 2:
        raise ValueError(f"rc_trajectory must have at least 2 timesteps for PCA, got {rc_trajectory.shape[1]}")
    
    # do pca on the trajectory
    pca = PCA(n_components=pca_components)
    pca.fit(rc_trajectory.T)
    pca_scores = pca.transform(rc_trajectory.T)
    explained_variance = pca.explained_variance_ratio_
    effective_dim = participation_ratio(explained_variance)
    return {
        'effective_dim': effective_dim,
        'explained_variance': explained_variance,
        'pca_scores': pca_scores,
    }


@dataclass
class ESNConfig:
    """Configuration for Echo State Network.
    
    Parameters
    ----------
    N : int
        Number of reservoir neurons.
    input_dim : int
        Dimensionality of the input signal.
    spectral_radius : float, default=0.9
        Target spectral radius of reservoir weight matrix.
    alpha : float, default=1e-6
        Ridge regression regularization parameter.
    sparsity : float, default=0.9
        Fraction of zero entries in reservoir weight matrix.
    input_scaling : float, default=0.5
        Scaling factor for input weights.
    bias_scaling : float, default=0.1
        Scaling factor for bias vector.
    seed : int | None, default=None
        Random seed for reproducibility.
    dtype : np.dtype, default=np.float64
        Data type for arrays.
    """
    # general hyperparameters
    N: int
    input_dim: int
    spectral_radius: float = 0.9
    alpha: float = 1e-6
    sparsity: float = 0.9
    input_scaling: float = 0.5
    bias_scaling: float = 0.1
    seed: int | None = None
    
    # weights generation strategy
    weights_generation_strategy: Literal["Gaussian", "Uniform", "Bernoulli", "Small-World", "Scale-Free"] | Callable = "Gaussian"
    bias_generation_strategy: Literal["Gaussian", "Uniform", "Bernoulli"] | Callable = "Uniform"
    input_generation_strategy: Literal["Gaussian", "Uniform", "Bernoulli"] | Callable = "Uniform"
    self_connections: bool = False
    
    # data type
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    
    # dynamics config (optional, for automatic generation of dynamics)
    mode: str = "standard"
    leaky_rate: float = 0.1
    beta: float = 0.5
    scale: float = 0.1
    
    def __post_init__(self):
        # validate N and input_dim
        if not isinstance(self.N, (int, np.integer)) or self.N <= 0:
            raise ValueError(f"N must be a positive integer, got {self.N}")
        if not isinstance(self.input_dim, (int, np.integer)) or self.input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {self.input_dim}")
        
        # validate spectral_radius
        if not isinstance(self.spectral_radius, (int, float)) or self.spectral_radius <= 0:
            raise ValueError(f"spectral_radius must be positive, got {self.spectral_radius}")
        
        # validate alpha (regularization)
        if not isinstance(self.alpha, (int, float)) or self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        
        # validate sparsity
        if not isinstance(self.sparsity, (int, float)) or not (0 <= self.sparsity < 1):
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")
        
        # validate scaling parameters
        if not isinstance(self.input_scaling, (int, float)) or self.input_scaling < 0:
            raise ValueError(f"input_scaling must be non-negative, got {self.input_scaling}")
        if not isinstance(self.bias_scaling, (int, float)) or self.bias_scaling < 0:
            raise ValueError(f"bias_scaling must be non-negative, got {self.bias_scaling}")
        
        # validate seed
        if self.seed is not None and (not isinstance(self.seed, (int, np.integer)) or self.seed < 0):
            raise ValueError(f"seed must be a non-negative integer or None, got {self.seed}")
        
        # validate dynamics parameters
        if not isinstance(self.leaky_rate, (int, float)) or not (0 < self.leaky_rate <= 1):
            raise ValueError(f"leaky_rate must be in (0, 1], got {self.leaky_rate}")
        if not isinstance(self.beta, (int, float)) or not (0 < self.beta <= 1):
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")
        if not isinstance(self.scale, (int, float)) or self.scale < 0:
            raise ValueError(f"scale must be non-negative, got {self.scale}")
        
        # validate mode
        valid_modes = {"standard", "leaky", "leakyrand", "es2n", "es2nrand"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")

@runtime_checkable
class ReservoirDynamics(Protocol):
    """protocol for reservoir update dynamics"""
    
    def update(self, r: NDArray, z: NDArray, activation: Callable[[NDArray], NDArray]) -> NDArray:
        """compute new reservoir state
        
        Parameters
        ----------
        r : ndarray of shape (N,)
            Current reservoir state.
        z : ndarray of shape (N,)
            Pre-activation: Wr @ r + Wx @ x + b
        activation : callable
            Activation function.
            
        Returns
        -------
        r_new : ndarray of shape (N,)
            Updated reservoir state.
        """
        ...
    
    def jacobian_update(self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray, WxWout: NDArray) -> tuple[NDArray, NDArray]:
        """Propagate tangent vectors for Lyapunov computation.
        
        Parameters
        ----------
        delta : ndarray of shape (N, k)
            Tangent vectors to propagate.
        r : ndarray of shape (N,)
            Current reservoir state.
        z : ndarray of shape (N,)
            Pre-activation values.
        Wr : ndarray of shape (N, N)
            Reservoir weight matrix.
        Wx : ndarray of shape (N, input_dim)
            Input weight matrix.
        Wout : ndarray of shape (input_dim, N)
            Output weight matrix (without bias).
            
        Returns
        -------
        delta_new : ndarray of shape (N, k)
            Updated tangent vectors.
        r_new : ndarray of shape (N,)
            Updated reservoir state.
        """
        ...
    
    def get_params(self) -> dict:
        """return mode-specific parameters for serialization"""
        ...
    
    @classmethod
    def from_params(cls, params: dict) -> "ReservoirDynamics":
        """reconstruct from serialized parameters"""
        ...
        
    def conditional_jacobian_update(self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray) -> tuple[NDArray, NDArray]:
        """Propagate tangent vectors for conditional Lyapunov computation.
        
        Unlike jacobian_update, this does NOT include output feedback (Wout),
        as the system is being driven by external data.
        """
        ...
    
    def conditional_jacobian_update_vector(self, g: NDArray, z: NDArray, Wr: NDArray) -> NDArray:
        """Propagate tangent vectors for conditional max CLE computation.
        """
        ...


class StandardDynamics:
    """standard ESN dynamics: r = tanh(Wr @ r + Wx @ x + b)
    
    Examples
    --------
    >>> from rc.esn import StandardDynamics, create_dynamics
    >>> dynamics = StandardDynamics()
    >>> dynamics = create_dynamics("standard", N=100)
    """
    __slots__ = ()
    def update(self, r: NDArray, z: NDArray, activation: Callable[[NDArray], NDArray]) -> NDArray: return activation(z)
    
    def jacobian_update(self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray, WxWout: NDArray) -> tuple[NDArray, NDArray]:
        s = np.tanh(z)
        D = 1.0 - s**2
        J_delta = Wr @ delta + (WxWout @ delta)
        delta_new = D[:, np.newaxis] * J_delta
        return delta_new, s
    
    def get_params(self) -> dict: return {"mode": "standard"}
    
    @classmethod
    def from_params(cls, params: dict) -> "StandardDynamics": return cls()
    
    def conditional_jacobian_update(self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray) -> tuple[NDArray, NDArray]:
        """jacobian for driven dynamics."""
        s = np.tanh(z)
        D = 1.0 - s**2
        delta_new = D[:, np.newaxis] * (Wr @ delta)
        return delta_new, s
    
    def conditional_jacobian_update_vector(self, g: NDArray, z: NDArray, Wr) -> NDArray:
        D = 1.0 - np.tanh(z)**2
        Wr_g = Wr.dot(g) if hasattr(Wr, 'dot') else Wr @ g
        return D * Wr_g

@dataclass
class LeakyDynamics:
    """leaky integrator ESN dynamics
    
    r = (1 - leak) * r + leak * tanh(Wr @ r + Wx @ x + b)
    
    Parameters
    ----------
    leaky_rate : ndarray of shape (N,)
        Per-neuron leaky integration rates. Values should be in (0, 1].
        
    Examples
    --------
    Create with uniform leak rate for all neurons:
    
    >>> import numpy as np
    >>> from rc.esn import LeakyDynamics, create_dynamics
    >>> N = 100
    >>> dynamics = LeakyDynamics(leaky_rate=np.full(N, 0.3))
    
    Create with per-neuron random leak rates:
    
    >>> dynamics = LeakyDynamics(leaky_rate=np.random.uniform(0.1, 0.5, N))
    
    Use via create_dynamics helper:
    
    >>> rng = np.random.default_rng(42)
    >>> dynamics = create_dynamics("leaky", N=100, leaky_rate=0.2, rng=rng)
    """
    __slots__ = ('leaky_rate', 'keep_rate')
    leaky_rate: NDArray
    
    def __post_init__(self): self.keep_rate = 1.0 - self.leaky_rate
    
    def update(self, r: NDArray, z: NDArray, activation: Callable[[NDArray], NDArray]) -> NDArray: return self.keep_rate * r + self.leaky_rate * activation(z)
    
    def jacobian_update(self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray, WxWout: NDArray) -> tuple[NDArray, NDArray]:
        s = np.tanh(z)
        D = 1.0 - s**2
        J_delta = Wr @ delta + (WxWout @ delta)
        delta_new = (self.keep_rate[:, np.newaxis] * delta + self.leaky_rate[:, np.newaxis] * (D[:, np.newaxis] * J_delta))
        r_new = self.keep_rate * r + self.leaky_rate * s
        return delta_new, r_new
    
    def get_params(self) -> dict: return {"mode": "leaky", "leaky_rate": self.leaky_rate}
    
    @classmethod
    def from_params(cls, params: dict) -> "LeakyDynamics": return cls(leaky_rate=params["leaky_rate"])

    def conditional_jacobian_update(self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray) -> tuple[NDArray, NDArray]:
        """jacobian for driven dynamics."""
        s = np.tanh(z)
        D = 1.0 - s**2
        delta_new = (self.keep_rate[:, np.newaxis] * delta + self.leaky_rate[:, np.newaxis] * (D[:, np.newaxis] * (Wr @ delta)))
        r_new = self.keep_rate * r + self.leaky_rate * s
        return delta_new, r_new
    
    def conditional_jacobian_update_vector(self, g: NDArray, z: NDArray, Wr) -> NDArray:
        D = 1.0 - np.tanh(z)**2
        Wr_g = Wr.dot(g) if hasattr(Wr, 'dot') else Wr @ g
        return self.keep_rate * g + self.leaky_rate * (D * Wr_g)
@dataclass 
class ES2NDynamics:
    """ES2N dynamics with orthogonal mixing.
    
    r = beta * tanh(z) + (1 - beta) * (O @ r)
    
    Parameters
    ----------
    beta : ndarray of shape (N,)
        Per-neuron nonlinearity mixing parameter. Values should be in (0, 1].
    O : ndarray of shape (N, N)
        Orthogonal transformation matrix.
        
    Examples
    --------
    Create with uniform beta and random orthogonal matrix:
    
    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> from rc.esn import ES2NDynamics, create_dynamics
    >>> N = 100
    >>> O = ortho_group.rvs(N)
    >>> dynamics = ES2NDynamics(beta=np.full(N, 0.5), O=O)
    
    Use via create_dynamics helper (recommended):
    
    >>> rng = np.random.default_rng(42)
    >>> dynamics = create_dynamics("es2n", N=100, beta=0.5, rng=rng)
    """
    __slots__ = ('beta', 'O', 'keep_rate')
    beta: NDArray
    O: NDArray
    
    def __post_init__(self): self.keep_rate = 1.0 - self.beta
    
    def update(self, r: NDArray, z: NDArray, activation: Callable[[NDArray], NDArray]) -> NDArray: return self.beta * activation(z) + self.keep_rate * (self.O @ r)
    
    def jacobian_update(self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray, WxWout: NDArray) -> tuple[NDArray, NDArray]:
        s = np.tanh(z)
        D = 1.0 - s**2
        J_delta = Wr @ delta + (WxWout @ delta)
        nonlin_term = self.beta[:, np.newaxis] * (D[:, np.newaxis] * J_delta)
        lin_term = self.keep_rate[:, np.newaxis] * (self.O @ delta)
        delta_new = nonlin_term + lin_term
        r_new = self.beta * s + self.keep_rate * (self.O @ r)
        return delta_new, r_new
    
    def get_params(self) -> dict: return {"mode": "es2n", "beta": self.beta, "O": self.O}
    
    @classmethod
    def from_params(cls, params: dict) -> "ES2NDynamics": return cls(beta=params["beta"], O=params["O"])
    
    def conditional_jacobian_update_vector(self, g: NDArray, z: NDArray, Wr) -> NDArray:
        D = 1.0 - np.tanh(z)**2
        Wr_g = Wr.dot(g) if hasattr(Wr, 'dot') else Wr @ g
        return self.beta * (D * Wr_g) + self.keep_rate * (self.O @ g)
    
def create_dynamics(mode: Literal["standard", "leaky", "leakyrand", "es2n", "es2nrand"] | str, N: int, dtype: np.dtype = np.float64, 
                  leaky_rate: float | NDArray = 0.1, beta: float | NDArray = 0.5, scale: float = 0.1, rng: np.random.Generator = None) -> ReservoirDynamics:
    """function to create reservoir dynamics
    
    Parameters
    ----------
    mode : str
        Dynamics mode: 'standard', 'leaky', 'leakyrand', 'es2n', 'es2nrand'.
    N : int
        Number of reservoir neurons.
    dtype : np.dtype
        Data type for arrays.
    leaky_rate : float or array-like
        Leaky rate for leaky modes.
    beta : float or array-like
        Beta parameter for ES2N modes.
    scale : float
        Scale for random parameter sampling.
    rng : np.random.Generator or None
        Random number generator.
        
    Returns
    -------
    dynamics : ReservoirDynamics
        Configured dynamics instance.
    """
    # validate N
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError(f"N must be a positive integer, got {N}")
    
    # validate scale
    if not isinstance(scale, (int, float)) or scale < 0:
        raise ValueError(f"scale must be non-negative, got {scale}")
    
    # validate rng for modes that need it
    if mode in ("leakyrand", "es2nrand", "es2n") and rng is None:
        raise ValueError(f"rng (random number generator) is required for mode '{mode}'")
    
    if mode == "standard": 
        return StandardDynamics()
    elif mode == "leaky":
        # validate leaky_rate
        if np.isscalar(leaky_rate):
            if not (0 < leaky_rate <= 1):
                raise ValueError(f"leaky_rate must be in (0, 1], got {leaky_rate}")
            lr = np.full(N, leaky_rate, dtype=dtype)
        else:
            lr = np.asarray(leaky_rate, dtype=dtype)
            if lr.shape != (N,):
                raise ValueError(f"leaky_rate array must have shape ({N},), got {lr.shape}")
            if np.any(lr <= 0) or np.any(lr > 1):
                raise ValueError("all leaky_rate values must be in (0, 1]")
        return LeakyDynamics(leaky_rate=np.clip(lr, 0, 1))
    elif mode == "es2n":
        # validate beta
        if np.isscalar(beta):
            if not (0 < beta <= 1):
                raise ValueError(f"beta must be in (0, 1], got {beta}")
            b = np.full(N, beta, dtype=dtype)
        else:
            b = np.asarray(beta, dtype=dtype)
            if b.shape != (N,):
                raise ValueError(f"beta array must have shape ({N},), got {b.shape}")
            if np.any(b <= 0) or np.any(b > 1):
                raise ValueError("all beta values must be in (0, 1]")
        O = ortho_group.rvs(N, random_state=rng).astype(dtype)
        return ES2NDynamics(beta=np.clip(b, 0.01, 1), O=O)
    elif mode == "leakyrand":
        if not (0 < leaky_rate <= 1):
            raise ValueError(f"leaky_rate must be in (0, 1], got {leaky_rate}")
        lr = rng.uniform(max(leaky_rate - scale, 0), min(leaky_rate + scale, 1), N).astype(dtype)
        return LeakyDynamics(leaky_rate=np.clip(lr, 0, 1))
    elif mode == "es2nrand":
        if not (0 < beta <= 1):
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        b = rng.uniform(max(beta - scale, 0), min(beta + scale, 1), N).astype(dtype)
        O = ortho_group.rvs(N, random_state=rng).astype(dtype)
        return ES2NDynamics(beta=np.clip(b, 0.01, 1), O=O)    
    else: 
        valid_modes = {"standard", "leaky", "leakyrand", "es2n", "es2nrand"}
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")


class ESN:
    
    """Echo State Network.
    
    Implements reservoir computing with pluggable dynamics for standard ESN,
    leaky integrator ESN, and ES2N variants.
    
    Parameters
    ----------
    config : ESNConfig
        Network configuration.
    dynamics : ReservoirDynamics
        Reservoir update dynamics.
        
    Attributes
    ----------
    Wr : ndarray or sparse matrix of shape (N, N)
        Reservoir weight matrix.
    Wx : ndarray of shape (N, input_dim)
        Input weight matrix.
    b : ndarray of shape (N,)
        Bias vector.
    Wout : ndarray of shape (input_dim, N) or None
        Output weights. None before training.
    Wout_bias : ndarray of shape (input_dim,) or None
        Output bias. None before training.
    r : ndarray of shape (N,)
        Current reservoir state.
        
    Examples
    --------
    >>> import numpy as np
    >>> from rc.esn import ESN, ESNConfig, StandardDynamics
    >>> config = ESNConfig(N=500, input_dim=3, spectral_radius=0.95)
    >>> dynamics = StandardDynamics()
    >>> esn = ESN(config, dynamics)
    >>> training_data = np.random.randn(3, 10000)  # (input_dim, T)
    >>> esn.train(training_data, washout=100)
    >>> warmup_data = np.random.randn(3, 100)  # (input_dim, warmup_length)
    >>> predictions, states = esn.predict(warmup_data, steps=1000)
    """
    __slots__ = ('config', 'rng', 'dynamics', 'Wr', 'Wx', 'b', 'r', 'Wout', 'Wout_bias', '_use_sparse')
    def __init__(self, config: ESNConfig | None = None, dynamics: ReservoirDynamics | None = None, *, N: int | None = None, input_dim: int | None = None, **kwargs) -> None:
        # build config
        if config is not None:
            if N is not None or input_dim is not None or kwargs: raise ValueError("Cannot specify both config and individual parameters")
            self.config = config
        else:
            if N is None or input_dim is None: raise ValueError("Must provide config or (N and input_dim)")
            self.config = ESNConfig(N=N, input_dim=input_dim, **kwargs)
        cfg = self.config
        self._use_sparse = self.config.sparsity > 0.9    

        # initialize random state
        self.rng = np.random.default_rng(cfg.seed)
        
        # build dynamics
        self.dynamics = dynamics if dynamics is not None else create_dynamics(cfg.mode, cfg.N, cfg.dtype, cfg.leaky_rate, cfg.beta, cfg.scale, self.rng)
        
        # weights
        self.Wr, self.Wx, self.b = self._create_reservoir_weights(), self._create_input_weights(), self._create_bias()
        
        # rc state
        self.r = self.rng.uniform(-1, 1, cfg.N).astype(cfg.dtype)
        
        # output weights
        self.Wout, self.Wout_bias = None, None
        
        # sparse 
        
    @property
    def N(self) -> int: return self.config.N
    
    @property
    def input_dim(self) -> int: return self.config.input_dim
    
    @property
    def is_trained(self) -> bool: return self.Wout is not None
    
    def _activation(self, z: NDArray) -> NDArray:
        """tanh activation function."""
        return np.tanh(z)
    
    def _create_small_world_weights(self) -> NDArray | csr_matrix:
        """Watts-Strogatz small-world reservoir.
        """
        cfg = self.config
        
        k = max(2, int(cfg.N * (1 - cfg.sparsity)))
        if k % 2 == 1: k += 1 
        
        rewire_prob = 0.1  
        
        row_indices = []
        col_indices = []
        
        for i in range(cfg.N):
            for j in range(1, k // 2 + 1):
                right = (i + j) % cfg.N
                left = (i - j) % cfg.N
                
                if self.rng.random() < rewire_prob:
                    candidates = [n for n in range(cfg.N) if n != i]
                    right = self.rng.choice(candidates)
                if self.rng.random() < rewire_prob:
                    candidates = [n for n in range(cfg.N) if n != i]
                    left = self.rng.choice(candidates)
                
                row_indices.extend([i, i])
                col_indices.extend([right, left])
        
        nnz = len(row_indices)
        data = self.rng.normal(0, 1, nnz).astype(cfg.dtype)
        
        J = csr_matrix((data, (row_indices, col_indices)), shape=(cfg.N, cfg.N), dtype=cfg.dtype)
        
        if not cfg.self_connections:
            J.setdiag(0)
        J.eliminate_zeros()
        
        if J.nnz > 0:
            eigvals, _ = sparse_eigs(J, k=1, which='LM')
            rho = np.max(np.abs(eigvals))
            if rho > 0:
                J = (J * (cfg.spectral_radius / rho)).tocsr()
        
        if not self._use_sparse:
            J = J.toarray()
        
        return J
    def _create_scale_free_weights(self) -> NDArray | csr_matrix:
        """Scale-free reservoir.
        """
        cfg = self.config
        
        m = max(1, int(cfg.N * (1 - cfg.sparsity) / 2))
        
        edges = set()
        degrees = np.zeros(cfg.N, dtype=np.int32)
        
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                edges.add((i, j))
                edges.add((j, i)) 
                degrees[i] += 1
                degrees[j] += 1
        
        for new_node in range(m + 1, cfg.N):
            existing_nodes = np.arange(new_node)
            probs = degrees[:new_node].astype(np.float64)
            
            if probs.sum() == 0:
                probs = np.ones(new_node)
            probs /= probs.sum()
            
            targets = self.rng.choice(existing_nodes, size=min(m, new_node), replace=False, p=probs)
            
            for target in targets:
                edges.add((new_node, target))
                edges.add((target, new_node)) 
                degrees[new_node] += 1
                degrees[target] += 1
        
        row_indices = [e[0] for e in edges]
        col_indices = [e[1] for e in edges]
        nnz = len(row_indices)
        
        data = self.rng.normal(0, 1, nnz).astype(cfg.dtype)
        
        J = csr_matrix((data, (row_indices, col_indices)), shape=(cfg.N, cfg.N), dtype=cfg.dtype)
        
        if not cfg.self_connections:
            J.setdiag(0)
        J.eliminate_zeros()
        
        if J.nnz > 0:
            eigvals, _ = sparse_eigs(J, k=1, which='LM')
            rho = np.max(np.abs(eigvals))
            if rho > 0:
                J = (J * (cfg.spectral_radius / rho)).tocsr()
        
        if not self._use_sparse:
            J = J.toarray()
        
        return J
    
    def _create_reservoir_weights(self) -> NDArray | csr_matrix:
        """create reservoir weight matrix"""
        cfg = self.config
        if cfg.weights_generation_strategy == "Gaussian":
            if self._use_sparse: 
                J = sparse_random(cfg.N, cfg.N, density=1 - cfg.sparsity, data_rvs=lambda s: self.rng.normal(0, cfg.spectral_radius / np.sqrt(cfg.N), s), format='csr', random_state=self.rng).astype(cfg.dtype)
                if not cfg.self_connections: J.setdiag(0)
                J.eliminate_zeros()
                vals, _ = sparse_eigs(J, k=1, which='LM')
                rho = np.max(np.abs(vals))
                if rho == 0: rho = 1.0
                J = (J * (cfg.spectral_radius / rho)).tocsr()
            else:
                J = sparse_random(cfg.N, cfg.N, density=1 - cfg.sparsity, data_rvs=lambda s: self.rng.normal(0, cfg.spectral_radius / np.sqrt(cfg.N), s), random_state=self.rng).toarray().astype(cfg.dtype)
                if not cfg.self_connections: np.fill_diagonal(J, 0)
                eigvals = np.linalg.eigvals(J)
                rho = np.max(np.abs(eigvals))
                if rho > 0: J *= cfg.spectral_radius / rho
            return J
        elif cfg.weights_generation_strategy == "Uniform":
            if self._use_sparse:  
                J = sparse_random(cfg.N, cfg.N, density=1 - cfg.sparsity, data_rvs=lambda s: self.rng.uniform(-1, 1, s), format='csr', random_state=self.rng).astype(cfg.dtype)
                if not cfg.self_connections:
                    J.setdiag(0)
                J.eliminate_zeros()
                
                eigvals, _ = sparse_eigs(J, k=1, which='LM')
                rho = np.max(np.abs(eigvals))
                if rho == 0: rho = 1.0
                J = (J * (cfg.spectral_radius / rho)).tocsr()
            else:
                J = sparse_random(cfg.N, cfg.N, density=1 - cfg.sparsity, data_rvs=lambda s: self.rng.uniform(-1, 1, s), random_state=self.rng).toarray().astype(cfg.dtype)
                if not cfg.self_connections: np.fill_diagonal(J, 0)
                eigvals = np.linalg.eigvals(J)
                rho = np.max(np.abs(eigvals))
                if rho > 0: J *= cfg.spectral_radius / rho
            return J
        elif cfg.weights_generation_strategy == "Bernoulli":
            if self._use_sparse:
                J = sparse_random(cfg.N, cfg.N, density=1 - cfg.sparsity, data_rvs=lambda s: self.rng.choice([-1, 1], size=s, p=[0.5, 0.5]), format='csr', random_state=self.rng).astype(cfg.dtype)
                if not cfg.self_connections: J.setdiag(0)
                J.eliminate_zeros()
                eigvals, _ = sparse_eigs(J, k=1, which='LM')
                rho = np.max(np.abs(eigvals))
                if rho == 0: rho = 1.0
                J = (J * (cfg.spectral_radius / rho)).tocsr()
            else:
                J = sparse_random(cfg.N, cfg.N, density=1 - cfg.sparsity, data_rvs=lambda s: self.rng.choice([-1, 1], size=s, p=[0.5, 0.5]), random_state=self.rng).toarray().astype(cfg.dtype)
                if not cfg.self_connections: np.fill_diagonal(J, 0)
                eigvals = np.linalg.eigvals(J)
                rho = np.max(np.abs(eigvals))
                if rho > 0: J *= cfg.spectral_radius / rho
            return J
        elif callable(cfg.weights_generation_strategy):
            J = cfg.weights_generation_strategy(self.rng, cfg.N, cfg.N, cfg.spectral_radius, cfg.sparsity, cfg.self_connections)
            if not cfg.self_connections: J.setdiag(0)
            J.eliminate_zeros()
            eigvals, _ = sparse_eigs(J, k=1, which='LM')
            rho = np.max(np.abs(eigvals))
            if rho == 0: rho = 1.0
            J = (J * (cfg.spectral_radius / rho)).tocsr()
            return J
        
        elif cfg.weights_generation_strategy == "Small-World": return self._create_small_world_weights()
        elif cfg.weights_generation_strategy == "Scale-Free": return self._create_scale_free_weights()
        else: raise ValueError(f"unknown weights generation strategy: {cfg.weights_generation_strategy}")
    
    def _create_input_weights(self) -> NDArray:
        """create input weight matrix"""
        cfg = self.config
        if cfg.input_generation_strategy == "Gaussian": return self.rng.normal(0, cfg.input_scaling, (cfg.N, cfg.input_dim)).astype(cfg.dtype)
        elif cfg.input_generation_strategy == "Uniform": return self.rng.uniform(-cfg.input_scaling, cfg.input_scaling, (cfg.N, cfg.input_dim)).astype(cfg.dtype)
        elif cfg.input_generation_strategy == "Bernoulli": return self.rng.choice([-1, 1], size=(cfg.N, cfg.input_dim), p=[0.5, 0.5]).astype(cfg.dtype) * cfg.input_scaling
        elif callable(cfg.input_generation_strategy): return cfg.input_generation_strategy(self.rng, cfg.N, cfg.input_dim, cfg.input_scaling)
        else: raise ValueError(f"unknown input generation strategy: {cfg.input_generation_strategy}")
    
    def _create_bias(self) -> NDArray:
        """create bias vector"""
        cfg = self.config
        if cfg.bias_generation_strategy == "Gaussian": return self.rng.normal(0, cfg.bias_scaling, cfg.N).astype(cfg.dtype)
        elif cfg.bias_generation_strategy == "Uniform": return self.rng.uniform(-cfg.bias_scaling, cfg.bias_scaling, cfg.N).astype(cfg.dtype)
        elif cfg.bias_generation_strategy == "Bernoulli": return self.rng.choice([-1, 1], size=cfg.N, p=[0.5, 0.5]).astype(cfg.dtype) * cfg.bias_scaling
        elif callable(cfg.bias_generation_strategy): return cfg.bias_generation_strategy(self.rng, cfg.N, cfg.bias_scaling)
        else: raise ValueError(f"unknown bias generation strategy: {cfg.bias_generation_strategy}")
    
    def _compute_preactivation(self, x: NDArray) -> NDArray:
        """compute preactivation (same for all networks)"""
        if self._use_sparse: return self.Wr.dot(self.r) + self.Wx @ x + self.b
        return self.Wr @ self.r + self.Wx @ x + self.b
    
    def step(self, x: NDArray) -> NDArray:
        """perform one reservoir update step
        
        Parameters
        ----------
        x : ndarray of shape (input_dim,)
            Input vector.
            
        Returns
        -------
        r : ndarray of shape (N,)
            Updated reservoir state.
        """
        x = np.asarray(x)
        if x.shape != (self.input_dim,):
            raise ValueError(f"input x must have shape ({self.input_dim},), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("input x contains NaN or infinite values")
        
        z = self._compute_preactivation(x)
        self.r = self.dynamics.update(self.r, z, self._activation)
        return self.r
    
    def reset_state(self, state: NDArray | None = None):
        """reset reservoir state
        
        Parameters
        ----------
        state : ndarray of shape (N,) or None
            New reservoir state. If None, initializes randomly.
        """
        if state is None: 
            self.r = self.rng.uniform(-1, 1, self.N).astype(self.config.dtype)
        else: 
            state = np.asarray(state)
            if state.shape != (self.N,):
                raise ValueError(f"state must have shape ({self.N},), got {state.shape}")
            if not np.isfinite(state).all():
                raise ValueError("state contains NaN or infinite values")
            self.r = state.astype(self.config.dtype)
    
    def train(self, x_train: NDArray, washout: int = 100, skip_indices: NDArray | None = None, skip_window: int = 20, return_states: bool = False) -> NDArray:
        """train output weights using ridge regression
        
        Parameters
        ----------
        x_train : ndarray of shape (input_dim, T)
            Training time series. Each column is one timestep.
        washout : int, default=100
            Initial timesteps to discard.
        skip_indices : array-like or None
            Indices to exclude from regression (e.g., dataset boundaries).
        skip_window : int, default=20
            window around skip_indices to exclude.
            
        Returns
        -------
        states : ndarray of shape (N, T - washout - 1)
            collected reservoir states.
        """
        # validate x_train
        x_train = np.asarray(x_train)
        if x_train.ndim != 2:
            raise ValueError(f"x_train must be 2D array of shape (input_dim, T), got shape {x_train.shape}")
        if x_train.shape[0] != self.input_dim:
            raise ValueError(f"x_train first dimension must match input_dim ({self.input_dim}), got {x_train.shape[0]}")
        if not np.isfinite(x_train).all():
            raise ValueError("x_train contains NaN or infinite values")
        
        T = x_train.shape[1]
        
        # validate washout
        if not isinstance(washout, (int, np.integer)) or washout < 0:
            raise ValueError(f"washout must be a non-negative integer, got {washout}")
        if washout >= T - 1:
            raise ValueError(f"washout ({washout}) must be less than T-1 ({T-1}) to have training samples")
        
        # validate skip_window
        if not isinstance(skip_window, (int, np.integer)) or skip_window < 0:
            raise ValueError(f"skip_window must be a non-negative integer, got {skip_window}")
        
        self.reset_state()
        
        # washout phase
        for i in range(min(washout, T - 1)):
            self.step(x_train[:, i])
        
        # collect states
        effective_T = T - washout - 1
        states = np.zeros((self.N, effective_T), dtype=self.config.dtype)
        
        for i in range(effective_T):
            self.step(x_train[:, washout + i])
            states[:, i] = self.r
        
        # prepare regression data
        states_with_bias = np.vstack([states, np.ones((1, effective_T), dtype=self.config.dtype)])
        targets = x_train[:, washout + 1:washout + 1 + effective_T]
        
        # handle skip indices
        if skip_indices is not None:
            mask = self._compute_skip_mask(skip_indices, washout, effective_T, skip_window)
            states_with_bias = states_with_bias[:, mask]
            targets = targets[:, mask]
        
        # solve ridge regression
        self._solve_ridge(states_with_bias, targets)
        
        return states if return_states else None
    def _compute_skip_mask(self, skip_indices: NDArray, washout: int, effective_T: int, skip_window: int) -> NDArray:
        """compute boolean mask for skipping indices"""
        skip_indices = np.asarray(skip_indices)
        skip_centers = skip_indices - (washout + 1)
        
        skip_set = set()
        for center in skip_centers:
            for offset in range(-skip_window, skip_window + 1):
                idx = center + offset
                if 0 <= idx < effective_T: skip_set.add(idx)
        
        mask = np.ones(effective_T, dtype=bool)
        if skip_set: mask[list(skip_set)] = False
        return mask
    
    def _solve_ridge(self, states_with_bias: NDArray, targets: NDArray) -> None:
        """ridge regression for output weights"""
        try:
            SS_t = states_with_bias @ states_with_bias.T
            reg = self.config.alpha * np.eye(SS_t.shape[0], dtype=self.config.dtype)
            Gram = SS_t + reg
            YS_t = targets @ states_with_bias.T
            
            c_factor = cho_factor(Gram, overwrite_a=False, check_finite=False)
            W_t = cho_solve(c_factor, YS_t.T, check_finite=False)
            Wout_full = W_t.T
        except Exception:
            ridge = Ridge(alpha=self.config.alpha, fit_intercept=False, solver='sparse_cg')
            ridge.fit(states_with_bias.T, targets.T)
            Wout_full = ridge.coef_
        
        if Wout_full.ndim == 1: Wout_full = Wout_full[np.newaxis, :]
        
        self.Wout = Wout_full[:, :-1]
        self.Wout_bias = Wout_full[:, -1]
    
    def predict(self, warmup: NDArray, steps: int) -> tuple[NDArray, NDArray]:
        """generate autonomous predictions.
        
        Parameters
        ----------
        warmup : ndarray of shape (input_dim, warmup_length)
            Sequence to initialize reservoir.
        steps : int
            Number of prediction steps.
            
        Returns
        -------
        predictions : ndarray of shape (input_dim, steps)
            Predicted time series.
        states : ndarray of shape (N, steps)
            Reservoir states during prediction.
        """
        if not self.is_trained:
            raise RuntimeError("ESN must be trained before prediction. Call train() first.")
        
        # validate warmup
        warmup = np.asarray(warmup)
        if warmup.ndim != 2:
            raise ValueError(f"warmup must be 2D array of shape (input_dim, warmup_length), got shape {warmup.shape}")
        if warmup.shape[0] != self.input_dim:
            raise ValueError(f"warmup first dimension must match input_dim ({self.input_dim}), got {warmup.shape[0]}")
        if warmup.shape[1] == 0:
            raise ValueError("warmup must have at least 1 timestep")
        if not np.isfinite(warmup).all():
            raise ValueError("warmup contains NaN or infinite values")
        
        # validate steps
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError(f"steps must be a positive integer, got {steps}")
        
        self.reset_state()
        
        # warmup
        for i in range(warmup.shape[1]):
            self.step(warmup[:, i])
        
        # autonomous prediction
        predictions = np.zeros((self.input_dim, steps), dtype=self.config.dtype)
        states = np.zeros((self.N, steps), dtype=self.config.dtype)
        
        for i in range(steps):
            output = self.Wout @ self.r + self.Wout_bias
            predictions[:, i] = output
            self.step(output)
            states[:, i] = self.r
        
        return predictions, states
    
    def predict_driven(self, data: NDArray) -> tuple[NDArray, NDArray]:
        """generate one-step-ahead predictions while driven by external data.
        
        Parameters
        ----------
        data : ndarray of shape (input_dim, steps)
            Input sequence to drive the reservoir.
            
        Returns
        -------
        predictions : ndarray of shape (input_dim, steps)
            One-step-ahead predicted time series.
        states : ndarray of shape (N, steps)
            Reservoir states during prediction.
        """
        if not self.is_trained: 
            raise RuntimeError("ESN must be trained before prediction. Call train() first.")
        
        # validate data
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"data must be 2D array of shape (input_dim, steps), got shape {data.shape}")
        if data.shape[0] != self.input_dim:
            raise ValueError(f"data first dimension must match input_dim ({self.input_dim}), got {data.shape[0]}")
        if data.shape[1] == 0:
            raise ValueError("data must have at least 1 timestep")
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or infinite values")
        
        self.reset_state()
        
        steps = data.shape[1]
        
        # driven prediction
        predictions = np.zeros((self.input_dim, steps), dtype=self.config.dtype)
        states = np.zeros((self.N, steps), dtype=self.config.dtype)
        
        for i in range(steps):
            self.step(data[:, i])
            states[:, i] = self.r
            predictions[:, i] = self.Wout @ self.r + self.Wout_bias
        
        return predictions, states
    
    def lyapunov_spectrum(self, initial_data: NDArray, num_lyaps: int = 40, steps: int = 10000, norm_time: int = 10, dt: float = 0.25, num_samples: int = 5, warmup: int = 100, transient: int = 100, calculate_convergence: bool = False) -> dict:
        """lyapunov spectrum of trained ESN dynamics.
        
        Uses QR decomposition with tangent space propagation.
        
        Parameters
        ----------
        initial_data : ndarray of shape (input_dim, T)
            Data for initializing reservoir state.
        num_lyaps : int, default=40
            Number of Lyapunov exponents to compute.
        steps : int, default=10000
            Autonomous steps for estimation.
        norm_time : int, default=10
            Steps between QR renormalizations.
        dt : float, default=0.25
            Time step for continuous-time conversion.
        num_samples : int, default=5
            Independent runs from different initial conditions.
        warmup : int, default=100
            Warmup length per sample.
        transient : int, default=100
            Autonomous steps after forcing before measurement.
        calculate_convergence : bool, default=False
            Whether to calculate convergence of the Lyapunov exponents.
            
        Returns
        -------
        dict with keys: 'mean', 'std', 'all_samples', 'convergence',
                        'num_valid_samples', 'max_lyapunov', 'distances'
        """
        if not self.is_trained: 
            raise RuntimeError("ESN must be trained before Lyapunov estimation. Call train() first.")
        
        # validate initial_data
        initial_data = np.asarray(initial_data)
        if initial_data.ndim != 2:
            raise ValueError(f"initial_data must be 2D array of shape (input_dim, T), got shape {initial_data.shape}")
        if initial_data.shape[0] != self.input_dim:
            raise ValueError(f"initial_data first dimension must match input_dim ({self.input_dim}), got {initial_data.shape[0]}")
        if not np.isfinite(initial_data).all():
            raise ValueError("initial_data contains NaN or infinite values")
        
        # validate num_lyaps
        if not isinstance(num_lyaps, (int, np.integer)) or num_lyaps <= 0:
            raise ValueError(f"num_lyaps must be a positive integer, got {num_lyaps}")
        if num_lyaps > self.N: 
            raise ValueError(f"num_lyaps ({num_lyaps}) cannot exceed N ({self.N})")
        
        # validate steps and norm_time
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError(f"steps must be a positive integer, got {steps}")
        if not isinstance(norm_time, (int, np.integer)) or norm_time <= 0:
            raise ValueError(f"norm_time must be a positive integer, got {norm_time}")
        
        # validate dt
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        # validate num_samples
        if not isinstance(num_samples, (int, np.integer)) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer, got {num_samples}")
        
        # validate warmup and transient
        if not isinstance(warmup, (int, np.integer)) or warmup < 0:
            raise ValueError(f"warmup must be a non-negative integer, got {warmup}")
        if not isinstance(transient, (int, np.integer)) or transient < 0:
            raise ValueError(f"transient must be a non-negative integer, got {transient}")
        
        # check data length
        min_data_length = warmup * num_samples + steps
        if initial_data.shape[1] < min_data_length:
            raise ValueError(f"initial_data length ({initial_data.shape[1]}) is too short for {num_samples} samples with warmup={warmup}")
        
        dtype = self.config.dtype
        
        Wr = self.Wr.toarray() if hasattr(self.Wr, 'toarray') else self.Wr
        
        all_lyap_exps = np.zeros((num_samples, num_lyaps))
        convergence_history = []
        distances = []
        
        init_length = min(warmup, initial_data.shape[1] // num_samples)
        random_starts = self.rng.integers(0, initial_data.shape[1] - init_length, size=num_samples)
        
        for sample_idx in range(num_samples):
            # initialize reservoir
            self.reset_state()
            
            start_idx = random_starts[sample_idx]
            end_idx = min(start_idx + init_length, initial_data.shape[1])
            init_segment = initial_data[:, start_idx:end_idx]
            
            # force with data
            for i in range(init_segment.shape[1]):
                self.step(init_segment[:, i])
            
            delta = ortho_group.rvs(self.N, random_state=self.rng).astype(dtype)[:, :num_lyaps]
            delta, _ = np.linalg.qr(delta, mode='reduced')
            WxWout = self.Wx @ self.Wout
            for t_step in range(transient):
                output = self.Wout @ self.r + self.Wout_bias
                z = self._compute_preactivation(output)
                delta, self.r = self.dynamics.jacobian_update(delta, self.r, z, Wr, WxWout)
                if (t_step + 1) % norm_time == 0:
                    Q, _ = np.linalg.qr(delta, mode='reduced')
                    delta = Q
                    
            delta, _ = np.linalg.qr(delta, mode='reduced')        
            R_ii_sum = np.zeros(num_lyaps, dtype=dtype)
            if calculate_convergence: local_convergence = []
            norm_count = 0
            
            trajectory = np.zeros((self.input_dim, steps), dtype=dtype)
            for step in range(steps):
                output = self.Wout @ self.r + self.Wout_bias
                trajectory[:, step] = output
                
                # compute pre-activation
                z = self._compute_preactivation(output)
                
                # propagate tangent vectors using dynamics
                delta, self.r = self.dynamics.jacobian_update(delta, self.r, z, Wr, WxWout)
                
                # qr renormalization
                if (step + 1) % norm_time == 0:
                    Q, R = np.linalg.qr(delta, mode='reduced')
                    R_ii_sum += np.log(np.maximum(np.abs(np.diag(R)), np.finfo(dtype).tiny))
                    delta = Q[:, :num_lyaps]
                    norm_count += 1
                    if calculate_convergence: local_convergence.append(R_ii_sum / (norm_count * norm_time * dt))
            
            # compute trajectory distance
            try:
                ref_segment = initial_data[:, end_idx:end_idx + steps]
                if ref_segment.shape[1] == steps: distances.append(ot.sliced_wasserstein_distance(ref_segment.T, trajectory.T, n_projections=50))
            except ImportError: pass
            
            if norm_count > 0:
                all_lyap_exps[sample_idx] = R_ii_sum / (norm_count * norm_time * dt)
                if calculate_convergence: convergence_history.append(np.array(local_convergence))
            else: all_lyap_exps[sample_idx] = np.nan
        
        valid_mask = ~np.any(np.isnan(all_lyap_exps), axis=1)
        all_lyap_exps_valid = all_lyap_exps[valid_mask]
        
        all_lyap_exps_sorted = np.sort(all_lyap_exps_valid, axis=1)[:, ::-1]
        mean_lyap = np.median(all_lyap_exps_sorted, axis=0)
        std_lyap = np.std(all_lyap_exps_sorted, axis=0)
        
        return {'mean': mean_lyap, 'std': std_lyap, 'all_samples': all_lyap_exps_valid, 'convergence': convergence_history, 'num_valid_samples': len(all_lyap_exps_valid), 
                'max_lyapunov': mean_lyap[0] if len(mean_lyap) > 0 else np.nan, 'distances': distances}
        
    def conditional_lyapunov_spectrum(self, data: NDArray, num_lyaps: int | None = None, 
                                   norm_time: int = 10, dt: float = 0.01, 
                                   warmup: int = 1000, transient: int = 1500, 
                                   calculate_convergence: bool = False) -> dict:        
        """conditional Lyapunov exponents while driven by data.
        
        computes CLEs for the driven system,
        where the reservoir receives external input rather than its own predictions.
        CLEs measure stability/chaos of the reservoir's response to input.
        
        Parameters
        ----------
        data : ndarray of shape (input_dim, T)
            Driving time series. Must be long enough for warmup + transient + measurement.
        num_lyaps : int or None
            Number of exponents to compute. If None, computes all N.
        norm_time : int, default=10
            Steps between QR renormalizations.
        dt : float, default=0.01
            Time step for continuous-time conversion.
        warmup : int, default=1000
            Steps to drive reservoir before starting (no Lyapunov computation).
        transient : int, default=1500
            Additional steps for tangent vectors to align before measurement.
        calculate_convergence : bool, default=False
            Whether to calculate convergence of the Lyapunov exponents.
            
        Returns
        -------
        dict with keys:
            'exponents': ndarray of shape (num_lyaps,) - sorted descending
            'convergence': ndarray of shape (num_renorms, num_lyaps) - running estimates
            'max_cle': float - largest conditional Lyapunov exponent
            'sum_cle': float - sum of all CLEs (related to information dimension)
        """
        # validate data
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"data must be 2D array of shape (input_dim, T), got shape {data.shape}")
        if data.shape[0] != self.input_dim:
            raise ValueError(f"data first dimension must match input_dim ({self.input_dim}), got {data.shape[0]}")
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or infinite values")
        
        T = data.shape[1]
        
        # validate num_lyaps
        if num_lyaps is None: 
            num_lyaps = self.N
        elif not isinstance(num_lyaps, (int, np.integer)) or num_lyaps <= 0:
            raise ValueError(f"num_lyaps must be a positive integer or None, got {num_lyaps}")
        if num_lyaps > self.N: 
            raise ValueError(f"num_lyaps ({num_lyaps}) cannot exceed N ({self.N})")
        
        # validate norm_time
        if not isinstance(norm_time, (int, np.integer)) or norm_time <= 0:
            raise ValueError(f"norm_time must be a positive integer, got {norm_time}")
        
        # validate dt
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        # validate warmup and transient
        if not isinstance(warmup, (int, np.integer)) or warmup < 0:
            raise ValueError(f"warmup must be a non-negative integer, got {warmup}")
        if not isinstance(transient, (int, np.integer)) or transient < 0:
            raise ValueError(f"transient must be a non-negative integer, got {transient}")
        
        total_needed = warmup + transient + norm_time * 10
        if T < total_needed: 
            raise ValueError(f"data length ({T}) too short. Need at least {total_needed} timesteps (warmup={warmup} + transient={transient} + {norm_time * 10} measurement steps)")
        
        dtype = self.config.dtype
        
        Wr = self.Wr  
        
        self.reset_state()
        for i in range(warmup):
            self.step(data[:, i])
        
        if num_lyaps == 1:
            return self._conditional_lyapunov_fast(data, Wr, warmup, transient, 
                                                    norm_time, dt, calculate_convergence)
        
        # Standard path for multiple exponents
        return self._conditional_lyapunov_full(data, Wr, num_lyaps, warmup, transient,
                                                norm_time, dt, calculate_convergence)

    def _conditional_lyapunov_fast(self, data: NDArray, Wr, warmup: int, transient: int,
                                    norm_time: int, dt: float, 
                                    calculate_convergence: bool) -> dict:
        """Fast max CLE using power iteration (single vector, sparse-compatible)."""
        dtype = self.config.dtype
        T = data.shape[1]
        measure_steps = T - warmup
        
        # Initialize single tangent vector
        g = self.rng.standard_normal(self.N).astype(dtype)
        g /= np.linalg.norm(g)
        
        # Drive reservoir and compute CLE simultaneously (no storage needed)
        r = self.r.copy()
        
        # Transient phase - let g align with dominant direction
        for i in range(transient):
            z = self.Wr.dot(r) + self.Wx @ data[:, warmup + i] + self.b
            g = self.dynamics.conditional_jacobian_update_vector(g, z, Wr)
            r = self.dynamics.update(r, z, self._activation)
            
            if (i + 1) % norm_time == 0:
                g /= np.linalg.norm(g)
        
        # Measurement phase
        log_sum = 0.0
        norm_count = 0
        convergence = [] if calculate_convergence else None
        
        for i in range(transient, measure_steps):
            z = self.Wr.dot(r) + self.Wx @ data[:, warmup + i] + self.b
            g = self.dynamics.conditional_jacobian_update_vector(g, z, Wr)
            r = self.dynamics.update(r, z, self._activation)
            
            if (i - transient + 1) % norm_time == 0:
                norm = np.linalg.norm(g)
                log_sum += np.log(max(norm, np.finfo(dtype).tiny))
                g /= norm
                norm_count += 1
                
                if calculate_convergence:
                    convergence.append(log_sum / (norm_count * norm_time * dt))
        
        if norm_count == 0:
            return {'exponents': np.array([np.nan]), 'convergence': None, 
                    'max_cle': np.nan, 'sum_cle': np.nan, 'num_renorms': 0}
        
        max_cle = log_sum / (norm_count * norm_time * dt)
        
        return {
            'exponents': np.array([max_cle]),
            'convergence': np.array(convergence)[:, np.newaxis] if calculate_convergence else None,
            'max_cle': max_cle,
            'sum_cle': max_cle,
            'num_renorms': norm_count
        }

    def _conditional_lyapunov_full(self, data: NDArray, Wr, num_lyaps: int, 
                                    warmup: int, transient: int, norm_time: int, 
                                    dt: float, calculate_convergence: bool) -> dict:
        """Full spectrum computation (optimized for sparse Wr)."""
        dtype = self.config.dtype
        T = data.shape[1]
        measure_steps = T - warmup
        
        # Pre-compute states and preactivations (needed for multiple vectors)
        states = np.zeros((self.N, measure_steps), dtype=dtype)
        preactivations = np.zeros((self.N, measure_steps), dtype=dtype)
        
        r = self.r.copy()
        for i in range(measure_steps):
            z = self.Wr.dot(r) + self.Wx @ data[:, warmup + i] + self.b
            r = self.dynamics.update(r, z, self._activation)
            states[:, i] = r
            preactivations[:, i] = z
        
        # Initialize tangent vectors
        G = ortho_group.rvs(self.N, random_state=self.rng).astype(dtype)[:, :num_lyaps]
        G, _ = np.linalg.qr(G, mode='reduced')
        
        # Transient phase
        for i in range(transient // norm_time):
            for j in range(norm_time):
                idx = i * norm_time + j
                if idx >= measure_steps: break
                G, _ = self.dynamics.conditional_jacobian_update(
                    G, states[:, idx], preactivations[:, idx], Wr)
            G, _ = np.linalg.qr(G, mode='reduced')
        
        # Measurement phase
        start_idx = transient
        end_idx = measure_steps
        num_renorms = (end_idx - start_idx) // norm_time
        
        R_log_sum = np.zeros(num_lyaps, dtype=dtype)
        convergence = np.zeros((num_renorms, num_lyaps), dtype=dtype) if calculate_convergence else None
        
        for i in range(num_renorms):
            for j in range(norm_time):
                idx = start_idx + i * norm_time + j
                if idx >= measure_steps: break
                G, _ = self.dynamics.conditional_jacobian_update(
                    G, states[:, idx], preactivations[:, idx], Wr)
            
            G, R = np.linalg.qr(G, mode='reduced')
            R_log_sum += np.log(np.maximum(np.abs(np.diag(R)), np.finfo(dtype).tiny))
            
            if calculate_convergence:
                convergence[i] = R_log_sum / ((i + 1) * norm_time * dt)
        
        # Final exponents (sorted descending)
        exponents = R_log_sum / (num_renorms * norm_time * dt)
        sort_idx = np.argsort(exponents)[::-1]
        exponents = exponents[sort_idx]
        
        return {
            'exponents': exponents,
            'convergence': convergence[:, sort_idx] if calculate_convergence else None,
            'max_cle': exponents[0],
            'sum_cle': np.sum(exponents),
            'num_renorms': num_renorms
        }    

    def save(self, path: str):
        """save model to file.
        
        Parameters
        ----------
        path : str
            Output file path (.npz).
        """
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        if not path.endswith('.npz'):
            raise ValueError(f"path must end with '.npz', got '{path}'")
        
        cfg = self.config
        
        # config
        config_data = {
            'N': cfg.N,
            'input_dim': cfg.input_dim,
            'spectral_radius': cfg.spectral_radius,
            'alpha': cfg.alpha,
            'sparsity': cfg.sparsity,
            'input_scaling': cfg.input_scaling,
            'bias_scaling': cfg.bias_scaling,
            'seed': cfg.seed if cfg.seed is not None else -1,
            'dtype': np.array(str(cfg.dtype)),
            'mode': np.array(cfg.mode),
            'leaky_rate': cfg.leaky_rate,
            'beta': cfg.beta,
            'scale': cfg.scale,
        }
        
        # dynamics parameters
        dynamics_params = self.dynamics.get_params()
        dynamics_data = {'dynamics_mode': np.array(dynamics_params.pop('mode'))}
        for k, v in dynamics_params.items():
            dynamics_data[f'dynamics_{k}'] = v
        
        # weights - handle sparse
        if hasattr(self.Wr, 'toarray'): weights_data = {'Wr_sparse': True, 'Wr_data': self.Wr.data, 'Wr_indices': self.Wr.indices, 'Wr_indptr': self.Wr.indptr, 'Wr_shape': np.array(self.Wr.shape)}
        else: weights_data = {'Wr_sparse': False, 'Wr': self.Wr}
        
        weights_data.update({'Wx': self.Wx, 'b': self.b, 'r': self.r})
        
        # output weights
        if self.is_trained:
            weights_data['Wout'] = self.Wout
            weights_data['Wout_bias'] = self.Wout_bias  
        np.savez_compressed(path, **config_data, **dynamics_data, **weights_data)

    @classmethod
    def load(cls, path: str) -> "ESN":
        """load model from file.
        
        Parameters
        ----------
        path : str
            Path to saved model (.npz).
            
        Returns
        -------
        esn : ESN
            Loaded model.
        """
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        if not path.endswith('.npz'):
            raise ValueError(f"path must end with '.npz', got '{path}'")
        
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"model file not found: '{path}'")
        
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as e:
            raise ValueError(f"failed to load model from '{path}': {e}")
        
        #  config
        seed = int(data['seed'])
        config = ESNConfig(
            N=int(data['N']),
            input_dim=int(data['input_dim']),
            spectral_radius=float(data['spectral_radius']),
            alpha=float(data['alpha']),
            sparsity=float(data['sparsity']),
            input_scaling=float(data['input_scaling']),
            bias_scaling=float(data['bias_scaling']),
            seed=seed if seed >= 0 else None,
            dtype=np.dtype(str(data['dtype'])),
            mode=str(data['mode']),
            leaky_rate=float(data['leaky_rate']),
            beta=float(data['beta']),
            scale=float(data['scale']),
        )
        
        #  dynamics
        dynamics_mode = str(data['dynamics_mode'])
        if dynamics_mode == 'standard': dynamics = StandardDynamics()
        elif dynamics_mode == 'leaky': dynamics = LeakyDynamics(leaky_rate=data['dynamics_leaky_rate'])
        elif dynamics_mode == 'es2n': dynamics = ES2NDynamics(beta=data['dynamics_beta'], O=data['dynamics_O'])
        else: raise ValueError(f"unknown mode: {dynamics_mode}")
        
        #  instance 
        esn = object.__new__(cls)
        esn.config = config
        esn.dynamics = dynamics
        esn.rng = np.random.default_rng(config.seed)
        
        #  weights
        if data['Wr_sparse']: esn.Wr = csr_matrix((data['Wr_data'], data['Wr_indices'], data['Wr_indptr']), shape=tuple(data['Wr_shape']))
        else: esn.Wr = data['Wr']
        
        esn.Wx, esn.b, esn.r = data['Wx'], data['b'], data['r']
        esn._use_sparse = config.sparsity > 0.95
        
        # output weights 
        if 'Wout' in data:
            esn.Wout, esn.Wout_bias = data['Wout'], data['Wout_bias']
        else: esn.Wout, esn.Wout_bias = None, None
        
        return esn