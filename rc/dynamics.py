from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Protocol, Callable, Literal, runtime_checkable
from scipy.stats import ortho_group


@runtime_checkable
class ReservoirDynamics(Protocol):
    """protocol for reservoir update dynamics"""

    def update(
        self, r: NDArray, z: NDArray, activation: Callable[[NDArray], NDArray]
    ) -> NDArray:
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

    def jacobian_update(
        self,
        delta: NDArray,
        r: NDArray,
        z: NDArray,
        Wr: NDArray,
        Wx: NDArray,
        Wout: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Propagate tangent vectors for Lyapunov computation.

        Parameters
        ----------
        delta : ndarray of shape (N, k)
            Tangent vectors to propagate.
        r : ndarray of shape (N,)
            Current reservoir state.
        z : ndarray of shape (N,)
            Pre-activation values.
        Wr : ndarray or csr_matrix of shape (N, N)
            Reservoir weight matrix.
        Wx : ndarray of shape (N, input_dim)
            Input weight matrix.
        Wout : ndarray of shape (input_dim, N)
            Output weight matrix.

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

    def conditional_jacobian_update(
        self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Propagate tangent vectors for conditional Lyapunov computation.

        The system is being driven by external data, so the output feedback is not included.
        """
        ...

    def conditional_jacobian_update_vector(
        self, g: NDArray, z: NDArray, Wr: NDArray
    ) -> NDArray:
        """Propagate tangent vectors for conditional max CLE computation."""
        ...


@dataclass
class StandardDynamics:
    """standard ESN dynamics: r = tanh(Wr @ r + Wx @ x + b)

    Examples
    --------
    >>> from rc import StandardDynamics, create_dynamics
    >>> dynamics = StandardDynamics()
    >>> dynamics = create_dynamics("standard", N=100)
    """

    __slots__ = ()

    def update(
        self, r: NDArray, z: NDArray, activation: Callable[[NDArray], NDArray]
    ) -> NDArray:
        return activation(z)

    def jacobian_update(
        self,
        delta: NDArray,
        r: NDArray,
        z: NDArray,
        Wr: NDArray,
        Wx: NDArray,
        Wout: NDArray,
    ) -> tuple[NDArray, NDArray]:
        s = np.tanh(z)
        D = 1.0 - s**2
        J_delta = Wr @ delta + Wx @ (Wout @ delta)
        delta_new = D[:, np.newaxis] * J_delta
        return delta_new, s

    def get_params(self) -> dict:
        return {"mode": "standard"}

    @classmethod
    def from_params(cls, params: dict) -> "StandardDynamics":
        return cls()

    def conditional_jacobian_update(
        self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray
    ) -> tuple[NDArray, NDArray]:
        """jacobian for driven dynamics."""
        s = np.tanh(z)
        D = 1.0 - s**2
        delta_new = D[:, np.newaxis] * (Wr @ delta)
        return delta_new, s

    def conditional_jacobian_update_vector(self, g: NDArray, z: NDArray, Wr) -> NDArray:
        D = 1.0 - np.tanh(z) ** 2
        Wr_g = Wr.dot(g) if hasattr(Wr, "dot") else Wr @ g
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
    >>> from rc import LeakyDynamics, create_dynamics
    >>> N = 100
    >>> dynamics = LeakyDynamics(leaky_rate=np.full(N, 0.3))

    Create with per-neuron random leak rates:

    >>> dynamics = LeakyDynamics(leaky_rate=np.random.uniform(0.1, 0.5, N))

    Use via create_dynamics helper:

    >>> rng = np.random.default_rng(42)
    >>> dynamics = create_dynamics("leaky", N=100, leaky_rate=0.2, rng=rng)
    """

    __slots__ = ("leaky_rate", "keep_rate")
    leaky_rate: NDArray

    def __post_init__(self):
        self.keep_rate = 1.0 - self.leaky_rate

    def update(
        self, r: NDArray, z: NDArray, activation: Callable[[NDArray], NDArray]
    ) -> NDArray:
        return self.keep_rate * r + self.leaky_rate * activation(z)

    def jacobian_update(
        self,
        delta: NDArray,
        r: NDArray,
        z: NDArray,
        Wr: NDArray,
        Wx: NDArray,
        Wout: NDArray,
    ) -> tuple[NDArray, NDArray]:
        s = np.tanh(z)
        D = 1.0 - s**2
        J_delta = Wr @ delta + Wx @ (Wout @ delta)
        delta_new = self.keep_rate[:, np.newaxis] * delta + self.leaky_rate[
            :, np.newaxis
        ] * (D[:, np.newaxis] * J_delta)
        r_new = self.keep_rate * r + self.leaky_rate * s
        return delta_new, r_new

    def get_params(self) -> dict:
        return {"mode": "leaky", "leaky_rate": self.leaky_rate}

    @classmethod
    def from_params(cls, params: dict) -> "LeakyDynamics":
        return cls(leaky_rate=params["leaky_rate"])

    def conditional_jacobian_update(
        self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray
    ) -> tuple[NDArray, NDArray]:
        """jacobian for driven dynamics."""
        s = np.tanh(z)
        D = 1.0 - s**2
        delta_new = self.keep_rate[:, np.newaxis] * delta + self.leaky_rate[
            :, np.newaxis
        ] * (D[:, np.newaxis] * (Wr @ delta))
        r_new = self.keep_rate * r + self.leaky_rate * s
        return delta_new, r_new

    def conditional_jacobian_update_vector(self, g: NDArray, z: NDArray, Wr) -> NDArray:
        D = 1.0 - np.tanh(z) ** 2
        Wr_g = Wr.dot(g) if hasattr(Wr, "dot") else Wr @ g
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
    >>> from rc import ES2NDynamics, create_dynamics
    >>> N = 100
    >>> O = ortho_group.rvs(N)
    >>> dynamics = ES2NDynamics(beta=np.full(N, 0.5), O=O)

    Use via create_dynamics helper (recommended):

    >>> rng = np.random.default_rng(42)
    >>> dynamics = create_dynamics("es2n", N=100, beta=0.5, rng=rng)
    """

    __slots__ = ("beta", "O", "keep_rate")
    beta: NDArray
    O: NDArray

    def __post_init__(self):
        self.keep_rate = 1.0 - self.beta

    def update(
        self, r: NDArray, z: NDArray, activation: Callable[[NDArray], NDArray]
    ) -> NDArray:
        return self.beta * activation(z) + self.keep_rate * (self.O @ r)

    def jacobian_update(
        self,
        delta: NDArray,
        r: NDArray,
        z: NDArray,
        Wr: NDArray,
        Wx: NDArray,
        Wout: NDArray,
    ) -> tuple[NDArray, NDArray]:
        s = np.tanh(z)
        D = 1.0 - s**2
        J_delta = Wr @ delta + Wx @ (Wout @ delta)
        nonlin_term = self.beta[:, np.newaxis] * (D[:, np.newaxis] * J_delta)
        lin_term = self.keep_rate[:, np.newaxis] * (self.O @ delta)
        delta_new = nonlin_term + lin_term
        r_new = self.beta * s + self.keep_rate * (self.O @ r)
        return delta_new, r_new

    def get_params(self) -> dict:
        return {"mode": "es2n", "beta": self.beta, "O": self.O}

    @classmethod
    def from_params(cls, params: dict) -> "ES2NDynamics":
        return cls(beta=params["beta"], O=params["O"])

    def conditional_jacobian_update(
        self, delta: NDArray, r: NDArray, z: NDArray, Wr: NDArray
    ) -> tuple[NDArray, NDArray]:
        """jacobian for driven dynamics."""
        s = np.tanh(z)
        D = 1.0 - s**2
        nonlin_term = self.beta[:, np.newaxis] * (D[:, np.newaxis] * (Wr @ delta))
        lin_term = self.keep_rate[:, np.newaxis] * (self.O @ delta)
        delta_new = nonlin_term + lin_term
        r_new = self.beta * s + self.keep_rate * (self.O @ r)
        return delta_new, r_new

    def conditional_jacobian_update_vector(self, g: NDArray, z: NDArray, Wr) -> NDArray:
        D = 1.0 - np.tanh(z) ** 2
        Wr_g = Wr.dot(g) if hasattr(Wr, "dot") else Wr @ g
        return self.beta * (D * Wr_g) + self.keep_rate * (self.O @ g)


def create_dynamics(
    mode: Literal["standard", "leaky", "leakyrand", "es2n", "es2nrand"] | str,
    N: int,
    dtype: np.dtype = np.float64,
    leaky_rate: float | NDArray = 0.1,
    beta: float | NDArray = 0.5,
    scale: float = 0.1,
    rng: np.random.Generator = None,
) -> ReservoirDynamics:
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
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError(f"N must be a positive integer, got {N}")

    if not isinstance(scale, (int, float)) or scale < 0:
        raise ValueError(f"scale must be non-negative, got {scale}")

    if mode in ("leakyrand", "es2nrand", "es2n") and rng is None:
        raise ValueError(f"rng (random number generator) is required for mode '{mode}'")

    if mode == "standard":
        return StandardDynamics()
    elif mode == "leaky":
        if np.isscalar(leaky_rate):
            if not (0 < leaky_rate <= 1):
                raise ValueError(f"leaky_rate must be in (0, 1], got {leaky_rate}")
            lr = np.full(N, leaky_rate, dtype=dtype)
        else:
            lr = np.asarray(leaky_rate, dtype=dtype)
            if lr.shape != (N,):
                raise ValueError(
                    f"leaky_rate array must have shape ({N},), got {lr.shape}"
                )
            if np.any(lr <= 0) or np.any(lr > 1):
                raise ValueError("all leaky_rate values must be in (0, 1]")
        return LeakyDynamics(leaky_rate=np.clip(lr, 0, 1))
    elif mode == "es2n":
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
        lr = rng.uniform(
            max(leaky_rate - scale, 0), min(leaky_rate + scale, 1), N
        ).astype(dtype)
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
