import numpy as np
from numpy.typing import NDArray

from rc.esn import ESN


def valid_prediction_time(true_data, predictions, threshold=0.4, dt=0.01):
    """Compute valid prediction time before trajectory diverges.

    Parameters
    ----------
    true_data : ndarray of shape (D, T)
        Ground truth trajectory.
    predictions : ndarray of shape (D, T)
        Predicted trajectory.
    threshold : float, default=0.4
        Normalized squared error threshold for divergence.
    dt : float, default=0.01
        Time step for converting steps to time.

    Returns
    -------
    vpt : float
        Valid prediction time in time units.
    """
    variance = np.var(np.sum(true_data, axis=0))
    squared_diff = np.sum((true_data - predictions) ** 2, axis=0)
    normalized_error = squared_diff / variance

    divergence_idx = np.where(normalized_error > threshold)[0]
    valid_steps = (
        divergence_idx[0] if len(divergence_idx) > 0 else len(normalized_error)
    )

    return valid_steps * dt


def calculate_max_conditional_lyapunov_exponent(
    esn: ESN, data: NDArray, dt: float, length: int
) -> float:
    """Calculate the max conditional Lyapunov exponent of the ESN.

    Parameters
    ----------
    esn : ESN
        Trained ESN model.
    data : NDArray
        Data of shape (input_dim, T).
    dt : float
        Time step for continuous-time conversion.
    length : int
        Length of data segment to use.

    Returns
    -------
    float
        Max conditional Lyapunov exponent.
    """
    if not isinstance(esn, ESN):
        raise TypeError(f"esn must be an ESN instance, got {type(esn).__name__}")
    if not esn.is_trained:
        raise RuntimeError(
            "ESN must be trained before computing conditional Lyapunov exponent. Call train() first."
        )

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D array of shape (input_dim, T), got shape {data.shape}"
        )
    if data.shape[0] != esn.input_dim:
        raise ValueError(
            f"data first dimension must match ESN input_dim ({esn.input_dim}), got {data.shape[0]}"
        )

    return esn.conditional_lyapunov_spectrum(
        data[:, :length], num_lyaps=1, norm_time=5, dt=dt
    )["exponents"][0]
