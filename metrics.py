import numpy as np
from numpy.typing import NDArray
import ot
from scipy.spatial import distance
from numba import jit, prange
from numpy.random import choice
import scipy
from typing import Callable, Any
from tqdm import tqdm
from esn import logger


def valid_prediction_time(true_data, predictions, threshold=0.4, dt=0.01):
    """
    Compute valid prediction time before trajectory diverges.
    
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
    squared_diff = np.sum((true_data - predictions)**2, axis=0)
    normalized_error = squared_diff / variance
    
    divergence_idx = np.where(normalized_error > threshold)[0]
    valid_steps = divergence_idx[0] if len(divergence_idx) > 0 else len(normalized_error)
    
    return valid_steps * dt


def sliced_wasserstein(true_data, predictions, n_projections=50):
    """
    Compute sliced Wasserstein distance between trajectories.
    
    Parameters
    ----------
    true_data : ndarray of shape (T, D)
        Ground truth trajectory (time × dimensions).
    predictions : ndarray of shape (T, D)
        Predicted trajectory (time × dimensions).
    n_projections : int, default=50
        Number of random projections.
    
    Returns
    -------
    distance : float
        Sliced Wasserstein distance.
    """
    return ot.sliced_wasserstein_distance(true_data, predictions, n_projections=n_projections)


def max_sliced_wasserstein(true_data, predictions, n_projections=50):
    """Max sliced Wasserstein distance (adversarial projection)."""
    return ot.sliced.max_sliced_wasserstein_distance(true_data, predictions, n_projections=n_projections)


def integral_timescale(trajectory, max_lag=80, dt=1/16):
    """
    Compute integral timescale from autocorrelation.
    
    Parameters
    ----------
    trajectory : ndarray of shape (D, T)
        Trajectory with dimensions × time.
    max_lag : int, default=80
        Maximum lag for autocorrelation.
    dt : float, default=1/16
        Time step.
    
    Returns
    -------
    tau : ndarray of shape (D,)
        Integral timescale per dimension.
    """
    D, N = trajectory.shape
    tau = np.zeros(D)
    
    for d in range(D):
        x = trajectory[d] - trajectory[d].mean()
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[N-1:N-1+max_lag]
        autocorr = autocorr / autocorr[0]
        tau[d] = np.trapz(np.abs(autocorr), np.arange(max_lag) * dt)
    
    return tau


def count_radius(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def sample_data(trajectory, N, sample_size):
    n_points = trajectory.shape[1]
    sampled_indices = choice(n_points, size=N, replace=False)
    sample = trajectory[:, sampled_indices]
    return sample, sampled_indices


def distance_matrix_scipy(sample, trajectory):
    return distance.cdist(sample.T, trajectory.T, metric='euclidean')


@jit(nopython=True, parallel=True)
def count_points_in_radius_numba(distances, radius_array, sample_indices):
    N, total_points = distances.shape
    radius_len = len(radius_array)
    n_inside = np.zeros((radius_len, N))
    for k in prange(radius_len):
        radi = radius_array[k]
        for i in prange(N):
            count = 0
            sample_idx = sample_indices[i]
            for j in range(total_points):
                if j != sample_idx and distances[i, j] < radi:
                    count += 1
            n_inside[k, i] = count
    return n_inside


def compute_slope(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sum((x - mean_x) ** 2)
    slope = numerator / denominator
    return slope


def compute_intercept(t, log_error, slope):
    mean_t = np.mean(t)
    mean_log_error = np.mean(log_error)
    intercept = mean_log_error - slope * mean_t
    return intercept


def corr_funct(trajectory, N, radius, line=False):
    radius = np.array(radius)
    if not line:
        start_five = int(round(trajectory.shape[1] * 0.05))
        end_five = int(round(trajectory.shape[1] * 0.95))
        sample, points = sample_data(trajectory[:, start_five:end_five], N, 20)
    else:
        mask = (trajectory[0, :] > 50) & (trajectory[0, :] < 150)
        filtered_line = trajectory[:, mask]
        sample, points = sample_data(trajectory, N, 20)
    total_points = trajectory.shape[1] - 1
    distances = distance_matrix_scipy(sample, trajectory)
    n_inside = count_points_in_radius_numba(distances, radius, points)
    return n_inside, total_points, sample


def bootstrap_samples(N, radius, v, samples, line=False):
    sampl_slope = np.zeros(samples)
    sampl_intercept = np.zeros(samples)
    all_sampled_points = []
    sample_of_averages = []
    for i in range(samples):
        n_inside, total, points = corr_funct(v.T, N, radius, line)
        all_sampled_points.append(points)
        normalised = n_inside / total
        avg_vec = np.mean(normalised, axis=1)
        sample_of_averages.append(avg_vec)
        x = np.log(radius)
        y = np.log(avg_vec)
        slope = compute_slope(x, y)
        intercept = compute_intercept(x, y, slope)
        sampl_slope[i] = slope
        sampl_intercept[i] = intercept
    return sampl_slope, sampl_intercept, all_sampled_points, sample_of_averages


def calculate_correlation_distance(data, start_radius, end_radius, samples, line=False):
    N = 100
    radius = np.logspace(np.log10(start_radius), np.log10(end_radius), samples)
    sampl_slope, sampl_intercept, all_sampled_points, sample_of_averages = bootstrap_samples(N, radius, data.T, samples, line)
    mean = np.mean(sample_of_averages, axis=0)
    std = np.std(sample_of_averages, axis=0)
    ci_lower = mean - 1.96 * std/np.sqrt(100)
    ci_upper = mean + 1.96 * std/np.sqrt(100)
    return sampl_slope, sampl_intercept, all_sampled_points, sample_of_averages, mean, std, ci_lower, ci_upper


    return sampl_slope, sampl_intercept, all_sampled_points, sample_of_averages, mean, std, ci_lower, ci_upper


def correlation_dimension_error(true_data, predictions, start_radius=1e-2, end_radius=10, samples=20):
    """ 
    сompute error in correlation dimension between ground truth and prediction.
    
    Parameters
    ----------
    true_data : ndarray of shape (T, D)
    predictions : ndarray of shape (T, D)
    
    Returns
    -------
    error : float
        Absolute difference in mean correlation dimension.
    """
    # Calculate for true data
    *_, mean_true, _, _, _ = calculate_correlation_distance(
        true_data, start_radius, end_radius, samples
    )
    # Calculate for predictions
    *_, mean_pred, _, _, _ = calculate_correlation_distance(
        predictions, start_radius, end_radius, samples
    )
    
    # Compare the mean slopes (dimensions)
    mean_slope_true = np.mean(mean_true)
    mean_slope_pred = np.mean(mean_pred)
    
    return np.abs(mean_slope_true - mean_slope_pred)


class CovarianceDistance:
    """Static methods for covariance distance"""
    @staticmethod
    def inverse_square_root_matrix(matrix: np.ndarray) -> np.ndarray:
        """Inverse square root matrix."""
        if isinstance(matrix, np.ma.MaskedArray):
            matrix = matrix.data
        return scipy.linalg.inv(scipy.linalg.sqrtm(matrix))

    @staticmethod
    def positive_definite_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Positive definite distance."""
        sqrt_inv_matrix1 = CovarianceDistance.inverse_square_root_matrix(matrix1)
        M = sqrt_inv_matrix1 @ matrix2 @ sqrt_inv_matrix1
        M = (M + M.T) / 2
        eigenvalues = np.linalg.eigvalsh(M)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        return np.sqrt(np.sum(np.log(eigenvalues)**2))

    @staticmethod
    def euclidean_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Euclidean distance."""
        return float(np.linalg.norm(matrix1 - matrix2))

    @staticmethod
    def cosine_distance_matrices(A: np.ndarray, B: np.ndarray) -> float:
        """Cosine distance between matrices."""
        inner_product = np.sum(A * B)
        norm_A = np.sqrt(np.sum(A * A))
        norm_B = np.sqrt(np.sum(B * B))
        cosine_similarity = inner_product / (norm_A * norm_B)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance

    @staticmethod
    def log_euclidean_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Log-Euclidean distance."""
        return float(np.linalg.norm(np.log(matrix1 + 0.00001) - np.log(matrix2 + 0.00001)))

    @staticmethod
    def spatial_correlation_function(correlation_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Spatial correlation function."""
        n_segments = correlation_matrix.shape[0]
        distances = np.arange(0, n_segments)
        avg_correlations = []
        for d in distances:
            correlations_at_d = []
            for i in range(n_segments - d):
                correlations_at_d.append(correlation_matrix[i, i + d])
            avg_correlations.append(np.mean(correlations_at_d))
        return distances, np.array(avg_correlations)

    @staticmethod
    def create_distance_m(covariance_matrix_list: list[np.ndarray], distance_function: Callable) -> np.ndarray:
        """Create distance matrix."""
        n = len(covariance_matrix_list)
        distance_matrix = np.zeros((n, n))
        for i in tqdm(range(n), desc="Computing distance matrix"):
            for j in range(n):
                distance_matrix[i, j] = distance_function(covariance_matrix_list[i], covariance_matrix_list[j])
        return distance_matrix

    @staticmethod
    def compute_error(true_data, predictions, method='positive_definite'):
        """
        Compute covariance distance between two trajectories.
        
        Parameters
        ----------
        true_data : ndarray of shape (T, D)
        predictions : ndarray of shape (T, D)
        method : str, default='positive_definite'
            Method to use: 'positive_definite', 'euclidean', 'cosine', 'log_euclidean'.
            
        Returns
        -------
        distance : float
        """
        cov_true = np.cov(true_data.T)
        cov_pred = np.cov(predictions.T)
        
        if method == 'positive_definite': return CovarianceDistance.positive_definite_distance(cov_true, cov_pred)
        elif method == 'euclidean': return CovarianceDistance.euclidean_distance(cov_true, cov_pred)
        elif method == 'cosine':  return CovarianceDistance.cosine_distance_matrices(cov_true, cov_pred)
        elif method == 'log_euclidean': return CovarianceDistance.log_euclidean_distance(cov_true, cov_pred)
        else: raise ValueError(f"Unknown covariance distance method: {method}")


def covariance_distance_error(true_data: NDArray, predictions: NDArray, method: str = 'positive_definite') -> float:
    """Wrapper for CovarianceDistance error."""
    return CovarianceDistance.compute_error(true_data, predictions, method=method)


def mse(true_data, predictions):
    """Mean Squared Error."""
    return np.mean((true_data - predictions) ** 2)


def rmse(true_data, predictions):
    """Root Mean Squared Error."""
    return np.sqrt(mse(true_data, predictions))


def nrmse(true_data, predictions):
    """Normalized Root Mean Squared Error (normalized by std of ground truth)."""
    rmse_val = rmse(true_data, predictions)
    std = np.std(true_data)
    if std == 0: return rmse_val
    return rmse_val / std


def mae(true_data, predictions):
    """Mean Absolute Error."""
    return np.mean(np.abs(true_data - predictions))

METRICS = {}

def register(name: str, expects_TD: bool = True, maximize: bool = False):
    def decorator(fn):
        def fixed(gt, pred, **kwargs):
            if expects_TD:
                gt, pred = gt.T, pred.T
            return fn(gt, pred, **kwargs)
        
        fixed.maximize = maximize
        fixed.name = name
        METRICS[name] = fixed
        return fixed
    return decorator


@register("wasserstein", expects_TD=True)
def _wasserstein(gt: NDArray, pred: NDArray, n_projections: int = 50) -> float:
    """Wasserstein distance.
    
    Parameters
    ----------
    gt : ndarray of shape (D, T)
    pred : ndarray of shape (D, T)
    n_projections : int, default=50
        Number of projections.
        
    Returns
    -------
    distance : float
        Wasserstein distance.
    """
    logger.debug(f"Computing Wasserstein distance with {n_projections} projections")
    return ot.sliced_wasserstein_distance(gt, pred, n_projections=n_projections)

@register("max_wasserstein", expects_TD=True)
def _max_wasserstein(gt: NDArray, pred: NDArray, n_projections: int = 50) -> float:
    """Max Wasserstein distance.
    
    Parameters
    ----------
    gt : ndarray of shape (D, T)
    pred : ndarray of shape (D, T)
    n_projections : int, default=50
        Number of projections.
        
    Returns
    -------
    distance : float
        Max Wasserstein distance.
    """
    logger.debug(f"Computing Max Wasserstein distance with {n_projections} projections")
    return ot.sliced.max_sliced_wasserstein_distance(gt, pred, n_projections=n_projections)

@register("vpt", expects_TD=False, maximize=True)
def _vpt(gt: NDArray, pred: NDArray, threshold: float = 0.4, dt: float = 0.01) -> float:
    """Valid Prediction Time.
    
    Parameters
    ----------
    gt : ndarray of shape (D, T)
    pred : ndarray of shape (D, T)
    threshold : float, default=0.4
        Threshold for valid prediction time.
    dt : float, default=0.01
        Time step.
        
    Returns
    -------
    distance : float
        Valid Prediction Time.
    """
    logger.debug(f"Computing Valid Prediction Time with threshold {threshold} and dt {dt}")
    return valid_prediction_time(gt, pred, threshold, dt)

@register("mse", expects_TD=False)
def _mse(gt: NDArray, pred: NDArray) -> float:
    """Mean Squared Error.
    
    Parameters
    ----------
    gt : ndarray of shape (D, T)
    pred : ndarray of shape (D, T)
        
    Returns
    -------
    distance : float
        Mean Squared Error.
    """
    logger.debug(f"Computing Mean Squared Error")
    return np.mean((gt - pred) ** 2)

@register("rmse", expects_TD=False)
def _rmse(gt: NDArray, pred: NDArray) -> float:
    """Root Mean Squared Error.
    
    Parameters
    ----------
    gt : ndarray of shape (D, T)
    pred : ndarray of shape (D, T)
        
    Returns
    -------
    distance : float
        Root Mean Squared Error.
    """
    logger.debug(f"Computing Root Mean Squared Error")
    return np.sqrt(np.mean((gt - pred) ** 2))

@register("nrmse", expects_TD=False)
def _nrmse(gt: NDArray, pred: NDArray) -> float:
    """Normalized Root Mean Squared Error.
    
    Parameters
    ----------
    gt : ndarray of shape (D, T)
    pred : ndarray of shape (D, T)
        
    Returns
    -------
    distance : float
        Normalized Root Mean Squared Error.
    """
    logger.debug(f"Computing Normalized Root Mean Squared Error")
    return np.sqrt(np.mean((gt - pred) ** 2)) / np.std(gt)

@register("cov_distance", expects_TD=True)
def _cov_distance(gt: NDArray, pred: NDArray, method: str = 'positive_definite') -> float:
    """Covariance Distance.
    
    Parameters
    ----------
    gt : ndarray of shape (D, T)
    pred : ndarray of shape (D, T)
    method : str, default='positive_definite'
        Method to use: 'positive_definite', 'euclidean', 'cosine', 'log_euclidean'.
        
    Returns
    -------
    distance : float
        Covariance Distance.
    """
    logger.debug(f"Computing Covariance Distance with method {method}")
    return CovarianceDistance.compute_error(gt, pred, method)

@register("corr_dim", expects_TD=True)
def _corr_dim(gt: NDArray, pred: NDArray, start_radius: float = 1e-2, end_radius: float = 10, samples: int = 20) -> float:
    """Correlation Dimension Error.
    
    Parameters
    ----------
    gt : ndarray of shape (D, T)
    pred : ndarray of shape (D, T)
    start_radius : float, default=1e-2
    end_radius : float, default=10
        End radius.
    samples : int, default=20
        Number of samples.
        
    Returns
    -------
    distance : float
        Correlation Dimension Error.
    """
    logger.debug(f"Computing Correlation Dimension Error with start radius {start_radius}, end radius {end_radius}, and {samples} samples")
    return correlation_dimension_error(gt, pred, start_radius, end_radius, samples)

@register("timescale_error", expects_TD=False)
def _timescale_error(gt: NDArray, pred: NDArray, max_lag: int = 80, dt: float = 1/16) -> float:
    """Timescale Error.
    
    Parameters
    ----------
    gt : ndarray of shape (D, T)
    pred : ndarray of shape (D, T)
    max_lag : int, default=80
        Maximum lag.
    dt : float, default=1/16
        Time step.
        
    Returns
    -------
    distance : float
        Timescale Error.
    """
    logger.debug(f"Computing Timescale Error with max lag {max_lag} and dt {dt}")
    tau_gt = integral_timescale(gt, max_lag, dt)
    tau_pred = integral_timescale(pred, max_lag, dt)
    return np.linalg.norm(tau_gt - tau_pred)


def compute_metric(name: str, gt: NDArray, pred: NDArray, **kwargs: Any) -> float:
    """Compute metric.
    
    Parameters
    ----------
    name : str
        Metric name.
    gt : ndarray of shape (D, T)
        Ground truth trajectory.
    pred : ndarray of shape (D, T)
        Predicted trajectory.
    **kwargs : dict
        Keyword arguments for the metric.
        
    Returns
    -------
    value : float
        Metric value.
        
    Raises
    ------
    ValueError
        If the metric name is not found.
    """
    logger.debug(f"Computing metric {name} with kwargs {kwargs}")
    if name not in METRICS: raise ValueError(f"unknown metric: {name}. available: {list(METRICS.keys())}")
    return METRICS[name](gt, pred, **kwargs)