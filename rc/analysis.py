import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


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
        raise ValueError(
            f"explained_variance must be 1D array, got shape {explained_variance.shape}"
        )
    if len(explained_variance) == 0:
        raise ValueError("explained_variance cannot be empty")
    if np.any(explained_variance < 0):
        raise ValueError("explained_variance values must be non-negative")

    sum_sq = np.sum(explained_variance**2)
    if sum_sq == 0:
        raise ValueError("explained_variance cannot be all zeros")

    return np.sum(explained_variance) ** 2 / sum_sq


def analyse_dynamics(
    rc_trajectory: NDArray, pca_components: int | float = 0.95
) -> dict:
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
        raise ValueError(
            f"rc_trajectory must be 2D array of shape (N, T), got shape {rc_trajectory.shape}"
        )
    if rc_trajectory.shape[0] == 0 or rc_trajectory.shape[1] == 0:
        raise ValueError(
            f"rc_trajectory cannot have zero-length dimensions, got shape {rc_trajectory.shape}"
        )
    if rc_trajectory.shape[1] < 2:
        raise ValueError(
            f"rc_trajectory must have at least 2 timesteps for PCA, got {rc_trajectory.shape[1]}"
        )

    pca = PCA(n_components=pca_components)
    pca.fit(rc_trajectory.T)
    pca_scores = pca.transform(rc_trajectory.T)
    explained_variance = pca.explained_variance_ratio_
    effective_dim = participation_ratio(explained_variance)
    return {
        "effective_dim": effective_dim,
        "explained_variance": explained_variance,
        "pca_scores": pca_scores,
    }
