import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def participation_ratio(explained_variance: NDArray) -> float:
    """Compute the participation ratio (effective dimension) from PCA variances.

    Defined as ``(sum(v))**2 / sum(v**2)`` where ``v`` is the per-component
    explained variance. Equals the number of components when variance is
    uniformly distributed, and 1 when a single component dominates.

    Parameters
    ----------
    explained_variance : array-like, 1D
        Per-component explained variance (or explained-variance ratios) of
        a PCA decomposition. All entries must be non-negative, and at least
        one entry must be positive.

    Returns
    -------
    participation_ratio : float
        Effective number of significantly contributing components.
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
    """Run PCA on a reservoir trajectory and report its effective dimension.

    Parameters
    ----------
    rc_trajectory : array-like
        Reservoir trajectory of shape ``(N, T)``, where ``N`` is the reservoir
        size and ``T`` is the number of timesteps. Must have ``T >= 2``.
    pca_components : int or float, default=0.95
        Forwarded to :class:`sklearn.decomposition.PCA` as ``n_components``.
        If an int, the number of components to keep. If a float in ``(0, 1]``,
        the minimum cumulative explained-variance ratio to retain.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'effective_dim'`` (float): participation ratio over the retained
          explained-variance ratios.
        - ``'explained_variance'`` (ndarray): explained-variance ratio per
          retained component.
        - ``'pca_scores'`` (ndarray of shape ``(T, n_components)``): projection
          of the trajectory onto the retained PCA components.
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
