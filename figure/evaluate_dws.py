from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rc import ESN 

INPUT_DIM = 5
WARMUP_LEN = 2000
PRED_STEPS = 5000
N_WINDOWS = 10
N_PROJ = 50
CRAWL = ROOT / "data_eigenworm/crawl.mat"
ESNS_ROOT = ROOT / "esns"


def filter_nans(pcs):
    nan_mask = np.isnan(pcs).any(axis=0)
    return pcs[:, ~nan_mask]


def process_worm(path, number):
    with h5py.File(path, "r") as f:
        tr_refs = f["tr"][number]
        tr_data = [f[ref][:] for ref in tr_refs.flatten()]
    all_pcs = np.hstack([filter_nans(d[:INPUT_DIM, :]) for d in tr_data])
    return StandardScaler().fit_transform(all_pcs.T).T


def make_projections(d, n_proj, rng):
    blocks = []
    for _ in range(0, n_proj, d):
        Z = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(Z)
        blocks.append(Q)
    return np.hstack(blocks)[:, :n_proj]


def sw_mean_max(ref, traj, projections, p: int = 2):
    """ref, traj: (T, D). projections: (D, n_proj). Returns (mean_sw, max_sw)."""
    rp = np.sort(ref @ projections, axis=0)
    tp = np.sort(traj @ projections, axis=0)
    w_p = np.mean(np.abs(rp - tp) ** p, axis=0)  # W_p^p per projection
    return float(np.mean(w_p) ** (1.0 / p)), float(np.max(w_p) ** (1.0 / p))


def eval_one_esn(esn_path, worm, starts, projections):
    stem = esn_path.stem
    t_load0 = time.perf_counter()
    esn = ESN.load(str(esn_path))
    if hasattr(esn.Wr, "tocsc"):
        esn.Wr = esn.Wr.tocsc()
    load_t = time.perf_counter() - t_load0

    means = np.zeros(len(starts))
    maxs = np.zeros(len(starts))
    pred_times = np.zeros(len(starts))
    for j, s in enumerate(starts):
        t0 = time.perf_counter()
        warmup = worm[:INPUT_DIM, s:s + WARMUP_LEN]
        ref = worm[:INPUT_DIM, s + WARMUP_LEN:s + WARMUP_LEN + PRED_STEPS]
        pred, _ = esn.predict(warmup, steps=PRED_STEPS, return_states=False)
        means[j], maxs[j] = sw_mean_max(ref.T, pred.T, projections)
        pred_times[j] = time.perf_counter() - t0
    return stem, means, maxs, pred_times, load_t


def main():
    proj_rng = np.random.default_rng(0)
    projections = make_projections(INPUT_DIM, N_PROJ, proj_rng)

    for n in range(12):
        worm_dir = ESNS_ROOT / f"worm_{n}"
        esn_files = sorted(worm_dir.glob("esn_*.npz"),
                           key=lambda p: int(p.stem.split("_")[1]))
        if not esn_files:
            print(f"worm {n}: no ESNs found, skipping")
            continue

        worm = process_worm(str(CRAWL), n)
        T = worm.shape[1]
        usable = T - WARMUP_LEN - PRED_STEPS
        if usable <= 0:
            print(f"worm {n}: too short (T={T}), skipping")
            continue
        starts = np.linspace(0, usable, N_WINDOWS).astype(int)
        worm_t0 = time.perf_counter()
        results = Parallel(n_jobs=-2)(
            delayed(eval_one_esn)(f, worm, starts, projections)
            for f in esn_files
        )

        stems = [r[0] for r in results]
        order = np.argsort([int(s.split("_")[1]) for s in stems])
        esn_indices = np.array([int(stems[k].split("_")[1]) for k in order])
        means = np.array([results[k][1] for k in order])
        maxs = np.array([results[k][2] for k in order])

        out = worm_dir / "dws.npz"
        np.savez(out, esn_indices=esn_indices, means=means, maxs=maxs, starts=starts)
        total_t = time.perf_counter() - worm_t0
        print(
            f"worm {n}: {len(esn_files)} ESNs evaluated in "
            f"{timedelta(seconds=int(total_t))}, "
            f"mean-SW median={np.median(means):.3f}",
        )


if __name__ == "__main__":
    main()
