from __future__ import annotations

import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rc import ESN, ESNConfig

WORM = 1
INPUT_DIM = 5
WARMUP_LEN = 2000
PRED_STEPS = 5000
N_WINDOWS = 10
N_PROJ = 50

CRAWL = ROOT / "data_eigenworm/crawl.mat"
OUT = ROOT / "esns" / f"worm_{WORM}" / "dws_resampled.npz"

SEEDS = range(100, 200)
WASHOUT = 2000
N_JOBS = 6

CONFIG = ESNConfig(
    N=10000,
    input_dim=INPUT_DIM,
    spectral_radius=0.1809616855907689,
    alpha=1.5,
    sparsity=0.99,
    input_scaling=1.2665236214415563,
    bias_scaling=0.01,
    mode="leaky",
    leaky_rate=0.7057809844406092,
    seed=42,
)


def filter_nans(pcs):
    return pcs[:, ~np.isnan(pcs).any(axis=0)]


def process_worm(path, number):
    with h5py.File(path, "r") as f:
        tr_refs = f["tr"][number]
        tr_data = [f[ref][:] for ref in tr_refs.flatten()]
    pcs = np.hstack([filter_nans(d[:INPUT_DIM, :]) for d in tr_data])
    return StandardScaler().fit_transform(pcs.T).T


def make_projections(d, n_proj, rng):
    blocks = []
    for _ in range(0, n_proj, d):
        Z = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(Z)
        blocks.append(Q)
    return np.hstack(blocks)[:, :n_proj]


def sw_mean_max(ref, traj, projections, p=2):
    rp = np.sort(ref @ projections, axis=0)
    tp = np.sort(traj @ projections, axis=0)
    w_p = np.mean(np.abs(rp - tp) ** p, axis=0)
    return float(np.mean(w_p) ** (1.0 / p)), float(np.max(w_p) ** (1.0 / p))


def eval_one(esn_idx, seed, worm_r, starts, projections):
    try:
        esn = ESN(replace(CONFIG, seed=seed))
        esn.train(worm_r[:INPUT_DIM, :], washout=WASHOUT)
        if hasattr(esn.Wr, "tocsc"):
            esn.Wr = esn.Wr.tocsc()
        means = np.zeros(len(starts))
        maxs = np.zeros(len(starts))
        for j, s in enumerate(starts):
            warmup = worm_r[:INPUT_DIM, s:s + WARMUP_LEN]
            ref = worm_r[:INPUT_DIM, s + WARMUP_LEN:s + WARMUP_LEN + PRED_STEPS]
            pred, _ = esn.predict(warmup, steps=PRED_STEPS, return_states=False)
            means[j], maxs[j] = sw_mean_max(ref.T, pred.T, projections)
        return (esn_idx, means, maxs, "ok")
    except Exception as e:
        return (esn_idx, None, None, f"fail: {type(e).__name__}: {e}")


def main():
    worm = process_worm(str(CRAWL), WORM)
    worm_r = worm[:, ::2]
    T = worm_r.shape[1]
    usable = T - WARMUP_LEN - PRED_STEPS
    if usable <= 0:
        raise SystemExit(f"worm {WORM}: resampled too short (T={T})")
    starts = np.linspace(0, usable, N_WINDOWS).astype(int)
    print(f"worm {WORM}: resampled T={T}, windows={N_WINDOWS}x"
          f"(warmup={WARMUP_LEN}+pred={PRED_STEPS}), starts={starts.tolist()}")

    proj_rng = np.random.default_rng(0)
    projections = make_projections(INPUT_DIM, N_PROJ, proj_rng)

    seeds = list(SEEDS)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(eval_one)(i, seed, worm_r, starts, projections)
        for i, seed in enumerate(seeds)
    )
    elapsed = timedelta(seconds=int(time.perf_counter() - t0))

    ok = [(i, m, mx) for i, m, mx, s in results if s == "ok"]
    fails = [(i, s) for i, m, mx, s in results if s.startswith("fail")]
    if fails:
        print(f"failures: {len(fails)}")
        for i, s in fails:
            print(f"  esn_{i}: {s}")

    ok.sort(key=lambda t: t[0])
    esn_indices = np.array([t[0] for t in ok])
    means = np.array([t[1] for t in ok])
    maxs = np.array([t[2] for t in ok])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, esn_indices=esn_indices, means=means, maxs=maxs, starts=starts)

    med_per_esn = np.median(means, axis=1)
    print(f"worm {WORM}: {len(ok)}/{len(seeds)} scored ({elapsed})")
    print(f"  mean-SW median across ESNs: {np.median(med_per_esn):.4f}  "
          f"(min={med_per_esn.min():.3f}, max={med_per_esn.max():.3f})")
    print(f"  saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
