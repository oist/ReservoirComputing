from __future__ import annotations

import sys
import time
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
CRAWL = ROOT / "data_eigenworm/crawl.mat"
OUT_DIR = ROOT / "esns" / f"worm_{WORM}" / "lyap_resampled"

SEEDS = range(100, 200)
WASHOUT = 2000
N_JOBS = 6

CONFIG = dict(
    N=10000,
    input_dim=INPUT_DIM,
    spectral_radius=0.1809616855907689,
    alpha=1.5,
    sparsity=0.99,
    input_scaling=1.2665236214415563,
    bias_scaling=0.01,
    mode="leaky",
    leaky_rate=0.7057809844406092,
)

LYAP_KWARGS = dict(
    num_lyaps=20,
    steps=2500,
    norm_time=1,
    dt=2 / 16,
    num_samples=15,
    warmup=500,
    transient=500,
    calculate_convergence=False,
    n_jobs=1,
)


def filter_nans(pcs):
    return pcs[:, ~np.isnan(pcs).any(axis=0)]


def load_worm(path, number):
    with h5py.File(path, "r") as f:
        tr_refs = f["tr"][number]
        tr_data = [f[ref][:] for ref in tr_refs.flatten()]
    pcs = np.hstack([filter_nans(d[:INPUT_DIM, :]) for d in tr_data])
    return StandardScaler().fit_transform(pcs.T).T


def compute_one(seed, worm_r, out_path):
    if out_path.exists():
        return (seed, "skip")
    try:
        esn = ESN(ESNConfig(seed=seed, **CONFIG))
        esn.train(worm_r[:INPUT_DIM, :], washout=WASHOUT)
        spectrum = esn.lyapunov_spectrum(worm_r[:INPUT_DIM, WASHOUT:], **LYAP_KWARGS)
        np.savez(
            out_path,
            all_samples=spectrum["all_samples"],
            max_lyapunov=spectrum["max_lyapunov"],
            num_valid_samples=spectrum["num_valid_samples"],
            distances=np.asarray(spectrum["distances"]),
        )
        return (seed, f"ok λ1={spectrum['max_lyapunov']:.3f}")
    except Exception as e:
        return (seed, f"fail: {type(e).__name__}: {e}")


def main():
    worm = load_worm(str(CRAWL), WORM)
    worm_r = worm[:, ::2]
    print(f"worm {WORM}: original {worm.shape}, resampled {worm_r.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(SEEDS)
    print(f"computing {len(seeds)} reservoirs (seeds {seeds[0]}..{seeds[-1]}) "
          f"with N_JOBS={N_JOBS}")

    t0 = time.perf_counter()
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(compute_one)(seed, worm_r, OUT_DIR / f"esn_{i}.npz")
        for i, seed in enumerate(seeds)
    )
    elapsed = timedelta(seconds=int(time.perf_counter() - t0))

    n_ok = sum(1 for _, s in results if s.startswith("ok"))
    n_skip = sum(1 for _, s in results if s == "skip")
    n_fail = sum(1 for _, s in results if s.startswith("fail"))
    print(f"worm {WORM}: {n_ok} computed, {n_skip} skipped, {n_fail} failed ({elapsed})")
    for seed, s in results:
        if s.startswith("fail"):
            print(f"  seed {seed}: {s}")


if __name__ == "__main__":
    main()
