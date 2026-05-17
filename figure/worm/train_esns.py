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

INPUT_DIM = 5
CRAWL = ROOT / "data_eigenworm/crawl.mat"
ESNS = ROOT / "esns"

N_ESNS_PER_WORM = 100
WASHOUT = 2000
N_JOBS = 8

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


def train_and_save(esn_idx, worm, out_dir, n):
    path = out_dir / f"esn_{esn_idx}.npz"
    if path.exists():
        return (esn_idx, "skip")
    try:
        cfg = replace(CONFIG, seed=n * 100 + esn_idx)
        esn = ESN(cfg)
        esn.train(worm[:INPUT_DIM, :], washout=WASHOUT)
        esn.save(str(path))
        return (esn_idx, "ok")
    except Exception as e:
        return (esn_idx, f"fail: {type(e).__name__}: {e}")


def main():
    for n in range(12):
        worm = process_worm(str(CRAWL), n)
        out_dir = ESNS / f"worm_{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        results = Parallel(n_jobs=N_JOBS)(
            delayed(train_and_save)(i, worm, out_dir, n)
            for i in range(N_ESNS_PER_WORM)
        )
        n_ok = sum(1 for _, s in results if s == "ok")
        n_skip = sum(1 for _, s in results if s == "skip")
        n_fail = sum(1 for _, s in results if s.startswith("fail"))
        elapsed = timedelta(seconds=int(time.perf_counter() - t0))
        print(f"worm {n}: {n_ok} trained, {n_skip} skipped, {n_fail} failed "
              f"(T={worm.shape[1]}, {elapsed})")
        for idx, s in results:
            if s.startswith("fail"):
                print(f"  esn_{idx}: {s}")


if __name__ == "__main__":
    main()
