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
from rc import ESN

INPUT_DIM = 5
CRAWL = ROOT / "data_eigenworm/crawl.mat"
ESNS = ROOT / "esns"

M_PIPELINE = 40
N_BOOTSTRAP = 2000
SEED = 42
N_JOBS = 8

LYAP_KWARGS = dict(
    num_lyaps=20,
    steps=2500,
    norm_time=2,
    dt=1 / 16,
    num_samples=15,
    warmup=500,
    transient=500,
    calculate_convergence=False,
    n_jobs=1,
)


def filter_nans(pcs):
    nan_mask = np.isnan(pcs).any(axis=0)
    return pcs[:, ~nan_mask]


def process_worm(path, number):
    with h5py.File(path, "r") as f:
        tr_refs = f["tr"][number]
        tr_data = [f[ref][:] for ref in tr_refs.flatten()]
    all_pcs = np.hstack([filter_nans(d[:INPUT_DIM, :]) for d in tr_data])
    return StandardScaler().fit_transform(all_pcs.T).T


def identify_winners(n):
    dws_path = ESNS / f"worm_{n}" / "dws.npz"
    if not dws_path.exists():
        return None
    d = np.load(dws_path)
    scores = np.median(d["means"], axis=1)
    idx = d["esn_indices"]
    M = len(scores)
    rng = np.random.default_rng(SEED + n)
    winners = set()
    for _ in range(N_BOOTSTRAP):
        if M_PIPELINE >= M:
            samp = rng.integers(0, M, size=M_PIPELINE)
        else:
            samp = rng.choice(M, size=M_PIPELINE, replace=False)
        j = samp[np.argmin(scores[samp])]
        winners.add(int(idx[j]))
    return sorted(winners)


def compute_one(esn_path, initial_data, out_path):
    if out_path.exists():
        return (esn_path.stem, "skip")
    try:
        esn = ESN.load(str(esn_path))
        spectrum = esn.lyapunov_spectrum(initial_data, **LYAP_KWARGS)
        np.savez(
            out_path,
            all_samples=spectrum["all_samples"],
            max_lyapunov=spectrum["max_lyapunov"],
            num_valid_samples=spectrum["num_valid_samples"],
            distances=np.array(spectrum["distances"]),
        )
        return (esn_path.stem, f"ok λ1={spectrum['max_lyapunov']:.3f}")
    except Exception as e:
        return (esn_path.stem, f"fail: {type(e).__name__}: {e}")


def main():
    for n in range(12):
        winners = identify_winners(n)
        if not winners:
            continue
        worm = process_worm(str(CRAWL), n)
        initial_data = worm[:INPUT_DIM, :]
        T = initial_data.shape[1]
        needed = LYAP_KWARGS["warmup"] * LYAP_KWARGS["num_samples"] + LYAP_KWARGS["steps"]
        if T < needed:
            continue
        out_dir = ESNS / f"worm_{n}" / "lyap_top"
        out_dir.mkdir(exist_ok=True)
        esn_files = [ESNS / f"worm_{n}" / f"esn_{i}.npz" for i in winners]
        esn_files = [f for f in esn_files if f.exists()]
        t0 = time.perf_counter()
        results = Parallel(n_jobs=N_JOBS)(
            delayed(compute_one)(f, initial_data, out_dir / f"{f.stem}.npz")
            for f in esn_files
        )
        n_ok = sum(1 for _, s in results if s.startswith("ok"))
        n_skip = sum(1 for _, s in results if s == "skip")
        elapsed = timedelta(seconds=int(time.perf_counter() - t0))
        print(f"worm {n}: {n_ok} computed, {n_skip} skipped ({elapsed})")


if __name__ == "__main__":
    main()
