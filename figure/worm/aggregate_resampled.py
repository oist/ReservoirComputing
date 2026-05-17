from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPECTRA = ROOT / "esns" / "worm_1" / "lyap_resampled"
OUT = ROOT / "results_data" / "spectrum_resampled.pkl"


def main():
    files = sorted(SPECTRA.glob("esn_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"no spectra found in {SPECTRA}")

    samples = []
    distances = []
    valid = 0
    for f in files:
        d = np.load(f)
        samples.append(d["all_samples"])
        distances.append(np.asarray(d["distances"]))
        valid += int(d["num_valid_samples"])

    all_samples = np.vstack(samples)
    mean = np.mean(all_samples, axis=0)

    spectrum = dict(
        mean=mean,
        std=np.std(all_samples, axis=0),
        all_samples=all_samples,
        convergence=None,
        num_valid_samples=valid,
        max_lyapunov=float(np.max(mean)),
        distances=np.concatenate(distances),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(spectrum, f)

    print(f"aggregated {len(files)} reservoirs → all_samples={all_samples.shape}  ")
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
