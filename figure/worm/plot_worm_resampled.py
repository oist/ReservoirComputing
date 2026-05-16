from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
WORM = 1
DWS = ROOT / "esns" / f"worm_{WORM}" / "dws_resampled.npz"
SPECTRA = ROOT / "esns" / f"worm_{WORM}" / "lyap_resampled"
OUT = ROOT / "figure" / "lyap_plots"

M_PIPELINE = 40
N_BOOTSTRAP_WINNERS = 2000
N_OUTER = 1000
SEED = 42


def load():
    dws = np.load(DWS)
    quality = np.median(dws["means"], axis=1)
    idx = dws["esn_indices"]
    trajs = {}
    for i, ei in enumerate(idx):
        f = SPECTRA / f"esn_{int(ei)}.npz"
        if not f.exists():
            continue
        s = np.sort(np.load(f)["all_samples"], axis=1)[:, ::-1]
        if s.size:
            trajs[int(ei)] = s
    common = sorted(set(int(i) for i in idx) & set(trajs))
    q_by_i = {int(i): float(q) for i, q in zip(idx, quality)}
    return dict(
        idx=np.array(common, dtype=int),
        quality=np.array([q_by_i[i] for i in common]),
        trajs=trajs,
    )


def identify_winners(data, rng):
    idx, quality = data["idx"], data["quality"]
    M = len(idx)
    counts = np.zeros(M, dtype=int)
    for _ in range(N_BOOTSTRAP_WINNERS):
        if M_PIPELINE >= M:
            samp = rng.integers(0, M, size=M_PIPELINE)
        else:
            samp = rng.choice(M, size=M_PIPELINE, replace=False)
        counts[samp[int(np.argmin(quality[samp]))]] += 1
    winners = sorted(int(idx[k]) for k in np.where(counts > 0)[0])
    return winners, counts


def nested_bootstrap(data, winners, rng):
    idx, quality, trajs = data["idx"], data["quality"], data["trajs"]
    pool_idx = np.array([i for i in idx if i in winners], dtype=int)
    quality_map = {int(i): float(q) for i, q in zip(idx, quality)}
    M = len(idx)
    num_lyaps = next(iter(trajs.values())).shape[1]
    boot = np.empty((N_OUTER, num_lyaps))
    for b in range(N_OUTER):
        if M_PIPELINE >= M:
            samp = rng.integers(0, M, size=M_PIPELINE)
        else:
            samp = rng.choice(M, size=M_PIPELINE, replace=False)
        chosen = int(idx[samp[int(np.argmin(quality[samp]))]])
        ts = trajs[chosen]
        resample = rng.integers(0, len(ts), size=len(ts))
        boot[b] = np.median(ts[resample], axis=0)
    return boot


def plot(boot, save_svg, save_pdf):
    median = np.median(boot, axis=0)
    ci_lo = np.percentile(boot, 2.5, axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)
    yerr = np.array([median - ci_lo, ci_hi - median])

    plt.figure()
    plt.errorbar(
        range(1, len(median) + 1), median,
        yerr=yerr, fmt="k.", ms=8, capsize=5, ecolor="gray",
    )
    plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
    plt.xticks([1, len(median)], ["", ""])
    plt.yticks([0, -3], ["", ""])
    plt.minorticks_off()
    plt.savefig(save_svg, format="svg", transparent=True)
    plt.savefig(save_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(str(save_pdf).replace(".pdf", ".png"),
                format="png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    if len(data["idx"]) < 10:
        raise SystemExit(f"too few ESNs with both score and spectrum: {len(data['idx'])}")

    rng = np.random.default_rng(SEED)
    winners, counts = identify_winners(data, rng)
    boot = nested_bootstrap(data, set(winners), rng)
    plot(boot, OUT / f"worm_{WORM}_resampled.svg",
              OUT / f"worm_{WORM}_resampled.pdf")
    print(f"worm {WORM} resampled: M={len(data['idx'])}, "
          f"{len(winners)} winners "
          f"(top: {sorted(zip(counts, data['idx']), reverse=True)[:5]})")
    print(f"saved {(OUT / f'worm_{WORM}_resampled.svg').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
