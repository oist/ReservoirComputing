from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/iliasoroka/RC_esn")
ESNS = ROOT / "esns"
OUT = ROOT / "figure" / "lyap_plots"
OUT.mkdir(parents=True, exist_ok=True)

N_WORMS = 12
M_PIPELINE = 40
N_OUTER = 1000
SEED = 42


def load_worm(n):
    dws_path = ESNS / f"worm_{n}" / "dws.npz"
    top_dir = ESNS / f"worm_{n}" / "lyap_top"
    all_dir = ESNS / f"worm_{n}" / "lyap_all"
    if not dws_path.exists():
        return None
    dws = np.load(dws_path)
    q_by_i = {int(i): float(q)
              for i, q in zip(dws["esn_indices"], np.median(dws["means"], axis=1))}
    trajs_by_i = {}
    if top_dir.exists():
        for f in top_dir.glob("esn_*.npz"):
            i = int(f.stem.split("_")[1])
            d = np.load(f)
            s = np.sort(d["all_samples"], axis=1)[:, ::-1]
            if s.size:
                trajs_by_i[i] = s
    if all_dir.exists():
        for f in all_dir.glob("esn_*.npz"):
            i = int(f.stem.split("_")[1])
            if i in trajs_by_i:
                continue
            d = np.load(f)
            s = np.sort(d["all_samples"], axis=1)[:, ::-1]
            if s.size:
                trajs_by_i[i] = s
    common = sorted(set(q_by_i) & set(trajs_by_i))
    if len(common) < 10:
        return None
    return dict(
        idx=np.array(common, dtype=int),
        quality=np.array([q_by_i[i] for i in common]),
        trajs=trajs_by_i,
    )


def nested_bootstrap(data, n_outer, rng):
    idx, quality, trajs = data["idx"], data["quality"], data["trajs"]
    M = len(idx)
    num_lyaps = next(iter(trajs.values())).shape[1]
    boot = np.empty((n_outer, num_lyaps))
    for b in range(n_outer):
        if M_PIPELINE >= M:
            samp = rng.integers(0, M, size=M_PIPELINE)
        else:
            samp = rng.choice(M, size=M_PIPELINE, replace=False)
        j = int(idx[samp[int(np.argmin(quality[samp]))]])
        ts = trajs[j]
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
    rng = np.random.default_rng(SEED)
    for n in range(N_WORMS):
        data = load_worm(n)
        if data is None:
            continue
        boot = nested_bootstrap(data, N_OUTER, rng)
        plot(boot, OUT / f"worm_{n}.svg", OUT / f"worm_{n}.pdf")
        print(f"worm {n}: M={len(data['idx'])}, saved worm_{n}.{{svg,pdf,png}}")


if __name__ == "__main__":
    main()
