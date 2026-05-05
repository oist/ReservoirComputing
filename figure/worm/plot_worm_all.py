from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/iliasoroka/RC_esn")
OUT = ROOT / "figure" / "lyap_plots"

sys.path.insert(0, str(ROOT / "figure"))
from plot_worm import load_worm, nested_bootstrap

N_WORMS = 12
N_INNER_BOOT = 2000
N_OUTER_BOOT = 5000
SEED = 42


def load_per_worm():
    per_worm = {}
    for n in range(N_WORMS):
        data = load_worm(n)
        if data is None:
            continue
        rng = np.random.default_rng(SEED + n)
        per_worm[n] = nested_bootstrap(data, N_INNER_BOOT, rng)
    return per_worm


def aggregate(per_worm, seed):
    worms = sorted(per_worm)
    pools = [per_worm[n] for n in worms]
    n_worms = len(pools)
    num_lyaps = pools[0].shape[1]
    medians = np.array([np.median(p, axis=0) for p in pools])
    mean_spec = np.mean(medians, axis=0)

    rng = np.random.default_rng(seed)
    boot = np.empty((N_OUTER_BOOT, num_lyaps))
    for b in range(N_OUTER_BOOT):
        worm_samp = rng.integers(0, n_worms, size=n_worms)
        iter_specs = np.empty((n_worms, num_lyaps))
        for k, w_idx in enumerate(worm_samp):
            pool = pools[w_idx]
            traj_idx = rng.integers(0, pool.shape[0])
            iter_specs[k] = pool[traj_idx]
        boot[b] = np.mean(iter_specs, axis=0)
    ci_lo = np.percentile(boot, 2.5, axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)
    return mean_spec, ci_lo, ci_hi, medians


def plot(mean_spec, ci_lo, ci_hi, medians, save_svg, save_pdf):
    x = np.arange(1, len(mean_spec) + 1)
    yerr = np.array([mean_spec - ci_lo, ci_hi - mean_spec])

    plt.figure()
    for med in medians:
        plt.plot(x, med, "-", color="lightgray", alpha=0.6, linewidth=0.8, zorder=1)
    plt.errorbar(
        x, mean_spec, yerr=yerr,
        fmt="k.", ms=8, capsize=5, ecolor="gray", zorder=3,
    )
    plt.axhline(y=0, color="black", linestyle="-", linewidth=1)
    plt.xticks([1, len(mean_spec)], ["", ""])
    plt.yticks([0, -3], ["", ""])
    plt.minorticks_off()
    plt.savefig(save_svg, format="svg", transparent=True)
    plt.savefig(save_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(str(save_pdf).replace(".pdf", ".png"),
                format="png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close()


def main():
    per_worm = load_per_worm()
    mean_spec, ci_lo, ci_hi, medians = aggregate(per_worm, SEED + 1000)
    plot(mean_spec, ci_lo, ci_hi, medians,
         OUT / "worm_all.svg", OUT / "worm_all.pdf")
    print(f"saved worm_all.{{svg,pdf,png}} (across {len(per_worm)} worms)")


if __name__ == "__main__":
    main()
