from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figure"))
sys.path.insert(0, str(ROOT / "figure" / "worm"))

from plot_worm import load_worm, nested_bootstrap as nested_unresampled
from plot_worm_resampled import load as load_resampled, \
    identify_winners as identify_winners_r, \
    nested_bootstrap as nested_resampled

WORM = 1
DTS = [1 / 16, 2 / 16]
SEED = 42
OUT = ROOT / "figure" / "lyap_plots"


def boot_unresampled(rng):
    data = load_worm(WORM)
    if data is None:
        raise SystemExit(f"no un-resampled data for worm {WORM}")
    return nested_unresampled(data, n_outer=1000, rng=rng)


def boot_resampled(rng):
    data = load_resampled()
    winners, _ = identify_winners_r(data, rng)
    return nested_resampled(data, set(winners), rng)


def summarise(boot):
    med = np.median(boot, axis=0)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return med, med - lo, hi - med


def plot_scaling(values, err_lo, err_hi, inv_dts, slopes, slope_colors, save):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for k in range(values.shape[0]):
        ax.errorbar(inv_dts, values[k], yerr=[err_lo[k], err_hi[k]],
                    marker="o", markersize=4, linewidth=1, alpha=0.7,
                    capsize=3, color=slope_colors[k])
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, format="svg", transparent=True)
    plt.savefig(str(save).replace(".svg", ".pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(str(save).replace(".svg", ".png"),
                format="png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close()


def plot_slopes(slopes, slope_errs, slope_colors, save):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    y = np.arange(len(slopes))
    ax.barh(y, slopes, xerr=slope_errs, color=slope_colors,
            edgecolor="gray", linewidth=0.5, capsize=3,
            error_kw={"elinewidth": 1, "alpha": 0.7})
    ax.axvline(x=0, color="green", linestyle="--", linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save, format="svg", transparent=True)
    plt.savefig(str(save).replace(".svg", ".pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(str(save).replace(".svg", ".png"),
                format="png", dpi=300, transparent=True, bbox_inches="tight")
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    boot_u = boot_unresampled(rng)
    boot_r = boot_resampled(rng)

    med_u, lo_u, hi_u = summarise(boot_u)
    med_r, lo_r, hi_r = summarise(boot_r)

    n_exp = min(len(med_u), len(med_r), 40)
    med_u, lo_u, hi_u = med_u[:n_exp], lo_u[:n_exp], hi_u[:n_exp]
    med_r, lo_r, hi_r = med_r[:n_exp], lo_r[:n_exp], hi_r[:n_exp]

    inv_dts = np.array([1 / d for d in DTS])
    values = np.stack([med_u, med_r], axis=1)
    err_lo = np.stack([lo_u, lo_r], axis=1)
    err_hi = np.stack([hi_u, hi_r], axis=1)

    slopes = (values[:, -1] - values[:, 0]) / (inv_dts[-1] - inv_dts[0])

    n_boot = 4000
    slope_boot = np.empty((n_exp, n_boot))
    for k in range(n_exp):
        se0 = (err_lo[k, 0] + err_hi[k, 0]) / (2 * 1.96)
        se1 = (err_lo[k, 1] + err_hi[k, 1]) / (2 * 1.96)
        s0 = rng.normal(values[k, 0], se0, size=n_boot)
        s1 = rng.normal(values[k, 1], se1, size=n_boot)
        slope_boot[k] = (s1 - s0) / (inv_dts[-1] - inv_dts[0])
    slope_lo = np.percentile(slope_boot, 2.5, axis=1)
    slope_hi = np.percentile(slope_boot, 97.5, axis=1)
    slope_errs = (slope_hi - slope_lo) / 2

    max_abs = np.max(np.abs(slopes)) if np.any(slopes) else 1.0
    slope_colors = plt.cm.RdYlGn_r(np.abs(slopes) / max_abs)

    plot_scaling(values, err_lo, err_hi, inv_dts, slopes, slope_colors,
                 OUT / f"lyapunov_dt_scaling_worm_{WORM}.svg")
    plot_slopes(slopes, slope_errs, slope_colors,
                OUT / f"lyapunov_dt_scaling_slopes_worm_{WORM}.svg")

    print(f"worm {WORM}: {n_exp} exponents")
    print(f"  λ1: unresampled={med_u[0]:.3f} [{med_u[0]-lo_u[0]:.3f}/{hi_u[0]+med_u[0]-med_u[0]:.3f}], "
          f"resampled={med_r[0]:.3f}")
    print(f"  slope range: {slopes.min():.4f} .. {slopes.max():.4f}")
    print(f"  saved {(OUT / f'lyapunov_dt_scaling_worm_{WORM}.svg').relative_to(ROOT)} "
          f"+ slopes figure")


if __name__ == "__main__":
    main()
