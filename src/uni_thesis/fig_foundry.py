"""Publication-grade figure foundry (headless, 600 DPI PNG + vector PDF).

All plotted values are taken from the arrays / statistical results passed in;
annotations such as phase delays, biases and significance stars are computed
from those inputs, never typed in by hand.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import mark_inset  # noqa: E402
from scipy import signal as sps  # noqa: E402
from scipy import stats  # noqa: E402

from .exp_harness import phase_delay_ms  # noqa: E402
from .stat_router import rsm_predict  # noqa: E402

__all__ = [
    "OKABE_ITO",
    "PUBLICATION_DPI",
    "set_publication_style",
    "save_figure",
    "plot_signal_comparison",
    "plot_bland_altman",
    "plot_ablation_and_baselines",
    "plot_response_surface",
]

PUBLICATION_DPI = 600
# Okabe & Ito (2008) colour-blind-safe palette.
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

PathLike = Union[str, Path]


def set_publication_style(base_size: float = 9.0) -> None:
    """Serif fonts, size hierarchy, Okabe-Ito cycle, inward ticks, no top/right spines."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1.5,
        "figure.titlesize": base_size + 2,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "savefig.dpi": PUBLICATION_DPI,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: "plt.Figure", out_base: PathLike) -> Dict[str, str]:
    """Write ``<out_base>.png`` (600 DPI) and ``<out_base>.pdf`` (vector); close the figure."""
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    fig.savefig(png, dpi=PUBLICATION_DPI)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _color(i: int) -> str:
    return OKABE_ITO[i % len(OKABE_ITO)]


# --------------------------------------------------------------------------- #
# 1. Signal comparison
# --------------------------------------------------------------------------- #
def plot_signal_comparison(t: Sequence[float], reference: Sequence[float], outputs: Mapping[str, Sequence[float]],
                           fs: float, out_base: PathLike, noisy: Optional[Sequence[float]] = None,
                           zoom: Optional[Tuple[float, float]] = None, ylabel: str = "Amplitude",
                           ref_label: str = "Reference") -> Dict[str, Any]:
    """Time-domain overlay with a zoomed inset plus Welch PSD overlay.

    Legend entries carry each output's phase delay versus the reference,
    computed by cross-correlation (``exp_harness.phase_delay_ms``)."""
    set_publication_style()
    t = np.asarray(t, float)
    ref = np.asarray(reference, float)
    delays = {name: phase_delay_ms(ref, np.asarray(y, float), fs) for name, y in outputs.items()}
    if zoom is None:
        span = min(t[-1] - t[0], max(5.0 / fs * 50, (t[-1] - t[0]) / 8))
        mid = t[0] + (t[-1] - t[0]) / 2
        zoom = (mid, mid + span)

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(7.0, 5.8), layout="constrained",
                                      gridspec_kw={"height_ratios": [1.3, 1.0]})
    series: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    if noisy is not None:
        series.append(("Noisy input", np.asarray(noisy, float), {"color": "#BBBBBB", "lw": 0.6}))
    series.append((ref_label, ref, {"color": _color(0), "lw": 1.3}))
    for i, (name, y) in enumerate(outputs.items(), start=1):
        series.append((f"{name} ($\\Delta t$ = {delays[name]:.2f} ms)", np.asarray(y, float),
                       {"color": _color(i), "lw": 1.0, "ls": "-" if i % 2 else "--"}))
    for label, y, kw in series:
        ax_t.plot(t, y, label=label, **kw)
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel(ylabel)
    ax_t.set_title("(a) Time-domain comparison", loc="left")
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(2, len(series)), fontsize=7)

    ins = ax_t.inset_axes([0.64, 0.52, 0.34, 0.44])
    mask = (t >= zoom[0]) & (t <= zoom[1])
    for _, y, kw in series:
        ins.plot(t[mask], y[mask], **kw)
    ins.set_xlim(*zoom)
    ins.tick_params(labelsize=6)
    ins.set_title("Zoom", fontsize=7)
    mark_inset(ax_t, ins, loc1=2, loc2=4, fc="none", ec="0.4", lw=0.6)

    nper = int(min(len(ref), max(64, 2 ** int(math.log2(max(len(ref) // 4, 64))))))
    for label, y, kw in series:
        f, pxx = sps.welch(y, fs=fs, nperseg=nper)
        ax_f.semilogy(f, np.maximum(pxx, np.finfo(float).tiny), label=label.split(" ($")[0], **kw)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("PSD (units$^2$/Hz)")
    ax_f.set_xlim(0, fs / 2)
    ax_f.set_title("(b) Power spectral density (Welch)", loc="left")
    paths = save_figure(fig, out_base)
    return {**paths, "phase_delay_ms": delays}


# --------------------------------------------------------------------------- #
# 2. Bland-Altman
# --------------------------------------------------------------------------- #
def plot_bland_altman(reference: Sequence[float], measured: Sequence[float], out_base: PathLike,
                      units: str = "", label_ref: str = "Reference", label_meas: str = "Method") -> Dict[str, Any]:
    """Bland-Altman plot: bias (solid), +-1.96 SD limits (dashed), 95% CI bands
    (bias: t-based; limits: SE = SD*sqrt(3/n), Bland & Altman 1999)."""
    set_publication_style()
    a, b = np.asarray(reference, float), np.asarray(measured, float)
    if a.shape != b.shape or a.size < 3:
        raise ValueError("Bland-Altman needs paired arrays of equal length >= 3")
    mean = (a + b) / 2.0
    diff = b - a
    n = diff.size
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    loa = (bias - 1.96 * sd, bias + 1.96 * sd)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    se_bias, se_loa = sd / math.sqrt(n), sd * math.sqrt(3.0 / n)

    fig, ax = plt.subplots(figsize=(4.6, 3.6), layout="constrained")
    ax.scatter(mean, diff, s=8, color=_color(5), alpha=0.6, edgecolors="none")
    xr = (float(mean.min()), float(mean.max()))
    for center, se, ls, lab in [(bias, se_bias, "-", f"Bias = {bias:.3g}"),
                                (loa[0], se_loa, "--", f"−1.96 SD = {loa[0]:.3g}"),
                                (loa[1], se_loa, "--", f"+1.96 SD = {loa[1]:.3g}")]:
        ax.axhline(center, color=_color(6) if ls == "-" else _color(0), ls=ls, lw=1.0, label=lab)
        ax.axhspan(center - tcrit * se, center + tcrit * se, color=_color(6) if ls == "-" else "0.5", alpha=0.15, lw=0)
    unit = f" ({units})" if units else ""
    ax.set_xlabel(f"Mean of {label_ref} and {label_meas}{unit}")
    ax.set_ylabel(f"{label_meas} − {label_ref}{unit}")
    ax.set_xlim(xr[0] - 0.02 * (xr[1] - xr[0]), xr[1] + 0.02 * (xr[1] - xr[0]))
    ax.legend(loc="best")
    paths = save_figure(fig, out_base)
    return {**paths, "bias": bias, "sd": sd, "loa": loa, "n": n}


# --------------------------------------------------------------------------- #
# 3. Ablation / baselines bars + boxes with significance brackets
# --------------------------------------------------------------------------- #
def _bracket(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color="0.15")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=8)


def plot_ablation_and_baselines(samples: Mapping[str, Sequence[float]], out_base: PathLike, metric_label: str,
                                tests: Sequence[Mapping[str, Any]] = (), show_ns: bool = True) -> Dict[str, Any]:
    """(a) Mean +- SD bars with significance brackets from pairwise ``tests``
    (dicts with method_a, method_b, significance); (b) box plots with points."""
    set_publication_style()
    names = list(samples)
    data = [np.asarray(samples[n], float) for n in names]
    means = np.array([d.mean() for d in data])
    sds = np.array([d.std(ddof=1) if d.size > 1 else 0.0 for d in data])
    x = np.arange(len(names))
    colors = [_color(i + 1) for i in range(len(names))]

    fig, (ax_b, ax_x) = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    ax_b.bar(x, means, yerr=sds, capsize=3, color=colors, edgecolor="0.15", linewidth=0.6,
             error_kw={"elinewidth": 0.8})
    lo = float(min((means - sds).min(), 0.0))
    top = float((means + sds).max())
    span = max(top - lo, abs(top), 1e-12)
    h, y = 0.03 * span, top + 0.05 * span
    drawn = 0
    idx = {n: i for i, n in enumerate(names)}
    pair_tests = [tt for tt in tests if tt.get("method_a") in idx and tt.get("method_b") in idx and "significance" in tt]
    for tt in sorted(pair_tests, key=lambda r: abs(idx[r["method_a"]] - idx[r["method_b"]])):
        if tt["significance"] == "ns" and not show_ns:
            continue
        _bracket(ax_b, idx[tt["method_a"]], idx[tt["method_b"]], y, h, tt["significance"])
        y += 0.12 * span
        drawn += 1
    ax_b.set_ylim(lo, y + 0.05 * span)
    ax_b.set_xticks(x, names, rotation=20, ha="right")
    ax_b.set_ylabel(metric_label)
    ax_b.set_title("(a) Mean ± SD", loc="left")

    ax_x.boxplot(data, positions=x, widths=0.5, showfliers=False,
                 medianprops={"color": "0.1"}, boxprops={"linewidth": 0.8}, whiskerprops={"linewidth": 0.8})
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax_x.scatter(i + rng.uniform(-0.12, 0.12, d.size), d, s=7, color=colors[i], alpha=0.75, edgecolors="none", zorder=3)
    ax_x.set_xticks(x, names, rotation=20, ha="right")
    ax_x.set_ylabel(metric_label)
    ax_x.set_title("(b) Distribution", loc="left")
    ax_x.yaxis.set_major_locator(MaxNLocator(6))
    paths = save_figure(fig, out_base)
    return {**paths, "brackets": drawn}


# --------------------------------------------------------------------------- #
# 4. Response surface
# --------------------------------------------------------------------------- #
def plot_response_surface(model: Mapping[str, Any], bounds: Mapping[str, Tuple[float, float]], out_base: PathLike,
                          response_label: str = "Response", maximize: bool = True,
                          observed: Optional[Tuple[Sequence[Sequence[float]], Sequence[float]]] = None,
                          resolution: int = 60) -> Dict[str, Any]:
    """3-D surface + 2-D contour of a two-factor RSM model (``stat_router.fit_response_surface``),
    marking the optimum: the stationary point if it is the right kind and inside
    ``bounds``, otherwise the best grid point."""
    names = list(model["factors"])
    if len(names) != 2:
        raise ValueError("plot_response_surface requires exactly two factors")
    set_publication_style()
    (x0, x1), (y0, y1) = bounds[names[0]], bounds[names[1]]
    gx, gy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    gz = rsm_predict(model, np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

    sp = model.get("stationary_point")
    wanted = "maximum" if maximize else "minimum"
    if sp and sp["nature"] == wanted and x0 <= sp["point"][names[0]] <= x1 and y0 <= sp["point"][names[1]] <= y1:
        opt = (sp["point"][names[0]], sp["point"][names[1]], sp["predicted_response"])
        source = "stationary_point"
    else:
        k = int(np.argmax(gz) if maximize else np.argmin(gz))
        opt = (float(gx.ravel()[k]), float(gy.ravel()[k]), float(gz.ravel()[k]))
        source = "grid_search"

    fig = plt.figure(figsize=(8.4, 3.6), layout="constrained")
    fig.get_layout_engine().set(wspace=0.08)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.set_box_aspect(None, zoom=0.82)
    ax3.plot_surface(gx, gy, gz, cmap="cividis", linewidth=0, antialiased=True, alpha=0.9)
    ax3.contour(gx, gy, gz, zdir="z", offset=float(gz.min()), cmap="cividis", levels=10, linewidths=0.6)
    ax3.scatter([opt[0]], [opt[1]], [opt[2]], color=_color(6), s=25, depthshade=False)
    ax3.set_xlabel(names[0], labelpad=2)
    ax3.set_ylabel(names[1], labelpad=2)
    ax3.set_zlabel(response_label, labelpad=2)
    ax3.tick_params(labelsize=6, pad=0)
    ax3.set_title("(a) Response surface", loc="left")

    ax2 = fig.add_subplot(1, 2, 2)
    cf = ax2.contourf(gx, gy, gz, levels=14, cmap="cividis")
    cs = ax2.contour(gx, gy, gz, levels=7, colors="k", linewidths=0.5)
    ax2.clabel(cs, fontsize=6, fmt="%.3g")
    if observed is not None:
        pts = np.asarray(observed[0], float)
        ax2.scatter(pts[:, 0], pts[:, 1], s=10, facecolors="none", edgecolors="w", linewidths=0.7, label="Design points")
    ax2.plot(opt[0], opt[1], marker="*", ms=11, color=_color(6), mec="k", mew=0.5, ls="none",
             label=f"Optimum ({opt[0]:.3g}, {opt[1]:.3g})")
    ax2.set_xlabel(names[0])
    ax2.set_ylabel(names[1])
    ax2.set_title(f"(b) Contour ($R^2_{{adj}}$ = {model['r2_adj']:.3f})", loc="left")
    ax2.legend(loc="upper right", fontsize=6.5, frameon=True)
    fig.colorbar(cf, ax=ax2, label=response_label, shrink=0.9)
    paths = save_figure(fig, out_base)
    return {**paths, "optimum": {names[0]: opt[0], names[1]: opt[1], "response": opt[2]}, "optimum_source": source}
