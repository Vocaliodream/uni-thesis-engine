"""Pluggable experiment harness.

Two domain-agnostic pipelines are registered out of the box:

* ``signal_processing`` - waveform denoising benchmark (external .csv/.npy/.npz
  data or deterministic synthetic ECG / PPG / ToF signals with baseline wander,
  power-line interference and white noise) comparing registered filters on
  delta-SNR, RMSE, Pearson r and phase delay.
* ``parameter_optimization`` / ``ablation_study`` - multi-factor grid / central
  composite design evaluation and module-removal ablation over a registered
  objective (built-in analytic objectives or user plugins).

Every number written to ``results/<profile_id>/metrics.json`` is produced by an
actual run and stamped with the git commit and the SHA-256 of the input data.
All experiment settings come from the ``TopicProfile``; nothing topic-specific
is hard-coded here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

from .schema import ExperimentDesign, ExperimentResultBundle, FactorRange, MetricItem, TopicProfile

__all__ = [
    "FILTER_REGISTRY",
    "SIGNAL_SIMULATOR_REGISTRY",
    "OBJECTIVE_REGISTRY",
    "PIPELINE_REGISTRY",
    "register_filter",
    "register_signal_simulator",
    "register_objective",
    "register_pipeline",
    "load_plugins",
    "snr_db",
    "rmse",
    "pearson_r",
    "phase_delay_ms",
    "summarize",
    "build_design",
    "git_commit_hash",
    "ExperimentHarness",
    "SignalProcessingPipeline",
    "MultiFactorPipeline",
]

FilterFn = Callable[[np.ndarray, float, Mapping[str, Any]], np.ndarray]
SignalSimFn = Callable[[Mapping[str, Any], float, np.random.Generator], Tuple[np.ndarray, np.ndarray, np.ndarray]]
ObjectiveFn = Callable[[Mapping[str, Any], Mapping[str, Any], np.random.Generator], Mapping[str, float]]

FILTER_REGISTRY: Dict[str, FilterFn] = {}
SIGNAL_SIMULATOR_REGISTRY: Dict[str, SignalSimFn] = {}
OBJECTIVE_REGISTRY: Dict[str, ObjectiveFn] = {}
PIPELINE_REGISTRY: Dict[str, Type["BasePipeline"]] = {}


def _registrar(registry: Dict[str, Any], kind: str):
    def register(name: str):
        def deco(obj):
            if name in registry and registry[name] is not obj:
                raise ValueError(f"{kind} '{name}' is already registered")
            registry[name] = obj
            return obj
        return deco
    return register


register_filter = _registrar(FILTER_REGISTRY, "filter")
register_signal_simulator = _registrar(SIGNAL_SIMULATOR_REGISTRY, "signal simulator")
register_objective = _registrar(OBJECTIVE_REGISTRY, "objective")
register_pipeline = _registrar(PIPELINE_REGISTRY, "pipeline")


def load_plugins(plugin_dir: Union[str, Path]) -> List[str]:
    """Import every ``*.py`` in ``plugin_dir`` so its ``register_*`` decorators run."""
    loaded = []
    for path in sorted(Path(plugin_dir).glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod_name = f"uni_thesis_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load plugin {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        loaded.append(mod_name)
    return loaded


def _require(params: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    missing = [k for k in keys if k not in params]
    if missing:
        raise ValueError(f"{where} requires parameters {missing}")


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
@register_filter("raw")
def _filter_raw(x: np.ndarray, fs: float, params: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(x, dtype=float).copy()


@register_filter("moving_average")
def _filter_moving_average(x: np.ndarray, fs: float, params: Mapping[str, Any]) -> np.ndarray:
    _require(params, ["window_s"], "moving_average")
    n = max(1, int(round(float(params["window_s"]) * fs)))
    kernel = np.ones(n) / n
    if bool(params.get("causal", False)):
        return sps.lfilter(kernel, [1.0], x)
    padded = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


@register_filter("savitzky_golay")
def _filter_savgol(x: np.ndarray, fs: float, params: Mapping[str, Any]) -> np.ndarray:
    _require(params, ["window_s", "polyorder"], "savitzky_golay")
    n = max(3, int(round(float(params["window_s"]) * fs)))
    n = n if n % 2 == 1 else n + 1
    order = int(params["polyorder"])
    if order >= n:
        raise ValueError("savitzky_golay polyorder must be smaller than the window length")
    return sps.savgol_filter(x, n, order, mode="interp")


def _butter_ba(fs: float, params: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    _require(params, ["order"], "butterworth")
    low, high = params.get("low_hz"), params.get("high_hz")
    nyq = fs / 2.0
    if low is not None and high is not None:
        if not 0 < float(low) < float(high) < nyq:
            raise ValueError("butterworth requires 0 < low_hz < high_hz < fs/2")
        return sps.butter(int(params["order"]), [float(low), float(high)], btype="bandpass", fs=fs)
    if high is not None:
        if not 0 < float(high) < nyq:
            raise ValueError("butterworth requires 0 < high_hz < fs/2")
        return sps.butter(int(params["order"]), float(high), btype="lowpass", fs=fs)
    if low is not None:
        if not 0 < float(low) < nyq:
            raise ValueError("butterworth requires 0 < low_hz < fs/2")
        return sps.butter(int(params["order"]), float(low), btype="highpass", fs=fs)
    raise ValueError("butterworth requires low_hz and/or high_hz")


@register_filter("butterworth_zero_phase")
def _filter_butter_zero_phase(x: np.ndarray, fs: float, params: Mapping[str, Any]) -> np.ndarray:
    b, a = _butter_ba(fs, params)
    return sps.filtfilt(b, a, x)


@register_filter("butterworth_causal")
def _filter_butter_causal(x: np.ndarray, fs: float, params: Mapping[str, Any]) -> np.ndarray:
    b, a = _butter_ba(fs, params)
    return sps.lfilter(b, a, x)


# --------------------------------------------------------------------------- #
# Synthetic signal models
# --------------------------------------------------------------------------- #
# Gaussian-kernel PQRST morphology from McSharry et al., IEEE TBME 2003
# (doi:10.1109/tbme.2003.808805), Table 1: (theta_i, a_i, b_i). Overridable via params["waves"].
_ECG_WAVES = [(-math.pi / 3, 1.2, 0.25), (-math.pi / 12, -5.0, 0.1), (0.0, 30.0, 0.1),
              (math.pi / 12, -7.5, 0.1), (math.pi / 2, 0.75, 0.4)]


def _beat_phase(params: Mapping[str, Any], n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    _require(params, ["heart_rate_bpm"], "cardiac signal model")
    mean_rr = 60.0 / float(params["heart_rate_bpm"])
    rr_std = float(params.get("rr_std_s", 0.0))
    duration = n / fs
    rrs = []
    while sum(rrs) < duration + mean_rr:
        rrs.append(max(0.2 * mean_rr, mean_rr + rr_std * rng.standard_normal()))
    t = np.arange(n) / fs
    edges = np.concatenate([[0.0], np.cumsum(rrs)])
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(rrs) - 1)
    frac = (t - edges[idx]) / np.asarray(rrs)[idx]
    return 2 * math.pi * frac - math.pi


def _gaussian_waves(phase: np.ndarray, waves: Sequence[Sequence[float]]) -> np.ndarray:
    z = np.zeros_like(phase)
    for theta, a, b in waves:
        d = np.angle(np.exp(1j * (phase - float(theta))))
        z += float(a) * np.exp(-(d ** 2) / (2 * float(b) ** 2))
    return z


def _add_interference(clean: np.ndarray, t: np.ndarray, params: Mapping[str, Any],
                      rng: np.random.Generator) -> np.ndarray:
    _require(params, ["noise_std"], "signal simulator")
    noisy = clean.copy()
    bw_amp = float(params.get("baseline_wander_amp", 0.0))
    if bw_amp:
        _require(params, ["baseline_wander_hz"], "baseline wander")
        noisy += bw_amp * np.sin(2 * math.pi * float(params["baseline_wander_hz"]) * t + rng.uniform(0, 2 * math.pi))
    pl_amp = float(params.get("powerline_amp", 0.0))
    if pl_amp:
        _require(params, ["powerline_hz"], "power-line interference")
        noisy += pl_amp * np.sin(2 * math.pi * float(params["powerline_hz"]) * t + rng.uniform(0, 2 * math.pi))
    noisy += float(params["noise_std"]) * rng.standard_normal(len(t))
    return noisy


def _time_axis(params: Mapping[str, Any], fs: float) -> np.ndarray:
    _require(params, ["duration_s"], "signal simulator")
    n = int(round(float(params["duration_s"]) * fs))
    if n < 16:
        raise ValueError("duration_s * sampling_rate_hz must yield at least 16 samples")
    return np.arange(n) / fs


@register_signal_simulator("ecg")
def _sim_ecg(params: Mapping[str, Any], fs: float, rng: np.random.Generator):
    t = _time_axis(params, fs)
    z = _gaussian_waves(_beat_phase(params, len(t), fs, rng), params.get("waves", _ECG_WAVES))
    clean = float(params.get("amplitude", 1.0)) * z / np.max(np.abs(z))
    return t, clean, _add_interference(clean, t, params, rng)


@register_signal_simulator("ppg")
def _sim_ppg(params: Mapping[str, Any], fs: float, rng: np.random.Generator):
    _require(params, ["waves"], "ppg simulator (list of (phase_rad, amplitude, width_rad))")
    t = _time_axis(params, fs)
    z = _gaussian_waves(_beat_phase(params, len(t), fs, rng), params["waves"])
    clean = float(params.get("amplitude", 1.0)) * z / np.max(np.abs(z))
    return t, clean, _add_interference(clean, t, params, rng)


@register_signal_simulator("tof")
def _sim_tof(params: Mapping[str, Any], fs: float, rng: np.random.Generator):
    _require(params, ["base_distance", "motion_amp", "motion_hz"], "tof simulator")
    t = _time_axis(params, fs)
    clean = float(params["base_distance"]) + float(params["motion_amp"]) * np.sin(2 * math.pi * float(params["motion_hz"]) * t)
    return t, clean, _add_interference(clean, t, params, rng)


# --------------------------------------------------------------------------- #
# Built-in generic objectives
# --------------------------------------------------------------------------- #
@register_objective("analytic_response_surface")
def _obj_response_surface(config: Mapping[str, Any], params: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, float]:
    """Second-order polynomial response y = b0 + sum b_i x_i + sum b_ii x_i^2 + sum b_ij x_i x_j + noise."""
    _require(params, ["intercept", "noise_std"], "analytic_response_surface")
    x = config.get("factors", {})
    y = float(params["intercept"])
    for f, c in params.get("linear", {}).items():
        y += float(c) * float(x[f])
    for f, c in params.get("quadratic", {}).items():
        y += float(c) * float(x[f]) ** 2
    for pair, c in params.get("interaction", {}).items():
        a, b = pair.split("*")
        y += float(c) * float(x[a]) * float(x[b])
    y += float(params["noise_std"]) * rng.standard_normal()
    return {str(params.get("metric_name", "response")): y}


@register_objective("additive_module_contributions")
def _obj_module_contrib(config: Mapping[str, Any], params: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, float]:
    """Score = base + sum of enabled modules' contributions + pairwise interactions + noise."""
    _require(params, ["base_value", "contributions", "noise_std"], "additive_module_contributions")
    on = config.get("modules", {})
    y = float(params["base_value"])
    for m, enabled in on.items():
        if m not in params["contributions"]:
            raise ValueError(f"no contribution configured for module '{m}'")
        if enabled:
            y += float(params["contributions"][m])
    for pair, c in params.get("interactions", {}).items():
        a, b = pair.split("*")
        if on.get(a) and on.get(b):
            y += float(c)
    y += float(params["noise_std"]) * rng.standard_normal()
    return {str(params.get("metric_name", "score")): y}


# --------------------------------------------------------------------------- #
# Metrics & statistics
# --------------------------------------------------------------------------- #
def snr_db(clean: np.ndarray, x: np.ndarray) -> float:
    clean = np.asarray(clean, float)
    sig = np.sum((clean - clean.mean()) ** 2)
    err = max(float(np.sum((np.asarray(x, float) - clean) ** 2)), np.finfo(float).tiny)
    return float(10.0 * np.log10(sig / err))


def rmse(clean: np.ndarray, x: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(x, float) - np.asarray(clean, float)) ** 2)))


def pearson_r(clean: np.ndarray, x: np.ndarray) -> float:
    a, b = np.asarray(clean, float), np.asarray(x, float)
    if np.std(a) == 0 or np.std(b) == 0:
        raise ValueError("pearson_r undefined for a constant signal")
    return float(np.corrcoef(a, b)[0, 1])


def phase_delay_ms(reference: np.ndarray, x: np.ndarray, fs: float) -> float:
    """Lag of ``x`` relative to ``reference`` at the cross-correlation peak
    (positive = x lags), refined by parabolic interpolation, in milliseconds."""
    a = np.asarray(reference, float) - np.mean(reference)
    b = np.asarray(x, float) - np.mean(x)
    corr = sps.correlate(b, a, mode="full", method="fft")
    lags = sps.correlation_lags(len(b), len(a), mode="full")
    k = int(np.argmax(corr))
    shift = 0.0
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
    return float((lags[k] + shift) / fs * 1000.0)


def summarize(name: str, values: Sequence[float], unit: str = "") -> MetricItem:
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError(f"metric '{name}' has no finite values")
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    if v.size > 1 and std > 0:
        half = float(stats.t.ppf(0.975, v.size - 1)) * std / math.sqrt(v.size)
        ci = (mean - half, mean + half)
    else:
        ci = (mean, mean)
    q75, q25 = np.percentile(v, [75, 25])
    return MetricItem(name=name, mean=mean, std=std, median=float(np.median(v)), iqr=float(q75 - q25),
                      unit=unit, ci_95=(float(ci[0]), float(ci[1])), is_best=False)


def _score(item: MetricItem, direction: str) -> float:
    if direction == "max":
        return item.mean
    if direction == "min":
        return -item.mean
    return -abs(item.mean)


def _mark_best(methods_metrics: Dict[str, Dict[str, MetricItem]], directions: Mapping[str, str]) -> None:
    metric_names = set().union(*[set(m) for m in methods_metrics.values()])
    for metric in metric_names:
        direction = directions.get(metric)
        if direction is None:
            continue
        holders = [(meth, mm[metric]) for meth, mm in methods_metrics.items() if metric in mm]
        best = max(_score(item, direction) for _, item in holders)
        for meth, item in holders:
            if _score(item, direction) == best:
                methods_metrics[meth][metric] = item.model_copy(update={"is_best": True})


def _paired_tests(raw: Mapping[str, Mapping[str, Sequence[float]]], methods_metrics: Mapping[str, Mapping[str, MetricItem]],
                  metrics: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for metric in metrics:
        best = [m for m, mm in methods_metrics.items() if metric in mm and mm[metric].is_best]
        if not best:
            continue
        ref = best[0]
        for other in methods_metrics:
            if other == ref or metric not in raw[other]:
                continue
            a, b = np.asarray(raw[ref][metric], float), np.asarray(raw[other][metric], float)
            if a.size < 2 or np.allclose(a - b, 0.0):
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            p = float(res.pvalue)
            if not math.isfinite(p):
                continue
            out.append({
                "test": "wilcoxon_signed_rank", "metric": metric, "method_a": ref, "method_b": other,
                "n": int(a.size), "statistic": float(res.statistic), "p_value": min(max(p, 0.0), 1.0),
                "notes": sorted({str(w.message) for w in caught}),
            })
    return out


# --------------------------------------------------------------------------- #
# Design of experiments
# --------------------------------------------------------------------------- #
def _to_space(fr: FactorRange, v: float) -> float:
    return math.log10(v) if fr.log_scale else v


def _from_space(fr: FactorRange, v: float) -> float:
    return 10 ** v if fr.log_scale else v


def build_design(factor_ranges: Mapping[str, FactorRange], design: Optional[ExperimentDesign]) -> List[Dict[str, float]]:
    """Grid (full factorial over ``steps``) or inscribed central composite design."""
    if not factor_ranges:
        raise ValueError("build_design requires at least one factor range")
    names = list(factor_ranges)
    design = design or ExperimentDesign()
    if design.kind == "grid":
        axes = []
        for n in names:
            fr = factor_ranges[n]
            if fr.steps is None:
                raise ValueError(f"grid design requires 'steps' for factor '{n}'")
            lo, hi = _to_space(fr, float(fr.low)), _to_space(fr, float(fr.high))
            axes.append([_from_space(fr, v) for v in np.linspace(lo, hi, fr.steps)])
        return [dict(zip(names, map(float, combo))) for combo in itertools.product(*axes)]

    k = len(names)
    alpha = float(design.ccd_alpha) if design.ccd_alpha is not None else (2 ** k) ** 0.25
    coded: List[Tuple[float, ...]] = [tuple(map(float, c)) for c in itertools.product([-1.0, 1.0], repeat=k)]
    for i in range(k):
        for s in (-alpha, alpha):
            pt = [0.0] * k
            pt[i] = s
            coded.append(tuple(pt))
    coded.extend([tuple([0.0] * k)] * design.center_points)
    scale = max(alpha, 1.0)
    points = []
    for c in coded:
        row = {}
        for n, cv in zip(names, c):
            fr = factor_ranges[n]
            lo, hi = _to_space(fr, float(fr.low)), _to_space(fr, float(fr.high))
            row[n] = float(_from_space(fr, (lo + hi) / 2 + (cv / scale) * (hi - lo) / 2))
        points.append(row)
    return points


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_commit_hash(repo_root: Union[str, Path, None] = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root or Path.cwd()),
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("git commit hash unavailable; results require a git checkout for provenance") from exc
    return out.stdout.strip()


def _sha256_bytes(chunks: Sequence[bytes]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
class PipelineOutput:
    def __init__(self, raw: Dict[str, Dict[str, List[float]]], units: Dict[str, str], directions: Dict[str, str],
                 dataset_sha256: str, arrays: Dict[str, np.ndarray]):
        self.raw = raw
        self.units = units
        self.directions = directions
        self.dataset_sha256 = dataset_sha256
        self.arrays = arrays


class BasePipeline:
    """Common state for registered pipelines; subclasses provide ``execute() -> PipelineOutput``."""

    def __init__(self, profile: TopicProfile):
        self.profile = profile


_SIGNAL_UNITS = {"delta_snr_db": "dB", "rmse": "signal units", "pearson_r": "", "phase_delay_ms": "ms", "runtime_s": "s"}
_SIGNAL_DIRECTIONS = {"delta_snr_db": "max", "rmse": "min", "pearson_r": "max", "phase_delay_ms": "min_abs", "runtime_s": "min"}


@register_pipeline("signal_processing")
class SignalProcessingPipeline(BasePipeline):
    """Pipeline A: filter benchmark on external or synthetic waveforms."""

    def _methods(self) -> List[str]:
        methods = list(dict.fromkeys(["raw", *self.profile.baselines]))
        unknown = [m for m in methods if m not in FILTER_REGISTRY]
        if unknown:
            raise KeyError(f"unregistered filter(s) {unknown}; available: {sorted(FILTER_REGISTRY)}")
        return methods

    def _load_external(self) -> Tuple[np.ndarray, np.ndarray, bytes]:
        path = Path(self.profile.data_config.input_path or "")
        blob = path.read_bytes()
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
            if not {"clean", "noisy"} <= set(df.columns):
                raise ValueError("CSV input requires 'clean' and 'noisy' columns")
            return df["clean"].to_numpy(float), df["noisy"].to_numpy(float), blob
        if suffix == ".npy":
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[0] != 2:
                raise ValueError(".npy input must have shape (2, N): rows [clean, noisy]")
            return arr[0].astype(float), arr[1].astype(float), blob
        if suffix == ".npz":
            with np.load(path) as z:
                return z["clean"].astype(float), z["noisy"].astype(float), blob
        raise ValueError(f"unsupported input format '{suffix}'")

    def execute(self) -> PipelineOutput:
        p = self.profile
        fs = float(p.data_config.sampling_rate_hz)
        methods = self._methods()
        raw: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in _SIGNAL_UNITS} for m in methods}
        hash_chunks: List[bytes] = []
        arrays: Dict[str, np.ndarray] = {}
        external = None
        if p.data_config.input_path:
            clean_e, noisy_e, blob = self._load_external()
            external = (clean_e, noisy_e)
            hash_chunks.append(blob)
        sim = p.data_config.simulator
        for r in range(p.n_repeats):
            if external is not None:
                clean, noisy = external
                t = np.arange(len(clean)) / fs
            else:
                if sim.name not in SIGNAL_SIMULATOR_REGISTRY:
                    raise KeyError(f"unregistered signal simulator '{sim.name}'; available: {sorted(SIGNAL_SIMULATOR_REGISTRY)}")
                rng = np.random.default_rng([p.random_seed, r])
                t, clean, noisy = SIGNAL_SIMULATOR_REGISTRY[sim.name](sim.params, fs, rng)
                hash_chunks += [np.ascontiguousarray(clean).tobytes(), np.ascontiguousarray(noisy).tobytes()]
            snr_in = snr_db(clean, noisy)
            if r == 0:
                arrays.update({"t": t, "clean": clean, "noisy": noisy})
            for m in methods:
                t0 = time.perf_counter()
                y = np.asarray(FILTER_REGISTRY[m](noisy, fs, p.method_params.get(m, {})), float)
                elapsed = time.perf_counter() - t0
                if y.shape != clean.shape or not np.all(np.isfinite(y)):
                    raise ValueError(f"filter '{m}' produced invalid output")
                raw[m]["delta_snr_db"].append(snr_db(clean, y) - snr_in)
                raw[m]["rmse"].append(rmse(clean, y))
                raw[m]["pearson_r"].append(pearson_r(clean, y))
                raw[m]["phase_delay_ms"].append(phase_delay_ms(clean, y, fs))
                raw[m]["runtime_s"].append(elapsed)
                if r == 0:
                    arrays[f"out__{m}"] = y
        return PipelineOutput(raw, dict(_SIGNAL_UNITS), dict(_SIGNAL_DIRECTIONS), _sha256_bytes(hash_chunks), arrays)


class MultiFactorPipeline(BasePipeline):
    """Pipeline B: multi-factor design or module ablation over a registered objective."""

    def configurations(self) -> List[Tuple[str, Dict[str, Any]]]:
        p = self.profile
        if p.domain_type == "ablation_study":
            modules = p.ablation.modules
            centers = {}
            if p.data_config.factor_ranges:
                centers = build_design(p.data_config.factor_ranges, ExperimentDesign(kind="ccd", center_points=1))[-1]
            configs = [("full", {"modules": {m: True for m in modules}, "factors": centers})]
            for m in modules:
                configs.append((f"w/o {m}", {"modules": {x: x != m for x in modules}, "factors": centers}))
            if p.ablation.include_baseline:
                configs.append(("baseline", {"modules": {m: False for m in modules}, "factors": centers}))
            return configs
        points = build_design(p.data_config.factor_ranges, p.design)
        configs, seen = [], {}
        for pt in points:
            label = ",".join(f"{k}={v:.6g}" for k, v in pt.items())
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1:
                label = f"{label}#{seen[label]}"
            configs.append((label, {"factors": pt}))
        return configs

    def execute(self) -> PipelineOutput:
        p = self.profile
        sim = p.data_config.simulator
        if sim is None:
            raise ValueError(f"{p.domain_type} requires data_config.simulator naming a registered objective")
        if sim.name not in OBJECTIVE_REGISTRY:
            raise KeyError(f"unregistered objective '{sim.name}'; available: {sorted(OBJECTIVE_REGISTRY)}")
        objective = OBJECTIVE_REGISTRY[sim.name]
        configs = self.configurations()
        raw: Dict[str, Dict[str, List[float]]] = {}
        metric_keys: Optional[List[str]] = None
        for ci, (label, cfg) in enumerate(configs):
            raw[label] = {}
            for r in range(p.n_repeats):
                rng = np.random.default_rng([p.random_seed, ci, r])
                t0 = time.perf_counter()
                result = dict(objective(cfg, sim.params, rng))
                elapsed = time.perf_counter() - t0
                if metric_keys is None:
                    metric_keys = sorted(result)
                if sorted(result) != metric_keys:
                    raise ValueError(f"objective returned inconsistent metric keys for '{label}'")
                for k, v in result.items():
                    fv = float(v)
                    if not math.isfinite(fv):
                        raise ValueError(f"objective returned non-finite '{k}' for '{label}'")
                    raw[label].setdefault(k, []).append(fv)
                raw[label].setdefault("runtime_s", []).append(elapsed)
        provenance = {"objective": sim.name, "params": _jsonable(sim.params),
                      "configs": _jsonable(configs), "seed": p.random_seed, "n_repeats": p.n_repeats}
        digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()
        arrays: Dict[str, np.ndarray] = {"config_labels": np.array([c[0] for c in configs])}
        if p.data_config.factor_ranges and p.domain_type != "ablation_study":
            names = list(p.data_config.factor_ranges)
            arrays["design_matrix"] = np.array([[c[1]["factors"][n] for n in names] for c in configs], float)
            arrays["factor_names"] = np.array(names)
        units = {k: "" for k in (metric_keys or [])}
        units["runtime_s"] = "s"
        return PipelineOutput(raw, units, {"runtime_s": "min"}, digest, arrays)


register_pipeline("parameter_optimization")(MultiFactorPipeline)
register_pipeline("ablation_study")(MultiFactorPipeline)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class ExperimentHarness:
    """Runs the pipeline registered for ``profile.domain_type`` and writes a
    schema-valid ``ExperimentResultBundle`` plus raw ``.npz`` series."""

    def __init__(self, profile: TopicProfile, results_dir: Union[str, Path] = "results",
                 repo_root: Union[str, Path, None] = None, plugin_dir: Union[str, Path, None] = None):
        self.profile = profile
        self.results_dir = Path(results_dir)
        self.repo_root = Path(repo_root) if repo_root else None
        if plugin_dir is not None:
            load_plugins(plugin_dir)

    @property
    def output_dir(self) -> Path:
        return self.results_dir / self.profile.profile_id

    def run(self) -> ExperimentResultBundle:
        p = self.profile
        if p.domain_type not in PIPELINE_REGISTRY:
            raise KeyError(f"no pipeline registered for domain '{p.domain_type}'")
        commit = git_commit_hash(self.repo_root)
        out = PIPELINE_REGISTRY[p.domain_type](p).execute()

        available = set().union(*[set(v) for v in out.raw.values()])
        missing = [m for m in p.metrics_target if m not in available]
        if missing:
            raise ValueError(f"metrics_target {missing} not produced; available: {sorted(available)}")

        directions = {**out.directions, **p.metrics_direction}
        methods_metrics: Dict[str, Dict[str, MetricItem]] = {
            method: {name: summarize(name, vals, out.units.get(name, "")) for name, vals in metrics.items()}
            for method, metrics in out.raw.items()
        }
        _mark_best(methods_metrics, directions)
        tests = _paired_tests(out.raw, methods_metrics, p.metrics_target)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self.output_dir / "raw_series.npz"
        arrays = dict(out.arrays)
        for method, metrics in out.raw.items():
            for name, vals in metrics.items():
                arrays[f"metric__{method}__{name}"] = np.asarray(vals, float)
        np.savez(npz_path, **arrays)
        metrics_path = self.output_dir / "metrics.json"

        bundle = ExperimentResultBundle(
            profile_id=p.profile_id,
            git_commit=commit,
            dataset_sha256=out.dataset_sha256,
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            methods_metrics=methods_metrics,
            statistical_tests=_jsonable(tests),
            artifacts_paths={"metrics": str(metrics_path), "raw_series": str(npz_path)},
        )
        bundle.save(metrics_path)
        return bundle
