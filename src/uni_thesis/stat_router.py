"""Automatic statistical test routing.

Given repeated-run samples per method/configuration, ``StatisticalRouter``
diagnoses distributions (Shapiro-Wilk per group, Levene across groups) and
routes to the appropriate test:

* two groups: paired / independent t-test (+ Cohen's d) when normal with equal
  variances, Welch t-test when normal with unequal variances (independent
  only), otherwise Wilcoxon signed-rank / Mann-Whitney U (+ rank-biserial r);
* k > 2 groups: one-way ANOVA + Tukey HSD, otherwise Kruskal-Wallis H;
* multi-factor designs: second-order response-surface OLS fit with R^2,
  adjusted R^2, model F, p and the analytic stationary point.

Every result is a JSON-serializable dict with ``test``, ``metric``, exact
``p_value`` and ``significance`` (``***``/``**``/``*``/``ns``) so it can be
written to ``ExperimentResultBundle.statistical_tests``.
"""

from __future__ import annotations

import itertools
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

from .schema import ExperimentResultBundle

__all__ = [
    "significance_stars",
    "cohens_d",
    "rank_biserial_paired",
    "rank_biserial_independent",
    "fit_response_surface",
    "rsm_predict",
    "StatisticalRouter",
    "load_raw_samples",
]


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _arr(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("samples must be a non-empty 1-D array of finite values")
    return a


def cohens_d(a: Sequence[float], b: Sequence[float], paired: bool) -> float:
    """Cohen's d_z for paired samples, pooled-SD Cohen's d for independent ones."""
    x, y = _arr(a), _arr(b)
    if paired:
        d = x - y
        sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
        return float(np.mean(d) / sd) if sd > 0 else 0.0
    nx, ny = x.size, y.size
    pooled = math.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else 0.0


def rank_biserial_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-) over non-zero differences."""
    d = _arr(a) - _arr(b)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos, w_neg = float(ranks[d > 0].sum()), float(ranks[d < 0].sum())
    return (w_pos - w_neg) / (w_pos + w_neg)


def rank_biserial_independent(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank-biserial r = 2*U_a/(n_a*n_b) - 1 (positive when a tends to exceed b)."""
    x, y = _arr(a), _arr(b)
    u_a = float(stats.mannwhitneyu(x, y, alternative="two-sided").statistic)
    return 2.0 * u_a / (x.size * y.size) - 1.0


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fn(*args, **kwargs)
    return res, sorted({str(w.message) for w in caught})


def _finite(p: float) -> bool:
    return p is not None and math.isfinite(float(p))


# --------------------------------------------------------------------------- #
# Response surface methodology
# --------------------------------------------------------------------------- #
def _rsm_terms(names: Sequence[str]) -> List[Tuple[str, Tuple[int, ...]]]:
    k = len(names)
    terms: List[Tuple[str, Tuple[int, ...]]] = [("1", ())]
    terms += [(names[i], (i,)) for i in range(k)]
    terms += [(f"{names[i]}^2", (i, i)) for i in range(k)]
    terms += [(f"{names[i]}*{names[j]}", (i, j)) for i, j in itertools.combinations(range(k), 2)]
    return terms


def _rsm_matrix(X: np.ndarray, terms: Sequence[Tuple[str, Tuple[int, ...]]]) -> np.ndarray:
    cols = []
    for _, idx in terms:
        col = np.ones(X.shape[0])
        for i in idx:
            col = col * X[:, i]
        cols.append(col)
    return np.column_stack(cols)


def fit_response_surface(X: Union[np.ndarray, Sequence[Sequence[float]]], y: Sequence[float],
                         names: Sequence[str]) -> Dict[str, Any]:
    """Second-order polynomial OLS: y = b0 + sum b_i x_i + sum b_ii x_i^2 + sum b_ij x_i x_j."""
    Xa = np.asarray(X, dtype=float)
    ya = _arr(y)
    if Xa.ndim != 2 or Xa.shape[0] != ya.size or Xa.shape[1] != len(names):
        raise ValueError("X must be (n, k) with k == len(names) and n == len(y)")
    terms = _rsm_terms(names)
    A = _rsm_matrix(Xa, terms)
    n, p_all = A.shape
    p = p_all - 1
    if n <= p_all:
        raise ValueError(f"RSM needs more than {p_all} observations, got {n}")
    if np.linalg.matrix_rank(A) < p_all:
        raise ValueError("RSM design matrix is rank deficient; add design points")
    beta, *_ = np.linalg.lstsq(A, ya, rcond=None)
    fitted = A @ beta
    sse = float(np.sum((ya - fitted) ** 2))
    sst = float(np.sum((ya - ya.mean()) ** 2))
    ssr = sst - sse
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    df_res = n - p - 1
    if sse > 0:
        f_val = (ssr / p) / (sse / df_res)
        p_val = float(stats.f.sf(f_val, p, df_res))
    else:
        f_val, p_val = float("inf"), 0.0

    k = len(names)
    b = beta[1:1 + k]
    B = np.zeros((k, k))
    for (label, idx), coef in zip(terms, beta):
        if len(idx) == 2:
            i, j = idx
            if i == j:
                B[i, i] = coef
            else:
                B[i, j] = B[j, i] = coef / 2.0
    stationary: Optional[Dict[str, Any]] = None
    if abs(np.linalg.det(B)) > 1e-12 * max(1.0, float(np.max(np.abs(B))) ** k):
        xs = -0.5 * np.linalg.solve(B, b)
        eig = np.linalg.eigvalsh(B)
        nature = "maximum" if np.all(eig < 0) else "minimum" if np.all(eig > 0) else "saddle"
        ys = float((_rsm_matrix(xs[None, :], terms) @ beta)[0])
        stationary = {"point": {nm: float(v) for nm, v in zip(names, xs)}, "predicted_response": ys,
                      "nature": nature, "eigenvalues": [float(e) for e in eig]}
    return {
        "test": "rsm_quadratic_ols",
        "factors": list(names),
        "terms": [t for t, _ in terms],
        "coefficients": {t: float(c) for (t, _), c in zip(terms, beta)},
        "n": int(n),
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "f_value": float(f_val) if math.isfinite(f_val) else None,
        "df_model": int(p),
        "df_resid": int(df_res),
        "p_value": float(min(max(p_val, 0.0), 1.0)),
        "significance": significance_stars(p_val),
        "stationary_point": stationary,
    }


def rsm_predict(model: Mapping[str, Any], X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
    names = model["factors"]
    terms = _rsm_terms(names)
    beta = np.array([model["coefficients"][t] for t, _ in terms])
    return _rsm_matrix(np.atleast_2d(np.asarray(X, dtype=float)), terms) @ beta


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
class StatisticalRouter:
    def __init__(self, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    # -- diagnostics ---------------------------------------------------------
    def normality(self, sample: Sequence[float]) -> Dict[str, Any]:
        x = _arr(sample)
        if x.size < 3 or np.ptp(x) == 0:
            return {"test": "shapiro_wilk", "statistic": None, "p_value": None, "normal": False,
                    "reason": "n < 3 or zero range: normality not assessable, treated as non-normal"}
        res, notes = _quiet(stats.shapiro, x)
        p = float(res.pvalue)
        return {"test": "shapiro_wilk", "statistic": float(res.statistic), "p_value": p,
                "normal": bool(p > self.alpha), "notes": notes}

    def homogeneity(self, groups: Sequence[Sequence[float]]) -> Dict[str, Any]:
        arrs = [_arr(g) for g in groups]
        if any(a.size < 2 for a in arrs) or all(np.ptp(a) == 0 for a in arrs):
            return {"test": "levene", "statistic": None, "p_value": None, "equal_var": False,
                    "reason": "insufficient variability for Levene test"}
        res, notes = _quiet(stats.levene, *arrs, center="median")
        p = float(res.pvalue)
        if not _finite(p):
            return {"test": "levene", "statistic": None, "p_value": None, "equal_var": False, "notes": notes}
        return {"test": "levene", "statistic": float(res.statistic), "p_value": p,
                "equal_var": bool(p > self.alpha), "notes": notes}

    def diagnose(self, groups: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
        norm = {k: self.normality(v) for k, v in groups.items()}
        lev = self.homogeneity(list(groups.values()))
        return {"normality": norm, "levene": lev,
                "all_normal": all(n["normal"] for n in norm.values()), "equal_var": lev["equal_var"]}

    # -- two groups ----------------------------------------------------------
    def compare_two(self, a: Sequence[float], b: Sequence[float], paired: bool, metric: str = "",
                    name_a: str = "a", name_b: str = "b") -> Dict[str, Any]:
        x, y = _arr(a), _arr(b)
        if paired and x.size != y.size:
            raise ValueError("paired comparison requires equal sample sizes")
        diag = self.diagnose({name_a: x, name_b: y})
        if paired:
            diag["normality"]["difference"] = self.normality(x - y)
            parametric = diag["normality"]["difference"]["normal"]
        else:
            parametric = diag["all_normal"]
        base = {"comparison": "pairwise", "metric": metric, "method_a": name_a, "method_b": name_b,
                "n_a": int(x.size), "n_b": int(y.size), "paired": paired,
                "mean_a": float(x.mean()), "mean_b": float(y.mean())}

        if paired and np.allclose(x - y, 0.0):
            return {**base, "test": "identical_samples", "statistic": 0.0, "p_value": 1.0,
                    "significance": "ns", "effect_size": {"name": "none", "value": 0.0}, "diagnostics": diag}

        if parametric and (paired or diag["equal_var"]):
            name = "paired_t_test" if paired else "student_t_test"
            res, notes = _quiet(stats.ttest_rel, x, y) if paired else _quiet(stats.ttest_ind, x, y, equal_var=True)
            effect = {"name": "cohens_d_z" if paired else "cohens_d", "value": cohens_d(x, y, paired)}
        elif parametric:
            name = "welch_t_test"
            res, notes = _quiet(stats.ttest_ind, x, y, equal_var=False)
            effect = {"name": "cohens_d", "value": cohens_d(x, y, False)}
        elif paired:
            name = "wilcoxon_signed_rank"
            res, notes = _quiet(stats.wilcoxon, x, y, zero_method="wilcox", alternative="two-sided")
            effect = {"name": "rank_biserial", "value": rank_biserial_paired(x, y)}
        else:
            name = "mann_whitney_u"
            res, notes = _quiet(stats.mannwhitneyu, x, y, alternative="two-sided")
            effect = {"name": "rank_biserial", "value": rank_biserial_independent(x, y)}
        p = float(res.pvalue)
        if not _finite(p):
            raise ValueError(f"{name} produced a non-finite p-value for metric '{metric}'")
        return {**base, "test": name, "statistic": float(res.statistic), "p_value": p,
                "significance": significance_stars(p), "effect_size": effect, "diagnostics": diag, "notes": notes}

    # -- k groups ------------------------------------------------------------
    def compare_multi(self, groups: Mapping[str, Sequence[float]], metric: str = "") -> Dict[str, Any]:
        if len(groups) < 3:
            raise ValueError("compare_multi requires at least three groups")
        names = list(groups)
        arrs = [_arr(groups[n]) for n in names]
        diag = self.diagnose(dict(zip(names, arrs)))
        n_total = sum(a.size for a in arrs)
        k = len(arrs)
        base = {"comparison": "multi_group", "metric": metric, "groups": names, "diagnostics": diag}
        if diag["all_normal"] and diag["equal_var"]:
            res, notes = _quiet(stats.f_oneway, *arrs)
            grand = np.concatenate(arrs).mean()
            ss_between = sum(a.size * (a.mean() - grand) ** 2 for a in arrs)
            ss_total = float(np.sum((np.concatenate(arrs) - grand) ** 2))
            tukey, tnotes = _quiet(stats.tukey_hsd, *arrs)
            posthoc = []
            for i, j in itertools.combinations(range(k), 2):
                pij = float(tukey.pvalue[i, j])
                posthoc.append({"method_a": names[i], "method_b": names[j], "mean_diff": float(tukey.statistic[i, j]),
                                "p_value": pij, "significance": significance_stars(pij)})
            p = float(res.pvalue)
            return {**base, "test": "one_way_anova", "statistic": float(res.statistic), "p_value": p,
                    "significance": significance_stars(p),
                    "effect_size": {"name": "eta_squared", "value": float(ss_between / ss_total) if ss_total > 0 else 0.0},
                    "posthoc": {"test": "tukey_hsd", "pairs": posthoc}, "notes": notes + tnotes}
        res, notes = _quiet(stats.kruskal, *arrs)
        h = float(res.statistic)
        p = float(res.pvalue)
        if not _finite(p):
            raise ValueError(f"kruskal produced a non-finite p-value for metric '{metric}'")
        return {**base, "test": "kruskal_wallis_h", "statistic": h, "p_value": p,
                "significance": significance_stars(p),
                "effect_size": {"name": "epsilon_squared", "value": float(h * (n_total + 1) / (n_total ** 2 - 1))},
                "notes": notes}

    # -- orchestration -------------------------------------------------------
    def analyze(self, samples: Mapping[str, Mapping[str, Sequence[float]]], metrics: Sequence[str],
                proposed: str, paired: bool, design: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        """``samples[method][metric]`` -> list of test dicts.  ``design`` (optional)
        = {"factor_names": [...], "points": {method: [x1, x2, ...]}} enables RSM."""
        if proposed not in samples:
            raise KeyError(f"proposed method '{proposed}' not in samples")
        out: List[Dict[str, Any]] = []
        for metric in metrics:
            holders = {m: s[metric] for m, s in samples.items() if metric in s}
            if metric not in samples[proposed]:
                continue
            for other in holders:
                if other == proposed:
                    continue
                out.append(self.compare_two(holders[proposed], holders[other], paired, metric, proposed, other))
            if len(holders) >= 3 and all(_arr(v).size >= 2 for v in holders.values()):
                out.append(self.compare_multi(holders, metric))
            if design:
                names = list(design["factor_names"])
                X, y = [], []
                for m, vals in holders.items():
                    for v in _arr(vals):
                        X.append(design["points"][m])
                        y.append(v)
                rsm = fit_response_surface(np.asarray(X, float), y, names)
                out.append({**rsm, "metric": metric, "comparison": "response_surface"})
        return json.loads(json.dumps(out, allow_nan=False))

    def analyze_results_dir(self, profile_dir: Union[str, Path], metrics: Sequence[str], proposed: str,
                            paired: bool) -> ExperimentResultBundle:
        """Re-route the statistics of an existing run and rewrite its metrics.json
        (statistical_tests + SHA-256)."""
        pdir = Path(profile_dir)
        bundle = ExperimentResultBundle.load(pdir / "metrics.json")
        samples, design = load_raw_samples(pdir / "raw_series.npz")
        bundle.statistical_tests = self.analyze(samples, metrics, proposed, paired, design)
        bundle.save(pdir / "metrics.json")
        return bundle


def load_raw_samples(npz_path: Union[str, Path]) -> Tuple[Dict[str, Dict[str, List[float]]], Optional[Dict[str, Any]]]:
    samples: Dict[str, Dict[str, List[float]]] = {}
    design = None
    with np.load(npz_path, allow_pickle=False) as z:
        for key in z.files:
            if key.startswith("metric__"):
                method, metric = key[len("metric__"):].rsplit("__", 1)
                samples.setdefault(method, {})[metric] = [float(v) for v in z[key]]
        if "design_matrix" in z.files:
            labels = [str(s) for s in z["config_labels"]]
            design = {"factor_names": [str(s) for s in z["factor_names"]],
                      "points": {lab: [float(v) for v in row] for lab, row in zip(labels, z["design_matrix"])}}
    return samples, design
