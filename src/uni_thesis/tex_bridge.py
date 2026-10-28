"""Zero-hallucination LaTeX bridge.

Reads a signed ``metrics.json`` (``ExperimentResultBundle``) and writes

* ``auto_metrics.tex`` – ``\\providecommand`` macros (letters-only names) for every
  metric statistic, proposed-vs-other differences, statistical test results and
  response-surface figures;
* ``auto_tables.tex`` – booktabs three-line tables (comparison and ablation)
  whose cells reference those macros, with the best value in ``\\mathbf``, the
  runner-up underlined and significance stars as superscripts.

Both files carry ``% metrics_sha256: <sha256 of metrics.json bytes>`` so the
anti-hallucination audit can detect stale macros.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .schema import ExperimentResultBundle

__all__ = [
    "MacroNameCollision",
    "macro_name",
    "format_number",
    "format_p_value",
    "file_sha256",
    "build_macros",
    "render_macros",
    "render_tables",
    "generate_tex_artifacts",
    "SHA_HEADER_KEY",
]

SHA_HEADER_KEY = "metrics_sha256"
PathLike = Union[str, Path]

_DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
_SYMBOL_WORDS = {"=": "Eq", "+": "Plus", "-": "Minus", ".": "Pt", "#": "Rep", "%": "Pct", "<": "Lt", ">": "Gt",
                 "*": "Times", "/": " ", "_": " ", ",": " And "}
_LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                  "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_TEST_LABELS = {
    "paired_t_test": "paired t-test", "student_t_test": "Student t-test", "welch_t_test": "Welch t-test",
    "wilcoxon_signed_rank": "Wilcoxon signed-rank test", "mann_whitney_u": "Mann-Whitney U test",
    "one_way_anova": "one-way ANOVA", "kruskal_wallis_h": "Kruskal-Wallis H test",
    "rsm_quadratic_ols": "second-order RSM (OLS)", "identical_samples": "identical samples",
}


class MacroNameCollision(ValueError):
    """Two distinct sources map to the same LaTeX macro name."""


def macro_name(*parts: str) -> str:
    """CamelCase, letters-only macro identifier built from arbitrary labels."""
    words: List[str] = []
    for part in parts:
        text = "".join(_SYMBOL_WORDS.get(ch, ch) for ch in str(part))
        text = re.sub(r"\d", lambda m: f" {_DIGIT_WORDS[int(m.group())]} ", text)
        for token in re.split(r"[^A-Za-z]+", text):
            if token:
                words.append(token if token.isupper() and len(token) > 1 else token[:1].upper() + token[1:])
    name = "".join(words)
    if not name or not name.isalpha():
        raise ValueError(f"cannot derive a letters-only macro name from {parts!r}")
    return name


def latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in text)


def format_number(x: float, sig: int = 4) -> str:
    """Significant-figure formatting; scientific notation for very small/large magnitudes."""
    v = float(x)
    if not math.isfinite(v):
        raise ValueError("cannot format a non-finite number")
    if v == 0:
        return "0"
    mag = math.floor(math.log10(abs(v)))
    if mag < -3 or mag >= 6:
        mant, exp = f"{v:.{sig - 1}e}".split("e")
        body = f"{mant}\\times 10^{{{int(exp)}}}"
        return f"\\ensuremath{{{body}}}"
    decimals = max(sig - 1 - mag, 0)
    s = f"{v:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return f"\\ensuremath{{{s}}}" if s.startswith("-") else s


def format_p_value(p: float) -> str:
    """``< 0.001`` below the reporting floor, otherwise three significant figures
    (so values near a threshold such as 0.0496 are never rounded across it)."""
    p = float(p)
    if p < 0.001:
        return r"\ensuremath{< 0.001}"
    return f"{p:#.3g}" if p < 1 else "1"


def file_sha256(path: PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Derived facts
# --------------------------------------------------------------------------- #
def _infer_proposed(bundle: ExperimentResultBundle) -> str:
    pairwise = [t["method_a"] for t in bundle.statistical_tests
                if t.get("comparison") == "pairwise" and t.get("method_a") in bundle.methods_metrics]
    if pairwise:
        return Counter(pairwise).most_common(1)[0][0]
    for method, metrics in bundle.methods_metrics.items():
        if any(m.is_best for m in metrics.values()):
            return method
    return next(iter(bundle.methods_metrics))


def _metric_names(bundle: ExperimentResultBundle) -> List[str]:
    names: List[str] = []
    for metrics in bundle.methods_metrics.values():
        for k in metrics:
            if k not in names:
                names.append(k)
    return names


def _direction(bundle: ExperimentResultBundle, metric: str, methods: Sequence[str]) -> Optional[str]:
    """'max' / 'min' / 'min_abs' inferred from which mean the engine flagged ``is_best``."""
    vals = {m: bundle.methods_metrics[m][metric].mean for m in methods if metric in bundle.methods_metrics[m]}
    best = [m for m in vals if bundle.methods_metrics[m][metric].is_best]
    if not best or len(set(vals.values())) < 2:
        return None
    b = vals[best[0]]
    if b >= max(vals.values()):
        return "max"
    if b <= min(vals.values()):
        return "min"
    if abs(b) <= min(abs(v) for v in vals.values()):
        return "min_abs"
    return None


def _ranking(bundle: ExperimentResultBundle, metric: str, methods: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """(best, runner-up) among ``methods`` for ``metric``."""
    direction = _direction(bundle, metric, list(bundle.methods_metrics))
    vals = [(bundle.methods_metrics[m][metric].mean, m) for m in methods if metric in bundle.methods_metrics[m]]
    if direction is None or len(vals) < 2:
        return None, None
    key = (lambda t: abs(t[0])) if direction == "min_abs" else (lambda t: t[0])
    ordered = sorted(vals, key=key, reverse=(direction == "max"))
    best, second = ordered[0], ordered[1]
    if key(best) == key(second):
        return None, None
    return best[1], second[1]


def _pairwise_index(bundle: ExperimentResultBundle) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for t in bundle.statistical_tests:
        if t.get("comparison") == "pairwise":
            idx[(t["method_a"], t["method_b"], t["metric"])] = t
    return idx


# --------------------------------------------------------------------------- #
# Macros
# --------------------------------------------------------------------------- #
class _MacroTable:
    def __init__(self) -> None:
        self.macros: Dict[str, str] = {}
        self.origin: Dict[str, str] = {}

    def add(self, name: str, value: str, origin: str) -> None:
        if name in self.macros and self.origin[name] != origin:
            raise MacroNameCollision(f"macro \\{name} produced by both '{self.origin[name]}' and '{origin}'")
        self.macros[name] = value
        self.origin[name] = origin


def build_macros(bundle: ExperimentResultBundle, proposed: Optional[str] = None,
                 sig: int = 4) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Return ``(macros, context)``; ``context`` holds the names the tables reuse."""
    proposed = proposed or _infer_proposed(bundle)
    if proposed not in bundle.methods_metrics:
        raise KeyError(f"proposed method '{proposed}' not in bundle")
    tab = _MacroTable()
    cell_macros: Dict[Tuple[str, str], Tuple[str, str]] = {}

    stat_fields = [("Mean", "mean"), ("Std", "std"), ("Median", "median"), ("Iqr", "iqr")]
    for method, metrics in bundle.methods_metrics.items():
        aliases = [method] + (["Proposed"] if method == proposed else [])
        for metric, item in metrics.items():
            for alias in aliases:
                origin = f"{method}|{metric}"
                base = "Metric" + macro_name(alias, metric)
                for suffix, field in stat_fields:
                    tab.add(base + suffix, format_number(getattr(item, field), sig), f"{origin}|{field}")
                tab.add(base + "CiLow", format_number(item.ci_95[0], sig), f"{origin}|ci_low")
                tab.add(base + "CiHigh", format_number(item.ci_95[1], sig), f"{origin}|ci_high")
                if item.unit:
                    tab.add(base + "Unit", latex_escape(item.unit), f"{origin}|unit")
                if alias == method:
                    cell_macros[(method, metric)] = (base + "Mean", base + "Std")

    prop_metrics = bundle.methods_metrics[proposed]
    for other, metrics in bundle.methods_metrics.items():
        if other == proposed:
            continue
        for metric, item in metrics.items():
            if metric not in prop_metrics:
                continue
            diff = prop_metrics[metric].mean - item.mean
            base = "Diff" + macro_name("Proposed", "Vs", other, metric)
            origin = f"diff|{proposed}|{other}|{metric}"
            tab.add(base + "Abs", format_number(diff, sig), origin + "|abs")
            if item.mean != 0:
                tab.add(base + "RelPct", format_number(100.0 * diff / abs(item.mean), sig), origin + "|rel")

    stars_lookup: Dict[Tuple[str, str], str] = {}
    for t in bundle.statistical_tests:
        kind = t.get("comparison")
        metric = t["metric"]
        if kind == "pairwise":
            a, b = t["method_a"], t["method_b"]
            heads = [macro_name(a, "Vs", b, metric)]
            if a == proposed:
                heads.append(macro_name("Proposed", "Vs", b, metric))
                stars_lookup[(b, metric)] = t["significance"]
            origin = f"pairwise|{a}|{b}|{metric}"
        elif kind == "multi_group":
            heads = [macro_name(metric, "Omnibus")]
            origin = f"multi|{metric}"
        elif kind == "response_surface":
            heads = [macro_name("Rsm", metric)]
            origin = f"rsm|{metric}"
        else:
            continue
        for h in heads:
            st = "Stat" + h
            tab.add(st + "Pval", format_p_value(t["p_value"]), origin + "|p")
            tab.add(st + "Stars", t["significance"], origin + "|stars")
            tab.add(st + "Test", latex_escape(_TEST_LABELS.get(t["test"], t["test"].replace("_", " "))), origin + "|test")
            if t.get("statistic") is not None:
                tab.add(st + "Statistic", format_number(t["statistic"], sig), origin + "|stat")
            eff = t.get("effect_size")
            if isinstance(eff, Mapping) and eff.get("value") is not None:
                tab.add(st + "Effect", format_number(eff["value"], sig), origin + "|eff")
                tab.add(st + "EffectName", latex_escape(str(eff["name"]).replace("_", " ")), origin + "|effname")
            if kind == "response_surface":
                tab.add(st + "Rsq", format_number(t["r2"], sig), origin + "|r2")
                tab.add(st + "RsqAdj", format_number(t["r2_adj"], sig), origin + "|r2adj")
                if t.get("f_value") is not None:
                    tab.add(st + "Fvalue", format_number(t["f_value"], sig), origin + "|f")
                sp = t.get("stationary_point")
                if sp:
                    for factor, val in sp["point"].items():
                        tab.add(st + "Opt" + macro_name(factor), format_number(val, sig), origin + f"|opt|{factor}")
                    tab.add(st + "OptResponse", format_number(sp["predicted_response"], sig), origin + "|optresp")
                    tab.add(st + "OptNature", sp["nature"], origin + "|nature")

    context = {"proposed": proposed, "cells": cell_macros, "stars": stars_lookup}
    return tab.macros, context


def _header(kind: str, metrics_sha: str, bundle: ExperimentResultBundle) -> str:
    return "\n".join([
        f"% {kind} -- AUTO-GENERATED by uni_thesis.tex_bridge. DO NOT EDIT BY HAND.",
        f"% {SHA_HEADER_KEY}: {metrics_sha}",
        f"% bundle_sha256: {bundle.compute_sha256()}",
        f"% profile_id: {bundle.profile_id}",
        f"% git_commit: {bundle.git_commit}",
        f"% generated_utc: {datetime.now(timezone.utc).isoformat()}",
        "",
    ])


def render_macros(macros: Mapping[str, str], header: str) -> str:
    lines = [header]
    for name in sorted(macros):
        lines.append(f"\\providecommand{{\\{name}}}{{{macros[name]}}}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _cell(method: str, metric: str, ctx: Mapping[str, Any], best: Optional[str], second: Optional[str]) -> str:
    mean_m, std_m = ctx["cells"][(method, metric)]
    body = f"\\{mean_m} \\pm \\{std_m}"
    if method == best:
        body = f"\\mathbf{{{body}}}"
    stars = ctx["stars"].get((method, metric))
    sup = f"^{{\\text{{{stars}}}}}" if stars and stars != "ns" else ""
    cell = f"${body}{sup}$"
    return f"\\underline{{{cell}}}" if method == second else cell


def _table(bundle: ExperimentResultBundle, methods: Sequence[str], metrics: Sequence[str], ctx: Mapping[str, Any],
           caption: str, label: str) -> str:
    rankings = {m: _ranking(bundle, m, methods) for m in metrics}
    head = ["Method"]
    for m in metrics:
        unit = next((bundle.methods_metrics[x][m].unit for x in methods if m in bundle.methods_metrics[x]
                     and bundle.methods_metrics[x][m].unit), "")
        head.append(latex_escape(m) + (f" ({latex_escape(unit)})" if unit else ""))
    rows = []
    for method in methods:
        name = latex_escape(method) + (" (proposed)" if method == ctx["proposed"] else "")
        cells = [_cell(method, m, ctx, *rankings[m]) if (method, m) in ctx["cells"] else "--" for m in metrics]
        rows.append(" & ".join([name] + cells) + r" \\")
    tests_used = sorted({_TEST_LABELS.get(t["test"], t["test"]) for t in bundle.statistical_tests
                         if t.get("comparison") == "pairwise" and t["method_a"] == ctx["proposed"]
                         and t["method_b"] in methods and t["metric"] in metrics})
    note = (r"\par\footnotesize Mean $\pm$ SD. Bold: best; underline: second best. "
            r"Superscripts: significance versus the proposed method ($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$"
            + (f"; {latex_escape(', '.join(tests_used))}" if tests_used else "") + ").")
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        " & ".join(head) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}}",
        note,
        r"\end{table}",
        "",
    ])


def render_tables(bundle: ExperimentResultBundle, ctx: Mapping[str, Any], header: str,
                  metrics: Optional[Sequence[str]] = None) -> str:
    metrics = list(metrics) if metrics else [m for m in _metric_names(bundle) if m != "runtime_s"] or _metric_names(bundle)
    methods = list(bundle.methods_metrics)
    ablation = [m for m in methods if m == "full" or m.startswith("w/o ")]
    if ablation and "baseline" in methods:
        ablation.append("baseline")
    is_ablation = "full" in ablation and any(m.startswith("w/o ") for m in ablation)
    parts = [header, "% Requires: \\usepackage{booktabs,amsmath,graphicx} and \\input{generated/auto_metrics.tex}.", ""]
    comparison = [m for m in methods if not is_ablation or m not in ablation]
    if len(comparison) >= 2:
        parts.append(_table(bundle, comparison, metrics, ctx, "Comparison with baseline methods",
                            "tab:auto-comparison"))
    if is_ablation:
        parts.append(_table(bundle, ablation, metrics, ctx, "Ablation study", "tab:auto-ablation"))
    return "\n".join(parts)


def generate_tex_artifacts(metrics_json_path: PathLike, out_dir: PathLike, proposed: Optional[str] = None,
                           metrics: Optional[Sequence[str]] = None, sig: int = 4) -> Dict[str, str]:
    """Verify ``metrics.json`` and write ``auto_metrics.tex`` + ``auto_tables.tex`` into ``out_dir``."""
    mpath = Path(metrics_json_path)
    bundle = ExperimentResultBundle.load(mpath)
    sha = file_sha256(mpath)
    macros, ctx = build_macros(bundle, proposed, sig)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp, tp = out / "auto_metrics.tex", out / "auto_tables.tex"
    mp.write_text(render_macros(macros, _header("auto_metrics.tex", sha, bundle)), encoding="utf-8")
    tp.write_text(render_tables(bundle, ctx, _header("auto_tables.tex", sha, bundle), metrics), encoding="utf-8")
    return {"auto_metrics": str(mp), "auto_tables": str(tp), SHA_HEADER_KEY: sha, "proposed": ctx["proposed"],
            "n_macros": str(len(macros))}
