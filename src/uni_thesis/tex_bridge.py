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
    "format_p_relation",
    "file_sha256",
    "build_macros",
    "render_macros",
    "render_tables",
    "generate_tex_artifacts",
    "method_groups",
    "rsm_equation",
    "render_flags",
    "render_figures",
    "render_literature",
    "SUPPORTED_LANGS",
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


_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "method": "Method", "proposed": " (proposed)", "comparison": "Comparison with baseline methods",
        "ablation": "Ablation study", "design": "Central composite design and observed responses",
        "run": "Run", "design_note": "Responses: mean $\\pm$ SD over repeated runs.",
        "note": "Mean $\\pm$ SD. Bold: best; underline: second best. Superscripts: significance versus the "
                "proposed method ($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$",
        "anova": "ANOVA of the second-order response-surface model ({m})",
        "coef": "Regression coefficients of the response-surface model ({m})",
        "source": "Source", "term": "Term", "coefficient": "Coefficient", "sig": "Sig.", "intercept": "Intercept",
        "adj": "$R^2$ = {r}, adjusted $R^2$ = {a}.",
        "Model": "Model", "Residual": "Residual", "Lack of fit": "Lack of fit", "Pure error": "Pure error",
        "Total": "Total", "maximum": "maximum", "minimum": "minimum", "saddle": "saddle point",
        "no_refs": "No DOI-verified reference was retrieved for this topic.",
        "limitation": " Reported limitation: ",
    },
    "zh": {
        "method": "方法", "proposed": "（本文）", "comparison": "与基线方法的对比", "ablation": "消融实验",
        "design": "中心复合设计及实测响应", "run": "试验号", "design_note": "响应值：重复试验的均值 $\\pm$ 标准差。",
        "note": "均值 $\\pm$ 标准差。粗体：最优；下划线：次优。上标为相对本文方法的显著性"
                "（$^{*}p<0.05$，$^{**}p<0.01$，$^{***}p<0.001$",
        "anova": "二阶响应面模型方差分析（{m}）", "coef": "响应面模型回归系数（{m}）",
        "source": "来源", "term": "项", "coefficient": "系数", "sig": "显著性", "intercept": "常数项",
        "adj": "$R^2$ = {r}，调整 $R^2$ = {a}。",
        "Model": "模型", "Residual": "残差", "Lack of fit": "失拟项", "Pure error": "纯误差", "Total": "总计",
        "maximum": "极大值点", "minimum": "极小值点", "saddle": "鞍点",
        "no_refs": "本课题未检索到经 DOI 核验的文献。", "limitation": " 文献自述局限：",
    },
}
_TEST_LABELS_ZH = {
    "paired_t_test": "配对 t 检验", "student_t_test": "Student t 检验", "welch_t_test": "Welch t 检验",
    "wilcoxon_signed_rank": "Wilcoxon 符号秩检验", "mann_whitney_u": "Mann-Whitney U 检验",
    "one_way_anova": "单因素方差分析", "kruskal_wallis_h": "Kruskal-Wallis H 检验",
    "identical_samples": "样本完全相同",
}
SUPPORTED_LANGS = tuple(_STRINGS)


def _s(lang: str, key: str) -> str:
    if lang not in _STRINGS:
        raise ValueError(f"unsupported language '{lang}'; choose from {SUPPORTED_LANGS}")
    return _STRINGS[lang][key]


def _test_label(test: str, lang: str = "en") -> str:
    table = _TEST_LABELS_ZH if lang == "zh" else _TEST_LABELS
    return table.get(test, _TEST_LABELS.get(test, test.replace("_", " ")))


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


def format_p_relation(p: float) -> str:
    """``= 0.0123`` or ``< 0.001``, to follow an italic $p$ in running text."""
    p = float(p)
    return r"\ensuremath{< 0.001}" if p < 0.001 else f"\\ensuremath{{= {format_p_value(p)}}}"


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
                 sig: int = 4, lang: str = "en") -> Tuple[Dict[str, str], Dict[str, Any]]:
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
            tab.add(st + "Test", latex_escape(_test_label(t["test"], lang)), origin + "|test")
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
                    tab.add(st + "OptNature", _s(lang, sp["nature"]), origin + "|nature")

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
    body = f"\\mathbf{{{body}}}" if method == best else f"{{{body}}}"
    stars = ctx["stars"].get((method, metric))
    sup = f"^{{\\text{{{stars}}}}}" if stars and stars != "ns" else ""
    cell = f"${body}{sup}$"
    return f"\\underline{{{cell}}}" if method == second else cell


def _table(bundle: ExperimentResultBundle, methods: Sequence[str], metrics: Sequence[str], ctx: Mapping[str, Any],
           caption: str, label: str, lang: str = "en") -> str:
    rankings = {m: _ranking(bundle, m, methods) for m in metrics}
    head = [_s(lang, "method")]
    for m in metrics:
        unit = next((bundle.methods_metrics[x][m].unit for x in methods if m in bundle.methods_metrics[x]
                     and bundle.methods_metrics[x][m].unit), "")
        head.append(latex_escape(m) + (f" ({latex_escape(unit)})" if unit else ""))
    rows = []
    for method in methods:
        name = latex_escape(method) + (_s(lang, "proposed") if method == ctx["proposed"] else "")
        cells = [_cell(method, m, ctx, *rankings[m]) if (method, m) in ctx["cells"] else "--" for m in metrics]
        rows.append(" & ".join([name] + cells) + r" \\")
    tests_used = sorted({_test_label(t["test"], lang) for t in bundle.statistical_tests
                         if t.get("comparison") == "pairwise" and t["method_a"] == ctx["proposed"]
                         and t["method_b"] in methods and t["metric"] in metrics})
    sep, close = ("；", "）。") if lang == "zh" else ("; ", ").")
    note = (r"\par\footnotesize " + _s(lang, "note")
            + (sep + latex_escape(", ".join(tests_used)) if tests_used else "") + close)
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


def _is_ablation_label(m: str) -> bool:
    return m in ("full", "baseline") or m.startswith("w/o ") or m.startswith("+")


def method_groups(bundle: ExperimentResultBundle, design_labels: Sequence[str] = (),
                  order: Sequence[str] = ()) -> Dict[str, List[str]]:
    """Split methods into ``design`` runs, ``ablation`` configurations and the ``comparison`` set,
    in run order when ``order`` (the harness' ``method_order``) is given."""
    known = list(bundle.methods_metrics)
    methods = [m for m in order if m in bundle.methods_metrics] + [m for m in known if m not in set(order)]
    design_set = set(design_labels)
    design = [m for m in methods if m in design_set]
    rest = [m for m in methods if m not in design_set]
    abl = [m for m in rest if _is_ablation_label(m)]
    is_abl = "full" in abl and any(m.startswith(("w/o ", "+")) for m in abl)
    if not is_abl:
        abl = []
    elif not order and "baseline" in abl:
        others = [m for m in abl if m != "baseline"]
        abl = (["baseline"] + others) if any(m.startswith("+") for m in others) else (others + ["baseline"])
    comparison = [m for m in rest if m not in abl]
    return {"design": design, "ablation": abl, "comparison": comparison}


def _raw_series_info(bundle: ExperimentResultBundle, raw_series: Optional[PathLike]
                     ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """(design, method run order) from the run's ``raw_series.npz`` when available."""
    import numpy as np

    from .stat_router import load_raw_samples

    path = Path(raw_series) if raw_series else (Path(bundle.artifacts_paths["raw_series"])
                                                  if "raw_series" in bundle.artifacts_paths else None)
    if path is None or not path.exists():
        return None, []
    design = load_raw_samples(path)[1]
    with np.load(path, allow_pickle=False) as z:
        order = [str(m) for m in z["method_order"]] if "method_order" in z.files else []
    return design, order


def _signed(v: float, sig: int) -> str:
    return ("- " if v < 0 else "+ ") + format_number(abs(v), sig)


def rsm_equation(test: Mapping[str, Any], sig: int = 4, per_line: int = 4) -> str:
    """Fitted second-order model as an ``aligned`` block, factors coded ``x_1..x_k``."""
    names = list(test["factors"])
    sym = {n: f"x_{{{i + 1}}}" for i, n in enumerate(names)}
    coef = test["coefficients"]
    pieces = []
    for term in test["terms"]:
        if term == "1":
            continue
        if term.endswith("^2"):
            var = sym[term[:-2]] + "^{2}"
        elif "*" in term:
            a, b = term.split("*")
            var = sym[a] + sym[b]
        else:
            var = sym[term]
        pieces.append(f"{_signed(coef[term], sig)}\\,{var}")
    b0 = format_number(coef["1"], sig)
    lines = [" ".join(pieces[i:i + per_line]) for i in range(0, len(pieces), per_line)]
    body = f"\\hat{{y}} &= {b0} " + (lines[0] if lines else "")
    for ln in lines[1:]:
        body += f" \\\\ &\\quad {ln}"
    return f"\\begin{{aligned}}{body}\\end{{aligned}}"


_NA = r"\textit{N/A}"


def _role_macros(bundle: ExperimentResultBundle, macros: Dict[str, str], ctx: Mapping[str, Any],
                 groups: Mapping[str, List[str]], primary: str, title: Optional[str], sig: int,
                 lang: str = "en") -> Dict[str, bool]:
    """Profile-agnostic macros for the generic thesis template; absent facts are ``N/A``."""
    proposed = ctx["proposed"]
    mm = bundle.methods_metrics
    group = groups["ablation"] or groups["comparison"]
    role: Dict[str, str] = {
        "ThesisTitle": latex_escape(title) if title else latex_escape(bundle.profile_id),
        "ThesisProfileId": latex_escape(bundle.profile_id),
        "MetricPrimaryName": latex_escape(primary.replace("_", " ")),
        "MetricPrimaryUnit": "", "MetricPrimaryProposedName": _NA, "MetricPrimaryProposedMean": _NA,
        "MetricPrimaryProposedStd": _NA, "MetricPrimaryProposedCiLow": _NA, "MetricPrimaryProposedCiHigh": _NA,
        "MetricPrimaryRunnerUpName": _NA, "MetricPrimaryRunnerUpMean": _NA, "MetricPrimaryRunnerUpStd": _NA,
        "DiffPrimaryAbs": _NA, "DiffPrimaryRelPct": _NA,
        "StatPrimaryPval": _NA, "StatPrimaryTest": _NA, "StatPrimaryStars": _NA, "StatPrimaryEffect": _NA,
        "StatPrimaryEffectName": _NA, "StatPrimaryN": _NA,
        "StatPrimaryOmnibusPval": _NA, "StatPrimaryOmnibusTest": _NA,
        "StatRsmPrimaryRsq": _NA, "StatRsmPrimaryRsqAdj": _NA, "StatRsmPrimaryFvalue": _NA,
        "StatRsmPrimaryPval": _NA, "StatRsmPrimaryLackOfFitPval": _NA, "StatRsmPrimaryOptResponse": _NA,
        "StatRsmPrimaryOptNature": _NA, "StatRsmPrimaryOptPoint": _NA, "StatRsmPrimaryEquation": _NA,
        "StatRsmPrimaryFactorLegend": _NA,
    }
    if proposed in mm and primary in mm[proposed]:
        item = mm[proposed][primary]
        role.update({"MetricPrimaryProposedName": latex_escape(proposed),
                     "MetricPrimaryUnit": latex_escape(item.unit),
                     "MetricPrimaryProposedMean": format_number(item.mean, sig),
                     "MetricPrimaryProposedStd": format_number(item.std, sig),
                     "MetricPrimaryProposedCiLow": format_number(item.ci_95[0], sig),
                     "MetricPrimaryProposedCiHigh": format_number(item.ci_95[1], sig)})
        others = [m for m in group if m != proposed and primary in mm[m]]
        direction = _direction(bundle, primary, list(mm)) or "max"
        if others:
            key = {"max": lambda m: -mm[m][primary].mean, "min": lambda m: mm[m][primary].mean,
                   "min_abs": lambda m: abs(mm[m][primary].mean)}[direction]
            ru = min(others, key=key)
            r_item = mm[ru][primary]
            diff = item.mean - r_item.mean
            role.update({"MetricPrimaryRunnerUpName": latex_escape(ru),
                         "MetricPrimaryRunnerUpMean": format_number(r_item.mean, sig),
                         "MetricPrimaryRunnerUpStd": format_number(r_item.std, sig),
                         "DiffPrimaryAbs": format_number(diff, sig)})
            if r_item.mean != 0:
                role["DiffPrimaryRelPct"] = format_number(100.0 * diff / abs(r_item.mean), sig)
            for t in bundle.statistical_tests:
                if (t.get("comparison") == "pairwise" and t["metric"] == primary and t["method_a"] == proposed
                        and t["method_b"] == ru):
                    role.update({"StatPrimaryPval": format_p_relation(t["p_value"]), "StatPrimaryStars": t["significance"],
                                 "StatPrimaryTest": latex_escape(_test_label(t["test"], lang)),
                                 "StatPrimaryN": str(t.get("n_a", ""))})
                    eff = t.get("effect_size") or {}
                    if eff.get("value") is not None and eff.get("name") != "none":
                        role["StatPrimaryEffect"] = format_number(eff["value"], sig)
                        role["StatPrimaryEffectName"] = latex_escape(str(eff["name"]).replace("_", " "))
    for t in bundle.statistical_tests:
        if t.get("comparison") == "multi_group" and t["metric"] == primary:
            role["StatPrimaryOmnibusPval"] = format_p_relation(t["p_value"])
            role["StatPrimaryOmnibusTest"] = latex_escape(_test_label(t["test"], lang))
        if t.get("comparison") == "response_surface" and t["metric"] == primary:
            role.update({"StatRsmPrimaryRsq": format_number(t["r2"], sig),
                         "StatRsmPrimaryRsqAdj": format_number(t["r2_adj"], sig),
                         "StatRsmPrimaryPval": format_p_relation(t["p_value"]),
                         "StatRsmPrimaryEquation": rsm_equation(t, sig),
                         "StatRsmPrimaryFactorLegend": ", ".join(
                             f"$x_{{{i + 1}}}$ = {latex_escape(n)}" for i, n in enumerate(t["factors"]))})
            if t.get("f_value") is not None:
                role["StatRsmPrimaryFvalue"] = format_number(t["f_value"], sig)
            lof = [r for r in t.get("anova", []) if r["source"] == "Lack of fit"]
            if lof:
                role["StatRsmPrimaryLackOfFitPval"] = format_p_relation(lof[0]["p_value"])
            sp = t.get("stationary_point")
            if sp:
                role["StatRsmPrimaryOptResponse"] = format_number(sp["predicted_response"], sig)
                role["StatRsmPrimaryOptNature"] = _s(lang, sp["nature"])
                role["StatRsmPrimaryOptPoint"] = ", ".join(
                    f"{latex_escape(k)} = {format_number(v, sig)}" for k, v in sp["point"].items())
    for k, v in role.items():
        if k in macros and macros[k] != v:
            raise MacroNameCollision(f"role macro \\{k} clashes with a generated result macro")
        macros[k] = v
    has_rsm = any(t.get("comparison") == "response_surface" for t in bundle.statistical_tests)
    return {"AutoHasComparison": len(groups["comparison"]) >= 2, "AutoHasAblation": bool(groups["ablation"]),
            "AutoHasRsm": has_rsm, "AutoHasDesign": bool(groups["design"])}


def _rsm_term_macros(bundle: ExperimentResultBundle, macros: Dict[str, str], sig: int) -> None:
    for t in bundle.statistical_tests:
        if t.get("comparison") != "response_surface":
            continue
        head = "Stat" + macro_name("Rsm", t["metric"])
        macros[head + "Equation"] = rsm_equation(t, sig)
        for row in t.get("anova", []):
            if row["source"] == "Lack of fit":
                macros[head + "LackOfFitPval"] = format_p_value(row["p_value"])
        for term, st in t.get("term_stats", {}).items():
            name = head + "Term" + ("Intercept" if term == "1" else macro_name(term.replace("^2", " Sq")))
            macros[name + "Coef"] = format_number(st["coef"], sig)
            macros[name + "Pval"] = format_p_value(st["p_value"])


def render_flags(flags: Mapping[str, bool]) -> str:
    lines = []
    for name, on in flags.items():
        lines.append(f"\\newif\\if{name}")
        lines.append(f"\\{name}{'true' if on else 'false'}")
    return "\n".join(lines) + "\n"


def _design_table(bundle: ExperimentResultBundle, design: Mapping[str, Any], labels: Sequence[str],
                  metrics: Sequence[str], ctx: Mapping[str, Any], sig: int, lang: str = "en") -> str:
    names = list(design["factor_names"])
    head = [_s(lang, "run"), *[latex_escape(n) for n in names], *[latex_escape(m) for m in metrics]]
    rows = []
    for i, lab in enumerate(labels, start=1):
        pt = design["points"][lab]
        cells = [str(i), *[format_number(v, sig) for v in pt]]
        for m in metrics:
            if (lab, m) in ctx["cells"]:
                mean_m, std_m = ctx["cells"][(lab, m)]
                cells.append(f"$\\{mean_m} \\pm \\{std_m}$")
            else:
                cells.append("--")
        rows.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", "\\caption{" + _s(lang, "design") + "}",
        r"\label{tab:auto-design}", r"\small", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        r"\begin{tabular}{c" + "c" * len(names) + "c" * len(metrics) + "}", r"\toprule",
        " & ".join(head) + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}",
        r"\par\footnotesize " + _s(lang, "design_note"), r"\end{table}", ""])


def _anova_tables(bundle: ExperimentResultBundle, metrics: Sequence[str], sig: int, lang: str = "en") -> List[str]:
    out = []
    for t in bundle.statistical_tests:
        if t.get("comparison") != "response_surface" or t["metric"] not in metrics:
            continue
        label = "" if t["metric"] == metrics[0] else "-" + macro_name(t["metric"]).lower()
        rows = []
        for r in t.get("anova", []):
            f = format_number(r["f"], sig) if r.get("f") is not None else ""
            ms = format_number(r["ms"], sig) if r.get("ms") is not None else ""
            pv = format_p_value(r["p_value"]) if r.get("p_value") is not None else ""
            rows.append(f"{_s(lang, r['source'])} & {format_number(r['ss'], sig)} & {r['df']} & {ms} & {f} & {pv} \\\\")
        out.append("\n".join([
            r"\begin{table}[htbp]", r"\centering",
            "\\caption{" + _s(lang, "anova").format(m=latex_escape(t["metric"])) + "}",
            f"\\label{{tab:auto-anova{label}}}", r"\small",
            r"\begin{tabular}{lrrrrr}", r"\toprule", _s(lang, "source") + r" & SS & df & MS & $F$ & $p$ \\", r"\midrule",
            *rows, r"\bottomrule", r"\end{tabular}",
            r"\par\footnotesize " + _s(lang, "adj").format(r=f"\\Stat{macro_name('Rsm', t['metric'])}Rsq",
                                                             a=f"\\Stat{macro_name('Rsm', t['metric'])}RsqAdj"), r"\end{table}", ""]))
        trows = []
        stats_by_term = t.get("term_stats", {})
        for term in [x for x in t["terms"] if x in stats_by_term]:
            st = stats_by_term[term]
            tv = format_number(st["t"], sig) if st.get("t") is not None else ""
            sym = {n: f"x_{{{i + 1}}}" for i, n in enumerate(t["factors"])}
            if term == "1":
                tex = _s(lang, "intercept")
            elif term.endswith("^2"):
                tex = f"${sym[term[:-2]]}^{{2}}$"
            elif "*" in term:
                a, b = term.split("*")
                tex = f"${sym[a]}{sym[b]}$"
            else:
                tex = f"${sym[term]}$"
            stars = st["significance"]
            trows.append(f"{tex} & {format_number(st['coef'], sig)} & {format_number(st['se'], sig)} & {tv} & "
                         f"{format_p_value(st['p_value'])} & {stars} \\\\")
        if trows:
            out.append("\n".join([
                r"\begin{table}[htbp]", r"\centering",
                "\\caption{" + _s(lang, "coef").format(m=latex_escape(t["metric"])) + "}",
                f"\\label{{tab:auto-coef{label}}}", r"\small",
                r"\begin{tabular}{lrrrrc}", r"\toprule", " & ".join([_s(lang, "term"), _s(lang, "coefficient"), "SE", "$t$", "$p$", _s(lang, "sig")]) + r" \\",
                r"\midrule", *trows, r"\bottomrule", r"\end{tabular}",
                r"\par\footnotesize " + ", ".join(f"$x_{{{i + 1}}}$: {latex_escape(n)}"
                                                  for i, n in enumerate(t["factors"])) + ".",
                r"\end{table}", ""]))
    return out


def render_tables(bundle: ExperimentResultBundle, ctx: Mapping[str, Any], header: str,
                  metrics: Optional[Sequence[str]] = None, design: Optional[Mapping[str, Any]] = None,
                  sig: int = 4, order: Sequence[str] = (), lang: str = "en") -> str:
    metrics = list(metrics) if metrics else [m for m in _metric_names(bundle) if m != "runtime_s"] or _metric_names(bundle)
    groups = method_groups(bundle, list(design["points"]) if design else [], order)
    parts = [header, "% Requires: \\usepackage{booktabs,amsmath,graphicx} and \\input{generated/auto_metrics.tex}.", ""]
    if len(groups["comparison"]) >= 2:
        parts.append(_table(bundle, groups["comparison"], metrics, ctx, _s(lang, "comparison"),
                            "tab:auto-comparison", lang))
    if groups["ablation"]:
        parts.append(_table(bundle, groups["ablation"], metrics, ctx, _s(lang, "ablation"), "tab:auto-ablation", lang))
    if design and groups["design"]:
        parts.append(_design_table(bundle, design, groups["design"], metrics, ctx, sig, lang))
    parts += _anova_tables(bundle, metrics, sig, lang)
    return "\n".join(parts)


def render_figures(figures: Sequence[Mapping[str, str]]) -> str:
    """``figures``: dicts with ``path`` (relative to the thesis root, no extension needed),
    ``caption`` and ``label``; optional ``width`` (fraction of ``\\linewidth``)."""
    parts = ["% auto_figures.tex -- AUTO-GENERATED by uni_thesis.tex_bridge. DO NOT EDIT BY HAND.", ""]
    for f in figures:
        width = f.get("width", "1.0")
        parts += [r"\begin{figure}[htbp]", r"\centering",
                  f"\\includegraphics[width={width}\\linewidth]{{{f['path']}}}",
                  f"\\caption{{{f['caption']}}}", f"\\label{{{f['label']}}}", r"\end{figure}", ""]
    return "\n".join(parts)


def render_literature(records: Sequence[Any], lang: str = "en") -> str:
    """Related-work digest citing only DOI-verified records (one ``\\item`` per record)."""
    verified = [r for r in records if r.verified and r.doi]
    parts = ["% auto_literature.tex -- AUTO-GENERATED by uni_thesis.tex_bridge. DO NOT EDIT BY HAND.", ""]
    if not verified:
        parts.append(_s(lang, "no_refs"))
        return "\n".join(parts) + "\n"
    parts.append(r"\begin{itemize}")
    for r in sorted(verified, key=lambda x: (x.year, x.cite_key)):
        colon, stop = ("：", "。") if lang == "zh" else (": ", ".")
        line = f"\\item \\citet{{{r.cite_key}}} ({latex_escape(r.venue)}){colon}{latex_escape(r.methodology_tag)}{stop}"
        if r.limitation_note:
            line += _s(lang, "limitation") + latex_escape(r.limitation_note)
        parts.append(line)
    parts.append(r"\end{itemize}")
    return "\n".join(parts) + "\n"


def generate_tex_artifacts(metrics_json_path: PathLike, out_dir: PathLike, proposed: Optional[str] = None,
                           metrics: Optional[Sequence[str]] = None, sig: int = 4, title: Optional[str] = None,
                           raw_series: Optional[PathLike] = None, lang: str = "en") -> Dict[str, str]:
    """Verify ``metrics.json`` and write ``auto_metrics.tex`` + ``auto_tables.tex`` into ``out_dir``.

    ``metrics`` (default: every metric except runtime) selects table columns and the
    primary metric (first entry) behind the profile-agnostic role macros."""
    mpath = Path(metrics_json_path)
    bundle = ExperimentResultBundle.load(mpath)
    sha = file_sha256(mpath)
    _s(lang, "method")
    macros, ctx = build_macros(bundle, proposed, sig, lang)
    design, order = _raw_series_info(bundle, raw_series)
    groups = method_groups(bundle, list(design["points"]) if design else [], order)
    cols = list(metrics) if metrics else [m for m in _metric_names(bundle) if m != "runtime_s"] or _metric_names(bundle)
    _rsm_term_macros(bundle, macros, sig)
    flags = _role_macros(bundle, macros, ctx, groups, cols[0], title, sig, lang)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp, tp = out / "auto_metrics.tex", out / "auto_tables.tex"
    mp.write_text(render_macros(macros, _header("auto_metrics.tex", sha, bundle)) + render_flags(flags),
                  encoding="utf-8")
    tp.write_text(render_tables(bundle, ctx, _header("auto_tables.tex", sha, bundle), cols, design, sig, order, lang),
                  encoding="utf-8")
    return {"auto_metrics": str(mp), "auto_tables": str(tp), SHA_HEADER_KEY: sha, "proposed": ctx["proposed"],
            "n_macros": str(len(macros)), **{k: str(v).lower() for k, v in flags.items()}}
