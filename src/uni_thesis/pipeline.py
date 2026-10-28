"""End-to-end orchestration behind ``uni-thesis run | audit | pdf``.

``run_profile`` chains the five engine layers for one ``TopicProfile``:

1. literature retrieval + DOI verification (``lit_engine``),
2. experiment execution (``exp_harness``),
3. statistical routing (``stat_router``, invoked by the harness),
4. 600-DPI / vector figures (``fig_foundry``),
5. LaTeX macros, three-line tables, figure and literature snippets (``tex_bridge``),

then assembles a self-contained thesis tree under ``results/<profile_id>/thesis``
and runs the anti-hallucination audit on it.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from . import fig_foundry as ff
from .audit import AuditReport, run_anti_hallucination_audit
from .exp_harness import ExperimentHarness
from .lit_engine import (LiteratureFetchError, export_bibtex, export_literature_matrix, fetch_and_verify_literature,
                         load_signed_cache, save_signed_cache)
from .schema import ExperimentResultBundle, LiteratureRecord, TopicProfile
from .stat_router import load_raw_samples, slice_response_surface
from .tex_bridge import generate_tex_artifacts, method_groups, render_figures, render_literature

__all__ = ["RunResult", "ProfilePaths", "run_profile", "audit_profile", "build_pdf", "find_tex_engine"]

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


@dataclass
class ProfilePaths:
    """Filesystem layout of one profile's outputs."""

    results_dir: Path
    profile_id: str

    @property
    def root(self) -> Path:
        return self.results_dir / self.profile_id

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def lit_cache(self) -> Path:
        return self.root / "lit_cache.json"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def thesis(self) -> Path:
        return self.root / "thesis"

    @property
    def bib(self) -> Path:
        return self.thesis / "refs.bib"

    @property
    def pdf(self) -> Path:
        return self.root / "thesis.pdf"


@dataclass
class RunResult:
    profile_id: str
    bundle: ExperimentResultBundle
    records: List[LiteratureRecord]
    figures: List[Dict[str, Any]]
    tex: Dict[str, str]
    audit: AuditReport
    manifest_path: Path
    timings_s: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.audit.passed


# --------------------------------------------------------------------------- #
# Step 1 - literature
# --------------------------------------------------------------------------- #
def _literature(profile: TopicProfile, paths: ProfilePaths, offline: Optional[bool]) -> List[LiteratureRecord]:
    try:
        records = fetch_and_verify_literature(profile.literature_query, profile.profile_id, paths.results_dir,
                                              offline=offline, metric_vocabulary=profile.metrics_target)
    except LiteratureFetchError as exc:
        logger.warning("%s; using offline caches", exc)
        records = fetch_and_verify_literature(profile.literature_query, profile.profile_id, paths.results_dir,
                                              offline=True, metric_vocabulary=profile.metrics_target)
    cached = load_signed_cache(paths.lit_cache) if paths.lit_cache.exists() else []
    if {r.cite_key for r in records} - {r.cite_key for r in cached}:
        save_signed_cache(records, paths.lit_cache, profile.literature_query)
    export_literature_matrix(records, paths.root / "literature_matrix.csv", paths.root / "literature_matrix.md")
    return records


# --------------------------------------------------------------------------- #
# Step 4 - figures
# --------------------------------------------------------------------------- #
def _pairwise(bundle: ExperimentResultBundle, metric: str, methods: Sequence[str]) -> List[Dict[str, Any]]:
    return [t for t in bundle.statistical_tests if t.get("comparison") == "pairwise" and t["metric"] == metric
            and t["method_a"] in methods and t["method_b"] in methods]


def _proposed(bundle: ExperimentResultBundle, profile: TopicProfile) -> str:
    if profile.proposed_method and profile.proposed_method in bundle.methods_metrics:
        return profile.proposed_method
    pw = [t["method_a"] for t in bundle.statistical_tests if t.get("comparison") == "pairwise"]
    return max(set(pw), key=pw.count) if pw else next(iter(bundle.methods_metrics))


def _metric_label(bundle: ExperimentResultBundle, metric: str) -> str:
    unit = next((mm[metric].unit for mm in bundle.methods_metrics.values() if metric in mm and mm[metric].unit), "")
    return metric.replace("_", " ") + (f" ({unit})" if unit else "")


def _make_figures(profile: TopicProfile, bundle: ExperimentResultBundle, paths: ProfilePaths) -> List[Dict[str, Any]]:
    paths.figures.mkdir(parents=True, exist_ok=True)
    npz = Path(bundle.artifacts_paths["raw_series"])
    samples, design = load_raw_samples(npz)
    proposed = _proposed(bundle, profile)
    primary = profile.metrics_target[0]
    groups = method_groups(bundle, list(design["points"]) if design else [])
    figs: List[Dict[str, Any]] = []

    if profile.domain_type == "signal_processing":
        fs = float(profile.data_config.sampling_rate_hz)
        with np.load(npz, allow_pickle=False) as z:
            arr = {k: z[k] for k in z.files}
        channels = [str(c) for c in arr["channel_names"]] if "channel_names" in arr else [""]
        shown = [m for m in bundle.methods_metrics if m != "raw"]
        for ch in channels:
            suffix = f"__{ch}" if ch else ""
            clean, noisy = arr[f"clean{suffix}"], arr[f"noisy{suffix}"]
            outs = {m: arr[f"out__{m}{suffix}"] for m in shown if f"out__{m}{suffix}" in arr}
            pretty = {m.replace("_", " "): y for m, y in outs.items()}
            tag = ch or "signal"
            res = ff.plot_signal_comparison(arr["t"], clean, pretty, fs, paths.figures / f"signal_{tag}", noisy=noisy,
                                            ref_label="Clean reference")
            figs.append({**res, "name": f"signal_{tag}", "label": f"fig:auto-signal-{tag}",
                         "caption": f"Time-domain and spectral comparison of the filters on the {tag.upper()} channel; "
                                    "legend entries report the cross-correlation phase delay."})
            if proposed in outs:
                res = ff.plot_bland_altman(clean, outs[proposed], paths.figures / f"bland_altman_{tag}",
                                           label_ref="Reference", label_meas=proposed.replace("_", " "))
                figs.append({**res, "name": f"bland_altman_{tag}", "label": f"fig:auto-bland-altman-{tag}",
                             "caption": f"Bland-Altman agreement between the clean {tag.upper()} reference and the "
                                        f"{proposed.replace('_', ' ')} output (bias, limits of agreement and their "
                                        "confidence bands).", "width": "0.62"})

    for group, title in (("comparison", "methods"), ("ablation", "ablation configurations")):
        methods = groups[group]
        if len(methods) < 2:
            continue
        data = {m: samples[m][primary] for m in methods if primary in samples.get(m, {})}
        res = ff.plot_ablation_and_baselines(data, paths.figures / f"{group}_{primary}", _metric_label(bundle, primary),
                                             tests=_pairwise(bundle, primary, methods))
        figs.append({**res, "name": f"{group}_{primary}", "label": f"fig:auto-{group}",
                     "caption": f"{primary.replace('_', ' ')} across {title}: mean and SD with significance "
                                "brackets versus the proposed configuration, and per-run distributions."})

    rsm = next((t for t in bundle.statistical_tests
                if t.get("comparison") == "response_surface" and t["metric"] == primary), None)
    if rsm is not None and design is not None and len(rsm["factors"]) >= 2:
        names = list(rsm["factors"])
        ranges = profile.data_config.factor_ranges
        bounds = {n: (float(ranges[n].low), float(ranges[n].high)) for n in names}
        sp = rsm.get("stationary_point")
        inside = sp is not None and all(bounds[n][0] <= sp["point"][n] <= bounds[n][1] for n in names)
        lin = rsm["coefficients"]
        keep = sorted(names, key=lambda n: -abs(lin.get(f"{n}^2", 0.0)) * (bounds[n][1] - bounds[n][0]) ** 2)[:2]
        keep = [n for n in names if n in keep]
        fixed = {n: (sp["point"][n] if inside else sum(bounds[n]) / 2) for n in names if n not in keep}
        model = slice_response_surface(rsm, keep, fixed) if fixed else rsm
        idx = [names.index(k) for k in keep]
        pts = np.array([[design["points"][lab][i] for i in idx] for lab in design["points"]])
        resp = [float(np.mean(samples[lab][primary])) for lab in design["points"]]
        maximize = profile.metrics_direction.get(primary, "max") != "min"
        res = ff.plot_response_surface(model, {k: bounds[k] for k in keep}, paths.figures / f"response_surface_{primary}",
                                       response_label=primary.replace("_", " "), maximize=maximize,
                                       observed=(pts, resp))
        held = ", ".join(f"{n.replace('_', ' ')} held at its {'stationary' if inside else 'central'} value"
                         for n in fixed) or "all factors shown"
        figs.append({**res, "name": f"response_surface_{primary}", "label": "fig:auto-rsm",
                     "caption": f"Fitted second-order response surface and contour of {primary.replace('_', ' ')} "
                                f"over {keep[0].replace('_', ' ')} and {keep[1].replace('_', ' ')} ({held}); "
                                "the star marks the optimum."})
    return figs


# --------------------------------------------------------------------------- #
# Step 5 - thesis tree
# --------------------------------------------------------------------------- #
def _escape_caption(text: str) -> str:
    return text.replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _assemble_thesis(profile: TopicProfile, paths: ProfilePaths, template_dir: Path, records: Sequence[LiteratureRecord],
                     figs: Sequence[Mapping[str, Any]], mirror_generated: bool) -> Dict[str, str]:
    if not (template_dir / "main.tex").exists():
        raise FileNotFoundError(f"thesis template not found: {template_dir / 'main.tex'}")
    if paths.thesis.exists():
        shutil.rmtree(paths.thesis)
    shutil.copytree(template_dir, paths.thesis, ignore=shutil.ignore_patterns("generated", "*.aux", "*.log", "*.pdf"))
    gen = paths.thesis / "generated"
    info = generate_tex_artifacts(paths.metrics, gen, proposed=profile.proposed_method,
                                  metrics=profile.metrics_target, title=profile.title)
    fig_dir = paths.thesis / "figures"
    fig_dir.mkdir(exist_ok=True)
    entries = []
    for f in figs:
        shutil.copy(f["pdf"], fig_dir / Path(f["pdf"]).name)
        entries.append({"path": f"figures/{Path(f['pdf']).name}", "caption": _escape_caption(f["caption"]),
                        "label": f["label"], "width": f.get("width", "1.0")})
    (gen / "auto_figures.tex").write_text(render_figures(entries), encoding="utf-8")
    (gen / "auto_literature.tex").write_text(render_literature(records), encoding="utf-8")
    export_bibtex(records, paths.bib)
    if mirror_generated:
        target = template_dir / "generated"
        target.mkdir(exist_ok=True)
        for p in gen.iterdir():
            shutil.copy(p, target / p.name)
    return info


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def audit_profile(profile: TopicProfile, results_dir: PathLike = "results") -> AuditReport:
    paths = ProfilePaths(Path(results_dir), profile.profile_id)
    return run_anti_hallucination_audit(paths.thesis, paths.bib, paths.metrics, paths.lit_cache)


def run_profile(profile_path: PathLike, results_dir: PathLike = "results",
                template_dir: PathLike = "templates/thesis_tex", offline: Optional[bool] = None,
                repo_root: Optional[PathLike] = None, plugin_dir: Optional[PathLike] = None,
                mirror_generated: bool = True) -> RunResult:
    profile = TopicProfile.from_yaml(profile_path)
    paths = ProfilePaths(Path(results_dir), profile.profile_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    records = _literature(profile, paths, offline)
    timings["literature"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if plugin_dir is None and Path("plugins").is_dir():
        plugin_dir = Path("plugins")
    bundle = ExperimentHarness(profile, results_dir=paths.results_dir, repo_root=repo_root,
                               plugin_dir=plugin_dir).run()
    timings["experiments_and_statistics"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    figs = _make_figures(profile, bundle, paths)
    timings["figures"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tex = _assemble_thesis(profile, paths, Path(template_dir), records, figs, mirror_generated)
    timings["latex"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    report = audit_profile(profile, paths.results_dir)
    timings["audit"] = time.perf_counter() - t0

    manifest = {
        "profile_id": profile.profile_id,
        "title": profile.title,
        "git_commit": bundle.git_commit,
        "bundle_sha256": bundle.compute_sha256(),
        "metrics": str(paths.metrics),
        "literature": {"records": len(records), "verified": sum(r.verified for r in records),
                       "cache": str(paths.lit_cache)},
        "figures": [{"name": f["name"], "png": f["png"], "pdf": f["pdf"]} for f in figs],
        "tex": {k: v for k, v in tex.items()},
        "thesis_dir": str(paths.thesis),
        "audit": report.to_dict(),
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
    }
    manifest_path = paths.root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return RunResult(profile.profile_id, bundle, records, figs, tex, report, manifest_path, timings)


def find_tex_engine(preferred: Optional[str] = None) -> Optional[str]:
    for engine in ([preferred] if preferred else []) + ["pdflatex", "xelatex", "lualatex"]:
        if engine and shutil.which(engine):
            return engine
    return None


def build_pdf(profile: TopicProfile, results_dir: PathLike = "results", engine: Optional[str] = None,
              timeout: float = 300.0) -> Path:
    """Compile ``results/<id>/thesis/main.tex`` (engine, bibtex, engine x2) into ``results/<id>/thesis.pdf``."""
    paths = ProfilePaths(Path(results_dir), profile.profile_id)
    if not (paths.thesis / "main.tex").exists():
        raise FileNotFoundError(f"{paths.thesis / 'main.tex'} missing; run `uni-thesis run` first")
    exe = find_tex_engine(engine)
    if exe is None:
        raise RuntimeError("no LaTeX engine (pdflatex/xelatex/lualatex) found on PATH")

    def _run(cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=paths.thesis, capture_output=True, text=True, timeout=timeout)

    tex_cmd = [exe, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    first = _run(tex_cmd)
    if first.returncode != 0:
        raise RuntimeError(f"{exe} failed:\n{first.stdout[-3000:]}")
    aux = (paths.thesis / "main.aux").read_text(encoding="utf-8", errors="replace")
    if shutil.which("bibtex") and "\\citation" in aux:
        bib = _run(["bibtex", "main"])
        if bib.returncode != 0:
            raise RuntimeError(f"bibtex failed:\n{bib.stdout[-3000:]}")
    for _ in range(2):
        res = _run(tex_cmd)
        if res.returncode != 0:
            raise RuntimeError(f"{exe} failed:\n{res.stdout[-3000:]}")
    shutil.copy(paths.thesis / "main.pdf", paths.pdf)
    return paths.pdf
