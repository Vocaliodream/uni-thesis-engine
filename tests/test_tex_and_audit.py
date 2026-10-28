"""Tests for the zero-hallucination LaTeX bridge and the anti-hallucination auditor."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from uni_thesis import exp_harness as eh
from uni_thesis import lit_engine as le
from uni_thesis import tex_bridge as tb
from uni_thesis.audit import run_anti_hallucination_audit, scan_raw_numbers
from uni_thesis.cli import main as cli_main
from uni_thesis.schema import ExperimentResultBundle, LiteratureRecord, TopicProfile

REPO_ROOT = Path(__file__).resolve().parents[1]
FS = 360.0


def _ablation_profile() -> TopicProfile:
    return TopicProfile.model_validate({
        "profile_id": "tex_abl", "title": "Ablation", "domain_type": "ablation_study",
        "literature_query": {"keywords": ["ablation"], "start_year": 2010, "max_papers": 3},
        "data_config": {"simulator": {"name": "additive_module_contributions", "params": {
            "base_value": 0.6, "contributions": {"A": 0.1, "B": 0.05}, "interactions": {"A*B": 0.02},
            "noise_std": 0.01, "metric_name": "accuracy"}}},
        "baselines": ["baseline"], "metrics_target": ["accuracy"], "random_seed": 3, "n_repeats": 8,
        "metrics_direction": {"accuracy": "max"}, "ablation": {"modules": ["A", "B"]},
    })


def _signal_profile() -> TopicProfile:
    return TopicProfile.model_validate({
        "profile_id": "tex_sig", "title": "Filters", "domain_type": "signal_processing",
        "literature_query": {"keywords": ["ecg"], "start_year": 2000, "max_papers": 5},
        "data_config": {"sampling_rate_hz": FS, "simulator": {"name": "ecg", "params": {
            "duration_s": 6.0, "heart_rate_bpm": 72, "rr_std_s": 0.02, "noise_std": 0.08,
            "baseline_wander_amp": 0.3, "baseline_wander_hz": 0.25, "powerline_amp": 0.15, "powerline_hz": 50.0}}},
        "baselines": ["moving_average", "butterworth_zero_phase", "butterworth_causal"],
        "metrics_target": ["delta_snr_db", "rmse", "phase_delay_ms"], "random_seed": 11, "n_repeats": 5,
        "method_params": {"moving_average": {"window_s": 0.02},
                          "butterworth_zero_phase": {"order": 2, "low_hz": 0.67, "high_hz": 40.0},
                          "butterworth_causal": {"order": 2, "low_hz": 0.67, "high_hz": 40.0}},
    })


@pytest.fixture(scope="module")
def ablation_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("abl")
    eh.ExperimentHarness(_ablation_profile(), results_dir=root / "results", repo_root=REPO_ROOT).run()
    return root / "results" / "tex_abl" / "metrics.json"


@pytest.fixture(scope="module")
def signal_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("sig")
    eh.ExperimentHarness(_signal_profile(), results_dir=root / "results", repo_root=REPO_ROOT).run()
    return root / "results" / "tex_sig" / "metrics.json"


# =============================================================== tex bridge units
def test_macro_names_are_letters_only():
    assert tb.macro_name("Proposed", "SNR", "Mean") == "ProposedSNRMean"
    assert tb.macro_name("w/o A", "accuracy") == "WOAAccuracy"
    assert tb.macro_name("butterworth_zero_phase", "delta_snr_db") == "ButterworthZeroPhaseDeltaSnrDb"
    assert tb.macro_name("x1=5,x2=-0.5").isalpha()
    with pytest.raises(ValueError):
        tb.macro_name("___")


def test_number_and_p_value_formatting():
    assert tb.format_number(24.85) == "24.85"
    assert tb.format_number(0.4213) == "0.4213"
    assert tb.format_number(-1.5) == r"\ensuremath{-1.5}"
    assert tb.format_number(0.00012) == r"\ensuremath{1.200\times 10^{-4}}"
    assert tb.format_p_value(0.0002) == r"\ensuremath{< 0.001}"
    assert tb.format_p_value(0.0496) == "0.0496"
    assert tb.format_p_value(1.0) == "1"
    with pytest.raises(ValueError):
        tb.format_number(float("nan"))


def test_generate_tex_artifacts_ablation(ablation_run, tmp_path):
    out = tmp_path / "generated"
    info = tb.generate_tex_artifacts(ablation_run, out)
    bundle = ExperimentResultBundle.load(ablation_run)
    text = (out / "auto_metrics.tex").read_text()
    assert f"% metrics_sha256: {tb.file_sha256(ablation_run)}" in text
    assert info["proposed"] == "full"
    full = bundle.methods_metrics["full"]["accuracy"]
    assert rf"\providecommand{{\MetricProposedAccuracyMean}}{{{tb.format_number(full.mean)}}}" in text
    assert rf"\providecommand{{\MetricFullAccuracyStd}}{{{tb.format_number(full.std)}}}" in text
    pw = next(t for t in bundle.statistical_tests if t.get("comparison") == "pairwise" and t["method_b"] == "baseline")
    assert rf"\providecommand{{\StatProposedVsBaselineAccuracyPval}}{{{tb.format_p_value(pw['p_value'])}}}" in text
    assert r"\providecommand{\StatAccuracyOmnibusPval}" in text
    for name in __import__("re").findall(r"\\providecommand\{\\([^}]*)\}", text):
        assert name.isalpha(), name

    tables = (out / "auto_tables.tex").read_text()
    for cmd in (r"\toprule", r"\midrule", r"\bottomrule", r"\label{tab:auto-ablation}"):
        assert cmd in tables
    assert r"$\mathbf{\MetricFullAccuracyMean \pm \MetricFullAccuracyStd}$" in tables
    second = sorted(((m["accuracy"].mean, k) for k, m in bundle.methods_metrics.items()), reverse=True)[1][1]
    assert rf"\underline{{$\Metric{tb.macro_name(second, 'accuracy')}Mean" in tables
    assert r"^{\text{***}}" in tables
    assert "tab:auto-comparison" not in tables  # every method belongs to the ablation


def test_generate_tex_artifacts_signal_comparison(signal_run, tmp_path):
    tb.generate_tex_artifacts(signal_run, tmp_path / "generated")
    tables = (tmp_path / "generated" / "auto_tables.tex").read_text()
    assert "tab:auto-comparison" in tables and "tab:auto-ablation" not in tables
    assert tables.count(r"\mathbf{") >= 3          # one best per metric column
    assert tables.count(r"\underline{") >= 3
    metrics = (tmp_path / "generated" / "auto_metrics.tex").read_text()
    assert r"\DiffProposedVsRawDeltaSnrDbAbs" in metrics


def test_macro_collision_detected():
    tab = tb._MacroTable()
    tab.add("MetricA", "1", "x")
    with pytest.raises(tb.MacroNameCollision):
        tab.add("MetricA", "2", "y")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_generated_tex_compiles(signal_run, tmp_path):
    tb.generate_tex_artifacts(signal_run, tmp_path / "generated")
    (tmp_path / "main.tex").write_text("\n".join([
        r"\documentclass{article}", r"\usepackage{booktabs,amsmath,graphicx}",
        r"\input{generated/auto_metrics.tex}", r"\begin{document}",
        r"Gain \DiffProposedVsRawDeltaSnrDbAbs, $p$ \StatProposedVsRawDeltaSnrDbPval.",
        r"\input{generated/auto_tables.tex}", r"\end{document}", ""]))
    proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert (tmp_path / "main.pdf").stat().st_size > 1000


# =============================================================== audit fixtures
def _verified_records():
    return le.load_baseline_cache()[:3]


def _thesis(tmp_path: Path, metrics_json: Path, chap4_body: str, extra_records=(), extra_bib: str = "") -> dict:
    tex = tmp_path / "thesis_tex"
    (tex / "chapters").mkdir(parents=True)
    tb.generate_tex_artifacts(metrics_json, tex / "generated")
    records = list(_verified_records()) + list(extra_records)
    cache = metrics_json.parent / "lit_cache.json"
    le.save_signed_cache(records, cache)
    bib = tex / "refs.bib"
    le.export_bibtex(records, bib)
    if extra_bib:
        bib.write_text(bib.read_text() + "\n" + extra_bib)
    k = [r.cite_key for r in _verified_records()]
    (tex / "chapters" / "chap3_method.tex").write_text(
        "\\chapter{Method}\\label{chap:method}\n"
        f"Wavelet shrinkage~\\cite{{{k[0]}}} and polynomial smoothing~\\cite{{{k[1]}, {k[2]}}} are compared.\n"
        "As shown in Fig.~3.1 and Table 3-2, see Eq.~(3.4) and Section 2.1; $x_{10}$ and $y^2$ are indices.\n"
        "\\includegraphics[width=0.8\\linewidth]{fig/flow.pdf}\n"
        "% a comment with 12.5 numbers is ignored\n")
    (tex / "chapters" / "chap4_experiments.tex").write_text(chap4_body)
    (tex / "chapters" / "chap5_conclusion.tex").write_text(
        "\\chapter{Conclusion}\nThe full model reaches \\MetricProposedAccuracyMean{} accuracy; "
        "removing a module costs up to \\DiffProposedVsBaselineAccuracyAbs{} (\\StatProposedVsBaselineAccuracyStars).\n")
    return {"tex_dir": tex, "bib_path": bib, "metrics_json_path": metrics_json}


COMPLIANT_CHAP4 = (
    "\\chapter{Experiments}\n"
    "\\input{generated/auto_tables.tex}\n"
    "Table~\\ref{tab:auto-ablation} reports \\MetricFullAccuracyMean{} $\\pm$ \\MetricFullAccuracyStd{} for the full "
    "model versus \\MetricBaselineAccuracyMean{} for the baseline ($p$ \\StatProposedVsBaselineAccuracyPval, "
    "\\StatProposedVsBaselineAccuracyTest), an improvement of \\DiffProposedVsBaselineAccuracyRelPct\\%.\n"
    "We repeat every run 3 times on 2 splits; 100\\% of runs converge (Chapter 4, 第3章).\n")


def _copy_run(metrics_json: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "results" / "tex_abl"
    dst.mkdir(parents=True)
    shutil.copy(metrics_json, dst / "metrics.json")
    return dst / "metrics.json"


# =============================================================== audit gates
def test_compliant_thesis_passes(ablation_run, tmp_path):
    paths = _thesis(tmp_path, _copy_run(ablation_run, tmp_path), COMPLIANT_CHAP4)
    report = run_anti_hallucination_audit(**paths)
    assert report.passed, report.format()
    assert report.citations_checked == 3
    assert "chapters/chap4_experiments.tex" in report.checked_files


def test_raw_numbers_are_blocked_with_file_and_line(ablation_run, tmp_path):
    body = COMPLIANT_CHAP4 + (
        "Our method achieves 24.85 dB and an accuracy of 98.5\\%.\n"
        "The difference is significant ($p < 0.05$) with error 0.042.\n"
        "提升了12.3dB，共使用 42 名受试者。\n")
    paths = _thesis(tmp_path, _copy_run(ablation_run, tmp_path), body)
    report = run_anti_hallucination_audit(**paths)
    raw = report.by_gate("raw_number")
    assert not report.passed and len(raw) == 5
    assert all(v.file == "chapters/chap4_experiments.tex" for v in raw)
    base = COMPLIANT_CHAP4.count("\n")
    assert {v.line for v in raw} == {base + 1, base + 2, base + 3}
    joined = " ".join(v.message for v in raw)
    for token in ("24.85", "98.5\\%", "p < 0.05", "0.042", "12.3"):
        assert token in joined, token
    assert "'42'" not in joined                       # plain counts pass by default
    assert not report.by_gate("citation") and not report.by_gate("provenance")
    strict = run_anti_hallucination_audit(**paths, strict_integers=True).by_gate("raw_number")
    assert len(strict) == 6 and any("'42'" in v.message for v in strict)


def test_scan_raw_numbers_ignores_references_and_comments(tmp_path):
    f = tmp_path / "c.tex"
    f.write_text("See Figure 4.2, 表 4-1, Algorithm 2, Eq. (5.3), \\ref{fig:12} and \\label{sec:3.2}.\n"
                 "\\vspace{12pt} \\includegraphics[scale=0.45]{a.png} % 77.7\n"
                 "In 2024 we used $\\alpha_{12}$ only.\n")
    assert scan_raw_numbers(f) == []
    f.write_text("freq 50.5 Hz and Sec 7 then 1e-3 with 42 subjects\n")
    msgs = [v.message for v in scan_raw_numbers(f)]
    assert any("50.5" in m for m in msgs) and any("1e-3" in m for m in msgs)
    assert not any("'7'" in m or "'42'" in m for m in msgs)
    strict = [v.message for v in scan_raw_numbers(f, strict_integers=True)]
    assert any("'42'" in m for m in strict) and not any("'7'" in m for m in strict)


def test_fake_citation_is_blocked(ablation_run, tmp_path):
    body = COMPLIANT_CHAP4 + "Recent work~\\cite{fake2026} claims otherwise.\n"
    paths = _thesis(tmp_path, _copy_run(ablation_run, tmp_path), body)
    report = run_anti_hallucination_audit(**paths)
    cites = report.by_gate("citation")
    assert len(cites) == 1 and "fake2026" in cites[0].message and "not found in refs.bib" in cites[0].message
    assert cites[0].line == COMPLIANT_CHAP4.count("\n") + 1


def test_unverified_or_doi_mismatched_citation_is_blocked(ablation_run, tmp_path):
    unverified = LiteratureRecord(cite_key="ghost2025method", title="A method", authors=["G. Host"], year=2025,
                                  doi=None, venue="Unknown", methodology_tag="unknown", verified=False)
    real = _verified_records()[0]
    bib_extra = ("@article{ghost2025method,\n  title = {A method},\n  author = {Host, G.},\n  year = {2025}\n}\n")
    body = COMPLIANT_CHAP4 + "Hallucinated~\\cite{ghost2025method}.\n"
    paths = _thesis(tmp_path, _copy_run(ablation_run, tmp_path), body, extra_records=[unverified], extra_bib=bib_extra)
    bib = paths["bib_path"]
    bib.write_text(bib.read_text().replace(real.doi, "10.9999/forged.doi"))
    report = run_anti_hallucination_audit(**paths)
    msgs = [v.message for v in report.by_gate("citation")]
    assert any("ghost2025method" in m and "not DOI-verified" in m for m in msgs)
    assert any("ghost2025method" in m and "lacks a valid DOI" in m for m in msgs)
    assert any(real.cite_key in m and "DOI mismatch" in m for m in msgs)


def test_tampered_lit_cache_is_blocked(ablation_run, tmp_path):
    paths = _thesis(tmp_path, _copy_run(ablation_run, tmp_path), COMPLIANT_CHAP4)
    cache = paths["metrics_json_path"].parent / "lit_cache.json"
    doc = json.loads(cache.read_text())
    doc["records"][0]["year"] = 2001
    cache.write_text(json.dumps(doc))
    report = run_anti_hallucination_audit(**paths)
    assert any("signature" in v.message for v in report.by_gate("citation"))


def test_stale_macros_are_blocked(ablation_run, tmp_path):
    paths = _thesis(tmp_path, _copy_run(ablation_run, tmp_path), COMPLIANT_CHAP4)
    mj = paths["metrics_json_path"]
    bundle = ExperimentResultBundle.load(mj)
    item = bundle.methods_metrics["full"]["accuracy"]
    bundle.methods_metrics["full"]["accuracy"] = item.model_copy(update={"std": item.std * 1.01})
    bundle.save(mj)                                    # a re-run: valid bundle, new hash
    report = run_anti_hallucination_audit(**paths)
    prov = report.by_gate("provenance")
    assert len(prov) == 2 and all("stale macros" in v.message for v in prov)
    tb.generate_tex_artifacts(mj, paths["tex_dir"] / "generated")
    assert run_anti_hallucination_audit(**paths).passed


def test_tampered_metrics_and_invented_macro_are_blocked(ablation_run, tmp_path):
    body = COMPLIANT_CHAP4 + "Precision was \\MetricProposedPrecisionMean.\n"
    paths = _thesis(tmp_path, _copy_run(ablation_run, tmp_path), body)
    mj = paths["metrics_json_path"]
    mj.write_text(mj.read_text().replace('"n_a": 8', '"n_a": 9', 1))
    report = run_anti_hallucination_audit(**paths)
    msgs = [v.message for v in report.by_gate("provenance")]
    assert any("integrity" in m for m in msgs)
    assert any("stale macros" in m for m in msgs)
    assert any("\\MetricProposedPrecisionMean is not defined" in m for m in msgs)


def test_missing_generated_file_and_chapter(ablation_run, tmp_path):
    paths = _thesis(tmp_path, _copy_run(ablation_run, tmp_path), COMPLIANT_CHAP4)
    (paths["tex_dir"] / "generated" / "auto_metrics.tex").unlink()
    (paths["tex_dir"] / "chapters" / "chap5_conclusion.tex").unlink()
    report = run_anti_hallucination_audit(**paths)
    assert any("missing" in v.message for v in report.by_gate("provenance"))
    assert any("not found" in v.message for v in report.by_gate("raw_number"))


# =============================================================== entry points
def test_script_and_cli(ablation_run, tmp_path, capsys):
    paths = _thesis(tmp_path, _copy_run(ablation_run, tmp_path), COMPLIANT_CHAP4)
    args = ["--tex-dir", str(paths["tex_dir"]), "--bib", str(paths["bib_path"]),
            "--metrics", str(paths["metrics_json_path"])]
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "audit_anti_hallucination.py"), *args, "--json"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["passed"] is True
    assert cli_main(["audit", *args]) == 0
    with (paths["tex_dir"] / "chapters" / "chap4_experiments.tex").open("a") as fh:
        fh.write("Accuracy 0.97 \\cite{fake2026}\n")
    assert cli_main(["audit", *args]) == 1
    out = capsys.readouterr().out
    assert "AUDIT FAILED" in out and "fake2026" in out
    assert cli_main(["build-tex", str(paths["metrics_json_path"]), "--out-dir", str(tmp_path / "gen2")]) == 0
    assert (tmp_path / "gen2" / "auto_tables.tex").exists()
    assert cli_main(["build-tex", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "gen3")]) == 1
