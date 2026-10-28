"""End-to-end tests of the unified CLI on the two bundled demo profiles."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from uni_thesis import exp_harness as eh
from uni_thesis import pipeline as pl
from uni_thesis import tex_bridge as tb
from uni_thesis.cli import main as cli_main
from uni_thesis.schema import ExperimentResultBundle, TopicProfile
from uni_thesis.stat_router import fit_response_surface, rsm_predict, slice_response_surface

REPO = Path(__file__).resolve().parents[1]
PROFILES = {"demo_biosignal": REPO / "profiles" / "demo_biosignal.yaml",
            "demo_rsm_opt": REPO / "profiles" / "demo_rsm_opt.yaml"}
HAS_TEX = pl.find_tex_engine() is not None


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    template = root / "template"
    shutil.copytree(REPO / "templates" / "thesis_tex", template, ignore=shutil.ignore_patterns("generated"))
    results = root / "results"
    codes = {}
    for pid, path in PROFILES.items():
        codes[pid] = cli_main(["run", "--profile", str(path), "--results-dir", str(results),
                               "--template-dir", str(template), "--offline"])
    return {"root": root, "template": template, "results": results, "codes": codes}


def _paths(ws, pid) -> pl.ProfilePaths:
    return pl.ProfilePaths(ws["results"], pid)


# =============================================================== full pipeline
@pytest.mark.parametrize("pid", list(PROFILES))
def test_run_produces_every_artifact(workspace, pid):
    assert workspace["codes"][pid] == 0
    p = _paths(workspace, pid)
    bundle = ExperimentResultBundle.load(p.metrics)            # verifies SHA-256
    for f in (p.root / "raw_series.npz", p.lit_cache, p.root / "literature_matrix.csv",
              p.root / "literature_matrix.md", p.bib, p.thesis / "main.tex",
              *(p.thesis / "generated" / n for n in ("auto_metrics.tex", "auto_tables.tex",
                                                     "auto_figures.tex", "auto_literature.tex"))):
        assert f.exists() and f.stat().st_size > 0, f
    manifest = json.loads((p.root / "run_manifest.json").read_text())
    assert manifest["audit"]["passed"] is True
    assert manifest["bundle_sha256"] == bundle.compute_sha256()
    assert manifest["literature"]["verified"] >= 1
    assert manifest["figures"]
    for fig in manifest["figures"]:
        png, pdf = Path(fig["png"]), Path(fig["pdf"])
        assert pdf.read_bytes()[:4] == b"%PDF"
        with Image.open(png) as im:
            assert round(im.info["dpi"][0]) == 600
        assert (p.thesis / "figures" / pdf.name).exists()
    text = (p.thesis / "generated" / "auto_metrics.tex").read_text()
    assert f"% metrics_sha256: {tb.file_sha256(p.metrics)}" in text
    mirrored = workspace["template"] / "generated" / "auto_metrics.tex"
    assert mirrored.exists()


def test_biosignal_profile_results(workspace):
    p = _paths(workspace, "demo_biosignal")
    profile = TopicProfile.from_yaml(PROFILES["demo_biosignal"])
    assert profile.data_config.sampling_rate_hz == 250
    bundle = ExperimentResultBundle.load(p.metrics)
    mm = bundle.methods_metrics
    assert set(mm) == {"raw", "moving_average", "savitzky_golay", "butterworth_zero_phase"}
    for ch in ("ecg", "ppg"):
        assert {f"delta_snr_db_{ch}", f"rmse_{ch}", f"pearson_r_{ch}", f"ba_loa_width_{ch}"} <= set(mm["raw"])
    bw = mm["butterworth_zero_phase"]
    for metric in profile.metrics_target:
        assert bw[metric].is_best, metric
    pw = [t for t in bundle.statistical_tests if t.get("comparison") == "pairwise"]
    assert pw and all(t["method_a"] == "butterworth_zero_phase" and t["paired"] for t in pw)
    with np.load(p.root / "raw_series.npz") as z:
        assert list(z["channel_names"]) == ["ecg", "ppg"]
        assert z["clean__ppg"].shape == z["out__butterworth_zero_phase__ppg"].shape
    metrics_tex = (p.thesis / "generated" / "auto_metrics.tex").read_text()
    assert "\\AutoHasComparisontrue" in metrics_tex and "\\AutoHasRsmfalse" in metrics_tex
    assert "\\providecommand{\\MetricPrimaryProposedName}{butterworth\\_zero\\_phase}" in metrics_tex
    names = {f["name"] for f in json.loads((p.root / "run_manifest.json").read_text())["figures"]}
    assert {"signal_ecg", "signal_ppg", "bland_altman_ecg", "bland_altman_ppg", "comparison_delta_snr_db"} <= names
    tables = (p.thesis / "generated" / "auto_tables.tex").read_text()
    assert tables.index("raw") < tables.index("butterworth\\_zero\\_phase (proposed)")  # run order kept


def test_rsm_profile_results(workspace):
    p = _paths(workspace, "demo_rsm_opt")
    bundle = ExperimentResultBundle.load(p.metrics)
    labels = list(bundle.methods_metrics)
    assert {"baseline", "+pretreatment", "+pretreatment +catalyst", "full"} <= set(labels)
    rsm = next(t for t in bundle.statistical_tests if t.get("comparison") == "response_surface")
    assert rsm["factors"] == ["temperature_c", "time_min", "ratio"]
    assert rsm["n"] == (8 + 6 + 3) * 3                      # CCD runs x repeats; ablation excluded
    assert {r["source"] for r in rsm["anova"]} >= {"Model", "Residual", "Lack of fit", "Pure error", "Total"}
    assert set(rsm["term_stats"]) == set(rsm["terms"])
    assert rsm["stationary_point"]["nature"] == "maximum"
    abl = [t for t in bundle.statistical_tests if t.get("comparison") == "pairwise"]
    assert {t["method_b"] for t in abl} == {"baseline", "+pretreatment", "+pretreatment +catalyst"}
    tables = (p.thesis / "generated" / "auto_tables.tex").read_text()
    for label in ("tab:auto-ablation", "tab:auto-design", "tab:auto-anova", "tab:auto-coef"):
        assert label in tables
    assert tables.index("baseline &") < tables.index("full (proposed)")
    metrics_tex = (p.thesis / "generated" / "auto_metrics.tex").read_text()
    assert "\\AutoHasRsmtrue" in metrics_tex and "\\AutoHasAblationtrue" in metrics_tex
    assert "\\providecommand{\\StatRsmPrimaryEquation}{\\begin{aligned}" in metrics_tex
    assert "\\StatRsmYieldPctTermTemperatureCSqCoef" in metrics_tex
    assert (p.figures / "response_surface_yield_pct.png").exists()
    assert (p.figures / "ablation_yield_pct.pdf").exists()


# =============================================================== audit / pdf
def test_audit_by_profile_and_tamper_detection(workspace, capsys):
    args = ["audit", "--profile", str(PROFILES["demo_rsm_opt"]), "--results-dir", str(workspace["results"])]
    assert cli_main(args) == 0
    chap = _paths(workspace, "demo_rsm_opt").thesis / "chapters" / "chap4_experiments.tex"
    original = chap.read_text()
    try:
        chap.write_text(original + "\nThe optimum yield reached 97.32\\% here \\cite{fake2026}.\n")
        assert cli_main(args + ["--json"]) == 1
        capsys.readouterr()
        assert cli_main(args + ["--json"]) == 1
        report = json.loads(capsys.readouterr().out)
        gates = {v["gate"] for v in report["violations"]}
        assert {"raw_number", "citation"} <= gates
    finally:
        chap.write_text(original)
    assert cli_main(["audit"]) == 2


@pytest.mark.skipif(not HAS_TEX, reason="no LaTeX engine installed")
@pytest.mark.parametrize("pid", list(PROFILES))
def test_pdf_compiles(workspace, pid):
    assert cli_main(["pdf", "--profile", str(PROFILES[pid]), "--results-dir", str(workspace["results"])]) == 0
    pdf = _paths(workspace, pid).pdf
    assert pdf.read_bytes()[:4] == b"%PDF" and pdf.stat().st_size > 20000
    log = (_paths(workspace, pid).thesis / "main.log").read_text(errors="replace")
    assert "Citation" not in log or "undefined" not in log.split("Citation", 1)[1][:200]


def test_cli_error_paths(tmp_path, capsys):
    assert cli_main(["run", "--profile", str(tmp_path / "missing.yaml")]) == 1
    assert cli_main(["pdf", "--profile", str(PROFILES["demo_biosignal"]), "--results-dir", str(tmp_path)]) == 1
    assert cli_main(["audit", "--profile", str(tmp_path / "missing.yaml")]) == 1
    assert cli_main(["pdf", "--profile", str(tmp_path / "missing.yaml")]) == 1
    assert cli_main(["run", "--profile", str(PROFILES["demo_biosignal"]), "--results-dir", str(tmp_path),
                     "--template-dir", str(tmp_path / "no_template"), "--offline", "--no-mirror"]) == 1
    assert "RUN FAILED" in capsys.readouterr().err


def test_run_reports_audit_failure(tmp_path, capsys):
    template = tmp_path / "template"
    shutil.copytree(REPO / "templates" / "thesis_tex", template, ignore=shutil.ignore_patterns("generated"))
    chap5 = template / "chapters" / "chap5_conclusion.tex"
    chap5.write_text(chap5.read_text() + "\nWe observed an accuracy of 0.9731.\n")
    code = cli_main(["run", "--profile", str(PROFILES["demo_rsm_opt"]), "--results-dir", str(tmp_path / "res"),
                     "--template-dir", str(template), "--offline", "--no-mirror"])
    assert code == 1
    captured = capsys.readouterr()
    assert '"status": "audit_failed"' in captured.out and "0.9731" in captured.err
    assert not (template / "generated").exists()


def test_build_pdf_without_engine(tmp_path, monkeypatch):
    profile = TopicProfile.from_yaml(PROFILES["demo_biosignal"])
    (tmp_path / profile.profile_id / "thesis").mkdir(parents=True)
    (tmp_path / profile.profile_id / "thesis" / "main.tex").write_text("x")
    monkeypatch.setattr(pl.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="no LaTeX engine"):
        pl.build_pdf(profile, tmp_path)
    assert pl.find_tex_engine("xelatex") is None


def test_literature_falls_back_offline_when_nothing_verifiable(tmp_path, monkeypatch):
    profile = TopicProfile.from_yaml(PROFILES["demo_rsm_opt"])
    calls = []
    real = pl.fetch_and_verify_literature

    def fake(query, pid, results_dir, offline=None, metric_vocabulary=()):
        calls.append(offline)
        if not offline:
            raise pl.LiteratureFetchError("nothing verifiable")
        return real(query, pid, results_dir, offline=True, metric_vocabulary=metric_vocabulary)

    monkeypatch.setattr(pl, "fetch_and_verify_literature", fake)
    paths = pl.ProfilePaths(tmp_path, profile.profile_id)
    paths.root.mkdir(parents=True)
    records = pl._literature(profile, paths, offline=None)
    assert calls == [None, True] and records and all(r.verified for r in records)
    assert paths.lit_cache.exists()


# =============================================================== engine units
def test_slice_response_surface_matches_full_model():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(40, 3))
    y = 1 + X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] - X[:, 0] ** 2 - X[:, 1] ** 2 - X[:, 2] ** 2 + 0.3 * X[:, 0] * X[:, 2]
    model = fit_response_surface(X, y + 0.01 * rng.standard_normal(40), ["a", "b", "c"])
    sl = slice_response_surface(model, ["a", "c"], {"b": 0.4})
    grid = rng.uniform(-1, 1, size=(10, 2))
    full = rsm_predict(model, np.column_stack([grid[:, 0], np.full(10, 0.4), grid[:, 1]]))
    np.testing.assert_allclose(rsm_predict(sl, grid), full, rtol=1e-10, atol=1e-10)
    assert sl["stationary_point"]["nature"] == "maximum"
    with pytest.raises(ValueError):
        slice_response_surface(model, ["a", "a"], {"b": 0.0, "c": 0.0})
    with pytest.raises(ValueError):
        slice_response_surface(model, ["a", "b"], {})


def test_multichannel_simulator_validation():
    sim = eh.SIGNAL_SIMULATOR_REGISTRY["multichannel"]
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        sim({"channels": {}}, 100.0, rng)
    with pytest.raises(KeyError):
        sim({"channels": {"x": {"simulator": "multichannel"}}}, 100.0, rng)
    base = {"heart_rate_bpm": 60, "noise_std": 0.01}
    with pytest.raises(ValueError):
        sim({"channels": {"a": {"simulator": "ecg", "params": {**base, "duration_s": 2.0}},
                          "b": {"simulator": "ecg", "params": {**base, "duration_s": 3.0}}}}, 100.0, rng)
    t, clean, noisy = sim({"channels": {"a": {"simulator": "ecg", "params": {**base, "duration_s": 2.0}}}}, 100.0, rng)
    assert clean.shape == noisy.shape == (1, t.size)


def test_ablation_modes_and_modular_objective():
    profile = TopicProfile.from_yaml(PROFILES["demo_rsm_opt"])
    pipe = eh.MultiFactorPipeline(profile)
    abl = pipe.ablation_configurations()
    assert [lab for lab, _ in abl] == ["baseline", "+pretreatment", "+pretreatment +catalyst", "full"]
    assert len(pipe.design_labels) == 17
    removal = TopicProfile.model_validate({**profile.model_dump(), "ablation": {"modules": ["a", "b"]}})
    labels = [lab for lab, _ in eh.MultiFactorPipeline(removal).ablation_configurations()]
    assert labels == ["full", "w/o a", "w/o b", "baseline"]
    obj = eh.OBJECTIVE_REGISTRY["modular_response_surface"]
    params = {"intercept": 1.0, "noise_std": 0.0, "contributions": {"m": 2.0}, "metric_name": "y"}
    rng = np.random.default_rng(0)
    assert obj({"factors": {}}, params, rng)["y"] == pytest.approx(3.0)
    assert obj({"factors": {}, "modules": {"m": False}}, params, rng)["y"] == pytest.approx(1.0)


def test_method_groups_and_literature_rendering():
    assert tb.render_literature([]).strip().endswith("No DOI-verified reference was retrieved for this topic.")
    fig = tb.render_figures([{"path": "figures/a.pdf", "caption": "Cap", "label": "fig:a"}])
    assert "\\includegraphics[width=1.0\\linewidth]{figures/a.pdf}" in fig and "\\label{fig:a}" in fig
    assert tb.format_p_relation(0.2) == "\\ensuremath{= 0.200}"
    assert tb.render_flags({"AutoHasX": False}) == "\\newif\\ifAutoHasX\n\\AutoHasXfalse\n"


def test_crossref_author_names_are_sanitised(tmp_path):
    from uni_thesis import lit_engine as le

    msg = {"author": [{"given": "C. U.,", "family": "Ameh,"}, {"family": "Lin Ma,"}, {"name": " Consortium , "},
                      {"given": "", "family": ""}]}
    assert le._crossref_authors(msg) == ["Ameh, C. U.", "Lin Ma", "Consortium"]
    assert le._bib_author("A., B., C.") == "{A., B., C.}"
    assert le._bib_author("Smith, John,") == "Smith, John"
