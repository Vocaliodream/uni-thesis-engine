"""Tests for the literature engine and the experiment harness.

Network access is replaced by a recorded fake session so the suite is
deterministic offline; the fake payloads mirror real Crossref/OpenAlex response
shapes for the real DOI 10.1109/18.382009 (Donoho, 1995).
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import requests
from scipy import signal as sps

from uni_thesis import exp_harness as eh
from uni_thesis.lit_engine import (
    CacheSignatureError,
    LiteratureFetchError,
    export_bibtex,
    export_literature_matrix,
    fetch_and_verify_literature,
    is_valid_doi_format,
    load_baseline_cache,
    load_signed_cache,
    make_cite_key,
    normalize_doi,
    verify_doi_exists,
)
from uni_thesis.schema import ExperimentResultBundle, LiteratureQuery, LiteratureRecord, TopicProfile

REPO_ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1109/18.382009"
CROSSREF_MSG = {
    "DOI": DOI,
    "title": ["De-noising by soft-thresholding"],
    "author": [{"family": "Donoho", "given": "D.L."}],
    "issued": {"date-parts": [[1995, 5]]},
    "container-title": ["IEEE Transactions on Information Theory"],
    "type": "journal-article",
    "abstract": "<jats:p>We measure the SNR and RMSE of soft thresholding.</jats:p>",
}
OPENALEX_WORK = {
    "doi": f"https://doi.org/{DOI}",
    "title": "De-noising by soft-thresholding",
    "publication_year": 1995,
    "authorships": [{"author": {"display_name": "David L. Donoho"}}],
    "primary_location": {"source": {"display_name": "IEEE Transactions on Information Theory"}},
    "primary_topic": {"display_name": "Wavelet denoising"},
    "type_crossref": "journal-article",
}
FAKE_DOI = "10.5555/hallucinated.2024.001"


class FakeResponse:
    def __init__(self, status: int, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Routes GETs to canned payloads; records every URL requested."""

    def __init__(self, works=None, openalex=None, crossref_search=None, fail=False):
        self.works = works if works is not None else {DOI: CROSSREF_MSG}
        self.openalex = openalex if openalex is not None else [OPENALEX_WORK]
        self.crossref_search = crossref_search if crossref_search is not None else [CROSSREF_MSG]
        self.fail = fail
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if self.fail:
            raise requests.ConnectionError("offline")
        if url.startswith("https://api.openalex.org/works"):
            return FakeResponse(200, {"results": self.openalex})
        if url == "https://api.crossref.org/works":
            return FakeResponse(200, {"message": {"items": self.crossref_search}})
        doi = url.split("/works/", 1)[1].lower()
        if doi in self.works:
            return FakeResponse(200, {"message": self.works[doi]})
        return FakeResponse(404)


def _query(**kw):
    base = {"keywords": ["soft-thresholding"], "start_year": 1990, "max_papers": 5}
    base.update(kw)
    return LiteratureQuery(**base)


# =============================================================== literature
@pytest.mark.parametrize("doi,ok", [
    ("10.1109/18.382009", True),
    ("https://doi.org/10.1038/s41592-019-0686-2", True),
    ("doi:10.1111/j.2517-6161.1951.tb00067.x", True),
    ("10.12/abc", False),
    ("11.1109/18.382009", False),
    ("10.1109/", False),
    ("not a doi", False),
    ("", False),
])
def test_doi_format_validation(doi, ok):
    assert is_valid_doi_format(doi) is ok


def test_normalize_doi_strips_resolver_and_case():
    assert normalize_doi(" https://doi.org/10.1109/TBME.1985.325532 ") == "10.1109/tbme.1985.325532"


def test_verify_doi_matches_metadata():
    s = FakeSession()
    assert verify_doi_exists(DOI, "De-noising by soft thresholding", ["David L. Donoho"], 1995, session=s)
    assert verify_doi_exists(DOI, session=s)


@pytest.mark.parametrize("kwargs", [
    {"expected_title": "Deep learning beats everything"},
    {"expected_authors": ["Jane Smith"]},
    {"expected_year": 2010},
])
def test_verify_doi_rejects_mismatched_metadata(kwargs):
    assert verify_doi_exists(DOI, session=FakeSession(), **kwargs) is False


def test_verify_doi_rejects_unregistered_and_malformed():
    s = FakeSession()
    assert verify_doi_exists(FAKE_DOI, session=s) is False
    n_calls = len(s.calls)
    assert verify_doi_exists("10.1/bad", session=s) is False
    assert len(s.calls) == n_calls, "malformed DOI must not hit the network"


def test_fetch_verifies_and_writes_signed_cache(tmp_path):
    hallucinated = dict(CROSSREF_MSG, DOI=FAKE_DOI, title=["A paper that does not exist"])
    s = FakeSession(crossref_search=[CROSSREF_MSG, hallucinated])
    recs = fetch_and_verify_literature(_query(), profile_id="p1", results_dir=tmp_path, session=s,
                                       offline=False, metric_vocabulary=["SNR", "RMSE", "PSNR"])
    assert [r.doi for r in recs] == [DOI]
    rec = recs[0]
    assert rec.verified and rec.cite_key == "donoho1995noising"
    assert rec.methodology_tag == "Wavelet denoising"
    assert rec.core_metrics == ["SNR", "RMSE"]
    cached = load_signed_cache(tmp_path / "p1" / "lit_cache.json")
    assert cached == recs


def test_fetch_refuses_to_fabricate_when_nothing_verifies(tmp_path):
    s = FakeSession(works={}, openalex=[], crossref_search=[dict(CROSSREF_MSG, DOI=FAKE_DOI)])
    with pytest.raises(LiteratureFetchError):
        fetch_and_verify_literature(_query(), results_dir=tmp_path, session=s, offline=False)


def test_offline_fallback_uses_signed_baseline(tmp_path):
    recs = fetch_and_verify_literature(_query(keywords=["denoising", "smoothing"], start_year=1950), profile_id="ci",
                                       results_dir=tmp_path, session=FakeSession(fail=True), offline=False)
    dois = {r.doi for r in recs}
    assert {"10.1109/18.382009", "10.1021/ac60214a047"} <= dois
    assert all(r.verified and r.doi for r in recs)
    assert all(r.verified for r in load_baseline_cache())


def test_tampered_cache_is_rejected(tmp_path):
    fetch_and_verify_literature(_query(), profile_id="p2", results_dir=tmp_path, session=FakeSession(), offline=False)
    path = tmp_path / "p2" / "lit_cache.json"
    doc = json.loads(path.read_text())
    doc["records"][0]["year"] = 2024
    path.write_text(json.dumps(doc))
    with pytest.raises(CacheSignatureError):
        load_signed_cache(path)


def test_bibtex_generation(tmp_path):
    recs = [
        LiteratureRecord(cite_key="smith2020fast", title="Fast & robust filters: 100% better",
                         authors=["Smith, John", "Doe, Jane"], year=2020, doi="10.1000/abc_1",
                         venue="Journal of Signals", methodology_tag="filtering", verified=True),
        LiteratureRecord(cite_key="lee2021conf", title="Conference work", authors=["Lee, Kim"], year=2021,
                         doi="10.1000/xyz", venue="Proceedings of the IEEE Conference on X",
                         methodology_tag="m", verified=True),
        LiteratureRecord(cite_key="unverified2022", title="Unverified", authors=["X"], year=2022,
                         venue="V", methodology_tag="m", verified=False),
    ]
    text = export_bibtex(recs, tmp_path / "refs.bib")
    assert (tmp_path / "refs.bib").read_text() == text
    entries = re.findall(r"@(\w+)\{([^,]+),", text)
    assert entries == [("article", "smith2020fast"), ("inproceedings", "lee2021conf")]
    assert r"Fast \& robust filters: 100\% better" in text
    assert "doi = {10.1000/abc_1}" in text and "year = {2020}" in text
    assert "journal = {Journal of Signals}" in text and "booktitle = {Proceedings of the IEEE Conference on X}" in text
    assert "author = {Smith, John and Doe, Jane}" in text
    assert text.count("DOI verified via Crossref") == 2


def test_literature_matrix_exports(tmp_path):
    recs = load_baseline_cache()
    rows = export_literature_matrix(recs, tmp_path / "m.csv", tmp_path / "m.md")
    with open(tmp_path / "m.csv", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["CiteKey", "Year", "Methodology", "CoreMetrics", "Limitations"]
        assert len(list(reader)) == len(recs) == len(rows)
    md = (tmp_path / "m.md").read_text(encoding="utf-8").splitlines()
    assert md[0].startswith("| CiteKey | Year |") and len(md) == len(recs) + 2
    assert [int(r["Year"]) for r in rows] == sorted(int(r["Year"]) for r in rows)


def test_cite_keys_are_unique_and_bibtex_safe():
    taken = set()
    keys = [make_cite_key(["Ünal, Ali"], 2020, "Robust filtering", taken) for _ in range(3)]
    assert keys == ["unal2020robust", "unal2020robusta", "unal2020robustb"]


# =============================================================== experiment harness
FS = 250.0


def _signal_profile(**over):
    d = {
        "profile_id": "sig_test",
        "title": "Synthetic ECG filter benchmark",
        "domain_type": "signal_processing",
        "literature_query": {"keywords": ["ecg"], "start_year": 2000, "max_papers": 5},
        "data_config": {
            "sampling_rate_hz": FS,
            "simulator": {"name": "ecg", "params": {
                "duration_s": 8.0, "heart_rate_bpm": 72, "rr_std_s": 0.02, "noise_std": 0.08,
                "baseline_wander_amp": 0.3, "baseline_wander_hz": 0.25,
                "powerline_amp": 0.15, "powerline_hz": 50.0}},
        },
        "baselines": ["moving_average", "savitzky_golay", "butterworth_zero_phase", "butterworth_causal"],
        "metrics_target": ["delta_snr_db", "rmse", "pearson_r", "phase_delay_ms"],
        "random_seed": 11,
        "n_repeats": 6,
        "method_params": {
            "moving_average": {"window_s": 0.02},
            "savitzky_golay": {"window_s": 0.05, "polyorder": 3},
            "butterworth_zero_phase": {"order": 2, "low_hz": 0.67, "high_hz": 40.0},
            "butterworth_causal": {"order": 2, "low_hz": 0.67, "high_hz": 40.0},
        },
    }
    d.update(over)
    return TopicProfile.model_validate(d)


def test_zero_phase_butterworth_has_no_phase_shift():
    t = np.arange(0, 10, 1 / FS)
    clean = np.sin(2 * np.pi * 2.0 * t) + 0.5 * np.sin(2 * np.pi * 5.0 * t)
    noisy = clean + 0.3 * np.sin(2 * np.pi * 60.0 * t)
    params = {"order": 4, "high_hz": 20.0}
    zp = eh.FILTER_REGISTRY["butterworth_zero_phase"](noisy, FS, params)
    causal = eh.FILTER_REGISTRY["butterworth_causal"](noisy, FS, params)
    assert abs(eh.phase_delay_ms(clean, zp, FS)) < 1.0
    assert eh.phase_delay_ms(clean, causal, FS) > 5.0
    b, a = sps.butter(4, 20.0, fs=FS)
    _, h = sps.freqz(b, a, worN=[2.0, 5.0], fs=FS)
    assert np.allclose(np.angle(h * np.conj(h)), 0.0)


def test_metric_functions_on_known_inputs():
    clean = np.sin(np.linspace(0, 20 * np.pi, 2000))
    assert eh.rmse(clean, clean + 0.1) == pytest.approx(0.1)
    assert eh.pearson_r(clean, 2 * clean + 3) == pytest.approx(1.0)
    noise = 0.1 * np.ones_like(clean)
    assert eh.snr_db(clean, clean + noise) == pytest.approx(10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2)), rel=1e-6)
    assert eh.phase_delay_ms(clean, np.roll(clean, 5), 1000.0) == pytest.approx(5.0, abs=0.05)


def test_signal_pipeline_end_to_end(tmp_path):
    profile = _signal_profile()
    bundle = eh.ExperimentHarness(profile, results_dir=tmp_path, repo_root=REPO_ROOT).run()
    restored = ExperimentResultBundle.load(tmp_path / "sig_test" / "metrics.json")
    assert restored == bundle
    assert set(bundle.methods_metrics) == {"raw", *profile.baselines}
    for method, metrics in bundle.methods_metrics.items():
        assert set(profile.metrics_target) <= set(metrics)
    mm = bundle.methods_metrics
    assert mm["butterworth_zero_phase"]["delta_snr_db"].mean > mm["raw"]["delta_snr_db"].mean + 3
    assert abs(mm["butterworth_zero_phase"]["phase_delay_ms"].mean) < 1000 / FS
    assert mm["butterworth_causal"]["phase_delay_ms"].mean > abs(mm["butterworth_zero_phase"]["phase_delay_ms"].mean)
    assert mm["raw"]["delta_snr_db"].mean == 0.0
    assert any(item.is_best for item in (m["delta_snr_db"] for m in mm.values()))
    assert all(0 <= t["p_value"] <= 1 for t in bundle.statistical_tests) and bundle.statistical_tests
    with np.load(bundle.artifacts_paths["raw_series"]) as z:
        assert {"t", "clean", "noisy", "out__butterworth_zero_phase"} <= set(z.files)
        assert z["metric__raw__rmse"].shape == (6,)
    again = eh.ExperimentHarness(profile, results_dir=tmp_path / "b", repo_root=REPO_ROOT).run()
    assert again.dataset_sha256 == bundle.dataset_sha256
    assert again.methods_metrics["savitzky_golay"]["rmse"] == mm["savitzky_golay"]["rmse"]


def test_signal_pipeline_reads_external_csv(tmp_path):
    t = np.arange(0, 4, 1 / FS)
    clean = np.sin(2 * np.pi * 1.5 * t)
    noisy = clean + 0.2 * np.random.default_rng(0).standard_normal(len(t))
    path = tmp_path / "wave.csv"
    np.savetxt(path, np.column_stack([clean, noisy]), delimiter=",", header="clean,noisy", comments="")
    profile = _signal_profile(data_config={"input_path": str(path), "sampling_rate_hz": FS}, n_repeats=1)
    bundle = eh.ExperimentHarness(profile, results_dir=tmp_path, repo_root=REPO_ROOT).run()
    import hashlib
    assert bundle.dataset_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert bundle.methods_metrics["butterworth_zero_phase"]["rmse"].mean < bundle.methods_metrics["raw"]["rmse"].mean


def test_unregistered_method_and_missing_target_fail_loudly(tmp_path):
    with pytest.raises(KeyError):
        eh.ExperimentHarness(_signal_profile(baselines=["magic_filter"]), results_dir=tmp_path, repo_root=REPO_ROOT).run()
    with pytest.raises(ValueError):
        eh.ExperimentHarness(_signal_profile(metrics_target=["accuracy"]), results_dir=tmp_path, repo_root=REPO_ROOT).run()


def _ablation_profile():
    return TopicProfile.model_validate({
        "profile_id": "abl_test", "title": "Ablation", "domain_type": "ablation_study",
        "literature_query": {"keywords": ["ablation"], "start_year": 2010, "max_papers": 3},
        "data_config": {"simulator": {"name": "additive_module_contributions", "params": {
            "base_value": 0.6, "contributions": {"A": 0.1, "B": 0.05}, "interactions": {"A*B": 0.02},
            "noise_std": 0.01, "metric_name": "accuracy"}}},
        "baselines": ["baseline"], "metrics_target": ["accuracy"], "random_seed": 3, "n_repeats": 8,
        "metrics_direction": {"accuracy": "max"},
        "ablation": {"modules": ["A", "B"]},
    })


def test_ablation_metrics_are_complete(tmp_path):
    bundle = eh.ExperimentHarness(_ablation_profile(), results_dir=tmp_path, repo_root=REPO_ROOT).run()
    mm = bundle.methods_metrics
    assert list(mm) == ["full", "w/o A", "w/o B", "baseline"]
    for cfg, metrics in mm.items():
        assert set(metrics) == {"accuracy", "runtime_s"}
        acc = metrics["accuracy"]
        assert acc.std > 0 and acc.iqr >= 0 and acc.ci_95[0] <= acc.mean <= acc.ci_95[1]
        assert metrics["runtime_s"].mean >= 0
    assert mm["full"]["accuracy"].is_best
    assert mm["full"]["accuracy"].mean > mm["w/o A"]["accuracy"].mean > mm["baseline"]["accuracy"].mean
    assert {t["method_b"] for t in bundle.statistical_tests} == {"w/o A", "w/o B", "baseline"}


def test_ccd_and_grid_designs(tmp_path):
    from uni_thesis.schema import ExperimentDesign, FactorRange
    fr = {"x1": FactorRange(low=0.0, high=10.0, steps=3), "x2": FactorRange(low=1.0, high=100.0, steps=3, log_scale=True)}
    grid = eh.build_design(fr, ExperimentDesign(kind="grid"))
    assert len(grid) == 9 and {round(p["x2"], 6) for p in grid} == {1.0, 10.0, 100.0}
    ccd = eh.build_design(fr, ExperimentDesign(kind="ccd", center_points=3))
    assert len(ccd) == 2 ** 2 + 2 * 2 + 3
    assert all(0.0 <= p["x1"] <= 10.0 and 1.0 <= p["x2"] <= 100.0 + 1e-9 for p in ccd)

    profile = TopicProfile.model_validate({
        "profile_id": "opt_test", "title": "RSM", "domain_type": "parameter_optimization",
        "literature_query": {"keywords": ["rsm"], "start_year": 2010, "max_papers": 3},
        "data_config": {"simulator": {"name": "analytic_response_surface", "params": {
            "intercept": 5.0, "linear": {"x1": 1.0}, "quadratic": {"x1": -0.1}, "noise_std": 0.05}},
            "factor_ranges": {"x1": {"low": 0.0, "high": 10.0}, "x2": {"low": -1.0, "high": 1.0}}},
        "baselines": ["center"], "metrics_target": ["response"], "random_seed": 1, "n_repeats": 3,
        "metrics_direction": {"response": "max"}, "design": {"kind": "ccd", "center_points": 2},
    })
    bundle = eh.ExperimentHarness(profile, results_dir=tmp_path, repo_root=REPO_ROOT).run()
    assert len(bundle.methods_metrics) == 10
    best = [k for k, v in bundle.methods_metrics.items() if v["response"].is_best]
    assert len(best) == 1 and best[0].startswith("x1=5")


def test_plugin_registry_loads_user_objective(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "my_objective.py").write_text(
        "from uni_thesis.exp_harness import register_objective\n"
        "@register_objective('plugin_sum_test')\n"
        "def f(config, params, rng):\n"
        "    return {'total': sum(config['factors'].values()) + params['offset']}\n",
        encoding="utf-8",
    )
    loaded = eh.load_plugins(plugin_dir)
    assert loaded == ["uni_thesis_plugin_my_objective"]
    assert "plugin_sum_test" in eh.OBJECTIVE_REGISTRY
    out = eh.OBJECTIVE_REGISTRY["plugin_sum_test"]({"factors": {"a": 1.0, "b": 2.0}}, {"offset": 0.5}, None)
    assert out == {"total": 3.5}
    with pytest.raises(ValueError):
        eh.register_objective("plugin_sum_test")(lambda c, p, r: {})
