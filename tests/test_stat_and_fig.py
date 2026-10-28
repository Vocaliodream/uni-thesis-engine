"""Tests for the statistical router, the figure foundry and the Step-3 literature enhancements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from uni_thesis import exp_harness as eh
from uni_thesis import fig_foundry as ff
from uni_thesis import lit_engine as le
from uni_thesis.schema import ExperimentResultBundle, LiteratureQuery, TopicProfile
from uni_thesis.stat_router import (
    StatisticalRouter,
    fit_response_surface,
    load_raw_samples,
    rank_biserial_paired,
    significance_stars,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20261028)


def _normal_sample(n, loc, scale, seed):
    """Exactly normal-shaped sample (normal quantiles) in a seeded random order."""
    from scipy import stats as _st
    q = _st.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    return loc + scale * np.random.default_rng(seed).permutation(q)


# =============================================================== literature enhancements
class RecordingSession:
    def __init__(self, abstract=None, inverted=None):
        self.requests = []
        self.abstract = abstract
        self.inverted = inverted

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, dict(params or {})))
        doi = "10.1109/18.382009"
        msg = {"DOI": doi, "title": ["De-noising by soft-thresholding"], "author": [{"family": "Donoho", "given": "D.L."}],
               "issued": {"date-parts": [[1995, 5]]}, "container-title": ["IEEE Transactions on Information Theory"],
               "type": "journal-article"}
        if self.abstract is not None:
            msg["abstract"] = self.abstract

        class R:
            status_code = 200

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        if "openalex" in url:
            work = {"doi": f"https://doi.org/{doi}", "title": msg["title"][0], "publication_year": 1995,
                    "authorships": [{"author": {"display_name": "David L. Donoho"}}],
                    "abstract_inverted_index": self.inverted}
            return R({"results": [work]})
        if url.endswith("/works"):
            return R({"message": {"items": [msg]}})
        return R({"message": msg})


def test_polite_pool_user_agent_and_mailto(monkeypatch):
    monkeypatch.delenv("RESEARCH_CONTACT_EMAIL", raising=False)
    assert le.polite_user_agent() == "uni-thesis-engine/0.1 (mailto:agent@uni-thesis-engine.org)"
    assert le._session(None).headers["User-Agent"] == le.polite_user_agent()
    monkeypatch.setenv("RESEARCH_CONTACT_EMAIL", "lab@example.edu")
    assert le.contact_email() == "lab@example.edu"
    monkeypatch.setenv("RESEARCH_CONTACT_EMAIL", "not-an-email")
    assert le.contact_email() == le.DEFAULT_CONTACT_EMAIL


def test_polite_pool_mailto_sent_to_both_apis(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_CONTACT_EMAIL", "lab@example.edu")
    monkeypatch.setenv("OPENALEX_API_KEY", "k-123")
    s = RecordingSession()
    le.fetch_and_verify_literature(LiteratureQuery(keywords=["thresholding"], start_year=1990, max_papers=2),
                                   results_dir=tmp_path, session=s, offline=False)
    oa = [p for u, p in s.requests if "openalex" in u]
    cr = [p for u, p in s.requests if "crossref" in u]
    assert oa and oa[0]["mailto"] == "lab@example.edu" and oa[0]["api_key"] == "k-123"
    assert cr and all(p.get("mailto") == "lab@example.edu" for p in cr)


def test_limitation_extracted_verbatim_from_abstract():
    abstract = ("<jats:p>We propose a denoiser. It improves SNR by a wide margin. However, it assumes "
                "stationary noise. The main bottleneck is computation. Future work will target embedded use.</jats:p>")
    note = le.extract_limitations(abstract)
    assert note.startswith("[Extracted from Abstract] ")
    for sentence in ["However, it assumes stationary noise.", "The main bottleneck is computation.",
                     "Future work will target embedded use."]:
        assert sentence in note
    assert "We propose a denoiser." not in note
    assert le.extract_limitations("We propose X. It works well.") == ""
    assert le.extract_limitations("") == ""


def test_limitation_from_openalex_inverted_index_and_pending_fallback(tmp_path):
    words = "Wavelet shrinkage works. However, the threshold is limited to Gaussian noise.".split()
    inverted = {}
    for i, w in enumerate(words):
        inverted.setdefault(w, []).append(i)
    recs = le.fetch_and_verify_literature(LiteratureQuery(keywords=["thresholding"], start_year=1990, max_papers=2),
                                          results_dir=tmp_path / "a", session=RecordingSession(inverted=inverted),
                                          offline=False)
    assert recs[0].limitation_note == "[Extracted from Abstract] However, the threshold is limited to Gaussian noise."
    recs2 = le.fetch_and_verify_literature(LiteratureQuery(keywords=["thresholding"], start_year=1990, max_papers=2),
                                           results_dir=tmp_path / "b", session=RecordingSession(), offline=False)
    assert recs2[0].limitation_note == ""
    rows = le.export_literature_matrix(recs2, tmp_path / "m.csv", tmp_path / "m.md")
    assert rows[0]["Limitations"] == le.PENDING_REVIEW == "待人工补充（Pending manual review）"


# =============================================================== statistical router
def test_significance_stars():
    assert [significance_stars(p) for p in (0.0001, 0.005, 0.03, 0.2)] == ["***", "**", "*", "ns"]


def test_normal_paired_data_routes_to_paired_t_test():
    base = _normal_sample(30, 10, 1, 3)
    a = base + _normal_sample(30, 1.0, 0.3, 4)
    res = StatisticalRouter().compare_two(a, base, paired=True, metric="m", name_a="prop", name_b="base")
    assert res["test"] == "paired_t_test"
    assert res["effect_size"]["name"] == "cohens_d_z" and res["effect_size"]["value"] > 1
    assert res["significance"] == "***" and 0 <= res["p_value"] < 1e-3


def test_normal_independent_equal_var_routes_to_student_t():
    a, b = _normal_sample(40, 5, 1, 1), _normal_sample(40, 5.8, 1, 2)
    res = StatisticalRouter().compare_two(a, b, paired=False)
    assert res["test"] == "student_t_test" and res["effect_size"]["name"] == "cohens_d"
    assert res["diagnostics"]["levene"]["equal_var"] is True


def test_non_normal_data_routes_to_nonparametric_tests():
    a = RNG.exponential(1.0, 40) ** 3
    b = RNG.exponential(1.0, 40) ** 3 + 2.0
    router = StatisticalRouter()
    ind = router.compare_two(a, b, paired=False)
    assert ind["test"] == "mann_whitney_u" and ind["effect_size"]["name"] == "rank_biserial"
    assert -1 <= ind["effect_size"]["value"] < 0
    par = router.compare_two(b, a, paired=True)
    assert par["test"] == "wilcoxon_signed_rank"
    assert par["effect_size"]["value"] == pytest.approx(rank_biserial_paired(b, a))


def test_multi_group_routing_anova_vs_kruskal():
    router = StatisticalRouter()
    normal = {f"g{i}": _normal_sample(30, i * 0.8, 1.0, i) for i in range(3)}
    res = router.compare_multi(normal, metric="m")
    assert res["test"] == "one_way_anova" and res["posthoc"]["test"] == "tukey_hsd"
    assert len(res["posthoc"]["pairs"]) == 3 and 0 < res["effect_size"]["value"] < 1
    skewed = {f"g{i}": RNG.lognormal(i * 0.5, 1.2, 30) for i in range(3)}
    res2 = router.compare_multi(skewed, metric="m")
    assert res2["test"] == "kruskal_wallis_h" and res2["effect_size"]["name"] == "epsilon_squared"


def test_response_surface_recovers_known_optimum():
    g = np.linspace(-2, 2, 7)
    X = np.array([(x1, x2) for x1 in g for x2 in g])
    y = 10 - (X[:, 0] - 0.5) ** 2 - 2 * (X[:, 1] + 0.3) ** 2 + 0.3 * X[:, 0] * X[:, 1] + RNG.normal(0, 0.05, len(X))
    m = fit_response_surface(X, y, ["a", "b"])
    assert m["r2"] > 0.99 and m["r2_adj"] > 0.99 and m["p_value"] < 1e-6 and m["significance"] == "***"
    sp = m["stationary_point"]
    assert sp["nature"] == "maximum"
    # true optimum: grad = 0 -> [[-2, 0.3], [0.3, -4]] x = [-1, 1.2]
    truth = np.linalg.solve(np.array([[-2.0, 0.3], [0.3, -4.0]]), np.array([-1.0, 1.2]))
    assert sp["point"]["a"] == pytest.approx(truth[0], abs=0.05)
    assert sp["point"]["b"] == pytest.approx(truth[1], abs=0.05)
    with pytest.raises(ValueError):
        fit_response_surface(X[:5], y[:5], ["a", "b"])


def test_harness_writes_routed_tests_into_metrics_json(tmp_path):
    profile = TopicProfile.model_validate({
        "profile_id": "rsm_route", "title": "RSM", "domain_type": "parameter_optimization",
        "literature_query": {"keywords": ["rsm"], "start_year": 2010, "max_papers": 3},
        "data_config": {"simulator": {"name": "analytic_response_surface", "params": {
            "intercept": 5.0, "linear": {"x1": 1.0, "x2": 0.5}, "quadratic": {"x1": -0.1, "x2": -0.4},
            "interaction": {"x1*x2": 0.05}, "noise_std": 0.05}},
            "factor_ranges": {"x1": {"low": 0.0, "high": 10.0}, "x2": {"low": -2.0, "high": 2.0}}},
        "baselines": ["center"], "metrics_target": ["response"], "random_seed": 5, "n_repeats": 4,
        "metrics_direction": {"response": "max"}, "design": {"kind": "ccd", "center_points": 3},
    })
    bundle = eh.ExperimentHarness(profile, results_dir=tmp_path, repo_root=REPO_ROOT).run()
    kinds = {t["comparison"] for t in bundle.statistical_tests}
    assert kinds == {"pairwise", "multi_group", "response_surface"}
    for t in bundle.statistical_tests:
        assert {"test", "metric", "p_value", "significance"} <= set(t)
    rsm = next(t for t in bundle.statistical_tests if t["comparison"] == "response_surface")
    assert rsm["r2_adj"] > 0.95 and rsm["stationary_point"]["nature"] == "maximum"
    reloaded = StatisticalRouter().analyze_results_dir(tmp_path / "rsm_route", ["response"],
                                                      proposed=next(iter(bundle.methods_metrics)), paired=False)
    assert ExperimentResultBundle.load(tmp_path / "rsm_route" / "metrics.json") == reloaded
    samples, design = load_raw_samples(tmp_path / "rsm_route" / "raw_series.npz")
    assert design["factor_names"] == ["x1", "x2"] and len(samples) == 11


# =============================================================== figures
def _assert_figure(paths):
    for key in ("png", "pdf"):
        p = Path(paths[key])
        assert p.exists() and p.stat().st_size > 10 * 1024, f"{p} missing or too small"
    with Image.open(paths["png"]) as im:
        dpi = im.info["dpi"]
    assert round(dpi[0]) == 600 and round(dpi[1]) == 600
    assert Path(paths["pdf"]).read_bytes()[:5] == b"%PDF-"


def test_publication_style_settings():
    ff.set_publication_style()
    import matplotlib
    import matplotlib.pyplot as plt
    assert matplotlib.get_backend().lower() == "agg"
    assert plt.rcParams["font.family"] == ["serif"] and plt.rcParams["xtick.direction"] == "in"
    assert not plt.rcParams["axes.spines.top"] and not plt.rcParams["axes.spines.right"]
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"][:3] == [c for c in ff.OKABE_ITO[:3]]


def test_plot_signal_comparison(tmp_path):
    fs = 250.0
    t = np.arange(0, 6, 1 / fs)
    clean = np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.sin(2 * np.pi * 7 * t)
    noisy = clean + 0.25 * np.sin(2 * np.pi * 50 * t) + 0.1 * RNG.standard_normal(t.size)
    params = {"order": 4, "high_hz": 20.0}
    outs = {"Butterworth zero-phase": eh.FILTER_REGISTRY["butterworth_zero_phase"](noisy, fs, params),
            "Butterworth causal": eh.FILTER_REGISTRY["butterworth_causal"](noisy, fs, params)}
    res = ff.plot_signal_comparison(t, clean, outs, fs, tmp_path / "sig", noisy=noisy, zoom=(2.0, 2.6))
    _assert_figure(res)
    assert abs(res["phase_delay_ms"]["Butterworth zero-phase"]) < 1.0
    assert res["phase_delay_ms"]["Butterworth causal"] > 5.0


def test_plot_bland_altman(tmp_path):
    ref = RNG.normal(70, 10, 120)
    meas = ref + RNG.normal(0.8, 2.0, 120)
    res = ff.plot_bland_altman(ref, meas, tmp_path / "ba", units="bpm")
    _assert_figure(res)
    diff = meas - ref
    assert res["bias"] == pytest.approx(diff.mean())
    assert res["loa"][1] - res["loa"][0] == pytest.approx(2 * 1.96 * diff.std(ddof=1))


def test_plot_ablation_and_baselines(tmp_path):
    samples = {"Full": RNG.normal(0.9, 0.01, 10), "w/o A": RNG.normal(0.85, 0.01, 10), "Baseline": RNG.normal(0.8, 0.01, 10)}
    router = StatisticalRouter()
    tests = [router.compare_two(samples["Full"], samples[o], paired=False, name_a="Full", name_b=o) for o in ("w/o A", "Baseline")]
    res = ff.plot_ablation_and_baselines(samples, tmp_path / "abl", "Accuracy", tests)
    _assert_figure(res)
    assert res["brackets"] == 2 and all(t["significance"] == "***" for t in tests)


def test_plot_response_surface(tmp_path):
    g = np.linspace(0, 10, 6)
    X = np.array([(a, b) for a in g for b in g])
    y = 50 - (X[:, 0] - 4) ** 2 - 0.5 * (X[:, 1] - 6) ** 2 + RNG.normal(0, 0.1, len(X))
    model = fit_response_surface(X, y, ["temp", "time"])
    res = ff.plot_response_surface(model, {"temp": (0, 10), "time": (0, 10)}, tmp_path / "rsm",
                                   response_label="Yield", observed=(X, y))
    _assert_figure(res)
    assert res["optimum_source"] == "stationary_point"
    assert abs(res["optimum"]["temp"] - 4) < 0.3 and abs(res["optimum"]["time"] - 6) < 0.3
