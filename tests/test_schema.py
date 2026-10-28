"""Unit tests for uni_thesis.schema data contracts."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import yaml
from pydantic import ValidationError

from uni_thesis.cli import main as cli_main
from uni_thesis.schema import (
    ExperimentResultBundle,
    IntegrityError,
    LiteratureRecord,
    MetricItem,
    TopicProfile,
)

FAKE_SHA = hashlib.sha256(b"unit-test-dataset").hexdigest()


def _profile_dict() -> dict:
    return {
        "profile_id": "demo_signal-01",
        "title": "Synthetic test profile",
        "domain_type": "signal_processing",
        "literature_query": {"keywords": ["denoising", "wavelet"], "start_year": 2015, "max_papers": 30},
        "data_config": {"input_path": "data/raw.csv", "sampling_rate_hz": 100.0},
        "baselines": ["method_a", "method_b"],
        "metrics_target": ["metric_x"],
        "random_seed": 7,
    }


def _metric(name: str = "metric_x", mean: float = 1.0, best: bool = False) -> MetricItem:
    return MetricItem(
        name=name, mean=mean, std=0.1, median=mean, iqr=0.2, unit="dB",
        ci_95=(mean - 0.2, mean + 0.2), is_best=best,
    )


def _bundle() -> ExperimentResultBundle:
    return ExperimentResultBundle(
        profile_id="demo_signal-01",
        git_commit="a1b2c3d",
        dataset_sha256=FAKE_SHA,
        timestamp_utc="2026-01-01T00:00:00Z",
        methods_metrics={
            "method_a": {"metric_x": _metric(mean=1.0)},
            "method_b": {"metric_x": _metric(mean=2.0, best=True)},
        },
        statistical_tests=[{"test": "wilcoxon", "metric": "metric_x", "p_value": 0.03}],
        artifacts_paths={"fig_main": "results/fig_main.pdf"},
    )


# ---------------------------------------------------------------- profiles
def test_valid_profile_parses_from_yaml(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(_profile_dict()), encoding="utf-8")
    profile = TopicProfile.from_yaml(path)
    assert profile.domain_type == "signal_processing"
    assert profile.literature_query.max_papers == 30
    assert profile.data_config.sampling_rate_hz == 100.0


def test_optimization_profile_with_simulator_and_factor_ranges():
    d = _profile_dict()
    d["domain_type"] = "parameter_optimization"
    d["data_config"] = {
        "simulator": {"name": "generic_sim", "params": {"n": 10}},
        "factor_ranges": {"f1": {"low": 0.1, "high": 10.0, "steps": 5, "log_scale": True}},
    }
    profile = TopicProfile.model_validate(d)
    assert profile.data_config.factor_ranges["f1"].log_scale is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(domain_type="deep_magic"),
        lambda d: d.update(unknown_field=1),
        lambda d: d.update(random_seed="7"),
        lambda d: d.update(profile_id="bad id!"),
        lambda d: d.update(baselines=[]),
        lambda d: d["literature_query"].update(max_papers=0),
        lambda d: d["data_config"].pop("sampling_rate_hz"),
        lambda d: d["data_config"].update(simulator={"name": "sim"}),
    ],
    ids=["bad_domain", "extra_field", "seed_as_str", "bad_id", "empty_baselines",
         "zero_max_papers", "signal_without_rate", "two_data_sources"],
)
def test_invalid_profile_fields_rejected(mutate):
    d = copy.deepcopy(_profile_dict())
    mutate(d)
    with pytest.raises(ValidationError):
        TopicProfile.model_validate(d)


def test_factor_range_must_be_ordered():
    d = _profile_dict()
    d["domain_type"] = "parameter_optimization"
    d["data_config"]["factor_ranges"] = {"f1": {"low": 5, "high": 1}}
    with pytest.raises(ValidationError):
        TopicProfile.model_validate(d)


# ---------------------------------------------------------------- literature / metric
def test_literature_record_rules():
    rec = LiteratureRecord(
        cite_key="smith2020wave", title="T", authors=["A. Smith"], year=2020,
        doi="10.1000/xyz123", venue="J", methodology_tag="wavelet", verified=True,
    )
    assert rec.verified
    with pytest.raises(ValidationError):
        LiteratureRecord(cite_key="k1", title="T", authors=["A"], year=2020,
                         venue="J", methodology_tag="m", verified=True)
    with pytest.raises(ValidationError):
        LiteratureRecord(cite_key="k1", title="T", authors=["A"], year=2020,
                         doi="not-a-doi", venue="J", methodology_tag="m")


def test_metric_item_invariants():
    with pytest.raises(ValidationError):
        MetricItem(name="m", mean=1.0, std=-0.1, median=1.0, iqr=0.1, ci_95=(0.5, 1.5))
    with pytest.raises(ValidationError):
        MetricItem(name="m", mean=3.0, std=0.1, median=1.0, iqr=0.1, ci_95=(0.5, 1.5))
    with pytest.raises(ValidationError):
        MetricItem(name="m", mean=1.0, std=0.1, median=1.0, iqr=0.1, ci_95=(1.5, 0.5))


# ---------------------------------------------------------------- bundle
def test_bundle_roundtrip_preserves_content_and_hash():
    bundle = _bundle()
    text = bundle.to_json()
    restored = ExperimentResultBundle.from_json(text)
    assert restored == bundle
    assert restored.compute_sha256() == bundle.compute_sha256()
    assert json.loads(text)["sha256"] == bundle.compute_sha256()


def test_bundle_hash_is_deterministic_and_sensitive():
    a, b = _bundle(), _bundle()
    assert a.compute_sha256() == b.compute_sha256()
    b.methods_metrics["method_a"]["metric_x"] = _metric(mean=1.01)
    assert a.compute_sha256() != b.compute_sha256()


def test_tampered_bundle_fails_integrity_check():
    data = json.loads(_bundle().to_json())
    data["bundle"]["methods_metrics"]["method_a"]["metric_x"]["mean"] = 1.1
    tampered = json.dumps(data)
    with pytest.raises(IntegrityError):
        ExperimentResultBundle.from_json(tampered)
    assert ExperimentResultBundle.from_json(tampered, verify=False).methods_metrics["method_a"]["metric_x"].mean == 1.1


def test_bundle_rejects_invalid_provenance_fields():
    base = _bundle().payload()
    for key, bad in [("git_commit", "XYZ"), ("dataset_sha256", "abc"),
                     ("timestamp_utc", "2026-01-01T00:00:00"), ("timestamp_utc", "2026-01-01T08:00:00+08:00")]:
        d = copy.deepcopy(base)
        d[key] = bad
        with pytest.raises(ValidationError):
            ExperimentResultBundle.model_validate(d)
    d = copy.deepcopy(base)
    d["statistical_tests"] = [{"test": "t", "metric": "metric_x", "p_value": 1.5}]
    with pytest.raises(ValidationError):
        ExperimentResultBundle.model_validate(d)


def test_bundle_save_load_and_cli_verify(tmp_path, capsys):
    path = tmp_path / "bundle.json"
    digest = _bundle().save(path)
    assert ExperimentResultBundle.load(path).compute_sha256() == digest
    assert cli_main(["verify-bundle", str(path)]) == 0
    assert digest in capsys.readouterr().out
