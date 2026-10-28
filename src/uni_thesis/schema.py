"""Strongly typed cross-module data contracts for uni-thesis-engine.

All models are strict: unknown fields are rejected, types are not silently
coerced, and invariants (ranges, ordering, statistical consistency) are
validated at construction time.  No topic-specific constants live here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

__all__ = [
    "DomainType",
    "LiteratureQuery",
    "SimulatorConfig",
    "FactorRange",
    "DataConfig",
    "ExperimentDesign",
    "AblationConfig",
    "TopicProfile",
    "LiteratureRecord",
    "MetricItem",
    "ExperimentResultBundle",
    "IntegrityError",
]

DomainType = Literal["signal_processing", "parameter_optimization", "ablation_study"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_CITE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_:\-]*$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

Number = Union[StrictInt, StrictFloat]


class IntegrityError(ValueError):
    """Raised when a serialized bundle fails its SHA-256 integrity check."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=False, validate_assignment=True)


# --------------------------------------------------------------------------- #
# TopicProfile
# --------------------------------------------------------------------------- #
class LiteratureQuery(_StrictModel):
    keywords: List[StrictStr] = Field(min_length=1)
    start_year: StrictInt = Field(ge=1900)
    max_papers: StrictInt = Field(gt=0, le=1000)

    @field_validator("keywords")
    @classmethod
    def _keywords_non_blank(cls, v: List[str]) -> List[str]:
        cleaned = [k.strip() for k in v]
        if any(not k for k in cleaned):
            raise ValueError("keywords must not contain blank entries")
        if len(set(k.lower() for k in cleaned)) != len(cleaned):
            raise ValueError("keywords must be unique (case-insensitive)")
        return cleaned

    @field_validator("start_year")
    @classmethod
    def _year_not_future(cls, v: int) -> int:
        if v > datetime.now(timezone.utc).year:
            raise ValueError("start_year cannot be in the future")
        return v


class FactorRange(_StrictModel):
    low: Number
    high: Number
    steps: Optional[StrictInt] = Field(default=None, gt=1)
    log_scale: StrictBool = False

    @model_validator(mode="after")
    def _ordered(self) -> "FactorRange":
        if not self.low < self.high:
            raise ValueError("factor range requires low < high")
        if self.log_scale and self.low <= 0:
            raise ValueError("log_scale factor range requires low > 0")
        return self


class SimulatorConfig(_StrictModel):
    name: StrictStr = Field(min_length=1)
    params: Dict[StrictStr, Any] = Field(default_factory=dict)


class DataConfig(_StrictModel):
    input_path: Optional[StrictStr] = None
    simulator: Optional[SimulatorConfig] = None
    sampling_rate_hz: Optional[Number] = None
    factor_ranges: Dict[StrictStr, FactorRange] = Field(default_factory=dict)

    @field_validator("sampling_rate_hz")
    @classmethod
    def _positive_rate(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (not math.isfinite(v) or v <= 0):
            raise ValueError("sampling_rate_hz must be a positive finite number")
        return v

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "DataConfig":
        has_path = bool(self.input_path)
        has_sim = self.simulator is not None
        if has_path == has_sim:
            raise ValueError("data_config requires exactly one of input_path or simulator")
        return self


class ExperimentDesign(_StrictModel):
    """Design of experiments for multi-factor evaluation."""

    kind: Literal["grid", "ccd"] = "grid"
    ccd_alpha: Optional[Number] = Field(default=None, gt=0)
    center_points: StrictInt = Field(default=1, ge=1)


class AblationConfig(_StrictModel):
    """Component ablation.  ``removal``: Full vs. w/o each module vs. Baseline;
    ``incremental``: Baseline, +m1, +m1+m2, ..., Full (cumulative stages)."""

    modules: List[StrictStr] = Field(min_length=1)
    include_baseline: StrictBool = True
    mode: Literal["removal", "incremental"] = "removal"

    @field_validator("modules")
    @classmethod
    def _unique_modules(cls, v: List[str]) -> List[str]:
        if any(not m.strip() for m in v) or len(set(v)) != len(v):
            raise ValueError("ablation modules must be unique and non-blank")
        return v


class TopicProfile(_StrictModel):
    profile_id: StrictStr
    title: StrictStr = Field(min_length=1)
    title_zh: Optional[StrictStr] = Field(default=None, min_length=1)
    domain_type: DomainType
    literature_query: LiteratureQuery
    data_config: DataConfig
    baselines: List[StrictStr] = Field(min_length=1)
    metrics_target: List[StrictStr] = Field(min_length=1)
    random_seed: StrictInt = Field(ge=0, le=2**32 - 1)
    n_repeats: StrictInt = Field(default=1, ge=1, le=10000)
    method_params: Dict[StrictStr, Dict[StrictStr, Any]] = Field(default_factory=dict)
    metrics_direction: Dict[StrictStr, Literal["max", "min", "min_abs"]] = Field(default_factory=dict)
    design: Optional[ExperimentDesign] = None
    ablation: Optional[AblationConfig] = None
    proposed_method: Optional[StrictStr] = None

    @field_validator("profile_id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError("profile_id must be alphanumeric with '_' or '-' and start alphanumeric")
        return v

    @field_validator("baselines", "metrics_target")
    @classmethod
    def _unique_non_blank(cls, v: List[str]) -> List[str]:
        if any(not s.strip() for s in v):
            raise ValueError("entries must not be blank")
        if len(set(v)) != len(v):
            raise ValueError("entries must be unique")
        return v

    @model_validator(mode="after")
    def _domain_requirements(self) -> "TopicProfile":
        if self.domain_type == "signal_processing" and self.data_config.sampling_rate_hz is None:
            raise ValueError("signal_processing profiles require data_config.sampling_rate_hz")
        if self.domain_type == "parameter_optimization" and not self.data_config.factor_ranges:
            raise ValueError("parameter_optimization profiles require data_config.factor_ranges")
        if self.domain_type == "ablation_study" and self.ablation is None:
            raise ValueError("ablation_study profiles require an ablation config")
        return self

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TopicProfile":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"profile file {path} must contain a mapping at top level")
        return cls.model_validate(raw)


# --------------------------------------------------------------------------- #
# LiteratureRecord
# --------------------------------------------------------------------------- #
class LiteratureRecord(_StrictModel):
    cite_key: StrictStr
    title: StrictStr = Field(min_length=1)
    authors: List[StrictStr] = Field(min_length=1)
    year: StrictInt = Field(ge=1800)
    doi: Optional[StrictStr] = None
    venue: StrictStr = Field(min_length=1)
    methodology_tag: StrictStr = Field(min_length=1)
    limitation_note: StrictStr = ""
    core_metrics: List[StrictStr] = Field(default_factory=list)
    verified: StrictBool = False

    @field_validator("cite_key")
    @classmethod
    def _cite_key_format(cls, v: str) -> str:
        if not _CITE_KEY_RE.match(v):
            raise ValueError("cite_key must be a BibTeX-safe identifier")
        return v

    @field_validator("doi")
    @classmethod
    def _doi_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _DOI_RE.match(v):
            raise ValueError("doi must look like '10.xxxx/...'")
        return v

    @field_validator("year")
    @classmethod
    def _year_not_future(cls, v: int) -> int:
        if v > datetime.now(timezone.utc).year + 1:
            raise ValueError("year is in the future")
        return v

    @model_validator(mode="after")
    def _verified_needs_doi(self) -> "LiteratureRecord":
        if self.verified and not self.doi:
            raise ValueError("a verified record must carry a DOI")
        return self


# --------------------------------------------------------------------------- #
# MetricItem
# --------------------------------------------------------------------------- #
class MetricItem(_StrictModel):
    name: StrictStr = Field(min_length=1)
    mean: Number
    std: Number = Field(ge=0)
    median: Number
    iqr: Number = Field(ge=0)
    unit: StrictStr = ""
    ci_95: Tuple[Number, Number]
    is_best: StrictBool = False

    @model_validator(mode="after")
    def _consistency(self) -> "MetricItem":
        values = [self.mean, self.std, self.median, self.iqr, *self.ci_95]
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("metric values must be finite")
        lo, hi = self.ci_95
        if lo > hi:
            raise ValueError("ci_95 lower bound exceeds upper bound")
        if not (lo <= self.mean <= hi):
            raise ValueError("mean must lie within ci_95")
        return self


# --------------------------------------------------------------------------- #
# ExperimentResultBundle
# --------------------------------------------------------------------------- #
def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class ExperimentResultBundle(_StrictModel):
    profile_id: StrictStr
    git_commit: StrictStr
    dataset_sha256: StrictStr
    timestamp_utc: StrictStr
    methods_metrics: Dict[StrictStr, Dict[StrictStr, MetricItem]] = Field(min_length=1)
    statistical_tests: List[Dict[StrictStr, Any]] = Field(default_factory=list)
    artifacts_paths: Dict[StrictStr, StrictStr] = Field(default_factory=dict)

    @field_validator("profile_id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError("profile_id has invalid format")
        return v

    @field_validator("git_commit")
    @classmethod
    def _commit_format(cls, v: str) -> str:
        if not _GIT_COMMIT_RE.match(v):
            raise ValueError("git_commit must be a 7-40 char lowercase hex SHA")
        return v

    @field_validator("dataset_sha256")
    @classmethod
    def _sha_format(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ValueError("dataset_sha256 must be a 64-char lowercase hex digest")
        return v

    @field_validator("timestamp_utc")
    @classmethod
    def _timestamp_format(cls, v: str) -> str:
        try:
            ts = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp_utc must be ISO-8601") from exc
        if ts.tzinfo is None or ts.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("timestamp_utc must be timezone-aware UTC")
        return v

    @field_validator("statistical_tests")
    @classmethod
    def _tests_shape(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, test in enumerate(v):
            for key in ("test", "metric", "p_value"):
                if key not in test:
                    raise ValueError(f"statistical_tests[{i}] missing required key '{key}'")
            p = test["p_value"]
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"statistical_tests[{i}].p_value must be a number in [0, 1]")
        return v

    @model_validator(mode="after")
    def _metric_keys_match(self) -> "ExperimentResultBundle":
        for method, metrics in self.methods_metrics.items():
            if not metrics:
                raise ValueError(f"method '{method}' has no metrics")
            for key, item in metrics.items():
                if key != item.name:
                    raise ValueError(f"metric key '{key}' does not match MetricItem.name '{item.name}'")
        return self

    # -- integrity ---------------------------------------------------------- #
    def payload(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def compute_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def to_json(self, indent: Optional[int] = 2) -> str:
        envelope = {"bundle": self.payload(), "sha256": self.compute_sha256()}
        return json.dumps(envelope, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, text: str, verify: bool = True) -> "ExperimentResultBundle":
        data = json.loads(text)
        if not isinstance(data, dict) or "bundle" not in data or "sha256" not in data:
            raise IntegrityError("serialized bundle must contain 'bundle' and 'sha256' keys")
        bundle_payload = data["bundle"]
        if isinstance(bundle_payload, dict):
            for metrics in bundle_payload.get("methods_metrics", {}).values():
                for item in metrics.values():
                    if isinstance(item, dict) and isinstance(item.get("ci_95"), list):
                        item["ci_95"] = tuple(item["ci_95"])
        bundle = cls.model_validate(bundle_payload)
        if verify:
            expected = data["sha256"]
            actual = bundle.compute_sha256()
            if expected != actual:
                raise IntegrityError(f"SHA-256 mismatch: stored {expected}, computed {actual}")
        return bundle

    def save(self, path: Union[str, Path]) -> str:
        Path(path).write_text(self.to_json(), encoding="utf-8")
        return self.compute_sha256()

    @classmethod
    def load(cls, path: Union[str, Path], verify: bool = True) -> "ExperimentResultBundle":
        return cls.from_json(Path(path).read_text(encoding="utf-8"), verify=verify)
