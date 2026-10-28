"""Command-line entry point: `uni-thesis`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from pydantic import ValidationError

from . import __version__
from .schema import ExperimentResultBundle, IntegrityError, TopicProfile


def _cmd_validate_profile(args: argparse.Namespace) -> int:
    try:
        profile = TopicProfile.from_yaml(args.path)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "profile_id": profile.profile_id, "domain_type": profile.domain_type}))
    return 0


def _cmd_verify_bundle(args: argparse.Namespace) -> int:
    try:
        bundle = ExperimentResultBundle.load(args.path)
    except (ValidationError, IntegrityError, ValueError, OSError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "profile_id": bundle.profile_id, "sha256": bundle.compute_sha256()}))
    return 0


def _cmd_build_tex(args: argparse.Namespace) -> int:
    from .tex_bridge import generate_tex_artifacts

    try:
        info = generate_tex_artifacts(args.metrics, args.out_dir, proposed=args.proposed)
    except (ValidationError, IntegrityError, ValueError, KeyError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", **info}))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .audit import run_anti_hallucination_audit

    report = run_anti_hallucination_audit(args.tex_dir, args.bib, args.metrics, args.lit_cache,
                                          strict_integers=args.strict_integers)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) if args.json else report.format())
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uni-thesis", description="Universal thesis automation engine")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_profile = sub.add_parser("validate-profile", help="Validate a TopicProfile YAML file")
    p_profile.add_argument("path")
    p_profile.set_defaults(func=_cmd_validate_profile)

    p_bundle = sub.add_parser("verify-bundle", help="Verify an ExperimentResultBundle JSON file and its SHA-256")
    p_bundle.add_argument("path")
    p_bundle.set_defaults(func=_cmd_verify_bundle)

    p_tex = sub.add_parser("build-tex", help="Generate auto_metrics.tex and auto_tables.tex from metrics.json")
    p_tex.add_argument("metrics")
    p_tex.add_argument("--out-dir", default="templates/thesis_tex/generated")
    p_tex.add_argument("--proposed", default=None)
    p_tex.set_defaults(func=_cmd_build_tex)

    p_audit = sub.add_parser("audit", help="Run the anti-hallucination audit (raw numbers, citations, provenance)")
    p_audit.add_argument("--tex-dir", required=True)
    p_audit.add_argument("--bib", required=True)
    p_audit.add_argument("--metrics", required=True)
    p_audit.add_argument("--lit-cache", default=None)
    p_audit.add_argument("--json", action="store_true")
    p_audit.add_argument("--strict-integers", action="store_true",
                         help="also flag integers other than 0, 1, 2, 3, 100 and years")
    p_audit.set_defaults(func=_cmd_audit)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
