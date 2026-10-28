"""Command-line entry point: `uni-thesis`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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


def _load_profile(path: str) -> Optional[TopicProfile]:
    try:
        return TopicProfile.from_yaml(path)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"INVALID PROFILE: {exc}", file=sys.stderr)
        return None


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_profile

    if _load_profile(args.profile) is None:
        return 1
    try:
        result = run_profile(args.profile, results_dir=args.results_dir, template_dir=args.template_dir,
                             offline=True if args.offline else None, plugin_dir=args.plugins,
                             mirror_generated=not args.no_mirror)
    except (ValidationError, IntegrityError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"RUN FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok" if result.passed else "audit_failed", "profile_id": result.profile_id,
                      "verified_references": sum(r.verified for r in result.records),
                      "figures": [f["png"] for f in result.figures],
                      "auto_metrics": result.tex["auto_metrics"], "auto_tables": result.tex["auto_tables"],
                      "manifest": str(result.manifest_path),
                      "timings_s": {k: round(v, 2) for k, v in result.timings_s.items()}}, indent=2))
    if not result.passed:
        print(result.audit.format(), file=sys.stderr)
    return 0 if result.passed else 1


def _cmd_pdf(args: argparse.Namespace) -> int:
    from .pipeline import build_pdf

    profile = _load_profile(args.profile)
    if profile is None:
        return 1
    try:
        pdf = build_pdf(profile, results_dir=args.results_dir, engine=args.engine)
    except (OSError, RuntimeError) as exc:
        print(f"PDF FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "pdf": str(pdf)}))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .audit import run_anti_hallucination_audit
    from .pipeline import ProfilePaths

    if args.profile:
        profile = _load_profile(args.profile)
        if profile is None:
            return 1
        paths = ProfilePaths(Path(args.results_dir), profile.profile_id)
        tex_dir, bib, metrics, cache = paths.thesis, paths.bib, paths.metrics, paths.lit_cache
    elif args.tex_dir and args.bib and args.metrics:
        tex_dir, bib, metrics, cache = args.tex_dir, args.bib, args.metrics, args.lit_cache
    else:
        print("audit requires --profile, or --tex-dir, --bib and --metrics", file=sys.stderr)
        return 2
    report = run_anti_hallucination_audit(tex_dir, bib, metrics, cache, strict_integers=args.strict_integers)
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

    p_run = sub.add_parser("run", help="Literature -> experiments -> statistics -> figures -> LaTeX -> audit")
    p_run.add_argument("--profile", required=True)
    p_run.add_argument("--results-dir", default="results")
    p_run.add_argument("--template-dir", default="templates/thesis_tex")
    p_run.add_argument("--plugins", default=None, help="plugin directory (default: ./plugins if present)")
    p_run.add_argument("--offline", action="store_true", help="skip OpenAlex/Crossref; use signed caches only")
    p_run.add_argument("--no-mirror", action="store_true",
                       help="do not mirror generated .tex files into <template-dir>/generated")
    p_run.set_defaults(func=_cmd_run)

    p_pdf = sub.add_parser("pdf", help="Compile results/<profile_id>/thesis.pdf with pdflatex/xelatex + bibtex")
    p_pdf.add_argument("--profile", required=True)
    p_pdf.add_argument("--results-dir", default="results")
    p_pdf.add_argument("--engine", default=None, choices=["pdflatex", "xelatex", "lualatex"])
    p_pdf.set_defaults(func=_cmd_pdf)

    p_audit = sub.add_parser("audit", help="Run the anti-hallucination audit (raw numbers, citations, provenance)")
    p_audit.add_argument("--profile", default=None, help="audit results/<profile_id>/thesis")
    p_audit.add_argument("--results-dir", default="results")
    p_audit.add_argument("--tex-dir", default=None)
    p_audit.add_argument("--bib", default=None)
    p_audit.add_argument("--metrics", default=None)
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
