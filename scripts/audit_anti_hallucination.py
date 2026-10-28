#!/usr/bin/env python3
"""Anti-hallucination audit for the thesis LaTeX sources.

Three modes (exit status 0 when every audited thesis passes all three gates, 1 otherwise):

1. Hermetic (no arguments, used by the pre-commit hook): every ``profiles/*.yaml`` is
   regenerated offline from its signed caches into a temporary directory with the
   *current* templates and code, then audited::

       python scripts/audit_anti_hallucination.py [--profiles-dir profiles] [--template-dir ...]

2. Existing results (``make audit``): audit every ``<results-dir>/<profile_id>/thesis``
   produced by an earlier ``uni-thesis run``::

       python scripts/audit_anti_hallucination.py --results-dir results

3. Explicit single thesis::

       python scripts/audit_anti_hallucination.py --tex-dir templates/thesis_tex \
           --bib templates/thesis_tex/refs.bib --metrics results/<profile_id>/metrics.json \
           [--lit-cache results/<profile_id>/lit_cache.json] [--json]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uni_thesis.audit import AuditReport, run_anti_hallucination_audit  # noqa: E402


def _audit_results_dir(results_dir: Path, strict: bool) -> List[Tuple[str, AuditReport]]:
    out = []
    for root in sorted(p for p in results_dir.iterdir() if p.is_dir()) if results_dir.is_dir() else []:
        metrics, thesis = root / "metrics.json", root / "thesis"
        if not (metrics.exists() and (thesis / "main.tex").exists()):
            continue
        lit = root / "lit_cache.json"
        out.append((str(root), run_anti_hallucination_audit(thesis, thesis / "refs.bib", metrics,
                                                            lit if lit.exists() else None,
                                                            strict_integers=strict)))
    return out


def _audit_hermetic(profiles_dir: Path, template_dir: Path, strict: bool) -> List[Tuple[str, AuditReport]]:
    from uni_thesis.pipeline import ProfilePaths, run_profile

    out = []
    with tempfile.TemporaryDirectory(prefix="uni-thesis-audit-") as tmp:
        for profile in sorted(profiles_dir.glob("*.yaml")):
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_profile(profile, results_dir=tmp, template_dir=template_dir, offline=True,
                                  repo_root=REPO, mirror_generated=False,
                                  plugin_dir=REPO / "plugins" if (REPO / "plugins").is_dir() else None)
            p = ProfilePaths(Path(tmp), res.profile_id)
            out.append((f"{profile.name} (fresh offline run, {template_dir.name})",
                        run_anti_hallucination_audit(p.thesis, p.bib, p.metrics, p.lit_cache,
                                                     strict_integers=strict)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tex-dir")
    ap.add_argument("--bib")
    ap.add_argument("--metrics")
    ap.add_argument("--lit-cache", default=None)
    ap.add_argument("--results-dir", default=None, help="audit every <results-dir>/<profile_id>/thesis")
    ap.add_argument("--profiles-dir", default=str(REPO / "profiles"))
    ap.add_argument("--template-dir", action="append", default=None,
                    help="template(s) for the hermetic mode (default: every templates/thesis_tex*)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--strict-integers", action="store_true",
                    help="also flag integers other than 0, 1, 2, 3, 100 and years")
    args = ap.parse_args(argv)

    explicit = [args.tex_dir, args.bib, args.metrics]
    if any(explicit):
        if not all(explicit):
            ap.error("--tex-dir, --bib and --metrics must be given together")
        reports = [(args.tex_dir, run_anti_hallucination_audit(args.tex_dir, args.bib, args.metrics, args.lit_cache,
                                                               strict_integers=args.strict_integers))]
    elif args.results_dir:
        reports = _audit_results_dir(Path(args.results_dir), args.strict_integers)
        if not reports:
            print(f"no generated thesis found under {args.results_dir}; run `uni-thesis run --profile ...` first",
                  file=sys.stderr)
            return 1
    else:
        templates = [Path(t) for t in args.template_dir] if args.template_dir else sorted(
            p for p in (REPO / "templates").glob("thesis_tex*") if (p / "main.tex").exists())
        reports = [r for t in templates for r in _audit_hermetic(Path(args.profiles_dir), t, args.strict_integers)]
        if not reports:
            print(f"no profile found under {args.profiles_dir}", file=sys.stderr)
            return 1

    failed = [label for label, r in reports if not r.passed]
    if args.json:
        doc = reports[0][1].to_dict() if any(explicit) else {
            "passed": not failed, "reports": [{"target": label, **r.to_dict()} for label, r in reports]}
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        for label, report in reports:
            print(f"== {label}")
            print(report.format())
        print(f"\n{len(reports) - len(failed)}/{len(reports)} thesis audit(s) passed"
              + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
