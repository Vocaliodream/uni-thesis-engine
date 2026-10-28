#!/usr/bin/env python3
"""Anti-hallucination audit for the thesis LaTeX sources.

Usage:
    python scripts/audit_anti_hallucination.py --tex-dir templates/thesis_tex \
        --bib templates/thesis_tex/refs.bib --metrics results/<profile_id>/metrics.json \
        [--lit-cache results/<profile_id>/lit_cache.json] [--json]

Exit status 0 when all three gates pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uni_thesis.audit import run_anti_hallucination_audit  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tex-dir", required=True)
    ap.add_argument("--bib", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--lit-cache", default=None)
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--strict-integers", action="store_true",
                    help="also flag integers other than 0, 1, 2, 3, 100 and years")
    args = ap.parse_args(argv)
    report = run_anti_hallucination_audit(args.tex_dir, args.bib, args.metrics, args.lit_cache,
                                          strict_integers=args.strict_integers)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) if args.json else report.format())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
