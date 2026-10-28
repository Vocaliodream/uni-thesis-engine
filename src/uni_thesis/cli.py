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
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
