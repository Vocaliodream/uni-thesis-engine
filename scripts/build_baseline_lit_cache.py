"""Rebuild the signed baseline literature cache from live Crossref metadata.

Every bibliographic field comes from Crossref; only `methodology_tag` is a
curated label (Crossref carries no method taxonomy).  Run with network access:

    python scripts/build_baseline_lit_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from uni_thesis.lit_engine import fetch_crossref_metadata, make_cite_key, save_signed_cache
from uni_thesis.schema import LiteratureRecord

CURATED = [
    ("10.1109/18.382009", "wavelet soft-thresholding denoising"),
    ("10.1021/ac60214a047", "Savitzky-Golay least-squares smoothing filter"),
    ("10.1109/tbme.1985.325532", "ECG QRS detection with band-pass filtering"),
    ("10.1109/tbme.2003.808805", "synthetic ECG dynamical signal model"),
    ("10.1111/j.2517-6161.1951.tb00067.x", "response surface methodology central composite design"),
    ("10.1038/s41592-019-0686-2", "scientific computing signal processing library"),
]


def main() -> int:
    out = Path(__file__).resolve().parents[1] / "src" / "uni_thesis" / "data" / "baseline_literature.json"
    taken: set = set()
    records = []
    for doi, tag in CURATED:
        meta = fetch_crossref_metadata(doi)
        if meta is None:
            print(f"FAILED to resolve {doi}", file=sys.stderr)
            return 1
        key = make_cite_key(meta["authors"], meta["year"], meta["title"], taken)
        records.append(LiteratureRecord(
            cite_key=key, title=meta["title"], authors=meta["authors"], year=meta["year"],
            doi=meta["doi"], venue=meta["venue"], methodology_tag=tag, verified=True,
        ))
    sig = save_signed_cache(records, out)
    print(f"wrote {len(records)} records to {out} (sha256={sig})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
