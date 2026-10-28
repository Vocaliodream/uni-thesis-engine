"""Verified literature engine.

Searches real bibliographic indexes (OpenAlex, Crossref), cross-checks every
DOI against Crossref metadata (title, first author, year) and only keeps
records that pass.  Verified results are cached with a SHA-256 signature so
offline / CI runs are deterministic.  Exports BibTeX, a CSV literature matrix
and a Markdown comparison table.

No record is ever synthesized locally: every field comes from an API response
or from a signed cache that was itself built from API responses.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import requests

from .schema import LiteratureQuery, LiteratureRecord

__all__ = [
    "OPENALEX_WORKS_URL",
    "CROSSREF_WORKS_URL",
    "LiteratureFetchError",
    "CacheSignatureError",
    "normalize_doi",
    "is_valid_doi_format",
    "fetch_crossref_metadata",
    "verify_doi_exists",
    "search_openalex",
    "search_crossref",
    "fetch_and_verify_literature",
    "save_signed_cache",
    "load_signed_cache",
    "load_baseline_cache",
    "export_bibtex",
    "export_literature_matrix",
    "make_cite_key",
]

logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CROSSREF_WORKS_URL = "https://api.crossref.org/works/"
_USER_AGENT = "uni-thesis-engine/0.1 (https://github.com/Vocaliodream/uni-thesis-engine)"
_DOI_STRICT_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:<>A-Za-z0-9\[\]]+$")
_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:")
_TITLE_SIMILARITY_MIN = 0.90
_YEAR_TOLERANCE = 1
_BASELINE_RESOURCE = "baseline_literature.json"
_TYPE_PRIORITY = {"journal-article": 0, "article": 0, "proceedings-article": 1, "book-chapter": 2}
_NETWORK_ERRORS = (requests.ConnectionError, requests.Timeout)

SessionLike = Any


class LiteratureFetchError(RuntimeError):
    """Raised when literature cannot be obtained from any source."""


class CacheSignatureError(ValueError):
    """Raised when a literature cache fails its SHA-256 signature check."""


# --------------------------------------------------------------------------- #
# DOI helpers
# --------------------------------------------------------------------------- #
def normalize_doi(doi: str) -> str:
    """Strip resolver prefixes / whitespace and lowercase a DOI."""
    if not isinstance(doi, str):
        raise TypeError("doi must be a string")
    d = doi.strip()
    low = d.lower()
    for prefix in _DOI_PREFIXES:
        if low.startswith(prefix):
            d = d[len(prefix):]
            break
    return d.strip().lower()


def is_valid_doi_format(doi: str) -> bool:
    """Syntactic DOI check (directory indicator 10., registrant code, suffix)."""
    if not isinstance(doi, str) or not doi.strip():
        return False
    return bool(_DOI_STRICT_RE.match(normalize_doi(doi)))


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def _title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def _family_name(full_name: str) -> str:
    name = full_name.strip()
    if "," in name:
        return _normalize_text(name.split(",", 1)[0])
    parts = _normalize_text(name).split()
    return parts[-1] if parts else ""


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _session(session: Optional[SessionLike]) -> SessionLike:
    if session is not None:
        return session
    s = requests.Session()
    mailto = os.environ.get("UNI_THESIS_MAILTO")
    ua = f"{_USER_AGENT[:-1]}; mailto:{mailto})" if mailto else _USER_AGENT
    s.headers.update({"User-Agent": ua, "Accept": "application/json"})
    return s


def _get_json(session: SessionLike, url: str, params: Optional[Mapping[str, Any]] = None,
              timeout: float = 20.0) -> Tuple[int, Optional[Dict[str, Any]]]:
    resp = session.get(url, params=dict(params) if params else None, timeout=timeout)
    status = int(resp.status_code)
    if status != 200:
        return status, None
    try:
        return status, resp.json()
    except ValueError:
        return status, None


# --------------------------------------------------------------------------- #
# Metadata parsing
# --------------------------------------------------------------------------- #
def _crossref_year(msg: Mapping[str, Any]) -> Optional[int]:
    for key in ("published-print", "published-online", "issued", "published", "created"):
        parts = (msg.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0] is not None:
            return int(parts[0][0])
    return None


def _crossref_authors(msg: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for a in msg.get("author") or []:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if family:
            out.append(f"{family}, {given}" if given else family)
        elif a.get("name"):
            out.append(str(a["name"]).strip())
    return out


def _first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value or "").strip()


def _parse_crossref_item(msg: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "doi": normalize_doi(msg.get("DOI", "")) if msg.get("DOI") else None,
        "title": re.sub(r"\s+", " ", _first(msg.get("title"))),
        "authors": _crossref_authors(msg),
        "year": _crossref_year(msg),
        "venue": _first(msg.get("container-title")) or _first(msg.get("publisher")),
        "type": str(msg.get("type") or ""),
        "abstract": re.sub(r"<[^>]+>", " ", str(msg.get("abstract") or "")),
        "subjects": [str(s) for s in (msg.get("subject") or [])],
        "source": "crossref",
    }


def _openalex_abstract(inv: Optional[Mapping[str, Sequence[int]]]) -> str:
    if not inv:
        return ""
    positions: Dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[int(i)] = word
    return " ".join(positions[i] for i in sorted(positions))


def _parse_openalex_item(work: Mapping[str, Any]) -> Dict[str, Any]:
    doi = work.get("doi")
    authors = [((a.get("author") or {}).get("display_name") or "").strip()
               for a in work.get("authorships") or []]
    source = ((work.get("primary_location") or {}).get("source") or {})
    topic = (work.get("primary_topic") or {}).get("display_name")
    concepts = [c.get("display_name") for c in (work.get("concepts") or []) if c.get("display_name")]
    return {
        "doi": normalize_doi(doi) if doi else None,
        "title": re.sub(r"\s+", " ", str(work.get("title") or work.get("display_name") or "").strip()),
        "authors": [a for a in authors if a],
        "year": int(work["publication_year"]) if work.get("publication_year") else None,
        "venue": str(source.get("display_name") or ""),
        "type": str(work.get("type_crossref") or work.get("type") or ""),
        "abstract": _openalex_abstract(work.get("abstract_inverted_index")),
        "subjects": ([topic] if topic else []) + concepts[:3],
        "source": "openalex",
    }


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def fetch_crossref_metadata(doi: str, session: Optional[SessionLike] = None) -> Optional[Dict[str, Any]]:
    """Return parsed Crossref metadata for a DOI, or None if it does not resolve."""
    if not is_valid_doi_format(doi):
        return None
    s = _session(session)
    status, data = _get_json(s, CROSSREF_WORKS_URL + quote(normalize_doi(doi), safe="/"))
    if status != 200 or not data or "message" not in data:
        return None
    return _parse_crossref_item(data["message"])


def _metadata_matches(meta: Mapping[str, Any], title: Optional[str], authors: Optional[Sequence[str]],
                      year: Optional[int]) -> bool:
    if title is not None and _title_similarity(title, meta.get("title") or "") < _TITLE_SIMILARITY_MIN:
        return False
    if year is not None:
        meta_year = meta.get("year")
        if meta_year is None or abs(int(meta_year) - int(year)) > _YEAR_TOLERANCE:
            return False
    if authors:
        meta_families = {_family_name(a) for a in meta.get("authors") or []}
        if _family_name(authors[0]) not in meta_families:
            return False
    return True


def verify_doi_exists(doi: str, expected_title: Optional[str] = None,
                      expected_authors: Optional[Sequence[str]] = None,
                      expected_year: Optional[int] = None,
                      session: Optional[SessionLike] = None) -> bool:
    """True iff the DOI is well formed, resolves on Crossref, and (when given)
    its title (>=90% similar), first-author family name and year (+-1) match."""
    if not is_valid_doi_format(doi):
        return False
    meta = fetch_crossref_metadata(doi, session=session)
    if meta is None:
        return False
    return _metadata_matches(meta, expected_title, expected_authors, expected_year)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def search_openalex(query: LiteratureQuery, per_page: int, session: Optional[SessionLike] = None) -> List[Dict[str, Any]]:
    s = _session(session)
    params: Dict[str, Any] = {
        "search": " ".join(query.keywords),
        "filter": f"from_publication_date:{query.start_year}-01-01,has_doi:true",
        "per-page": max(1, min(per_page, 200)),
        "sort": "relevance_score:desc",
    }
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    status, data = _get_json(s, OPENALEX_WORKS_URL, params)
    if status != 200 or not data:
        logger.warning("OpenAlex search unavailable (HTTP %s)", status)
        return []
    return [_parse_openalex_item(w) for w in data.get("results") or []]


def search_crossref(query: LiteratureQuery, rows: int, session: Optional[SessionLike] = None) -> List[Dict[str, Any]]:
    s = _session(session)
    params = {
        "query.bibliographic": " ".join(query.keywords),
        "filter": f"from-pub-date:{query.start_year}-01-01",
        "rows": max(1, min(rows, 1000)),
        "select": "DOI,title,author,issued,published-print,published-online,container-title,type,abstract,subject,publisher",
    }
    status, data = _get_json(s, CROSSREF_WORKS_URL.rstrip("/"), params)
    if status != 200 or not data:
        logger.warning("Crossref search unavailable (HTTP %s)", status)
        return []
    return [_parse_crossref_item(m) for m in (data.get("message") or {}).get("items") or []]


# --------------------------------------------------------------------------- #
# Record construction
# --------------------------------------------------------------------------- #
def make_cite_key(authors: Sequence[str], year: int, title: str, taken: Optional[set] = None) -> str:
    family = _family_name(authors[0]).replace(" ", "") if authors else "anon"
    family = family or "anon"
    words = [w for w in _normalize_text(title).split() if len(w) > 3]
    base = f"{family}{year}{words[0] if words else 'work'}"
    base = re.sub(r"[^a-z0-9]", "", base)
    if not base[0].isalpha():
        base = "ref" + base
    key, suffix = base, 0
    taken = taken if taken is not None else set()
    while key in taken:
        suffix += 1
        key = f"{base}{chr(ord('a') + suffix - 1)}" if suffix <= 26 else f"{base}{suffix}"
    taken.add(key)
    return key


def _detect_metrics(text: str, vocabulary: Iterable[str]) -> List[str]:
    norm = f" {_normalize_text(text)} "
    found = []
    for term in vocabulary:
        t = _normalize_text(term)
        if t and f" {t} " in norm and term not in found:
            found.append(term)
    return found


def _to_record(meta: Mapping[str, Any], cite_key: str, metric_vocabulary: Sequence[str]) -> LiteratureRecord:
    subjects = [s for s in meta.get("subjects") or [] if s]
    return LiteratureRecord(
        cite_key=cite_key,
        title=meta["title"],
        authors=list(meta["authors"]),
        year=int(meta["year"]),
        doi=meta["doi"],
        venue=meta["venue"] or "Unknown venue",
        methodology_tag=subjects[0] if subjects else f"unclassified ({meta.get('type') or 'unknown type'})",
        limitation_note="",
        core_metrics=_detect_metrics(f"{meta['title']} {meta.get('abstract', '')}", metric_vocabulary),
        verified=True,
    )


# --------------------------------------------------------------------------- #
# Signed cache
# --------------------------------------------------------------------------- #
def _records_signature(records_payload: List[Dict[str, Any]]) -> str:
    canon = json.dumps(records_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def save_signed_cache(records: Sequence[LiteratureRecord], path: Union[str, Path],
                      query: Optional[LiteratureQuery] = None) -> str:
    payload = [r.model_dump(mode="json") for r in records]
    sig = _records_signature(payload)
    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "query": query.model_dump(mode="json") if query else None,
        "records": payload,
        "sha256": sig,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return sig


def _parse_signed_doc(doc: Mapping[str, Any], origin: str) -> List[LiteratureRecord]:
    if "records" not in doc or "sha256" not in doc:
        raise CacheSignatureError(f"{origin}: missing 'records' or 'sha256'")
    if _records_signature(list(doc["records"])) != doc["sha256"]:
        raise CacheSignatureError(f"{origin}: SHA-256 signature mismatch")
    return [LiteratureRecord.model_validate(r) for r in doc["records"]]


def load_signed_cache(path: Union[str, Path]) -> List[LiteratureRecord]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return _parse_signed_doc(doc, str(path))


def load_baseline_cache() -> List[LiteratureRecord]:
    """Load the signed, built-in benchmark literature shipped with the package."""
    text = resources.files("uni_thesis.data").joinpath(_BASELINE_RESOURCE).read_text(encoding="utf-8")
    return _parse_signed_doc(json.loads(text), f"package:{_BASELINE_RESOURCE}")


def _filter_offline(records: Sequence[LiteratureRecord], query: LiteratureQuery) -> List[LiteratureRecord]:
    kws = [_normalize_text(k) for k in query.keywords]
    out = []
    for r in records:
        if r.year < query.start_year:
            continue
        hay = _normalize_text(" ".join([r.title, r.methodology_tag, r.venue, *r.core_metrics]))
        if any(k and k in hay for k in kws):
            out.append(r)
    return out[: query.max_papers]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def fetch_and_verify_literature(query_cfg: Union[LiteratureQuery, Mapping[str, Any]],
                                profile_id: str = "default",
                                results_dir: Union[str, Path] = "results",
                                session: Optional[SessionLike] = None,
                                offline: Optional[bool] = None,
                                metric_vocabulary: Sequence[str] = ()) -> List[LiteratureRecord]:
    """Search OpenAlex + Crossref, verify every DOI on Crossref, cache and return
    verified records.  Offline (``offline=True``, env ``UNI_THESIS_OFFLINE=1``,
    or a network failure) falls back to the profile cache, then the signed
    built-in baseline cache."""
    query = query_cfg if isinstance(query_cfg, LiteratureQuery) else LiteratureQuery.model_validate(dict(query_cfg))
    cache_path = Path(results_dir) / profile_id / "lit_cache.json"
    if offline is None:
        offline = os.environ.get("UNI_THESIS_OFFLINE", "") == "1"

    if not offline:
        try:
            return _fetch_online(query, cache_path, _session(session), metric_vocabulary)
        except _NETWORK_ERRORS as exc:
            logger.warning("Network unavailable (%s); falling back to offline caches", exc)

    if cache_path.exists():
        cached = load_signed_cache(cache_path)
        if cached:
            return cached[: query.max_papers]
    return _filter_offline(load_baseline_cache(), query)


def _fetch_online(query: LiteratureQuery, cache_path: Path, s: SessionLike,
                  metric_vocabulary: Sequence[str]) -> List[LiteratureRecord]:
    pool = query.max_papers * 2
    candidates = search_openalex(query, pool, s) + search_crossref(query, pool, s)
    merged: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        if not c.get("doi") or not c.get("title") or not c.get("authors") or not c.get("year"):
            continue
        if c["year"] < query.start_year:
            continue
        existing = merged.get(c["doi"])
        if existing is None:
            merged[c["doi"]] = dict(c)
        else:
            if not existing.get("abstract") and c.get("abstract"):
                existing["abstract"] = c["abstract"]
            existing["subjects"] = list(dict.fromkeys((existing.get("subjects") or []) + (c.get("subjects") or [])))

    records: List[LiteratureRecord] = []
    taken: set = set()
    seen_titles: set = set()
    ordered = sorted(merged.items(), key=lambda kv: _TYPE_PRIORITY.get(kv[1].get("type", ""), len(_TYPE_PRIORITY)))
    for doi, cand in ordered:
        if len(records) >= query.max_papers:
            break
        title_key = _normalize_text(cand["title"])
        if title_key in seen_titles:
            continue
        meta = fetch_crossref_metadata(doi, session=s)
        if meta is None or not _metadata_matches(meta, cand["title"], cand["authors"], cand["year"]):
            logger.info("Rejected unverifiable DOI %s", doi)
            continue
        verified_meta = dict(meta)
        verified_meta["abstract"] = meta.get("abstract") or cand.get("abstract", "")
        verified_meta["subjects"] = list(dict.fromkeys((cand.get("subjects") or []) + (meta.get("subjects") or [])))
        if not verified_meta.get("authors") or not verified_meta.get("year") or not verified_meta.get("title"):
            continue
        seen_titles.add(title_key)
        key = make_cite_key(verified_meta["authors"], verified_meta["year"], verified_meta["title"], taken)
        records.append(_to_record(verified_meta, key, metric_vocabulary))

    if not records:
        raise LiteratureFetchError("no verifiable literature found for query; refusing to fabricate records")
    save_signed_cache(records, cache_path, query)
    return records


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
_BIB_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _bib_escape(text: str) -> str:
    return "".join(_BIB_ESCAPES.get(ch, ch) for ch in text)


def _bib_author(name: str) -> str:
    return _bib_escape(name.strip())


def _is_proceedings(venue: str) -> bool:
    v = venue.lower()
    return any(w in v for w in ("proceedings", "conference", "symposium", "workshop", "congress"))


def export_bibtex(records: Sequence[LiteratureRecord], path: Union[str, Path]) -> str:
    """Write a BibTeX file; only verified records are exported."""
    entries = []
    for r in records:
        if not r.verified or not r.doi:
            continue
        proc = _is_proceedings(r.venue)
        venue_field = "booktitle" if proc else "journal"
        fields = [
            ("title", "{" + _bib_escape(r.title) + "}"),
            ("author", " and ".join(_bib_author(a) for a in r.authors)),
            ("year", str(r.year)),
            (venue_field, _bib_escape(r.venue)),
            ("doi", r.doi),
            ("note", "DOI verified via Crossref"),
        ]
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
        entries.append(f"@{'inproceedings' if proc else 'article'}{{{r.cite_key},\n{body}\n}}")
    text = "\n\n".join(entries) + ("\n" if entries else "")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return text


_MATRIX_COLUMNS = ["CiteKey", "Year", "Methodology", "CoreMetrics", "Limitations"]
_UNANNOTATED = "未标注（需人工阅读后补充）"


def _md_cell(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", " ")


def export_literature_matrix(records: Sequence[LiteratureRecord], csv_path: Union[str, Path],
                             md_path: Union[str, Path]) -> List[Dict[str, str]]:
    rows = []
    for r in sorted(records, key=lambda x: (x.year, x.cite_key)):
        rows.append({
            "CiteKey": r.cite_key,
            "Year": str(r.year),
            "Methodology": r.methodology_tag,
            "CoreMetrics": "; ".join(r.core_metrics) if r.core_metrics else _UNANNOTATED,
            "Limitations": r.limitation_note or _UNANNOTATED,
        })
    cp, mp = Path(csv_path), Path(md_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MATRIX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(_MATRIX_COLUMNS) + " |", "|" + "---|" * len(_MATRIX_COLUMNS)]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(row[c]) for c in _MATRIX_COLUMNS) + " |")
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
