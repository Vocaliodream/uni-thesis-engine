"""Anti-hallucination static auditor for the thesis LaTeX sources.

Three gates:

1. **Raw Number Gate** – results chapters may not contain hand-written numbers
   (floats, percentages, p-values, scientific notation; integers only in
   ``strict_integers`` mode); every figure must come
   from an ``\\Metric...`` / ``\\Stat...`` / ``\\Diff...`` macro.  Figure, table,
   section and equation references, labels, lengths and sub/superscript
   indices are ignored.
2. **Citation Integrity Gate** – every ``\\cite`` key must exist in ``refs.bib``
   and in the signed ``lit_cache.json`` as a ``verified`` record carrying a
   valid DOI (matching the BibTeX DOI when present).
3. **Provenance Gate** – ``auto_metrics.tex`` / ``auto_tables.tex`` must carry
   the SHA-256 of the current ``metrics.json``; the bundle must pass its own
   integrity check; every result macro used in the chapters must be defined.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .lit_engine import CacheSignatureError, load_signed_cache
from .schema import ExperimentResultBundle, IntegrityError
from .tex_bridge import SHA_HEADER_KEY, file_sha256

__all__ = [
    "Violation",
    "AuditReport",
    "DEFAULT_RESULT_CHAPTERS",
    "DEFAULT_ALLOWED_INTEGERS",
    "scan_raw_numbers",
    "check_citations",
    "check_provenance",
    "run_anti_hallucination_audit",
]

PathLike = Union[str, Path]
DEFAULT_RESULT_CHAPTERS = ("chapters/chap3_method.tex", "chapters/chap4_experiments.tex",
                           "chapters/chap5_conclusion.tex")
DEFAULT_ALLOWED_INTEGERS = frozenset({0, 1, 2, 3, 100})
RESULT_MACRO_PREFIXES = ("Metric", "Stat", "Diff")


@dataclass(frozen=True)
class Violation:
    gate: str
    file: str
    line: int
    message: str
    snippet: str = ""

    def __str__(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        return f"[{self.gate}] {loc}: {self.message}" + (f" | {self.snippet}" if self.snippet else "")


@dataclass
class AuditReport:
    violations: List[Violation] = field(default_factory=list)
    checked_files: List[str] = field(default_factory=list)
    citations_checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.violations

    def by_gate(self, gate: str) -> List[Violation]:
        return [v for v in self.violations if v.gate == gate]

    def to_dict(self) -> Dict[str, object]:
        return {"passed": self.passed, "checked_files": self.checked_files,
                "citations_checked": self.citations_checked, "violations": [asdict(v) for v in self.violations]}

    def format(self) -> str:
        head = "AUDIT PASSED" if self.passed else f"AUDIT FAILED: {len(self.violations)} violation(s)"
        return "\n".join([head, *[str(v) for v in self.violations]])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_COMMENT = re.compile(r"(?<!\\)%.*$")
_STRIP_CMDS = re.compile(
    r"\\(?:label|ref|eqref|autoref|cref|Cref|pageref|cite[a-zA-Z]*|input|include|includegraphics|"
    r"bibliography|bibliographystyle|usepackage|documentclass|begin|end|url|href|hspace|vspace|setlength|"
    r"addtolength|newcommand|renewcommand|providecommand)(?![A-Za-z])\*?(?:\[[^\]]*\])*(?:\{[^{}]*\})?")
_REF_WORDS = re.compile(
    r"(?:(?<![A-Za-z])(?:Fig(?:ure)?s?\.?|Tab(?:le)?s?\.?|Sec(?:tion)?s?\.?|Chap(?:ter)?s?\.?|Eqs?\.?|Equations?|Algorithms?|"
    r"Appendix|Step|Stage)|图|表|式|算法|章|节|附录|第)[\s~]*[(（]?\s*\d+(?:[.\-–]\d+)*\s*[)）]?", re.IGNORECASE)
_LENGTHS = re.compile(r"-?\d*\.?\d+\s*\\(?:textwidth|linewidth|columnwidth|textheight|hsize|baselineskip)"
                      r"|(?<=[\[{=,])\s*-?\d*\.?\d+(?:pt|cm|mm|em|ex|bp)\b")
_INDEX = re.compile(r"[_^]\s*(?:\{[^{}]*\}|\d)")
_PVALUE = re.compile(r"\bp\s*(?:\\(?:leq?|geq?|lt|gt)|[<>=≤≥])\s*\$?\s*\d*\.?\d+", re.IGNORECASE)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*(?:\\%|％|%)")
_NUMBER = re.compile(r"(?<![A-Za-z\d.])(\d+\.\d+|\.\d+|\d+(?:[eE][-+]?\d+)?)(?!\d)")
_YEAR = re.compile(r"^(?:19|20)\d{2}$")


def _strip_comment(line: str) -> str:
    return _COMMENT.sub("", line)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Gate 1
# --------------------------------------------------------------------------- #
def scan_raw_numbers(tex_path: PathLike, root: Optional[PathLike] = None,
                     allowed_integers: Iterable[int] = DEFAULT_ALLOWED_INTEGERS,
                     allow_years: bool = True, strict_integers: bool = False) -> List[Violation]:
    """Flag hand-written result numbers in one chapter.

    Floats, percentages (other than allowed integers), p-value expressions and
    scientific notation are always flagged.  Plain integers (counts such as
    subjects, folds or repeats) pass unless ``strict_integers`` is set, in which
    case only ``allowed_integers`` and, with ``allow_years``, 4-digit years pass.
    """
    path = Path(tex_path)
    rel = _rel(path, Path(root)) if root else str(path)
    allowed = set(allowed_integers)
    out: List[Violation] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        flagged: List[str] = []
        for m in _PVALUE.finditer(line):
            flagged.append(f"hand-written p-value '{m.group().strip()}'")
        for m in _PERCENT.finditer(line):
            num = re.match(r"\d+(?:\.\d+)?", m.group()).group()
            if num.isdigit() and int(num) in allowed:
                continue
            flagged.append(f"hand-written percentage '{m.group().strip()}'")
        clean = _PVALUE.sub(" ", _PERCENT.sub(" ", line))
        clean = _STRIP_CMDS.sub(" ", clean)
        clean = _REF_WORDS.sub(" ", clean)
        clean = _LENGTHS.sub(" ", clean)
        clean = _INDEX.sub(" ", clean)
        for m in _NUMBER.finditer(clean):
            tok = m.group(1)
            if re.fullmatch(r"\d+", tok):
                n = int(tok)
                if not strict_integers or n in allowed or (allow_years and _YEAR.match(tok)):
                    continue
                flagged.append(f"hand-written integer '{tok}'")
            else:
                flagged.append(f"hand-written number '{tok}'")
        for msg in flagged:
            out.append(Violation("raw_number", rel, lineno, msg + " (use an auto-generated \\Metric/\\Stat macro)",
                                 raw.strip()[:160]))
    return out


# --------------------------------------------------------------------------- #
# Gate 2
# --------------------------------------------------------------------------- #
_CITE = re.compile(r"\\(?:no)?cite[a-zA-Z]*\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}")
_BIB_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n@|\Z)", re.DOTALL)
_BIB_DOI = re.compile(r"\bdoi\s*=\s*[{\"]\s*([^}\"]+?)\s*[}\"]", re.IGNORECASE)
_DOI_OK = re.compile(r"^10\.\d{4,9}/\S+$")


def _parse_bib(bib_path: Path) -> Dict[str, Optional[str]]:
    text = bib_path.read_text(encoding="utf-8")
    entries: Dict[str, Optional[str]] = {}
    for m in _BIB_ENTRY.finditer(text):
        if m.group(1).lower() in {"comment", "string", "preamble"}:
            continue
        doi = _BIB_DOI.search(m.group(3))
        entries[m.group(2)] = doi.group(1).strip() if doi else None
    return entries


def _collect_cites(tex_files: Sequence[Path], root: Path) -> List[Tuple[str, str, int]]:
    cites: List[Tuple[str, str, int]] = []
    for f in tex_files:
        for lineno, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            for m in _CITE.finditer(_strip_comment(raw)):
                for key in m.group(1).split(","):
                    key = key.strip()
                    if key and key != "*":
                        cites.append((key, _rel(f, root), lineno))
    return cites


def check_citations(tex_files: Sequence[Path], root: Path, bib_path: Path,
                    lit_cache_path: Path) -> Tuple[List[Violation], int]:
    out: List[Violation] = []
    cites = _collect_cites(tex_files, root)
    try:
        bib = _parse_bib(bib_path)
    except OSError as exc:
        return [Violation("citation", str(bib_path), 0, f"cannot read bibliography: {exc}")], len(cites)
    try:
        records = {r.cite_key: r for r in load_signed_cache(lit_cache_path)}
    except (OSError, CacheSignatureError, ValueError, json.JSONDecodeError) as exc:
        records = {}
        out.append(Violation("citation", str(lit_cache_path), 0, f"literature cache unusable: {exc}"))
    for key, f, line in cites:
        if key not in bib:
            out.append(Violation("citation", f, line, f"\\cite key '{key}' not found in {bib_path.name}"))
            continue
        rec = records.get(key)
        if rec is None:
            out.append(Violation("citation", f, line, f"'{key}' has no record in the signed lit_cache.json"))
            continue
        if not rec.verified:
            out.append(Violation("citation", f, line, f"'{key}' is not DOI-verified (verified == False)"))
        if not rec.doi or not _DOI_OK.match(rec.doi):
            out.append(Violation("citation", f, line, f"'{key}' lacks a valid DOI in lit_cache.json"))
        elif bib[key] and bib[key].lower() != rec.doi.lower():
            out.append(Violation("citation", f, line,
                                 f"'{key}' DOI mismatch: refs.bib '{bib[key]}' vs lit_cache '{rec.doi}'"))
    return out, len(cites)


# --------------------------------------------------------------------------- #
# Gate 3
# --------------------------------------------------------------------------- #
_SHA_LINE = re.compile(rf"^%\s*{SHA_HEADER_KEY}:\s*([0-9a-f]{{64}})\s*$", re.MULTILINE)
_DEFINED = re.compile(r"\\(?:providecommand|newcommand|renewcommand)\*?\s*\{?\\([A-Za-z]+)\}?")
_USED_MACRO = re.compile(r"\\((?:%s)[A-Za-z]+)" % "|".join(RESULT_MACRO_PREFIXES))


def check_provenance(tex_dir: Path, metrics_json_path: Path, tex_files: Sequence[Path]) -> List[Violation]:
    out: List[Violation] = []
    try:
        ExperimentResultBundle.load(metrics_json_path)
    except (IntegrityError, ValueError, OSError) as exc:
        out.append(Violation("provenance", str(metrics_json_path), 0, f"metrics.json failed integrity check: {exc}"))
    try:
        actual = file_sha256(metrics_json_path)
    except OSError as exc:
        return out + [Violation("provenance", str(metrics_json_path), 0, f"cannot read metrics.json: {exc}")]

    gen = tex_dir / "generated"
    defined: Set[str] = set()
    for name, required in (("auto_metrics.tex", True), ("auto_tables.tex", False)):
        p = gen / name
        if not p.exists():
            if required:
                out.append(Violation("provenance", _rel(p, tex_dir), 0, "auto-generated macro file is missing"))
            continue
        text = p.read_text(encoding="utf-8")
        m = _SHA_LINE.search(text)
        if not m:
            out.append(Violation("provenance", _rel(p, tex_dir), 1, f"header lacks '% {SHA_HEADER_KEY}: <hex>'"))
        elif m.group(1) != actual:
            line = text[:m.start()].count("\n") + 1
            out.append(Violation("provenance", _rel(p, tex_dir), line,
                                 f"stale macros: header {SHA_HEADER_KEY} {m.group(1)[:12]}... != "
                                 f"current metrics.json {actual[:12]}...; regenerate with tex_bridge"))
        if name == "auto_metrics.tex":
            defined = set(_DEFINED.findall(text))

    for f in tex_files:
        if gen in f.parents:
            continue
        for lineno, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            for name in _USED_MACRO.findall(_strip_comment(raw)):
                if name not in defined:
                    out.append(Violation("provenance", _rel(f, tex_dir), lineno,
                                         f"result macro \\{name} is not defined in auto_metrics.tex"))
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_anti_hallucination_audit(tex_dir: PathLike, bib_path: PathLike, metrics_json_path: PathLike,
                                 lit_cache_path: Optional[PathLike] = None,
                                 result_chapters: Sequence[str] = DEFAULT_RESULT_CHAPTERS,
                                 allowed_integers: Iterable[int] = DEFAULT_ALLOWED_INTEGERS,
                                 strict_integers: bool = False) -> AuditReport:
    """Run all three gates.  ``lit_cache_path`` defaults to ``lit_cache.json`` next to ``metrics.json``."""
    root = Path(tex_dir)
    mjson = Path(metrics_json_path)
    cache = Path(lit_cache_path) if lit_cache_path else mjson.parent / "lit_cache.json"
    report = AuditReport()
    tex_files = sorted(p for p in root.rglob("*.tex") if p.is_file())
    report.checked_files = [_rel(p, root) for p in tex_files]

    for chap in result_chapters:
        p = root / chap
        if p.exists():
            report.violations += scan_raw_numbers(p, root, allowed_integers, strict_integers=strict_integers)
        else:
            report.violations.append(Violation("raw_number", chap, 0, "results chapter not found"))

    cite_violations, n = check_citations(tex_files, root, Path(bib_path), cache)
    report.violations += cite_violations
    report.citations_checked = n
    report.violations += check_provenance(root, mjson, tex_files)
    return report
