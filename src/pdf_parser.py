"""Layout-aware PDF parsing with PyMuPDF (fitz).

Pipeline per page:

1. **Tables** - ``page.find_tables()`` detects ruled tables; each becomes one Markdown
   element (cells cleaned, header row first) so that row/column structure survives.
2. **Lines** - ``page.get_text("dict")`` gives lines with font metadata; lines that fall
   inside a table bbox are dropped (no duplicated text).
3. **Header / footer stripping** - lines inside the top/bottom margin bands that repeat
   on most pages (digits masked, so "Page 3 of 7" == "Page 4 of 7") or look like page
   numbers are removed.
4. **Reading order** - lines are classified as left / right / full-width relative to
   the page centre. Consecutive non-full-width lines that include a right-column line
   form a two-column band and are emitted column by column (left first), which fixes
   MuPDF merging side-by-side columns into a single block.
5. **Paragraphs & headings** - consecutive lines are merged into paragraphs; bold short
   lines become headings. Every element keeps its 1-based page number for citations.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

LOG = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(ARTICLE\s+\d+|SCHEDULE\s+\d+|\d+(\.\d+)+\s+[A-Z]|TABLE OF CONTENTS)")
_PAGE_NUM_RE = re.compile(r"^(page\s*#+(\s*of\s*#+)?|#+\s*/\s*#+|-\s*#+\s*-)$", re.IGNORECASE)


@dataclass
class ParserConfig:
    header_band: float = 0.07  # fraction of page height treated as header zone
    footer_band: float = 0.07
    repeat_min_frac: float = 0.5  # strip band lines repeated on >= this share of pages
    detect_tables: bool = True
    table_use_layout: bool = False  # PyMuPDF>=1.26 layout model: slower, not needed for ruled tables
    column_tolerance: float = 0.02  # fraction of page width around the centre line


@dataclass
class Element:
    kind: str  # "heading" | "text" | "table"
    text: str
    page: int  # 1-based
    bbox: tuple[float, float, float, float]
    font_size: float = 0.0
    bold: bool = False
    rows: list[list[str]] | None = None


@dataclass
class ParsedPage:
    number: int
    width: float
    height: float
    elements: list[Element] = field(default_factory=list)
    stripped: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(e.text for e in self.elements)

    @property
    def tables(self) -> list[Element]:
        return [e for e in self.elements if e.kind == "table"]


@dataclass
class ParsedDocument:
    doc_id: str
    source: str
    pages: list[ParsedPage]

    @property
    def elements(self) -> list[Element]:
        return [e for p in self.pages for e in p.elements]

    @property
    def n_tables(self) -> int:
        return sum(len(p.tables) for p in self.pages)

    @property
    def n_stripped(self) -> int:
        return sum(len(p.stripped) for p in self.pages)

    def page_texts(self) -> dict[int, str]:
        return {p.number: p.text for p in self.pages}

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


@dataclass
class _Line:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool
    block: int
    column: str = "full"


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", cell.replace("|", "/")).strip()


def table_to_markdown(rows: list[list[str]]) -> str:
    """Render rows as a GitHub-flavoured Markdown table (first row = header)."""
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    lines = ["|" + "|".join(padded[0]) + "|", "|" + "|".join(["---"] * width) + "|"]
    lines += ["|" + "|".join(r) + "|" for r in padded[1:]]
    return "\n".join(lines)


def _find_tables(page: pymupdf.Page, cfg: ParserConfig) -> list[tuple[pymupdf.Rect, list[list[str]]]]:
    if not cfg.detect_tables:
        return []
    try:
        finder = page.find_tables(use_layout=cfg.table_use_layout)
    except TypeError:  # older PyMuPDF without the layout switch
        finder = page.find_tables()
    out: list[tuple[pymupdf.Rect, list[list[str]]]] = []
    for tab in finder.tables:
        rows = [[_clean_cell(c) for c in row] for row in tab.extract()]
        rows = [r for r in rows if any(r)]
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
            out.append((pymupdf.Rect(tab.bbox), rows))
    return out


# --------------------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------------------
def _page_lines(page: pymupdf.Page, table_rects: list[pymupdf.Rect]) -> list[_Line]:
    data = page.get_text("dict", sort=True)
    lines: list[_Line] = []
    for b_idx, block in enumerate(data["blocks"]):
        if block.get("type", 0) != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            bbox = pymupdf.Rect(line["bbox"])
            centre = pymupdf.Point((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
            if any(r.contains(centre) for r in table_rects):
                continue
            parts: list[str] = []
            prev_x1: float | None = None
            for s in spans:
                gap = 0.0 if prev_x1 is None else s["bbox"][0] - prev_x1
                if parts and gap > 0.25 * s["size"] and not parts[-1].endswith(" "):
                    parts.append(" ")
                parts.append(s["text"])
                prev_x1 = s["bbox"][2]
            text = re.sub(r"\s+", " ", "".join(parts)).strip()
            bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
            size = max(s["size"] for s in spans)
            lines.append(_Line(text, tuple(bbox), size, bold, b_idx))
    return lines


def _band_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _strip_headers_footers(
    pages_lines: list[list[_Line]], heights: list[float], cfg: ParserConfig
) -> tuple[list[list[_Line]], list[list[str]]]:
    def in_band(line: _Line, h: float) -> bool:
        return line.bbox[3] <= h * cfg.header_band or line.bbox[1] >= h * (1 - cfg.footer_band)

    counts: Counter[str] = Counter()
    for lines, h in zip(pages_lines, heights):
        counts.update({_band_key(l.text) for l in lines if in_band(l, h)})
    n = len(pages_lines)
    min_repeats = max(2, int(round(cfg.repeat_min_frac * n)))
    kept_all: list[list[_Line]] = []
    stripped_all: list[list[str]] = []
    for lines, h in zip(pages_lines, heights):
        kept, stripped = [], []
        for l in lines:
            key = _band_key(l.text)
            if in_band(l, h) and (counts[key] >= min_repeats or _PAGE_NUM_RE.match(key)):
                stripped.append(l.text)
            else:
                kept.append(l)
        kept_all.append(kept)
        stripped_all.append(stripped)
    return kept_all, stripped_all


# --------------------------------------------------------------------------------------
# Reading order
# --------------------------------------------------------------------------------------
def order_lines(lines: list[_Line], page_width: float, tolerance: float = 0.02) -> list[_Line]:
    """Return lines in reading order, handling two-column bands (see module docstring)."""
    mid = page_width / 2
    tol = tolerance * page_width
    for l in lines:
        x0, _, x1, _ = l.bbox
        if x1 <= mid + tol:
            l.column = "left"
        elif x0 >= mid - tol:
            l.column = "right"
        else:
            l.column = "full"
    ordered: list[_Line] = []
    band: list[_Line] = []

    def flush() -> None:
        if any(l.column == "right" for l in band):
            # short single-column lines sitting above the start of the right column
            # (e.g. the last line of a paragraph) still belong to the normal flow
            top = min(l.bbox[1] for l in band if l.column == "right")
            lead = [l for l in band if l.bbox[3] <= top + 1]
            for l in lead:
                l.column = "full"
            ordered.extend(sorted(lead, key=lambda l: (l.bbox[1], l.bbox[0])))
            rest = [l for l in band if l not in lead]
            ordered.extend(sorted((l for l in rest if l.column == "left"), key=lambda l: (l.bbox[1], l.bbox[0])))
            ordered.extend(sorted((l for l in rest if l.column == "right"), key=lambda l: (l.bbox[1], l.bbox[0])))
        else:
            # no right column -> these are just short lines of the single-column flow
            for l in band:
                l.column = "full"
            ordered.extend(sorted(band, key=lambda l: (l.bbox[1], l.bbox[0])))
        band.clear()

    for l in sorted(lines, key=lambda l: (round(l.bbox[1], 1), l.bbox[0])):
        if l.column == "full":
            flush()
            ordered.append(l)
        else:
            band.append(l)
    flush()
    return ordered


def _is_heading(text: str, bold: bool) -> bool:
    if len(text) > 120:
        return False
    if _HEADING_RE.match(text) and bold:
        return True
    return bold and text.isupper() and len(text.split()) <= 12 and not text.endswith(".")


def _group_paragraphs(lines: list[_Line], page_no: int) -> list[Element]:
    elements: list[Element] = []
    current: list[_Line] = []

    def emit() -> None:
        if not current:
            return
        text = " ".join(l.text for l in current)
        x0 = min(l.bbox[0] for l in current)
        y0 = min(l.bbox[1] for l in current)
        x1 = max(l.bbox[2] for l in current)
        y1 = max(l.bbox[3] for l in current)
        bold = all(l.bold for l in current)
        kind = "heading" if _is_heading(text, bold) else "text"
        elements.append(Element(kind, text, page_no, (x0, y0, x1, y1), max(l.size for l in current), bold))
        current.clear()

    for line in lines:
        if current:
            prev = current[-1]
            gap = line.bbox[1] - prev.bbox[3]
            new_par = (
                line.block != prev.block
                or line.column != prev.column
                or line.bold != prev.bold
                or gap > 0.45 * line.size
                or line.bbox[1] < prev.bbox[1]  # jumped back up (next column)
            )
            if new_par:
                emit()
        current.append(line)
    emit()
    return elements


def _merge_broken_paragraphs(elements: list[Element]) -> list[Element]:
    """Glue text elements split mid-sentence (no terminal punctuation, next starts lower-case)."""
    merged: list[Element] = []
    for el in elements:
        if (
            merged
            and el.kind == "text"
            and merged[-1].kind == "text"
            and merged[-1].page == el.page
            and not re.search(r"[.:;!?]$", merged[-1].text)
            and el.text[:1].islower()
        ):
            prev = merged[-1]
            prev.text = f"{prev.text} {el.text}"
            prev.bbox = (
                min(prev.bbox[0], el.bbox[0]),
                min(prev.bbox[1], el.bbox[1]),
                max(prev.bbox[2], el.bbox[2]),
                max(prev.bbox[3], el.bbox[3]),
            )
        else:
            merged.append(el)
    return merged


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def parse_pdf(source: str | Path | bytes, doc_id: str | None = None, cfg: ParserConfig | None = None) -> ParsedDocument:
    """Parse a PDF (path or raw bytes) into ordered, page-tagged elements."""
    cfg = cfg or ParserConfig()
    if isinstance(source, (bytes, bytearray)):
        doc = pymupdf.open(stream=bytes(source), filetype="pdf")
        src_name = doc_id or "uploaded.pdf"
    else:
        doc = pymupdf.open(str(source))
        src_name = str(source)
    doc_id = doc_id or Path(src_name).stem
    try:
        page_tables: list[list[tuple[pymupdf.Rect, list[list[str]]]]] = []
        page_lines: list[list[_Line]] = []
        dims: list[tuple[float, float]] = []
        for page in doc:
            tables = _find_tables(page, cfg)
            page_tables.append(tables)
            page_lines.append(_page_lines(page, [r for r, _ in tables]))
            dims.append((page.rect.width, page.rect.height))
    finally:
        doc.close()

    kept, stripped = _strip_headers_footers(page_lines, [h for _, h in dims], cfg)
    pages: list[ParsedPage] = []
    for i, (lines, tables, (w, h), strip) in enumerate(zip(kept, page_tables, dims, stripped), start=1):
        ordered = order_lines(lines, w, cfg.column_tolerance)
        text_elements = _merge_broken_paragraphs(_group_paragraphs(ordered, i))
        table_elements = [
            Element("table", table_to_markdown(rows), i, tuple(rect), rows=rows) for rect, rows in tables
        ]
        # interleave tables by vertical position (tables are full-width blocks)
        elements: list[Element] = []
        pending = sorted(table_elements, key=lambda e: e.bbox[1])
        for el in text_elements:
            while pending and pending[0].bbox[1] <= el.bbox[1]:
                elements.append(pending.pop(0))
            elements.append(el)
        elements.extend(pending)
        pages.append(ParsedPage(number=i, width=w, height=h, elements=elements, stripped=strip))
    parsed = ParsedDocument(doc_id=doc_id, source=src_name, pages=pages)
    LOG.debug("parsed %s: %d pages, %d tables, %d header/footer lines stripped",
              doc_id, len(pages), parsed.n_tables, parsed.n_stripped)
    return parsed
