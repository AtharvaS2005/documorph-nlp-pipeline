"""
Digital-PDF table extraction via pdfplumber.

Reads table structure directly from a digital PDF's real vector line-drawing
instructions and text positions -- no rasterization or OCR involved, since
the page already has genuine embedded text. Tries a lines-based strategy
first (works when the PDF actually draws table borders), then falls back to
a text-alignment-based strategy for tables with no visible gridlines, so
this generalizes to arbitrary digital PDFs rather than assuming one table
style.
"""

from typing import Any, Dict, List, Optional
from collections import Counter


def _tables_to_dicts(tables: List[List[List[Optional[str]]]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for table in tables:
        rows = [[cell if cell is not None else "" for cell in row] for row in table]
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if rows:
            result.append({"title": "", "headers": [], "rows": rows})
    return result


def _is_reliable(tables: List[Dict[str, Any]]) -> bool:
    """A pdfplumber result is considered reliable only if it looks like a
    real rigid grid, not misdetected prose. Ordinary paragraph text (bullet
    lists especially) can get misread as a table by the text-alignment
    strategy, so row/column presence alone isn't enough -- real tables have
    a consistent column count across most rows and short cell content
    (labels/numbers), while misdetected prose has ragged column counts and
    full sentences in "cells"."""
    if not tables:
        return False
    for table in tables:
        rows = table.get("rows", [])
        if len(rows) < 2:
            continue

        col_counts = [len(r) for r in rows]
        if max(col_counts, default=0) < 2:
            continue

        most_common_count, freq = Counter(col_counts).most_common(1)[0]
        if most_common_count < 2 or freq / len(col_counts) < 0.7:
            continue

        total_cells = sum(len(r) for r in rows)
        non_empty_texts = [str(c).strip() for r in rows for c in r if str(c).strip()]
        if total_cells == 0 or (len(non_empty_texts) / total_cells) < 0.5:
            continue

        avg_words = sum(len(c.split()) for c in non_empty_texts) / max(1, len(non_empty_texts))
        if avg_words > 6:
            continue

        return True
    return False


def extract_tables_from_pdf_page(page: Any) -> List[Dict[str, Any]]:
    """Extract tables from a pdfplumber Page using its lines-based strategy
    (real PDF ruling lines only).

    A text-alignment-based strategy was tried as a fallback for borderless
    tables, but on real prose pages it fragments ordinary sentences into
    spurious multi-column "rows" (e.g. splitting "INTRODUCTION" into
    "INTRODU" / "CTION" as separate cells) that are hard to distinguish from
    genuine tables by inspecting the output alone, since the fragments are
    individually short and get padded to a consistent column count -- which
    is exactly what a reliability check would otherwise look for. Borderless
    digital-PDF tables are a known gap for now rather than risking false
    positives; TATR (already used for scanned pages) is the natural fit to
    revisit this later by rendering the page to an image."""
    try:
        raw_tables = page.extract_tables(
            {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
        )
    except Exception:
        raw_tables = []

    tables = _tables_to_dicts(raw_tables)
    if _is_reliable(tables):
        return tables
    return []