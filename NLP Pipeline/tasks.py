"""
Celery task(s) for the document-processing pipeline.

Model instances (PaddleOCR, DeepSeek-OCR) are loaded once per worker
process, lazily on first task execution -- not eagerly at import time, so
Celery can introspect this module without paying model-load cost, and not
per-task, since reloading a ~9GB model per document would be absurd.

DeepSeek is initialized before Paddle, matching the Windows-safe order
established elsewhere in this project (accuracy_eval.py, OCR_Text_Extraction
main()) -- DeepSeek's init imports torch, which must load its DLLs before
PaddleOCR tries to load shm.dll, or Paddle fails with WinError 127.
"""

import csv
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "Text Extraction"))
sys.path.insert(0, _THIS_DIR)

from celery_app import celery_app  # noqa: E402

import OCR_Text_Extraction as ocr_pipeline  # noqa: E402
import field_locator  # noqa: E402
import llm_pipeline  # noqa: E402

_ENGINES: Dict[str, Any] = {}


def _get_engines() -> Dict[str, Any]:
    if not _ENGINES:
        _ENGINES["deepseek"] = ocr_pipeline.DeepSeekTableEngine()
        _ENGINES["paddle"] = ocr_pipeline.PaddleEngine()
    return _ENGINES


def _read_tables_csv(output_dir: str, original_path: str) -> List[Dict[str, Any]]:
    base = ocr_pipeline._safe_name(os.path.splitext(os.path.basename(original_path))[0])
    csv_path = os.path.join(output_dir, base, "DeepSeekOCR_table_result.csv")
    if not os.path.isfile(csv_path):
        return []

    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _render_tables_as_text(tables: List[Dict[str, Any]]) -> str:
    """Render extracted table rows as simple pipe-separated text so the LLM
    step can see structured data too -- extractedText deliberately excludes
    table regions (masked out during OCR to avoid duplication/noise there),
    which otherwise makes totals/amounts/line-items invisible to the LLM
    even when table extraction itself worked correctly."""
    if not tables:
        return ""

    by_page: Dict[str, List[Dict[str, Any]]] = {}
    for row in tables:
        by_page.setdefault(str(row.get("page", "")), []).append(row)

    lines: List[str] = []
    for page, rows in sorted(by_page.items()):
        lines.append(f"\n[Table on page {page}]")
        for row in rows:
            cells = [str(v) for k, v in row.items() if k != "page" and str(v).strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


@celery_app.task(bind=True, name="tasks.process_document_task")
def process_document_task(
    self,
    file_path: str,
    mode: str,
    custom_prompt: str = "",
) -> Dict[str, Any]:
    """Run the full OCR/table/LLM pipeline on an already-saved file, updating
    task progress after each page. Returns the same result shape as the old
    synchronous /extract response."""
    engines = _get_engines()

    tmp_dir = tempfile.mkdtemp(prefix="docupipeline_task_")
    try:
        item = {
            "file": os.path.basename(file_path),
            "abs_path": file_path,
            "file_type": "digital",
        }

        def _progress(pages_done: int, pages_total: int) -> None:
            fraction = (pages_done / pages_total) if pages_total else 0.0
            self.update_state(state="PROGRESS", meta={"progress": round(fraction, 3)})

        ocr_pipeline.process_document(
            item=item,
            output_dir=tmp_dir,
            paddle=engines["paddle"],
            deepseek=engines["deepseek"],
            pdf_dpi=200,
            max_pdf_pages=0,
            deskew=True,
            progress_cb=_progress,
        )

        base = ocr_pipeline._safe_name(os.path.splitext(os.path.basename(file_path))[0])
        result_path = os.path.join(tmp_dir, base, "ocr_result.json")
        if not os.path.isfile(result_path):
            raise RuntimeError("OCR pipeline did not produce a result")

        with open(result_path, "r", encoding="utf-8") as f:
            ocr_result = json.load(f)

        extracted_text = ocr_result.get("text_outputs", {}).get("combined_standard", "")
        tables = _read_tables_csv(tmp_dir, file_path)

        if not extracted_text.strip() and not tables:
            raise RuntimeError(
                "No text or tables could be extracted from this document -- "
                "it may be corrupt, empty, or an unsupported format."
            )

        table_text = _render_tables_as_text(tables)

        mode_upper = (mode or "FULL").upper()
        if mode_upper == "FULL":
            # FULL's job is faithful reproduction of the extracted text --
            # appending the rendered table (useful context for the other
            # three modes) makes it duplicate that appendix verbatim instead.
            llm_input_text = extracted_text
        else:
            llm_input_text = extracted_text + ("\n\n" + table_text if table_text else "")

        try:
            llm_result = llm_pipeline.run_llm(mode_upper, llm_input_text, user_prompt=custom_prompt)
            if mode_upper == "CUSTOM_FIELDS":
                structured_json = llm_result.get("json")
            else:
                structured_json = {"result": llm_result.get("text", "")}
        except Exception as exc:
            structured_json = {"error": str(exc)}

        # Where each field sits on the original page, for highlighting in the
        # UI. Best effort: a failure here must not fail the extraction.
        field_locations: Dict[str, Any] = {}
        if mode_upper == "CUSTOM_FIELDS" and isinstance(structured_json, dict):
            try:
                field_locations = field_locator.locate_fields(file_path, structured_json, engines["paddle"])
            except Exception:
                field_locations = {}

        return {
            "extractedText": extracted_text,
            "structuredJson": structured_json,
            "tables": tables,
            "mode": mode_upper,
            "fieldLocations": field_locations,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            os.remove(file_path)
        except OSError:
            pass

