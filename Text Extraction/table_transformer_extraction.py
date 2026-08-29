"""
Table Transformer (TATR) based table extraction.

Uses the real microsoft/table-transformer-* object detection models:
- microsoft/table-transformer-detection            : finds table regions on a page
- microsoft/table-transformer-structure-recognition-v1.1-all : finds rows/columns
  inside a table region

This is a general-purpose, pretrained neural detector -- it does not assume any
particular document layout, language, or table style, and does not require
visible ruling lines (unlike the project's OpenCV line-map heuristic). Cell
text is read out via an OCR callback (PaddleOCR by default) applied to each
detected cell crop.

Known limitation: merged/spanning cells ("table spanning cell" detections) are
not specially reconstructed in this version -- each cell is derived purely
from the intersection of one detected row band and one detected column band.
A spanning cell's text will only show up in whichever single grid cell its
OCR crop happens to align with best; this mirrors a similar documented
limitation in the OpenCV-fallback table reconstruction used elsewhere in this
pipeline.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

DETECTION_MODEL_NAME = os.getenv("TATR_DETECTION_MODEL", "microsoft/table-transformer-detection")
STRUCTURE_MODEL_NAME = os.getenv(
    "TATR_STRUCTURE_MODEL", "microsoft/table-transformer-structure-recognition-v1.1-all"
)

_MODEL_CACHE: Dict[str, Any] = {}
_PADDLE_OCR_CACHE: Dict[str, Any] = {}

Box = Tuple[float, float, float, float]


def _get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(model_name: str):
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    device = _get_device()
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = TableTransformerForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()

    _MODEL_CACHE[model_name] = (processor, model, device)
    return processor, model, device


def _run_object_detection(image_bgr: np.ndarray, model_name: str, score_thresh: float) -> List[Dict[str, Any]]:
    """Run a table-transformer object-detection model on a BGR image, returning
    detections as {"label": str, "score": float, "box": (x1, y1, x2, y2)}
    in the image's own pixel coordinates."""
    import torch
    from PIL import Image as PILImage

    processor, model, device = _load_model(model_name)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([pil_img.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=score_thresh, target_sizes=target_sizes
    )[0]

    id2label = model.config.id2label
    detections: List[Dict[str, Any]] = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        detections.append(
            {
                "label": id2label.get(int(label_id), str(int(label_id))),
                "score": float(score),
                "box": (x1, y1, x2, y2),
            }
        )
    return detections


def detect_tables_tatr(image_bgr: np.ndarray, score_thresh: float = 0.7) -> List[Box]:
    """Find table bounding boxes anywhere on the page using the real
    detection model, regardless of whether the table has visible ruling
    lines. Complements/replaces line-map-based region detection, which
    misses borderless tables entirely."""
    if image_bgr is None:
        return []
    try:
        detections = _run_object_detection(image_bgr, DETECTION_MODEL_NAME, score_thresh)
    except Exception:
        return []
    return [d["box"] for d in detections if "table" in d["label"].lower()]


def _pad_box(box: Box, pad_ratio: float, max_w: int, max_h: int) -> Box:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(max_w), x2 + pad_x),
        min(float(max_h), y2 + pad_y),
    )


def extract_table_structure_tatr(
    image_bgr: np.ndarray,
    table_box: Optional[Box] = None,
    structure_thresh: float = 0.7,
    pad_ratio: float = 0.05,
) -> Dict[str, Any]:
    """Detect row/column structure inside a table region (or the whole image
    if table_box is None) and build a grid of cell boxes (in the coordinate
    space of the cropped table image used for detection).

    Returns a dict with: crop (the cropped BGR image the grid is relative
    to), grid (list of rows, each a list of cell boxes), num_rows, num_cols,
    avg_row_score, avg_col_score.
    """
    h, w = image_bgr.shape[:2]
    if table_box is None:
        crop = image_bgr
        origin = (0, 0)
    else:
        px1, py1, px2, py2 = _pad_box(table_box, pad_ratio, w, h)
        x1, y1, x2, y2 = int(px1), int(py1), int(px2), int(py2)
        if x2 <= x1 or y2 <= y1:
            return {"crop": None, "grid": [], "num_rows": 0, "num_cols": 0, "avg_row_score": 0.0, "avg_col_score": 0.0}
        crop = image_bgr[y1:y2, x1:x2]
        origin = (x1, y1)

    try:
        detections = _run_object_detection(crop, STRUCTURE_MODEL_NAME, structure_thresh)
    except Exception:
        detections = []

    rows = [d for d in detections if d["label"] == "table row"]
    cols = [d for d in detections if d["label"] == "table column"]

    rows_sorted = sorted(rows, key=lambda d: d["box"][1])
    cols_sorted = sorted(cols, key=lambda d: d["box"][0])

    grid: List[List[Box]] = []
    for row in rows_sorted:
        _, ry1, _, ry2 = row["box"]
        row_cells: List[Box] = []
        for col in cols_sorted:
            cx1, _, cx2, _ = col["box"]
            row_cells.append((cx1, ry1, cx2, ry2))
        grid.append(row_cells)

    avg_row_score = float(np.mean([r["score"] for r in rows_sorted])) if rows_sorted else 0.0
    avg_col_score = float(np.mean([c["score"] for c in cols_sorted])) if cols_sorted else 0.0

    return {
        "crop": crop,
        "origin": origin,
        "grid": grid,
        "num_rows": len(rows_sorted),
        "num_cols": len(cols_sorted),
        "avg_row_score": avg_row_score,
        "avg_col_score": avg_col_score,
    }


def _default_ocr_fn(crop_bgr: np.ndarray) -> str:
    """Lazily-created PaddleOCR instance used when the caller doesn't supply
    its own ocr_fn, so this module is usable standalone."""
    ocr = _PADDLE_OCR_CACHE.get("engine")
    if ocr is None:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(lang="en", use_textline_orientation=True)
        _PADDLE_OCR_CACHE["engine"] = ocr

    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    try:
        try:
            result = ocr.ocr(crop_bgr, cls=True)
        except TypeError:
            result = ocr.ocr(crop_bgr)
    except Exception:
        return ""

    texts: List[str] = []
    if isinstance(result, list):
        for block in result:
            if isinstance(block, dict):
                txt = block.get("rec_text") or block.get("text")
                if txt:
                    texts.append(str(txt))
                continue
            if isinstance(block, (list, tuple)):
                for item in block:
                    if isinstance(item, dict):
                        txt = item.get("rec_text") or item.get("text")
                        if txt:
                            texts.append(str(txt))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        txt_part = item[1]
                        if isinstance(txt_part, (list, tuple)) and txt_part:
                            texts.append(str(txt_part[0]))
                        elif isinstance(txt_part, str):
                            texts.append(txt_part)
    return " ".join(t.strip() for t in texts if t and t.strip())


def _fill_grid_text(
    crop_bgr: np.ndarray, grid: List[List[Box]], ocr_fn: Callable[[np.ndarray], str]
) -> List[List[str]]:
    h, w = crop_bgr.shape[:2]
    text_rows: List[List[str]] = []
    for row_cells in grid:
        row_text: List[str] = []
        for (x1, y1, x2, y2) in row_cells:
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w, int(x2)), min(h, int(y2))
            if ix2 <= ix1 or iy2 <= iy1:
                row_text.append("")
                continue
            cell_crop = crop_bgr[iy1:iy2, ix1:ix2]
            row_text.append(ocr_fn(cell_crop).strip())
        text_rows.append(row_text)
    return text_rows


def is_tatr_result_reliable(structure_result: Dict[str, Any], text_rows: Optional[List[List[str]]] = None) -> bool:
    """Decide whether a TATR structure-recognition result is trustworthy
    enough to use, versus falling back to another method. Combines detection
    confidence, grid-shape sanity, and content yield -- no single signal is
    trusted alone, since each can be individually misleading."""
    if not structure_result:
        return False

    num_rows = structure_result.get("num_rows", 0)
    num_cols = structure_result.get("num_cols", 0)
    if num_rows < 2 or num_cols < 2:
        return False

    min_confidence = float(os.getenv("TATR_MIN_STRUCTURE_CONFIDENCE", "0.6") or "0.6")
    if structure_result.get("avg_row_score", 0.0) < min_confidence:
        return False
    if structure_result.get("avg_col_score", 0.0) < min_confidence:
        return False

    if text_rows is not None:
        total_cells = sum(len(r) for r in text_rows)
        non_empty = sum(1 for r in text_rows for c in r if c)
        min_fill_ratio = float(os.getenv("TATR_MIN_CELL_FILL_RATIO", "0.3") or "0.3")
        if total_cells > 0 and (non_empty / total_cells) < min_fill_ratio:
            return False

    return True


def extract_tables_from_image_tatr(
    image_bgr: np.ndarray,
    table_boxes: Optional[List[Box]] = None,
    ocr_fn: Optional[Callable[[np.ndarray], str]] = None,
    detection_thresh: float = 0.7,
    structure_thresh: float = 0.7,
) -> Tuple[bool, List[Dict[str, Any]], str]:
    """Top-level entry point: detect table region(s) if not given, recognize
    structure, OCR each cell, and return (ok, tables, err) -- the same shape
    used elsewhere in this pipeline (e.g. DeepSeekTableEngine.extract_tables_from_image),
    so this can be dropped into the same call pattern."""
    if image_bgr is None:
        return False, [], "no image provided"

    if ocr_fn is None:
        ocr_fn = _default_ocr_fn

    boxes = table_boxes
    if not boxes:
        boxes = detect_tables_tatr(image_bgr, score_thresh=detection_thresh)
        if not boxes:
            return False, [], "no table detected"

    tables: List[Dict[str, Any]] = []
    unreliable_count = 0
    for box in boxes:
        structure_result = extract_table_structure_tatr(image_bgr, box, structure_thresh=structure_thresh)
        crop = structure_result.get("crop")
        grid = structure_result.get("grid") or []
        if crop is None or not grid:
            unreliable_count += 1
            continue

        text_rows = _fill_grid_text(crop, grid, ocr_fn)
        if not is_tatr_result_reliable(structure_result, text_rows):
            unreliable_count += 1
            continue

        tables.append({"title": "", "headers": [], "rows": text_rows})

    if tables:
        return True, tables, ""
    if unreliable_count:
        return False, [], "TATR structure result unreliable"
    return False, [], "no table detected"
