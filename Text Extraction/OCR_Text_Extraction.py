"""
OCR pipeline for text extraction from documents.

Workflow:
1) Reads images/PDFs from an input directory (recommended mode)
2) Optionally reads files from a pre-existing CSV report (--report mode)
3) Runs DeepSeek table OCR on pages where table regions are detected
4) Runs PaddleOCR for non-table text only (table regions masked)
5) Writes per-document outputs and consolidated outputs

Default inputs/outputs:
- Optional input report: Text Extraction/digital_vs_scanned_report.csv
- Output folder         : Text Extraction/OCR Results

Usage:
    python "Text Extraction/OCR_Text_Extraction.py" --input-dir "Data/OCR Test"
    python "Text Extraction/OCR_Text_Extraction.py" --max-files 10
    python "Text Extraction/OCR_Text_Extraction.py" --input-dir "Data/Denoised" --max-files 20 --deskew

DeepSeek table backend (free/local by default):
    Put settings in Text Extraction/.env.extraction (or Text Extraction/.env)
    using keys:
    - DEEPSEEK_TABLE_BACKEND=local|api   (default: local)
    - DEEPSEEK_TABLE_MODEL               (default local: deepseek-ai/DeepSeek-OCR)
    Optional API-only keys:
    - DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL

Outputs:
    Per-document folder: standard_text_paddle.txt, DeepSeekOCR_table_result.csv, meta.json, ocr_result.json
    OCR Results root    : OCR_Summary.csv, Standard_text_all.txt, tables_all.csv
"""

from __future__ import annotations
import argparse
import base64
import contextlib
import csv
import html
import io
import json
import math
import os
import re
import sys
import tempfile
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import table_transformer_extraction
import doctr_extraction
import pdfplumber_table_extraction


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _join_non_empty(parts: List[str], sep: str = "\n") -> str:
    clean = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return sep.join(clean)


def _short_error_text(msg: str, max_lines: int = 8) -> str:
    lines = [ln.strip() for ln in (msg or "").splitlines() if ln.strip()]
    if not lines:
        return "unknown error"
    if len(lines) <= max_lines:
        return " | ".join(lines)
    return " | ".join(lines[-max_lines:])


def _box_from_points(box: Any) -> Optional[Tuple[float, float, float, float]]:
    try:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return (min(xs), min(ys), max(xs), max(ys))
    except Exception:
        return None


def _scale_boxes(boxes: List[Tuple[int, int, int, int]], scale: float) -> List[Tuple[int, int, int, int]]:
    if abs(scale - 1.0) < 1e-3:
        return boxes
    scaled: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        scaled.append(
            (
                int(round(x1 * scale)),
                int(round(y1 * scale)),
                int(round(x2 * scale)),
                int(round(y2 * scale)),
            )
        )
    return scaled


def _box_overlap_ratio(a: Tuple[float, float, float, float], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    return float(inter) / float(area)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _env_bool(name: str, default: str = "") -> Optional[bool]:
    val = (os.getenv(name, default) or "").strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _paddle_can_use_gpu() -> bool:
    try:
        import paddle  # type: ignore

        if hasattr(paddle, "is_compiled_with_cuda") and paddle.is_compiled_with_cuda():
            try:
                return bool(paddle.device.cuda.device_count())
            except Exception:
                return True
    except Exception:
        return False
    return False

def _estimate_skew_angle_hough(image_bgr: np.ndarray) -> Optional[float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = gray.shape
    min_len = max(60, int(min(h, w) * 0.25))
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, threshold=100, minLineLength=min_len, maxLineGap=20)

    if lines is None:
        return None

    angles: List[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if -45.0 <= angle <= 45.0:
            angles.append(angle)

    if len(angles) < 3:
        return None

    return float(np.median(np.array(angles)))


def _estimate_skew_angle_minarea(image_bgr: np.ndarray) -> Optional[float]:
    """Fallback when Hough finds no reliable lines (sparse/low-contrast pages).
    Derives tilt from boxPoints() edge vectors directly instead of trusting
    minAreaRect's raw angle value — that angle's convention/range changed in
    OpenCV 4.5+, and trusting it directly caused ~90 deg over-rotation on 4.6.0."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    if coords is None or len(coords) < 1000:
        return None

    rect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(rect)
    (x1, y1), (x2, y2) = box[0], box[1]
    raw = math.degrees(math.atan2(y2 - y1, x2 - x1))

    tilt = raw % 90.0
    if tilt > 45.0:
        tilt -= 90.0
    return float(tilt)


def _estimate_skew_angle(image_bgr: np.ndarray) -> float:
    angle = _estimate_skew_angle_hough(image_bgr)
    if angle is None:
        angle = _estimate_skew_angle_minarea(image_bgr)
    return float(angle) if angle is not None else 0.0



def _load_env_if_available(env_path: str) -> bool:
    """Load .env values when python-dotenv is installed."""
    if not os.path.isfile(env_path):
        return False
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return True
    except Exception:
        return False


def _load_env_files_if_available(env_paths: List[str]) -> List[str]:
    """Load environment values from multiple files in order, without overriding existing vars."""
    loaded: List[str] = []
    for env_path in env_paths:
        if not env_path or not os.path.isfile(env_path):
            continue
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(env_path, override=False)
            loaded.append(env_path)
        except Exception:
            continue
    return loaded


def _is_pdf(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".pdf"


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _normalize_extensions(exts: str) -> Optional[set[str]]:
    if not exts:
        return None
    parts = [p.strip().lower() for p in re.split(r"[,;\s]+", exts) if p.strip()]
    if not parts:
        return None
    normalized: set[str] = set()
    for p in parts:
        normalized.add(p if p.startswith(".") else f".{p}")
    return normalized


def _rotate_image(image_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.01:
        return image_bgr
    h, w = image_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    mat[0, 2] += (new_w / 2) - center[0]
    mat[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image_bgr, mat, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 10 or max_height < 10:
        return image_bgr

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        image_bgr,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def detect_document_contour(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Find a 4-point document boundary when the page doesn't already fill the frame
    (phone/camera captures). Returns None for already-flat/full-frame pages (printer
    scans, digital renders) or for contours that are already a near-perfect rectangle
    (a false match on an internal design element, not real perspective skew)."""
    h, w = image_bgr.shape[:2]
    frame_area = float(h * w)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.25 or area > frame_area * 0.98:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype("float32")
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect

        top_w = np.linalg.norm(tr - tl)
        bottom_w = np.linalg.norm(br - bl)
        left_h = np.linalg.norm(bl - tl)
        right_h = np.linalg.norm(br - tr)

        width_ratio = min(top_w, bottom_w) / max(top_w, bottom_w, 1e-6)
        height_ratio = min(left_h, right_h) / max(left_h, right_h, 1e-6)

        # Real perspective skew makes opposite sides visibly unequal (a trapezoid).
        # A near-perfect rectangle here is almost always a false match on an
        # internal design element of an already-flat scan — skip it.
        if width_ratio > 0.92 and height_ratio > 0.92:
            continue

        return pts

    return None


def correct_perspective(image_bgr: np.ndarray) -> np.ndarray:
    """Warp a phone/camera-captured document to a flat top-down rectangle.
    No-op (returns original) if no reliable 4-point document boundary is found."""
    if image_bgr is None:
        return image_bgr
    try:
        pts = detect_document_contour(image_bgr)
        if pts is None:
            return image_bgr
        return _four_point_transform(image_bgr, pts)
    except Exception:
        return image_bgr


def remove_shadow(image_bgr: np.ndarray) -> np.ndarray:
    """Normalize uneven illumination/shadows common in phone-camera captures.
    Estimates the smooth illumination field with a large-kernel blur, then divides
    the image by it and rescales, which flattens gradual shadows without touching
    sharp text edges."""
    if image_bgr is None:
        return image_bgr
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        kernel_size = max(21, (min(h, w) // 15) | 1)
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        background = np.maximum(background, 1).astype(np.float32)
        normalized = (gray.astype(np.float32) / background) * 255.0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    except Exception:
        return image_bgr

def classify_pdf_page_text(page: Any) -> bool:
    """True if a PyMuPDF page already has enough embedded text to skip OCR
    entirely (a digital PDF page, as opposed to a scanned image inside a PDF)."""
    try:
        text = page.get_text() or ""
    except Exception:
        return False
    min_chars = int(os.getenv("OCR_DIGITAL_TEXT_MIN_CHARS", "40") or "40")
    return len(text.strip()) >= min_chars


def _brightness_gradient_score(image_bgr: np.ndarray) -> float:
    """Rough measure of uneven lighting across a coarse grid of the page.
    Flatbed/printer scans are near-uniformly lit; phone/camera photos often
    show a shadow or lighting gradient across the frame."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    rows, cols = 4, 4
    means: List[float] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            cell = gray[y0:y1, x0:x1]
            if cell.size:
                means.append(float(cell.mean()))
    if not means:
        return 0.0
    return float(max(means) - min(means))


def classify_scan_type(image_bgr: np.ndarray) -> str:
    """Distinguish a flatbed/printer scan from a phone/camera capture for a page
    image that has no usable embedded PDF text. Returns "phone_scan" or
    "printer_scan"."""
    try:
        if detect_document_contour(image_bgr) is not None:
            return "phone_scan"
        gradient_thresh = float(os.getenv("OCR_SHADOW_GRADIENT_THRESH", "35") or "35")
        if _brightness_gradient_score(image_bgr) >= gradient_thresh:
            return "phone_scan"
    except Exception:
        pass
    return "printer_scan"

_SWINUNET_DENOISE_FN = None


def _get_swinunet_denoise_fn():
    """Lazily import Preprocessing/swinunet_test.py's denoise_image, adding
    the sibling Preprocessing/ folder to sys.path since it isn't a package."""
    global _SWINUNET_DENOISE_FN
    if _SWINUNET_DENOISE_FN is not None:
        return _SWINUNET_DENOISE_FN

    preprocessing_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Preprocessing")
    if preprocessing_dir not in sys.path:
        sys.path.insert(0, preprocessing_dir)
    import swinunet_test  # type: ignore

    _SWINUNET_DENOISE_FN = swinunet_test.denoise_image
    return _SWINUNET_DENOISE_FN


def preprocess_image_for_ocr(image_bgr: np.ndarray, use_swinunet: bool = False) -> np.ndarray:
    """Deskew and denoise for scanned/digital documents to improve OCR quality."""
    if image_bgr is None:
        return image_bgr

    max_side = int(os.getenv("OCR_MAX_SIDE", "2400") or "2400")
    denoise_on = os.getenv("OCR_DENOISE", "1") != "0"
    deskew_on = os.getenv("OCR_DESKEW", "1") != "0"

    processed = image_bgr.copy()
    h, w = processed.shape[:2]
    if max(h, w) > max_side > 0:
        scale = float(max_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if denoise_on:
        if use_swinunet:
            try:
                denoise_fn = _get_swinunet_denoise_fn()
                processed = denoise_fn(processed)
            except Exception:
                processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        else:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)

    if deskew_on:
        angle = _estimate_skew_angle(processed)
        if abs(angle) >= 0.5:
            processed = _rotate_image(processed, angle_deg=angle)

    return processed


def preprocess_image_for_paddle(image_bgr: np.ndarray, use_swinunet: bool = False) -> np.ndarray:
    """Lightweight preprocess for PaddleOCR on scans (avoid over-denoise)."""
    if image_bgr is None:
        return image_bgr

    max_side = int(os.getenv("PADDLE_MAX_SIDE", os.getenv("OCR_MAX_SIDE", "2400")) or "2400")
    denoise_on = os.getenv("PADDLE_DENOISE", "0") != "0"
    deskew_pref = _env_bool("PADDLE_DESKEW", "")
    if deskew_pref is None:
        deskew_on = os.getenv("OCR_DESKEW", "1") != "0"
    else:
        deskew_on = bool(deskew_pref)

    processed = image_bgr.copy()
    h, w = processed.shape[:2]
    if max(h, w) > max_side > 0:
        scale = float(max_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if denoise_on:
        if use_swinunet:
            try:
                denoise_fn = _get_swinunet_denoise_fn()
                processed = denoise_fn(processed)
            except Exception:
                processed = cv2.fastNlMeansDenoisingColored(processed, None, 7, 7, 7, 21)
        else:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 7, 7, 7, 21)

    if deskew_on:
        angle = _estimate_skew_angle(processed)
        if abs(angle) >= 0.5:
            processed = _rotate_image(processed, angle_deg=angle)

    return processed


def _maybe_upscale_for_paddle(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    if os.getenv("PADDLE_SMALL_TEXT_UPSCALE", "1") == "0":
        return None

    min_side = int(os.getenv("PADDLE_SMALL_TEXT_MIN_SIDE", "1000") or "1000")
    target_side = int(os.getenv("PADDLE_SMALL_TEXT_TARGET_SIDE", "1400") or "1400")
    max_scale = float(os.getenv("PADDLE_SMALL_TEXT_MAX_SCALE", "2.5") or "2.5")

    h, w = image_bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= 0 or max_side >= min_side or target_side <= 0:
        return None

    scale = min(max_scale, float(target_side) / float(max_side))
    if scale <= 1.05:
        return None

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _paddle_variant_images(image_bgr: np.ndarray) -> List[np.ndarray]:
    """Generate a small set of enhanced variants to improve PaddleOCR recall."""
    variants: List[np.ndarray] = [image_bgr]

    extra_passes = os.getenv("PADDLE_EXTRA_PASSES", "1") != "0"
    if not extra_passes:
        return variants

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement (CLAHE) for faint text.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR))

    # Adaptive threshold for uneven lighting.
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # Mild sharpening to recover soft edges.
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.4, blur, -0.4, 0)
    variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

    return variants


def _extract_paddle_items(result: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    if not isinstance(result, list):
        return items

    for block in result:
        if isinstance(block, dict):
            txt = block.get("rec_text") or block.get("text") or block.get("label")
            score = block.get("score", 1.0)
            if isinstance(txt, str) and txt.strip():
                try:
                    score_val = float(score)
                except Exception:
                    score_val = 1.0
                items.append({"text": txt.strip(), "score": score_val, "box": None})
            continue

        if not isinstance(block, (list, tuple)):
            continue

        for item in block:
            if isinstance(item, dict):
                txt = item.get("rec_text") or item.get("text") or item.get("label")
                score = item.get("score", 1.0)
                if isinstance(txt, str) and txt.strip():
                    try:
                        score_val = float(score)
                    except Exception:
                        score_val = 1.0
                    items.append({"text": txt.strip(), "score": score_val, "box": None})
                continue

            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            box = item[0]
            txt_part = item[1]
            if isinstance(txt_part, (list, tuple)) and len(txt_part) >= 1:
                txt = str(txt_part[0]).strip()
                if not txt:
                    continue
                score_val = 1.0
                if len(txt_part) > 1 and isinstance(txt_part[1], (float, int)):
                    score_val = float(txt_part[1])
                items.append({"text": txt, "score": score_val, "box": box})
            elif isinstance(txt_part, str) and txt_part.strip():
                items.append({"text": txt_part.strip(), "score": 1.0, "box": box})

    return items


def _extract_paddle_lines(result: Any) -> List[Tuple[str, float]]:
    lines: List[Tuple[str, float]] = []
    for item in _extract_paddle_items(result):
        text = item.get("text") or ""
        score = item.get("score", 1.0)
        if isinstance(text, str) and text.strip():
            try:
                score_val = float(score)
            except Exception:
                score_val = 1.0
            lines.append((text.strip(), score_val))
    return lines


def _filter_items_outside_boxes(
    items: List[Dict[str, Any]],
    table_boxes: List[Tuple[int, int, int, int]],
) -> List[Dict[str, Any]]:
    if not table_boxes:
        return items
    min_overlap = float(os.getenv("PADDLE_TABLE_OVERLAP_THRESH", "0.2") or "0.2")
    drop_no_box = os.getenv("PADDLE_DROP_ITEMS_WITHOUT_BOX", "0") == "1"

    filtered: List[Dict[str, Any]] = []
    for item in items:
        box = _box_from_points(item.get("box"))
        if box is None:
            if not drop_no_box:
                filtered.append(item)
            continue
        overlaps = any(_box_overlap_ratio(box, tb) >= min_overlap for tb in table_boxes)
        if not overlaps:
            filtered.append(item)
    return filtered


@dataclass
class OcrDocResult:
    file: str
    label: str
    paddle_ok: bool
    deepseek_ok: bool
    pages: int
    standard_chars: int
    table_chars: int
    note: str


class PaddleEngine:
    def __init__(self) -> None:
        self.ready = False
        self.err = ""
        self.ocr = None
        self.use_gpu = False
        try:
            from paddleocr import PaddleOCR  # type: ignore

            # Avoid startup model-host checks in constrained networks.
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

            use_gpu_pref = _env_bool("PADDLE_USE_GPU", "auto")
            if use_gpu_pref is True:
                requested_gpu = True
            elif use_gpu_pref is False:
                requested_gpu = False
            else:
                requested_gpu = _paddle_can_use_gpu()

            # PaddleOCR constructor options vary by version.
            # Try modern signatures first, then older fallback.
            base_kwargs_list = [
                {"lang": "en", "use_textline_orientation": True},
                {"lang": "en"},
                {"lang": "en", "use_angle_cls": True},
            ]
            kwargs_candidates: List[Dict[str, Any]] = []
            if requested_gpu:
                for kw in base_kwargs_list:
                    kw_gpu = dict(kw)
                    kw_gpu["use_gpu"] = True
                    kwargs_candidates.append(kw_gpu)
            kwargs_candidates.extend(base_kwargs_list)

            init_errors: List[str] = []
            for kwargs in kwargs_candidates:
                try:
                    self.ocr = PaddleOCR(**kwargs)
                    self.ready = True
                    self.use_gpu = bool(kwargs.get("use_gpu"))
                    break
                except Exception as ie:
                    init_errors.append(f"{kwargs}: {ie}")

            if not self.ready:
                self.err = " | ".join(init_errors)
        except Exception as exc:
            self.err = str(exc)

    def extract_text(
        self,
        image_bgr: np.ndarray,
        table_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Tuple[bool, str, str]:
        if not self.ready or self.ocr is None:
            return False, "", f"PaddleOCR unavailable: {self.err}"

        errors: List[str] = []
        ordered_lines: List[str] = []
        seen: Dict[str, float] = {}

        early_stop_on = os.getenv("PADDLE_EARLY_STOP", "1") != "0"
        min_lines = int(os.getenv("PADDLE_EARLY_STOP_MIN_LINES", "20") or "20")
        min_chars = int(os.getenv("PADDLE_EARLY_STOP_MIN_CHARS", "500") or "500")
        min_avg_score = float(os.getenv("PADDLE_EARLY_STOP_MIN_AVG_SCORE", "0.86") or "0.86")

        variants = _paddle_variant_images(image_bgr)
        variant_pairs: List[Tuple[np.ndarray, float]] = [(v, 1.0) for v in variants]
        for idx, (variant, scale) in enumerate(variant_pairs):
            try:
                # PaddleOCR signatures differ across major versions.
                try:
                    result = self.ocr.ocr(variant, cls=True)
                except TypeError:
                    try:
                        result = self.ocr.ocr(variant)
                    except Exception:
                        result = self.ocr.predict(variant)

                items = _extract_paddle_items(result)
                if table_boxes:
                    items = _filter_items_outside_boxes(items, _scale_boxes(table_boxes, scale))

                for item in items:
                    text = item.get("text") or ""
                    score = item.get("score", 1.0)
                    if not isinstance(text, str) or not text.strip():
                        continue
                    try:
                        score_val = float(score)
                    except Exception:
                        score_val = 1.0
                    if text not in seen:
                        ordered_lines.append(text)
                        seen[text] = score_val
                    elif score_val > seen.get(text, 0.0):
                        seen[text] = score_val

                if idx == 0:
                    min_lines = int(os.getenv("PADDLE_SMALL_TEXT_MIN_LINES", "4") or "4")
                    min_chars = int(os.getenv("PADDLE_SMALL_TEXT_MIN_CHARS", "80") or "80")
                    char_count = sum(len(line) for line in ordered_lines)
                    if len(ordered_lines) < min_lines or char_count < min_chars:
                        upscaled = _maybe_upscale_for_paddle(image_bgr)
                        if upscaled is not None:
                            scale = float(upscaled.shape[1]) / float(image_bgr.shape[1])
                            variant_pairs.append((upscaled, scale))

                if early_stop_on and idx == 0 and ordered_lines:
                    avg_score = sum(seen.values()) / max(1, len(seen))
                    char_count = sum(len(line) for line in ordered_lines)
                    if len(ordered_lines) >= min_lines and char_count >= min_chars and avg_score >= min_avg_score:
                        return True, "\n".join(ordered_lines), ""
            except Exception as exc:
                errors.append(str(exc))

        if ordered_lines:
            return True, "\n".join(ordered_lines), ""

        if errors:
            return False, "", f"PaddleOCR failed: {errors[-1]}"
        return False, "", "PaddleOCR returned no text"


TABLE_CSV_MAX_COLS = 20


def _extract_json_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
    return None


def _to_cell_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() if v is not None else "" for v in values]


def _normalize_deepseek_tables(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_tables = payload.get("tables", [])
    if not isinstance(raw_tables, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for t in raw_tables:
        if not isinstance(t, dict):
            continue
        title = str(t.get("title", "")).strip()
        headers = _to_cell_list(t.get("headers", []))

        rows_raw = t.get("rows", [])
        rows: List[List[str]] = []
        if isinstance(rows_raw, list):
            for r in rows_raw:
                if isinstance(r, dict):
                    ordered = [r[k] for k in sorted(r.keys())]
                    rows.append([str(v).strip() if v is not None else "" for v in ordered])
                elif isinstance(r, list):
                    rows.append([str(v).strip() if v is not None else "" for v in r])

        normalized.append({"title": title, "headers": headers, "rows": rows})
    return normalized


def _strip_html_tags(text: str) -> str:
    clean = re.sub(r"<br\s*/?>", "\n", text or "", flags=re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = html.unescape(clean)
    return clean.replace("\r", "").strip()


def _extract_html_tables(raw_text: str) -> List[Dict[str, Any]]:
    text = raw_text or ""
    if "<table" not in text.lower():
        return []

    tables: List[Dict[str, Any]] = []
    table_blocks = re.findall(r"<table\b[^>]*>(.*?)</table>", text, flags=re.IGNORECASE | re.DOTALL)

    for ti, block in enumerate(table_blocks, start=1):
        row_blocks = re.findall(r"<tr\b[^>]*>(.*?)</tr>", block, flags=re.IGNORECASE | re.DOTALL)
        parsed_rows: List[List[str]] = []

        for rb in row_blocks:
            cells_raw = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", rb, flags=re.IGNORECASE | re.DOTALL)
            if cells_raw:
                cells = [_strip_html_tags(c) for c in cells_raw]
                parsed_rows.append(cells)

        if parsed_rows:
            tables.append({"title": f"html_table_{ti}", "headers": [], "rows": parsed_rows})

    return tables


def _extract_tables_from_generated_text(raw_text: str, allow_fallback: bool = False) -> List[Dict[str, Any]]:
    payload_json = _extract_json_payload(raw_text)
    if payload_json is not None:
        parsed = _normalize_deepseek_tables(payload_json)
        if parsed:
            return parsed

    html_tables = _extract_html_tables(raw_text)
    if html_tables:
        return html_tables

    if allow_fallback:
        return _fallback_tables_from_text(raw_text)
    return []

def _is_deepseek_table_result_reliable(tables: List[Dict[str, Any]]) -> bool:
    """A DeepSeek-OCR table result is only trustworthy if it looks like a real
    rigid grid, not a hallucinated table forced onto a non-tabular page (e.g.
    a dense form of unrelated labeled fields, which DeepSeek will still try
    to comply with since it's generative and doesn't self-reject like TATR
    does). Mirrors the same column-consistency + fill-ratio check already
    used for pdfplumber results."""
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

        return True
    return False


def _clean_cell_text(text: str) -> str:
    text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>", " ", text or "")
    return re.sub(r"\s+", " ", text.strip())
    

def _split_table_like_line(raw_line: str) -> List[str]:
    line = (raw_line or "").rstrip("\r\n")
    if not line.strip():
        return []
    if re.fullmatch(r"[-_=|:\s]+", line.strip()):
        return []

    if "|" in line:
        cells = [_clean_cell_text(c) for c in line.split("|")]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            return cells

    if "\t" in line:
        cells = [_clean_cell_text(c) for c in line.split("\t")]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            return cells

    if re.search(r"\s{2,}", line):
        cells = [_clean_cell_text(c) for c in re.split(r"\s{2,}", line.strip())]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            return cells

    return []


def _row_is_probably_header(row: List[str]) -> bool:
    if not row:
        return False
    non_empty = [c for c in row if c]
    if not non_empty:
        return False

    numeric_cells = 0
    long_cells = 0
    for cell in non_empty:
        if re.search(r"\d", cell):
            numeric_cells += 1
        if len(cell.split()) > 8:
            long_cells += 1

    if long_cells > max(1, len(non_empty) // 2):
        return False
    return numeric_cells <= max(1, len(non_empty) // 3)


def _merge_multiline_rows(rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return []

    col_freq: Dict[int, int] = {}
    for r in rows:
        c = len([x for x in r if str(x).strip()])
        if c >= 2:
            col_freq[c] = col_freq.get(c, 0) + 1
    target_cols = max(col_freq.items(), key=lambda kv: kv[1])[0] if col_freq else max(2, max(len(r) for r in rows))

    merged: List[List[str]] = []
    for raw in rows:
        cells = [_clean_cell_text(c) for c in raw if _clean_cell_text(c)]
        if not cells:
            continue

        if len(cells) >= target_cols:
            merged.append(cells)
            continue

        if not merged:
            merged.append(cells)
            continue

        prev = merged[-1]
        if len(cells) == 1:
            last_non_empty = -1
            for i in range(len(prev) - 1, -1, -1):
                if prev[i]:
                    last_non_empty = i
                    break
            if last_non_empty == -1:
                prev.append(cells[0])
            else:
                prev[last_non_empty] = _clean_cell_text(f"{prev[last_non_empty]} {cells[0]}")
            continue

        for i, val in enumerate(cells):
            if i < len(prev):
                prev[i] = _clean_cell_text(f"{prev[i]} {val}") if prev[i] else val
            else:
                prev.append(val)

    return merged


def _split_side_by_side_rows(rows: List[List[str]]) -> List[List[List[str]]]:
    if not rows:
        return []

    header = rows[0]
    if len(header) < 6:
        return [rows]

    best_split = -1
    best_score = -1
    for split in range(2, len(header) - 1):
        left = header[:split]
        right = header[split:]
        if min(len(left), len(right)) < 2:
            continue

        equal_tokens = sum(
            1 for a, b in zip(left, right) if _clean_cell_text(a).lower() == _clean_cell_text(b).lower() and _clean_cell_text(a)
        )
        left_words = set(" ".join(left).lower().split())
        right_words = set(" ".join(right).lower().split())
        overlap = len(left_words & right_words)
        score = equal_tokens * 3 + overlap
        if score > best_score:
            best_score = score
            best_split = split

    if best_split < 0 or best_score < 2:
        return [rows]

    left_rows: List[List[str]] = []
    right_rows: List[List[str]] = []
    for r in rows:
        l = [_clean_cell_text(c) for c in r[:best_split] if _clean_cell_text(c)]
        rr = [_clean_cell_text(c) for c in r[best_split:] if _clean_cell_text(c)]
        if l:
            left_rows.append(l)
        if rr:
            right_rows.append(rr)

    if len(left_rows) >= 2 and len(right_rows) >= 2:
        return [left_rows, right_rows]
    return [rows]


def _rows_to_table(rows: List[List[str]], title: str) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    normalized: List[List[str]] = []
    for r in rows:
        cells = [_clean_cell_text(c) for c in r if _clean_cell_text(c)]
        if len(cells) >= 2:
            normalized.append(cells)

    if not normalized:
        return None

    merged = _merge_multiline_rows(normalized)
    if not merged:
        return None

    if len(merged) >= 2 and _row_is_probably_header(merged[0]):
        headers = merged[0]
        rows_data = merged[1:]
    else:
        col_count = max(2, max(len(r) for r in merged))
        headers = [f"column_{i + 1}" for i in range(col_count)]
        rows_data = merged

    if not headers:
        return None

    padded_rows: List[List[str]] = []
    for r in rows_data:
        rr = list(r)
        if len(rr) < len(headers):
            rr += [""] * (len(headers) - len(rr))
        elif len(rr) > len(headers):
            rr = rr[: len(headers) - 1] + [_clean_cell_text(" ".join(rr[len(headers) - 1 :]))]
        if any(c.strip() for c in rr):
            padded_rows.append(rr)

    return {"title": title, "headers": headers, "rows": padded_rows}


def _looks_like_key_cell(text: str) -> bool:
    t = _clean_cell_text(text)
    if not t or len(t) > 50:
        return False
    words = t.split()
    if len(words) > 5:
        return False
    if re.search(r"[\.;]$", t):
        return False
    letters = sum(1 for ch in t if ch.isalpha())
    return letters >= max(1, len(t) // 3)


def _should_try_fallback_table_parser(page_text: str) -> bool:
    raw_lines = [(ln or "") for ln in (page_text or "").splitlines() if (ln or "").strip()]
    lines = [_clean_cell_text(ln) for ln in raw_lines if _clean_cell_text(ln)]
    if not raw_lines or not lines:
        return False

    if any(("|" in ln) or ("\t" in ln) or bool(re.search(r"\s{2,}", ln)) for ln in raw_lines):
        return True

    colon_pairs = sum(1 for ln in lines if bool(re.match(r"^([^:]{1,80})\s*[:\-]\s*(.+)$", ln)))
    if colon_pairs >= 2:
        return True

    if len(lines) >= 6:
        short_lines = sum(1 for ln in lines if len(ln) <= 32 and len(ln.split()) <= 4)
        if (short_lines / max(1, len(lines))) >= 0.65:
            return True

    return False


def _extract_key_value_table(page_text: str) -> Optional[Dict[str, Any]]:
    lines = [_clean_cell_text(ln) for ln in (page_text or "").splitlines() if _clean_cell_text(ln)]
    if not lines:
        return None

    pairs: List[List[str]] = []

    for ln in lines:
        m = re.match(r"^([^:]{1,80})\s*[:\-]\s*(.+)$", ln)
        if m:
            k = _clean_cell_text(m.group(1))
            v = _clean_cell_text(m.group(2))
            if k and v and _looks_like_key_cell(k):
                pairs.append([k, v])

    for ln in lines:
        cells = _split_table_like_line(ln)
        if (
            len(cells) == 2
            and _looks_like_key_cell(cells[0])
            and len(cells[0]) <= 40
            and len(cells[0].split()) <= 4
            and cells[1]
            and len(cells[1]) <= 120
        ):
            pairs.append([cells[0], cells[1]])

    if len(pairs) < 3:
        alt: List[List[str]] = []
        i = 0
        while i + 1 < len(lines):
            k = lines[i]
            v = lines[i + 1]
            if _looks_like_key_cell(k) and v:
                alt.append([k, v])
                i += 2
            else:
                i += 1
        if len(alt) >= 3:
            pairs = alt

    if len(pairs) < 2:
        return None

    # Deduplicate key-value pairs while preserving order.
    seen: Dict[str, bool] = {}
    unique_pairs: List[List[str]] = []
    for p in pairs:
        sig = f"{p[0]}||{p[1]}"
        if sig in seen:
            continue
        seen[sig] = True
        unique_pairs.append(p)

    return {
        "title": "fallback_key_value_table",
        "headers": ["field", "value"],
        "rows": unique_pairs,
    }


def _fallback_tables_from_text(page_text: str) -> List[Dict[str, Any]]:
    """Best-effort local fallback to recover structured tables from difficult OCR text layouts."""
    raw_lines = (page_text or "").splitlines()
    indexed_rows: List[Tuple[int, List[str]]] = []
    for idx, raw in enumerate(raw_lines):
        cells = _split_table_like_line(raw)
        if len(cells) >= 2:
            indexed_rows.append((idx, cells))

    tables: List[Dict[str, Any]] = []

    if indexed_rows:
        blocks: List[List[Tuple[int, List[str]]]] = []
        current_block: List[Tuple[int, List[str]]] = [indexed_rows[0]]
        for idx, cells in indexed_rows[1:]:
            if idx - current_block[-1][0] <= 2:
                current_block.append((idx, cells))
            else:
                blocks.append(current_block)
                current_block = [(idx, cells)]
        blocks.append(current_block)

        for bi, block in enumerate(blocks, start=1):
            rows = [r for _, r in block]
            split_groups = _split_side_by_side_rows(rows)
            for gi, grp in enumerate(split_groups, start=1):
                table = _rows_to_table(grp, title=f"fallback_table_{bi}_{gi}")
                if table and (table.get("headers") or table.get("rows")):
                    tables.append(table)

    kv_table = _extract_key_value_table(page_text)
    if kv_table and not tables:
        tables.append(kv_table)

    deduped: List[Dict[str, Any]] = []
    seen_signatures: Dict[str, bool] = {}
    for t in tables:
        sig_obj = {
            "headers": t.get("headers", []),
            "rows": t.get("rows", [])[:12],
        }
        sig = json.dumps(sig_obj, ensure_ascii=False, sort_keys=True)
        if sig in seen_signatures:
            continue
        seen_signatures[sig] = True
        deduped.append(t)

    return deduped


class DeepSeekTableEngine:
    def __init__(self, disable: bool = False, timeout_sec: int = 120) -> None:
        self.disable = disable
        self.timeout_sec = timeout_sec
        self.backend = (os.getenv("DEEPSEEK_TABLE_BACKEND", "local").strip().lower() or "local")
        if self.backend not in {"local", "api"}:
            self.backend = "local"

        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions").strip()
        legacy_model = os.getenv("DEEPSEEK_MODEL", "").strip()
        if self.backend == "api":
            self.model = os.getenv("DEEPSEEK_TABLE_MODEL", "").strip() or legacy_model or "deepseek-vl2"
        else:
            self.model = os.getenv("DEEPSEEK_TABLE_MODEL", "").strip() or "deepseek-ai/DeepSeek-OCR"

        self.local_prompt = (
            os.getenv("DEEPSEEK_LOCAL_PROMPT", "").strip()
            or "<image>\n<|grounding|>Extract all tables from this page as markdown with HTML table tags."
        )
        self.local_base_size = int(os.getenv("DEEPSEEK_LOCAL_BASE_SIZE", "1024") or "1024")
        self.local_image_size = int(os.getenv("DEEPSEEK_LOCAL_IMAGE_SIZE", "640") or "640")

        self.local_tokenizer = None
        self.local_model = None
        self.local_ready = False
        self.local_err = ""
        self.use_gpu = False

        self.balance_blocked = False
        self.balance_blocked_reason = ""

        if self.backend == "local":
            self._init_local_model()

    def _init_local_model(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "use_safetensors": True,
            }

            # Avoid hard dependency on flash-attn; eager works in more environments.
            if torch.cuda.is_available():
                model_kwargs["_attn_implementation"] = "eager"

            model = AutoModel.from_pretrained(self.model, **model_kwargs)
            model = model.eval()

            use_gpu_pref = _env_bool("DEEPSEEK_USE_GPU", "auto")
            if use_gpu_pref is True and not torch.cuda.is_available():
                raise RuntimeError("DEEPSEEK_USE_GPU=1 but CUDA is not available")
            use_gpu = torch.cuda.is_available() if use_gpu_pref is None else bool(use_gpu_pref)
            if use_gpu:
                model = model.cuda().to(torch.bfloat16)
            self.use_gpu = use_gpu

            self.local_tokenizer = tokenizer
            self.local_model = model

            # DeepSeek-OCR's own infer() hardcodes max_new_tokens=8192 in the
            # generate() calls it makes on itself -- infer() doesn't expose a
            # way to override this, and 8192 is far more than a table needs
            # (it's the dominant cost on a slow page). Wrap generate() directly
            # instead of editing the cached model file: infer() calls
            # self.generate(...), which resolves to this instance attribute
            # before the class method, so this is robust to the model being
            # re-downloaded or updated.
            _orig_generate = model.generate
            max_tokens_cap = int(os.getenv("DEEPSEEK_MAX_NEW_TOKENS", "1024") or "1024")

            def _capped_generate(*args, **kwargs):
                if "max_new_tokens" in kwargs:
                    kwargs["max_new_tokens"] = min(kwargs["max_new_tokens"], max_tokens_cap)
                return _orig_generate(*args, **kwargs)

            model.generate = _capped_generate

            self.local_ready = True
        except Exception as exc:
            self.local_err = f"Local DeepSeek-OCR init failed: {exc}"
            self.local_ready = False

    @property
    def ready(self) -> bool:
        if self.disable:
            return False
        if self.backend == "local":
            return self.local_ready
        return (not self.balance_blocked) and bool(self.api_key)

    def _post_chat_completion(self, messages: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
        try:
            import requests  # type: ignore
        except Exception as exc:
            return False, "", f"requests not available: {exc}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": messages,
        }

        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout_sec)
        except Exception as exc:
            return False, "", f"DeepSeek call failed: {exc}"

        if resp.status_code >= 400:
            body = (resp.text or "")[:800]
            body_lower = body.lower()
            if resp.status_code == 402 or "insufficient balance" in body_lower:
                self.balance_blocked = True
                self.balance_blocked_reason = "DeepSeek disabled for this run: insufficient balance"
                return False, "", f"DeepSeek HTTP {resp.status_code}: insufficient balance"
            return False, "", f"DeepSeek HTTP {resp.status_code}: {body[:300]}"

        try:
            data = resp.json()
        except Exception as exc:
            return False, "", f"DeepSeek response JSON parse failed: {exc}"

        choices = data.get("choices", [])
        if not choices:
            return False, "", "DeepSeek response has no choices"

        msg = choices[0].get("message", {})
        content = msg.get("content", "")

        # Some providers return content as list of chunks.
        if isinstance(content, list):
            chunks: List[str] = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("text", "")
                    if isinstance(t, str) and t.strip():
                        chunks.append(t)
            content = "\n".join(chunks)

        if not isinstance(content, str):
            content = str(content)

        return True, content, ""

    def validate_access(self) -> Tuple[bool, str]:
        if self.disable:
            return False, "DeepSeek disabled by flag"
        if self.backend == "local":
            if self.local_ready:
                return True, ""
            return False, self.local_err or "Local DeepSeek-OCR backend is not ready"
        if self.balance_blocked:
            msg = self.balance_blocked_reason or "DeepSeek temporarily disabled after insufficient balance response"
            return False, msg
        if not self.api_key:
            return False, "DEEPSEEK_API_KEY not set"

        ok, content, err = self._post_chat_completion(
            [
                {"role": "system", "content": "Reply with OK only."},
                {"role": "user", "content": "OK"},
            ]
        )
        if not ok:
            return False, err
        if not (content or "").strip():
            return False, "DeepSeek preflight returned empty response"
        return True, ""

    @staticmethod
    def _encode_image_to_data_url(image_bgr: np.ndarray, max_side: int = 1800, jpeg_quality: int = 90) -> Tuple[bool, str, str]:
        try:
            h, w = image_bgr.shape[:2]
            if max(h, w) > max_side:
                scale = float(max_side) / float(max(h, w))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(
                ".jpg",
                image_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, int(max(60, min(98, jpeg_quality)))],
            )
            if not ok:
                return False, "", "Failed to encode page image for DeepSeek"

            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return True, f"data:image/jpeg;base64,{b64}", ""
        except Exception as exc:
            return False, "", f"Failed to prepare image for DeepSeek: {exc}"

    def _extract_tables_from_image_local(self, page_image_bgr: np.ndarray) -> Tuple[bool, List[Dict[str, Any]], str]:
        if self.disable:
            return False, [], "DeepSeek disabled by flag"
        if not self.local_ready or self.local_model is None or self.local_tokenizer is None:
            return False, [], self.local_err or "Local DeepSeek-OCR backend is not ready"

        tmp_path = ""
        try:
            ok, buf = cv2.imencode(".jpg", page_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                return False, [], "Failed to encode image for local DeepSeek-OCR"

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tf.write(buf.tobytes())
                tmp_path = tf.name

            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                result = self.local_model.infer(
                    self.local_tokenizer,
                    prompt=self.local_prompt,
                    image_file=tmp_path,
                    output_path=tempfile.gettempdir(),
                    base_size=self.local_base_size,
                    image_size=self.local_image_size,
                    crop_mode=True,
                    save_results=False,
                    test_compress=False,
                )

            captured_text = capture.getvalue()

            raw_text = ""
            if isinstance(result, str):
                raw_text = result
            elif isinstance(result, (list, tuple)) and result:
                raw_text = str(result[0])
            elif isinstance(result, dict):
                for key in ("text", "content", "output", "markdown", "result"):
                    val = result.get(key)
                    if isinstance(val, str) and val.strip():
                        raw_text = val
                        break
                if not raw_text:
                    raw_text = str(result)
            else:
                raw_text = str(result)

            if not raw_text.strip() or raw_text.strip().lower() == "none":
                raw_text = captured_text

            tables = _extract_tables_from_generated_text(raw_text, allow_fallback=False)
            if not tables:
                return True, [], ""
            return True, tables, ""
        except Exception as exc:
            return False, [], f"Local DeepSeek-OCR inference failed: {exc}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _extract_tables_from_image_api(self, page_image_bgr: np.ndarray) -> Tuple[bool, List[Dict[str, Any]], str]:
        if self.disable:
            return False, [], "DeepSeek disabled by flag"
        if self.balance_blocked:
            msg = self.balance_blocked_reason or "DeepSeek temporarily disabled after insufficient balance response"
            return False, [], msg
        if not self.api_key:
            return False, [], "DEEPSEEK_API_KEY not set"

        ok_img, image_data_url, enc_err = self._encode_image_to_data_url(page_image_bgr)
        if not ok_img:
            return False, [], enc_err

        prompt = (
            "Extract EVERY table from this single page image. "
            "Treat this as direct table OCR from image pixels. "
            "Include bordered tables, borderless tables, multiline cells, key-value tables, and side-by-side tables. "
            "Return ONLY valid JSON with this schema exactly: "
            "{\"tables\":[{\"title\":\"\",\"headers\":[\"\"],\"rows\":[[\"\"]]}]}. "
            "If no tables are visible, return {\"tables\":[]}. "
            "No markdown, no explanation, no extra keys."
        )

        message_variants: List[Any] = [
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_data_url},
            ],
        ]

        errors: List[str] = []
        for user_content in message_variants:
            ok, content, err = self._post_chat_completion(
                [
                    {"role": "system", "content": "Extract structured tables only. Never return prose."},
                    {"role": "user", "content": user_content},
                ]
            )
            if not ok:
                errors.append(err)
                if self.balance_blocked:
                    break
                continue

            tables = _extract_tables_from_generated_text(content, allow_fallback=False)
            return True, tables, ""

        joined = " | ".join(e for e in errors if e) or "DeepSeek table OCR failed"
        return False, [], _short_error_text(joined)

    def extract_tables_from_image(self, page_image_bgr: np.ndarray) -> Tuple[bool, List[Dict[str, Any]], str]:
        if self.backend == "local":
            return self._extract_tables_from_image_local(page_image_bgr)
        return self._extract_tables_from_image_api(page_image_bgr)


def _compute_table_line_maps(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    h, w = gray.shape
    hk = max(10, w // 40)
    vk = max(10, h // 40)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    return bw, h_lines, v_lines


def detect_table_like_page(
    image_bgr: np.ndarray,
    line_maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> float:
    """Return heuristic table-likelihood score in [0,1]."""
    if line_maps is None:
        bw, h_lines, v_lines = _compute_table_line_maps(image_bgr)
    else:
        bw, h_lines, v_lines = line_maps
    h, w = bw.shape

    h_ratio = float(np.count_nonzero(h_lines)) / max(1, h * w)
    v_ratio = float(np.count_nonzero(v_lines)) / max(1, h * w)

    intersections = cv2.bitwise_and(h_lines, v_lines)
    inter_ratio = float(np.count_nonzero(intersections)) / max(1, h * w)

    score = min(1.0, 2.8 * h_ratio + 2.8 * v_ratio + 18.0 * inter_ratio)
    return float(score)


def _merge_nearby_boxes(boxes: List[Tuple[int, int, int, int]], pad: int = 16) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    pending = [tuple(int(v) for v in b) for b in boxes]
    changed = True

    while changed:
        changed = False
        merged: List[Tuple[int, int, int, int]] = []
        for box in pending:
            x1, y1, x2, y2 = box
            merged_into = False
            for i, m in enumerate(merged):
                mx1, my1, mx2, my2 = m
                overlap = not (x2 + pad < mx1 or x1 - pad > mx2 or y2 + pad < my1 or y1 - pad > my2)
                if overlap:
                    merged[i] = (min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2))
                    merged_into = True
                    changed = True
                    break
            if not merged_into:
                merged.append((x1, y1, x2, y2))
        pending = merged

    return sorted(pending, key=lambda b: (b[1], b[0]))


def detect_table_regions(
    image_bgr: np.ndarray,
    min_region_area_ratio: float = 0.01,
    line_maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> List[Tuple[int, int, int, int]]:
    """Detect coarse table regions for masking text OCR outside table areas."""
    if line_maps is None:
        bw, h_lines, v_lines = _compute_table_line_maps(image_bgr)
    else:
        bw, h_lines, v_lines = line_maps
    h, w = bw.shape
    page_area = max(1, h * w)

    grid = cv2.bitwise_or(h_lines, v_lines)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []

    for cnt in contours:
        x, y, bw_box, bh_box = cv2.boundingRect(cnt)
        area = bw_box * bh_box
        if area < int(page_area * min_region_area_ratio):
            continue
        if bw_box < max(80, w // 8):
            continue
        if bh_box < max(40, h // 25):
            continue

        roi = grid[y : y + bh_box, x : x + bw_box]
        density = float(np.count_nonzero(roi)) / max(1, area)
        if density < 0.008:
            continue

        if os.getenv("TABLE_REGION_SKIP_BORDER", "1") != "0":
            margin = int(os.getenv("TABLE_REGION_BORDER_MARGIN", "5") or "5")
            max_ratio = float(os.getenv("TABLE_REGION_BORDER_MAX_RATIO", "0.65") or "0.65")
            touches_border = x <= margin or y <= margin or (x + bw_box) >= (w - margin) or (y + bh_box) >= (h - margin)
            if touches_border and (area / max(1, page_area)) >= max_ratio:
                continue

        boxes.append((x, y, x + bw_box, y + bh_box))

    return _merge_nearby_boxes(boxes, pad=20)


def mask_table_regions_for_text(
    image_bgr: np.ndarray,
    table_boxes: List[Tuple[int, int, int, int]],
    pad_px: int = 10,
) -> np.ndarray:
    """White-out table regions so regular OCR returns non-table text only."""
    if not table_boxes:
        return image_bgr

    masked = image_bgr.copy()
    h, w = masked.shape[:2]
    for x1, y1, x2, y2 in table_boxes:
        xa = max(0, x1 - pad_px)
        ya = max(0, y1 - pad_px)
        xb = min(w - 1, x2 + pad_px)
        yb = min(h - 1, y2 + pad_px)
        cv2.rectangle(masked, (xa, ya), (xb, yb), (255, 255, 255), thickness=-1)
    return masked


def _mask_coverage_ratio(
    image_bgr: np.ndarray,
    table_boxes: List[Tuple[int, int, int, int]],
) -> float:
    if not table_boxes:
        return 0.0
    h, w = image_bgr.shape[:2]
    page_area = max(1, h * w)
    covered = 0
    for x1, y1, x2, y2 in table_boxes:
        covered += max(0, x2 - x1) * max(0, y2 - y1)
    return float(covered) / float(page_area)


def _merge_text_lines(primary: str, secondary: str) -> str:
    seen: Dict[str, bool] = {}
    merged: List[str] = []
    for block in (primary or "").splitlines():
        line = block.strip()
        if not line or line in seen:
            continue
        seen[line] = True
        merged.append(line)
    for block in (secondary or "").splitlines():
        line = block.strip()
        if not line or line in seen:
            continue
        seen[line] = True
        merged.append(line)
    return "\n".join(merged)


def render_pdf_to_images(pdf_path: str, dpi: int = 200, max_pages: int = 0) -> List[np.ndarray]:
    """Render PDF pages to BGR images. Requires PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF (fitz) is required for PDF rendering: {exc}")

    doc = fitz.open(pdf_path)
    images: List[np.ndarray] = []

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    page_total = doc.page_count
    use_pages = page_total if max_pages <= 0 else min(page_total, max_pages)

    for i in range(use_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif pix.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        images.append(arr)

    doc.close()
    return images


def load_image_as_bgr(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def load_digital_docs(report_csv: str, workspace_root: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    with open(report_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label") or "").strip().lower()
            if label != "digital":
                continue

            rel_path = (row.get("file") or "").strip()
            if not rel_path:
                continue

            abs_path = os.path.join(workspace_root, rel_path)
            if not os.path.isfile(abs_path):
                continue

            if not (_is_pdf(abs_path) or _is_image(abs_path)):
                continue

            docs.append(
                {
                    "file": rel_path,
                    "abs_path": abs_path,
                    "file_type": (row.get("file_type") or "").strip().lower() or "unknown",
                }
            )
    return docs


def load_docs_from_dir(
    input_dir: str,
    workspace_root: str,
    extensions: Optional[set[str]] = None,
) -> List[Dict[str, str]]:
    abs_dir = os.path.abspath(input_dir)
    if not os.path.isdir(abs_dir):
        return []

    docs: List[Dict[str, str]] = []
    for name in sorted(os.listdir(abs_dir)):
        abs_path = os.path.join(abs_dir, name)
        if not os.path.isfile(abs_path):
            continue
        ext = os.path.splitext(abs_path)[1].lower()
        if extensions is not None and ext not in extensions:
            continue
        if not (_is_pdf(abs_path) or _is_image(abs_path)):
            continue
        rel_path = os.path.relpath(abs_path, workspace_root)
        docs.append(
            {
                "file": rel_path,
                "abs_path": abs_path,
                "file_type": os.path.splitext(abs_path)[1].lower().lstrip(".") or "unknown",
            }
        )
    return docs


def write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


def _table_csv_headers() -> List[str]:
    return ["page"]


def _table_csv_headers_all() -> List[str]:
    return ["file", "page"]


def _write_table_rows_csv(path: str, rows: List[Dict[str, object]], include_file: bool = False) -> None:
    if not rows:
        headers = _table_csv_headers_all() if include_file else _table_csv_headers()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        return

    max_cols = max(len(r.get("cells", [])) for r in rows)
    max_cols = max(1, min(TABLE_CSV_MAX_COLS, max_cols))

    headers = ["page"] + [f"col_{i + 1}" for i in range(max_cols)]
    if include_file:
        headers = ["file"] + headers

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            cells = list(row.get("cells", []))[:max_cols]
            if len(cells) < max_cols:
                cells += [""] * (max_cols - len(cells))

            page = row.get("page", "")
            if include_file:
                file_val = row.get("file", "")
                writer.writerow([file_val, page, *cells])
            else:
                writer.writerow([page, *cells])

def _load_all_doc_metas(output_dir: str) -> List[Dict[str, Any]]:
    """Read meta.json from every existing per-document output folder, so
    consolidated reports can cover all runs on disk, not just the current
    invocation's docs."""
    metas: List[Dict[str, Any]] = []
    if not os.path.isdir(output_dir):
        return metas

    for name in sorted(os.listdir(output_dir)):
        doc_out = os.path.join(output_dir, name)
        if not os.path.isdir(doc_out):
            continue
        meta_path = os.path.join(doc_out, "meta.json")
        meta: Dict[str, Any] = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        meta["_doc_out"] = doc_out
        meta.setdefault("file", name)
        metas.append(meta)
    return metas


def _build_cumulative_results(output_dir: str) -> List["OcrDocResult"]:
    """Reconstruct OcrDocResult rows from every doc's meta.json on disk, so the
    summary CSV reflects all runs, not just the current invocation."""
    cumulative: List[OcrDocResult] = []
    for meta in _load_all_doc_metas(output_dir):
        notes = meta.get("notes", [])
        if not isinstance(notes, list):
            notes = []
        cumulative.append(
            OcrDocResult(
                file=str(meta.get("file", "")),
                label="digital",
                paddle_ok=bool((meta.get("paddle") or {}).get("ok")),
                deepseek_ok=bool((meta.get("deepseek") or {}).get("ok")),
                pages=int(meta.get("pages_processed", 0) or 0),
                standard_chars=int(meta.get("standard_chars", 0) or 0),
                table_chars=int(meta.get("table_rows_written", 0) or 0),
                note="; ".join(str(n) for n in notes[:8]),
            )
        )
    return cumulative


def write_consolidated_outputs(processed_docs: List[Dict[str, str]], output_dir: str) -> Tuple[str, str]:
    """Write workspace-level outputs covering every processed doc found under
    output_dir (not just processed_docs from the current invocation), so
    re-running the pipeline on a subset of files doesn't discard previously
    consolidated results.

    - Standard_text_all.txt: combined regular text
    - tables_all.csv      : combined extracted table rows
    """
    combined_text_parts: List[str] = []
    all_table_rows: List[Dict[str, object]] = []

    for meta in _load_all_doc_metas(output_dir):
        doc_out = meta["_doc_out"]
        rel_file = str(meta.get("file", os.path.basename(doc_out)))

        text_path = os.path.join(doc_out, "standard_text_paddle.txt")
        if os.path.isfile(text_path):
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text = (f.read() or "").strip()
                if text:
                    combined_text_parts.append(f"===== File: {rel_file} =====\n{text}")
            except Exception:
                pass

        tables_path = os.path.join(doc_out, "DeepSeekOCR_table_result.csv")
        if os.path.isfile(tables_path):
            try:
                with open(tables_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    headers = next(reader, [])
                    if not headers or headers[0].lower() != "page":
                        headers = []
                    for row in reader:
                        if not row:
                            continue
                        page = row[0]
                        cells = row[1:]
                        all_table_rows.append({"file": rel_file, "page": page, "cells": cells})
            except Exception:
                pass

    all_text_path = os.path.join(output_dir, "Standard_text_all.txt")
    write_text(all_text_path, _join_non_empty(combined_text_parts, sep="\n\n"))

    all_tables_path = os.path.join(output_dir, "tables_all.csv")
    _ensure_dir(os.path.dirname(all_tables_path))
    _write_table_rows_csv(all_tables_path, all_table_rows, include_file=True)

    return all_text_path, all_tables_path


def _find_row_anchor_column(rows: List[List[str]]) -> Optional[int]:
    """Find the column that marks where a new logical row begins.

    In most tables one column -- a serial number, item code or date -- is
    filled on the first printed line of a row and left blank on the lines
    that continue it. That column is the anchor. If the leftmost populated
    column is filled on *every* row then each printed line really is its own
    row, and None is returned so nothing gets merged."""
    if not rows:
        return None
    width = max(len(r) for r in rows)
    for c in range(width):
        filled = sum(1 for r in rows if c < len(r) and str(r[c]).strip())
        if filled == len(rows):
            return None
        if filled >= 1:
            return c
    return None


def _combine_row_cells(base: List[str], extra: List[str]) -> List[str]:
    """Fold extra's non-empty cells into base, column by column."""
    out = list(base)
    for i, val in enumerate(extra):
        text = str(val).strip()
        if not text:
            continue
        if i < len(out):
            existing = str(out[i]).strip()
            out[i] = f"{existing} {text}" if existing else text
        else:
            out.append(text)
    return out


def _merge_ocr_row_fragments(rows: List[List[str]]) -> List[List[str]]:
    """Fold continuation lines back into the logical row they belong to.

    Position clustering treats every printed line as a row, so a line item
    whose description, code and tax breakdown are printed on separate lines
    arrives as several sparse rows. This regroups them around the anchor
    column: lines below an anchored row are appended to it, and lines that
    appear above their anchor (a description printed before the line
    carrying the serial number) are held and attached to the anchored row
    that follows.

    Tables where every printed line is genuinely its own row have no anchor
    column and are returned untouched. Set OCR_TABLE_MERGE_ROWS=0 to disable
    entirely."""
    if _env_bool("OCR_TABLE_MERGE_ROWS", "1") is False:
        return rows
    if len(rows) < 3:
        return rows

    anchor = _find_row_anchor_column(rows)
    if anchor is None:
        return rows

    width = max(len(r) for r in rows)
    padded = [list(r) + [""] * (width - len(r)) for r in rows]
    if not any(str(r[anchor]).strip() for r in padded):
        return rows

    out: List[List[str]] = []
    pending: List[List[str]] = []
    last_is_data = False

    for row in padded:
        if str(row[anchor]).strip():
            merged = row
            for held in pending:
                merged = _combine_row_cells(merged, held)
            pending = []
            out.append(merged)
            last_is_data = True
        elif last_is_data:
            out[-1] = _combine_row_cells(out[-1], row)
        elif not out and _row_is_probably_header([c for c in row if str(c).strip()]):
            # only the very first line may be kept as a standalone header;
            # the header detector is too permissive for single-cell rows
            out.append(row)
        else:
            pending.append(row)

    out.extend(pending)
    return out


def _reconstruct_table_from_ocr_boxes(
    im_paddle: np.ndarray,
    table_box: Tuple[int, int, int, int],
    paddle: "PaddleEngine",
) -> List[List[str]]:
    """Reconstruct a basic table grid by clustering OCR item positions inside a
    detected table region into rows/columns. Used as a fallback when DeepSeek
    fails to find any structured table despite a real table region being
    detected by the line-map heuristic (e.g. the VISHAL_RANE e-ticket case,
    where DeepSeek returned tables_found=0 for a genuine passenger/seat grid)."""
    if not paddle.ready or paddle.ocr is None:
        return []

    h, w = im_paddle.shape[:2]
    x1 = max(0, int(table_box[0]))
    y1 = max(0, int(table_box[1]))
    x2 = min(w, int(table_box[2]))
    y2 = min(h, int(table_box[3]))
    if x2 <= x1 or y2 <= y1:
        return []

    crop = im_paddle[y1:y2, x1:x2]
    try:
        try:
            result = paddle.ocr.ocr(crop, cls=True)
        except TypeError:
            result = paddle.ocr.ocr(crop)
    except Exception:
        return []

    entries: List[Dict[str, float]] = []
    for item in _extract_paddle_items(result):
        text = str(item.get("text") or "").strip()
        box = _box_from_points(item.get("box"))
        if not text or box is None:
            continue
        bx1, by1, bx2, by2 = box
        entries.append({"text": text, "cy": (by1 + by2) / 2.0, "cx": (bx1 + bx2) / 2.0, "h": by2 - by1})

    if not entries:
        return []

    entries.sort(key=lambda e: e["cy"])
    avg_h = sum(e["h"] for e in entries) / len(entries) or 10.0
    row_gap = max(4.0, avg_h * 0.6)

    rows: List[List[Dict[str, float]]] = []
    for e in entries:
        if rows and abs(e["cy"] - rows[-1][-1]["cy"]) <= row_gap:
            rows[-1].append(e)
        else:
            rows.append([e])
    for row in rows:
        row.sort(key=lambda e: e["cx"])

    # Cluster x-centers across every row into shared column bins, so every
    # row gets the same column count aligned by actual horizontal position --
    # not just however many fragments happened to land in that one row.
    all_cx = sorted(e["cx"] for row in rows for e in row)
    avg_w = sum(e["h"] for row in rows for e in row) / max(1, sum(len(r) for r in rows)) or 10.0
    col_gap = max(8.0, avg_w * 2.0)

    col_centers: List[float] = []
    for cx in all_cx:
        if col_centers and abs(cx - col_centers[-1]) <= col_gap:
            continue
        col_centers.append(cx)

    def _col_index(cx: float) -> int:
        best_idx = 0
        best_dist = float("inf")
        for idx, center in enumerate(col_centers):
            dist = abs(cx - center)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    table_rows: List[List[str]] = []
    for row in rows:
        out_row = [""] * len(col_centers)
        for e in row:
            idx = _col_index(e["cx"])
            if out_row[idx]:
                out_row[idx] += " " + str(e["text"])
            else:
                out_row[idx] = str(e["text"])
        table_rows.append(out_row)
    table_rows.append(out_row)
    return _merge_ocr_row_fragments(table_rows)


def process_document(
    item: Dict[str, str],
    output_dir: str,
    paddle: PaddleEngine,
    deepseek: DeepSeekTableEngine,
    pdf_dpi: int,
    max_pdf_pages: int,
    deskew: bool,
    perspective: Optional[bool] = None,
    shadow_removal: Optional[bool] = None,
    swinunet: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> OcrDocResult:
    rel_file = item["file"]
    abs_path = item["abs_path"]
    label = item.get("file_type", "digital") or "digital"

    base = _safe_name(os.path.splitext(os.path.basename(abs_path))[0])
    doc_out = os.path.join(output_dir, base)
    _ensure_dir(doc_out)

    notes: List[str] = []
    page_rows: List[Dict[str, object]] = []
    table_rows: List[Dict[str, object]] = []
    standard_pages: List[Dict[str, object]] = []
    paddle_ok_any = False
    deepseek_ok_any = False
    table_count = 0
    expected_pages = 0
    pages_failed = 0

    def _write_per_doc_exports(
        page_data: List[Dict[str, object]],
        table_data: List[Dict[str, object]],
        standard_page_data: List[Dict[str, object]],
        paddle_text: str,
        combined_text: str,
        pages_processed: int,
        pages_expected: int,
        pages_failed_count: int,
        error: str = "",
    ) -> None:
        _ensure_dir(doc_out)
        csv_path = os.path.join(doc_out, "DeepSeekOCR_table_result.csv")
        json_path = os.path.join(doc_out, "ocr_result.json")

        _write_table_rows_csv(csv_path, table_data, include_file=False)

        payload = {
            "file": rel_file,
            "pages_expected": pages_expected,
            "pages_processed": pages_processed,
            "pages_failed": pages_failed_count,
            "error": error,
            "summary": {
                "paddle_ok": paddle_ok_any,
                "deepseek_ok": deepseek_ok_any,
                "standard_chars": len(combined_text),
                "tables_found": table_count,
                "table_rows_written": len(table_data),
            },
            "text_outputs": {
                "paddle": paddle_text,
                "combined_standard": combined_text,
            },
            "pages": page_data,
            "standard_text_pages": standard_page_data,
            "tables_csv_file": os.path.basename(csv_path),
            "notes": notes,
        }

        with open(json_path, "w", encoding="utf-8") as fjson:
            json.dump(payload, fjson, indent=2)

    
    # Load pages/images
    images: List[np.ndarray] = []
    digital_page_texts: List[Optional[str]] = []
    digital_page_tables: List[List[Dict[str, Any]]] = []
    try:
        if _is_pdf(abs_path):
            try:
                import fitz  # type: ignore

                with fitz.open(abs_path) as doc:
                    page_total = int(doc.page_count)
                    expected_pages = page_total if max_pdf_pages <= 0 else min(page_total, max_pdf_pages)
                    for p_idx in range(expected_pages):
                        page = doc.load_page(p_idx)
                        digital_page_texts.append(page.get_text(sort=True) if classify_pdf_page_text(page) else None)
            except Exception:
                expected_pages = 0

            try:
                import pdfplumber  # type: ignore

                with pdfplumber.open(abs_path) as pdf:
                    for p_idx in range(expected_pages):
                        is_digital = p_idx < len(digital_page_texts) and digital_page_texts[p_idx] is not None
                        if is_digital and p_idx < len(pdf.pages):
                            digital_page_tables.append(
                                pdfplumber_table_extraction.extract_tables_from_pdf_page(pdf.pages[p_idx])
                            )
                        else:
                            digital_page_tables.append([])
            except Exception:
                pass

            images = render_pdf_to_images(abs_path, dpi=pdf_dpi, max_pages=max_pdf_pages)
            if not images:
                raise RuntimeError("no pages rendered")
            if expected_pages <= 0:
                expected_pages = len(images)
            while len(digital_page_texts) < len(images):
                digital_page_texts.append(None)
            while len(digital_page_tables) < len(images):
                digital_page_tables.append([])
        else:
            img = load_image_as_bgr(abs_path)
            if img is None:
                raise RuntimeError("cannot load image")
            images = [img]
            expected_pages = 1
    except Exception as exc:
        notes.append(str(exc))
        pages_failed = 1
        meta = {
            "file": rel_file,
            "pages_expected": expected_pages,
            "pages_failed": pages_failed,
            "error": str(exc),
            "paddle": {"ok": False},
            "deepseek": {"ok": False},
        }
        with open(os.path.join(doc_out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        page_rows.append(
            {
                "file": rel_file,
                "page": 0,
                "table_score": "",
                "paddle_ok": False,
                "deepseek_ok": False,
                "page_note": str(exc),
            }
        )
        _write_per_doc_exports(
            page_data=page_rows,
            table_data=table_rows,
            standard_page_data=standard_pages,
            paddle_text="",
            combined_text="",
            pages_processed=0,
            pages_expected=expected_pages,
            pages_failed_count=pages_failed,
            error=str(exc),
        )

        return OcrDocResult(
            file=rel_file,
            label=label,
            paddle_ok=False,
            deepseek_ok=False,
            pages=0,
            standard_chars=0,
            table_chars=0,
            note="; ".join(notes),
        )

    paddle_text_parts: List[str] = []

    def _append_page_tables(page: int, tables: List[Dict[str, Any]]) -> int:
        written = 0
        for t_idx, table in enumerate(tables, start=1):
            headers = _to_cell_list(table.get("headers", []))
            rows = table.get("rows", [])

            if headers:
                table_rows.append({"page": page, "cells": headers})

            if isinstance(rows, list):
                for raw_row in rows:
                    if isinstance(raw_row, list):
                        row_cells = [str(v) if v is not None else "" for v in raw_row]
                    elif isinstance(raw_row, dict):
                        row_cells = [str(raw_row[k]) if raw_row[k] is not None else "" for k in sorted(raw_row.keys())]
                    else:
                        row_cells = [str(raw_row)]

                    table_rows.append({"page": page, "cells": row_cells})
            written += 1
        return written

    for pi, im in enumerate(images, start=1):
        if progress_cb:
            try:
                progress_cb(pi - 1, len(images))
            except Exception:
                pass
        digital_text = digital_page_texts[pi - 1] if pi - 1 < len(digital_page_texts) else None
        if digital_text is not None:
            paddle_ok_any = True
            page_text = digital_text.strip()
            paddle_text_parts.append(f"\n===== Page {pi} =====\n{page_text}")
            standard_pages.append(
                {
                    "file": rel_file,
                    "page": pi,
                    "source": "digital_pdf_text",
                    "text": page_text,
                }
            )

            digital_tables = digital_page_tables[pi - 1] if pi - 1 < len(digital_page_tables) else []
            table_note = "digital_pdf_bypass=direct_text_extraction"
            deepseek_used_digital = False
            if digital_tables:
                table_count += _append_page_tables(pi, digital_tables)
                table_note += " | table_source=pdfplumber"
            else:
                # pdfplumber only finds ruling-line tables; a digital PDF page
                # can still contain a borderless table with no visible lines.
                # Render this page as an image (already available as `im`)
                # and reuse the same TATR -> DeepSeek -> OCR-clustering chain
                # used for scanned pages, since TATR doesn't need visible
                # lines to find tables.
                digital_table_ok = False
                try:
                    tatr_boxes_digital = table_transformer_extraction.detect_tables_tatr(im)
                except Exception as exc:
                    tatr_boxes_digital = []
                    notes.append(f"TATR detect p{pi} (digital): {exc}")

                if tatr_boxes_digital:
                    try:
                        t_ok, t_tables, t_err = table_transformer_extraction.extract_tables_from_image_tatr(
                            im, table_boxes=tatr_boxes_digital
                        )
                    except Exception as exc:
                        t_ok, t_tables, t_err = False, [], str(exc)

                    if t_ok and t_tables:
                        table_count += _append_page_tables(pi, t_tables)
                        table_note += " | table_source=tatr"
                        digital_table_ok = True
                    elif t_err:
                        notes.append(f"TATR p{pi} (digital): {t_err}")

                    if not digital_table_ok:
                        d_ok_digital, d_tables_digital, d_err_digital = deepseek.extract_tables_from_image(im)
                        if d_ok_digital and d_tables_digital and _is_deepseek_table_result_reliable(d_tables_digital):
                            table_count += _append_page_tables(pi, d_tables_digital)
                            table_note += " | table_source=deepseek"
                            digital_table_ok = True
                            deepseek_used_digital = True
                            deepseek_ok_any = True
                        elif d_err_digital:
                            notes.append(f"DeepSeek p{pi} (digital): {d_err_digital}")

                    if not digital_table_ok:
                        fallback_rows_digital: List[List[str]] = []
                        for tb in tatr_boxes_digital:
                            fallback_rows_digital.extend(
                                _reconstruct_table_from_ocr_boxes(
                                    im,
                                    (int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3])),
                                    paddle,
                                )
                            )
                        if fallback_rows_digital:
                            table_count += _append_page_tables(pi, [{"headers": [], "rows": fallback_rows_digital}])
                            table_note += " | table_source=ocr_fallback"

            page_rows.append(
                {
                    "file": rel_file,
                    "page": pi,
                    "table_score": 0.0,
                    "table_regions": 0,
                    "paddle_ok": True,
                    "deepseek_ok": deepseek_used_digital,
                    "page_note": table_note,
                }
            )
            continue 

        scan_type = classify_scan_type(im)

        use_perspective = perspective if perspective is not None else (scan_type == "phone_scan")
        use_shadow_removal = shadow_removal if shadow_removal is not None else (scan_type == "phone_scan")

        page_im = correct_perspective(im) if use_perspective else im
        if use_shadow_removal:
            page_im = remove_shadow(page_im)
        if deskew:
            im_proc = preprocess_image_for_ocr(page_im, use_swinunet=swinunet)
            im_paddle = preprocess_image_for_paddle(page_im, use_swinunet=swinunet)
        else:
            im_proc = page_im
            im_paddle = page_im
        line_maps = _compute_table_line_maps(im_paddle)
        table_score = detect_table_like_page(im_paddle, line_maps=line_maps)
        table_boxes = detect_table_regions(im_paddle, line_maps=line_maps)
        has_table_regions = len(table_boxes) > 0
        deepseek_score_fallback = os.getenv("DEEPSEEK_TABLE_SCORE_FALLBACK", "1") != "0"
        deepseek_score_thresh = float(
            os.getenv("DEEPSEEK_TABLE_SCORE_FALLBACK_THRESH", "0.05") or "0.05"
        )
        deepseek_score_triggered = False
        if (not has_table_regions) and deepseek_score_fallback and table_score >= deepseek_score_thresh:
            has_table_regions = True
            deepseek_score_triggered = True

        mask_boxes = list(table_boxes)
        mask_boxes = list(table_boxes)
        mask_ratio = _mask_coverage_ratio(im_paddle, mask_boxes)
        max_mask_ratio = float(os.getenv("PADDLE_TABLE_MASK_MAX_RATIO", "0.45") or "0.45")
        if mask_ratio > max_mask_ratio:
            mask_boxes = []

        # Geometric box filtering of OCR results is independent of pixel masking:
        # even when masking itself is skipped (too much of the page would be
        # blanked out), still exclude OCR results whose box falls inside a real
        # detected table region, using actual table_boxes geometry.
        text_filter_boxes = list(table_boxes)
        text_only_image = mask_table_regions_for_text(im_paddle, mask_boxes)

        # Standard text OCR from non-table regions only.
        p_ok, p_txt, p_err = paddle.extract_text(text_only_image, table_boxes=text_filter_boxes)
        fallback_used = False
        rescue_used = False
        if (not p_ok) or (not (p_txt or "").strip()):
            min_lines = int(os.getenv("PADDLE_MASKED_FALLBACK_MIN_LINES", "3") or "3")
            min_chars = int(os.getenv("PADDLE_MASKED_FALLBACK_MIN_CHARS", "60") or "60")
            line_count = len([ln for ln in (p_txt or "").splitlines() if ln.strip()])
            char_count = len((p_txt or "").strip())
            if line_count < min_lines or char_count < min_chars:
                pu_ok, pu_txt, pu_err = paddle.extract_text(im_paddle, table_boxes=text_filter_boxes)
                if pu_ok and (pu_txt or "").strip():
                    p_txt = _merge_text_lines(p_txt, pu_txt)
                    p_ok = True
                    fallback_used = True
                elif pu_err:
                    notes.append(f"Paddle unmasked p{pi}: {pu_err}")

        # Final rescue pass for standard text: no table filtering.
        if (not p_ok) or (not (p_txt or "").strip()):
            pr_ok, pr_txt, pr_err = paddle.extract_text(im_paddle, table_boxes=None)
            if pr_ok and (pr_txt or "").strip():
                p_txt = _merge_text_lines(p_txt, pr_txt)
                p_ok = True
                rescue_used = True
            elif pr_err:
                notes.append(f"Paddle rescue p{pi}: {pr_err}")

        # DocTR fallback: only when Paddle (including its own fallback/rescue
        # passes above) still produced nothing usable.
        doctr_used = False
        if (not p_ok) or (not (p_txt or "").strip()):
            try:
                dt_ok, dt_txt, dt_err = doctr_extraction.extract_text(im_paddle)
            except Exception as exc:
                dt_ok, dt_txt, dt_err = False, "", str(exc)

            if dt_ok and (dt_txt or "").strip():
                p_txt = _merge_text_lines(p_txt, dt_txt)
                p_ok = True
                doctr_used = True
            elif dt_err:
                notes.append(f"DocTR p{pi}: {dt_err}")

        d_ok = False
        d_err = ""
        page_note_parts: List[str] = []
        if doctr_used:
            page_note_parts.append("text_source=doctr_fallback")

        if p_ok and p_txt.strip():
            paddle_ok_any = True
            paddle_text_parts.append(f"\n===== Page {pi} =====\n{p_txt}")
        elif p_err:
            notes.append(f"Paddle p{pi}: {p_err}")

        page_standard_text = (p_txt or "").strip()
        source = "none"
        if p_ok and (p_txt or "").strip():
            source = "paddle"
        # A blank text page is not necessarily a processing failure.
        # Count failures only when PaddleOCR fails for the page.
        if not page_standard_text and (not p_ok):
            pages_failed += 1

        standard_pages.append(
            {
                "file": rel_file,
                "page": pi,
                "source": source,
                "text": page_standard_text,
            }
        )

        # Table extraction: TATR (primary, general-purpose neural detector) ->
        # DeepSeek VLM (fallback when TATR's structure is unreliable) ->
        # OCR-position-clustering (last resort). TATR runs its own table
        # detection rather than depending on the OpenCV line-map heuristic
        # (table_boxes/has_table_regions), since that heuristic requires
        # visible ruling lines and would miss exactly the borderless/dynamic
        # tables TATR exists to handle. table_boxes is still used further
        # below for Paddle text masking, a separate concern.
        d_ok = False
        d_err = ""
        table_ok_any = False
        table_source = ""

        try:
            tatr_boxes = table_transformer_extraction.detect_tables_tatr(im_proc)
        except Exception as exc:
            tatr_boxes = []
            notes.append(f"TATR detect p{pi}: {exc}")

        if tatr_boxes or has_table_regions:
            if tatr_boxes:
                try:
                    t_ok, t_tables, t_err = table_transformer_extraction.extract_tables_from_image_tatr(
                        im_proc, table_boxes=tatr_boxes
                    )
                except Exception as exc:
                    t_ok, t_tables, t_err = False, [], str(exc)

                if t_ok and t_tables:
                    table_count += _append_page_tables(pi, t_tables)
                    table_ok_any = True
                    table_source = "tatr"
                elif t_err:
                    notes.append(f"TATR p{pi}: {t_err}")

            if not table_ok_any:
                d_ok, tables, d_err = deepseek.extract_tables_from_image(im_proc)
                if d_ok and tables and _is_deepseek_table_result_reliable(tables):
                    table_count += _append_page_tables(pi, tables)
                    deepseek_ok_any = True
                    table_ok_any = True
                    table_source = "deepseek"
                elif d_err:
                    notes.append(f"DeepSeek p{pi}: {d_err}")

            if not table_ok_any:
                fallback_rows: List[List[str]] = []
                for tb in (tatr_boxes or table_boxes):
                    fallback_rows.extend(
                        _reconstruct_table_from_ocr_boxes(
                            im_paddle,
                            (int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3])),
                            paddle,
                        )
                    )
                if fallback_rows:
                    table_count += _append_page_tables(pi, [{"headers": [], "rows": fallback_rows}])
                    table_ok_any = True
                    table_source = "ocr_fallback"
        else:
            page_note_parts.append("table=skipped_no_table_detected")

        if deepseek_score_triggered:
            page_note_parts.append(f"deepseek=score_fallback_trigger_{table_score:.3f}")

        if mask_ratio > max_mask_ratio:
            page_note_parts.append(f"mask=skipped_ratio_{mask_ratio:.2f}")
        if fallback_used:
            page_note_parts.append("paddle=unmasked_fallback")
        if rescue_used:
            page_note_parts.append("paddle=unrestricted_rescue")
        if p_err:
            page_note_parts.append(f"paddle={p_err}")
        if d_err:
            page_note_parts.append(f"deepseek={d_err}")
        if table_source:
            page_note_parts.append(f"table_source={table_source}")
        page_note_parts.append(f"tatr_regions={len(tatr_boxes)}")
        page_note_parts.append(f"table_regions={len(table_boxes)}")

        page_rows.append(
            {
                "file": rel_file,
                "page": pi,
                "table_score": round(float(table_score), 3),
                "table_regions": len(table_boxes),
                "paddle_ok": bool(p_ok and p_txt.strip()),
                "deepseek_ok": bool(d_ok),
                "page_note": " | ".join(page_note_parts),
            }
        )

        # Rebuild standard text from post-processed page text.
    paddle_sections: List[str] = []
    for p in standard_pages:
        page_no = p.get("page", "")
        page_text = str(p.get("text", "") or "").strip()
        if page_text:
            paddle_sections.append(f"===== Page {page_no} =====\n{page_text}")

    paddle_text = _join_non_empty(paddle_sections)
    combined_standard = paddle_text

    write_text(os.path.join(doc_out, "standard_text_paddle.txt"), paddle_text)

    for legacy_name in ("standard_text_combined.txt", "ocr_result.csv"):
        legacy_path = os.path.join(doc_out, legacy_name)
        if os.path.isfile(legacy_path):
            try:
                os.remove(legacy_path)
            except Exception:
                pass

    meta = {
        "file": rel_file,
        "pages_expected": expected_pages,
        "pages_processed": len(images),
        "pages_failed": pages_failed,
        "paddle": {"ok": paddle_ok_any},
        "deepseek": {"ok": deepseek_ok_any},
        "standard_chars": len(combined_standard),
        "tables_found": table_count,
        "table_rows_written": len(table_rows),
        "notes": notes,
    }
    _ensure_dir(doc_out)
    with open(os.path.join(doc_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _write_per_doc_exports(
        page_data=page_rows,
        table_data=table_rows,
        standard_page_data=standard_pages,
        paddle_text=paddle_text,
        combined_text=combined_standard,
        pages_processed=len(images),
        pages_expected=expected_pages,
        pages_failed_count=pages_failed,
    )

    return OcrDocResult(
        file=rel_file,
        label=label,
        paddle_ok=paddle_ok_any,
        deepseek_ok=deepseek_ok_any,
        pages=len(images),
        standard_chars=len(combined_standard),
        table_chars=len(table_rows),
        note="; ".join(notes[:8]),
    )


def write_summary_csv(results: List[OcrDocResult], output_csv: str) -> None:
    _ensure_dir(os.path.dirname(output_csv))
    headers = [
        "file",
        "label",
        "paddle_ok",
        "deepseek_ok",
        "pages",
        "standard_chars",
        "table_chars",
        "note",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "file": r.file,
                    "label": r.label,
                    "paddle_ok": r.paddle_ok,
                    "deepseek_ok": r.deepseek_ok,
                    "pages": r.pages,
                    "standard_chars": r.standard_chars,
                    "table_chars": r.table_chars,
                    "note": r.note,
                }
            )


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
    env_paths = [
        os.path.join(script_dir, ".env.environment"),
        os.path.join(script_dir, ".env.extraction"),
        os.path.join(script_dir, ".env"),
        os.path.join(script_dir, ".env.example"),
    ]
    env_loaded_paths = _load_env_files_if_available(env_paths)

    parser = argparse.ArgumentParser(description="OCR documents with PaddleOCR + DeepSeek")
    parser.add_argument(
        "--report",
        default=os.path.join(script_dir, "digital_vs_scanned_report.csv"),
        help="Path to digital-vs-scanned CSV report",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(script_dir, "OCR Results"),
        help="Directory to store OCR outputs",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Optional input directory for scanned images/PDFs (skips report)",
    )
    parser.add_argument(
        "--extensions",
        default="",
        help="Optional extensions filter for --input-dir (e.g. .jpg,.jpeg)",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Process first N docs (0 = all)")
    parser.add_argument("--pdf-dpi", type=int, default=200, help="PDF render DPI")
    parser.add_argument("--max-pdf-pages", type=int, default=0, help="Max pages per PDF (0 = all)")
    parser.add_argument("--deepseek-timeout", type=int, default=120, help="DeepSeek timeout seconds")
    parser.add_argument("--deskew", action="store_true", help="Deskew pages before OCR")
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskew even for scanned input")
    parser.add_argument("--swinunet-denoise", action="store_true", help="Use the SwinUNet model for denoising instead of classical OpenCV denoising")
    parser.add_argument("--no-swinunet-denoise", action="store_true", help="Disable SwinUNet denoising, use classical OpenCV denoising instead")
    parser.add_argument("--perspective-correct", action="store_true", help="Correct perspective for phone/camera-captured pages")
    parser.add_argument("--no-perspective-correct", action="store_true", help="Disable perspective correction")
    parser.add_argument("--shadow-removal", action="store_true", help="Normalize uneven lighting/shadows for phone-camera pages")
    parser.add_argument("--no-shadow-removal", action="store_true", help="Disable shadow removal")

    args = parser.parse_args()

    use_input_dir = bool((args.input_dir or "").strip())
    if use_input_dir:
        ext_filter = _normalize_extensions(args.extensions)
        docs = load_docs_from_dir(args.input_dir, workspace_root, extensions=ext_filter)
        if not docs:
            print("No input files found in directory. Nothing to process.")
            return
    else:
        if not os.path.isfile(args.report):
            raise FileNotFoundError(f"Detection report not found: {args.report}")
        docs = load_digital_docs(args.report, workspace_root)
        if not docs:
            print("No digital files found in report. Nothing to process.")
            return

    if args.max_files and args.max_files > 0:
        docs = docs[: args.max_files]

    _ensure_dir(args.output_dir)

    print(f"Docs selected: {len(docs)}")
    if env_loaded_paths:
        print("Loaded environment values from:")
        for p in env_loaded_paths:
            print(f"- {p}")
    print("Initializing OCR engines ...")

    # On Windows, loading Paddle before torch can break local DeepSeek-OCR DLL loading.
    deepseek = DeepSeekTableEngine(timeout_sec=args.deepseek_timeout)
    paddle = PaddleEngine()

    print(f"- Paddle ready   : {paddle.ready}")
    print(f"- Paddle use_gpu : {paddle.use_gpu}")
    print(f"- DeepSeek ready : {deepseek.ready}")
    print(f"- DeepSeek use_gpu : {deepseek.use_gpu}")
    print(f"- DeepSeek mode  : {deepseek.backend}")

    # DeepSeek is used for table extraction; if it's unavailable, keep going —
    # text extraction still works normally, and tables are simply marked
    # unavailable per document instead of aborting the whole run.
    if not deepseek.ready:
        print(
            "WARNING: DeepSeek is not ready — table extraction will be skipped for "
            "this run. Text extraction will still proceed normally. "
            "For free local mode set DEEPSEEK_TABLE_BACKEND=local and "
            "DEEPSEEK_TABLE_MODEL=deepseek-ai/DeepSeek-OCR."
        )
    else:
        preflight_ok, preflight_err = deepseek.validate_access()
        if not preflight_ok:
            print(f"WARNING: DeepSeek preflight failed: {preflight_err} — table extraction will be skipped for this run.")

    deskew = True
    if args.no_deskew:
        deskew = False
    if args.deskew:
        deskew = True
    
    swinunet = os.getenv("OCR_SWINUNET_DENOISE", "0") != "0"
    if args.no_swinunet_denoise:
        swinunet = False
    if args.swinunet_denoise:
        swinunet = True
    
    perspective: Optional[bool] = None
    _perspective_env = os.getenv("OCR_PERSPECTIVE_CORRECT")
    if _perspective_env is not None:
        perspective = _perspective_env != "0"
    if args.no_perspective_correct:
        perspective = False
    if args.perspective_correct:
        perspective = True

    shadow_removal: Optional[bool] = None
    _shadow_env = os.getenv("OCR_SHADOW_REMOVAL")
    if _shadow_env is not None:
        shadow_removal = _shadow_env != "0"
    if args.no_shadow_removal:
        shadow_removal = False
    if args.shadow_removal:
        shadow_removal = True

    results: List[OcrDocResult] = []

    for i, item in enumerate(docs, start=1):
        print(f"[{i}/{len(docs)}] OCR -> {item['file']}")
        try:
            res = process_document(
                item=item,
                output_dir=args.output_dir,
                paddle=paddle,
                deepseek=deepseek,
                pdf_dpi=args.pdf_dpi,
                max_pdf_pages=args.max_pdf_pages,
                deskew=deskew,
                perspective=perspective,
                shadow_removal=shadow_removal,
                swinunet=swinunet,
            )
        except Exception as exc:
            notes = _short_error_text(traceback.format_exc())
            print(f"ERROR: document failed -> {item['file']} :: {notes}")
            label = item.get("file_type", "digital") or "digital"
            res = OcrDocResult(
                file=item["file"],
                label=label,
                paddle_ok=False,
                deepseek_ok=False,
                pages=0,
                standard_chars=0,
                table_chars=0,
                note=f"fatal: {exc}",
            )
        results.append(res)

    summary_csv = os.path.join(args.output_dir, "OCR_Summary.csv")
    write_summary_csv(_build_cumulative_results(args.output_dir), summary_csv)
    all_text_path, all_tables_path = write_consolidated_outputs(docs, args.output_dir)

    ok_standard = sum(1 for r in results if r.paddle_ok)
    ok_table = sum(1 for r in results if r.deepseek_ok)

    print("\nDone")
    print("-" * 60)
    print(f"Processed docs         : {len(results)}")
    print(f"Standard OCR success   : {ok_standard}")
    print(f"Table OCR success      : {ok_table}")
    print(f"Summary CSV            : {summary_csv}")
    print(f"All standard text      : {all_text_path}")
    print(f"All tables CSV         : {all_tables_path}")


if __name__ == "__main__":
    main()
