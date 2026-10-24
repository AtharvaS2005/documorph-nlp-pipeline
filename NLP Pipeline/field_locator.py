"""
Find where each extracted field value appears on the original document, so
the UI can highlight it on a page preview.

Extraction itself can't supply positions: OCR runs on deskewed,
perspective-corrected, shadow-removed copies of each page, so its boxes
don't line up with the file the user uploaded. Instead, once the values are
known, they are located on the untouched original:

- PDF pages with a text layer: exact word boxes from PyMuPDF.
- Scanned PDF pages and images: one OCR pass over the unmodified page
  (EXIF orientation applied, as a browser would), using line boxes narrowed
  to the matched characters.

Boxes are normalised to 0-1 of the page width/height with the origin at
the top left, so they don't depend on the resolution the page is shown at.

When a value occurs more than once (a total repeated on several lines, a
date in the header and the footer), the occurrence nearest a label that
matches the field name wins: "total_amount" prefers the 250.00 next to
"Total Amount:" over a line-item 250.00.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# How far back from a match to look for its label, in characters.
_LABEL_WINDOW = 60
_MAX_PAGES = 30

Segment = Dict[str, Any]  # {"text": str, "box": (x0, y0, x1, y1) normalised}


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _value_variants(value: str) -> List[str]:
    """Spellings a value may take on the page: as extracted, without
    thousands separators, and with or without trailing .00 / currency."""
    v = _norm_space(str(value))
    out = [v]
    bare = re.sub(r"^[^\w]+|[^\w.]+$", "", v)  # strip leading symbols / trailing units
    out.append(bare)
    nums = re.sub(r"(?<=\d),(?=\d{3}\b)", "", bare)
    out.append(nums)
    m = re.fullmatch(r"(-?\d+)(?:\.(\d+))?", nums)
    if m:
        whole, frac = m.group(1), m.group(2)
        n = int(whole)
        with_commas = f"{n:,}"
        if frac is None or set(frac) == {"0"}:
            out += [whole, f"{whole}.00", with_commas, f"{with_commas}.00"]
        else:
            out += [f"{with_commas}.{frac}"]
    seen, result = set(), []
    for x in out:
        if x and len(x) >= 2 and x.lower() not in seen:
            seen.add(x.lower())
            result.append(x)
    return sorted(result, key=len, reverse=True)


def _label_words(field: str) -> List[str]:
    words = [w for w in re.split(r"[_\W]+", field.lower()) if len(w) >= 2]
    stop = {"no", "of", "the", "to", "id"}
    return [w for w in words if w not in stop] or words


def _page_string(segments: List[Segment]) -> Tuple[str, List[Tuple[int, int, int]]]:
    """Join segments into one searchable string, remembering which character
    range belongs to which segment."""
    parts, spans, pos = [], [], 0
    for i, seg in enumerate(segments):
        t = seg["text"]
        spans.append((pos, pos + len(t), i))
        parts.append(t)
        pos += len(t) + 1
    return " ".join(parts), spans


def _box_for_range(segments, spans, start, end):
    boxes = []
    for s0, s1, i in spans:
        if s1 <= start or s0 >= end:
            continue
        x0, y0, x1, y1 = segments[i]["box"]
        if segments[i].get("line"):
            # OCR line box: keep only the matched characters' share of the width
            n = max(1, s1 - s0)
            a = (max(start, s0) - s0) / n
            b = (min(end, s1) - s0) / n
            x0, x1 = x0 + (x1 - x0) * a, x0 + (x1 - x0) * b
        boxes.append((x0, y0, x1, y1))
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _find(pages: List[List[Segment]], field: str, value: Any) -> Optional[Dict[str, Any]]:
    if value is None or isinstance(value, (dict, list)) or not str(value).strip():
        return None
    labels = _label_words(field)
    best = None  # (score, -page, -start, result)
    for p_idx, segments in enumerate(pages):
        if not segments:
            continue
        text, spans = _page_string(segments)
        low = text.lower()
        for variant in _value_variants(str(value)):
            pat = re.compile(r"(?<![\w.])" + re.escape(variant.lower()) + r"(?![\w])")
            for m in pat.finditer(low):
                box = _box_for_range(segments, spans, m.start(), m.end())
                if box is None:
                    continue
                before = low[max(0, m.start() - _LABEL_WINDOW):m.start()]
                score = sum(1 for w in labels if w in before) * 10 + len(variant) / 100.0
                key = (score, -p_idx, -m.start())
                if best is None or key > best[0]:
                    best = (key, {
                        "page": p_idx + 1,
                        "box": [round(v, 4) for v in box],
                        "matched": text[m.start():m.end()],
                    })
            if best is not None and best[1]["page"] == p_idx + 1:
                break  # a longer spelling already matched on this page
    return best[1] if best else None


def _text_layer_segments(page) -> List[Segment]:
    w, h = float(page.rect.width), float(page.rect.height)
    segs = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words", sort=True):
        if word.strip():
            segs.append({"text": word, "box": (x0 / w, y0 / h, x1 / w, y1 / h)})
    return segs


def _ocr_segments(image_bgr, paddle) -> List[Segment]:
    if paddle is None or not getattr(paddle, "ready", False):
        return []
    import OCR_Text_Extraction as ocr_pipeline  # noqa: E402

    h, w = image_bgr.shape[:2]
    try:
        try:
            result = paddle.ocr.ocr(image_bgr, cls=True)
        except TypeError:
            result = paddle.ocr.ocr(image_bgr)
    except Exception:
        return []
    segs = []
    for item in ocr_pipeline._extract_paddle_items(result):
        box = ocr_pipeline._box_from_points(item.get("box"))
        text = _norm_space(item.get("text") or "")
        if box is None or not text:
            continue
        x0, y0, x1, y1 = box
        segs.append({"text": text, "box": (x0 / w, y0 / h, x1 / w, y1 / h), "line": True})
    # reading order: top to bottom, then left to right
    segs.sort(key=lambda s: (round(s["box"][1], 2), s["box"][0]))
    return segs


def _load_image_bgr(path: str):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.array(im)[:, :, ::-1].copy()


def page_segments(file_path: str, paddle=None) -> Tuple[List[List[Segment]], List[str]]:
    """Searchable segments for each page, and how each page was read
    ("text-layer" or "ocr")."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _IMAGE_EXTS:
        return [_ocr_segments(_load_image_bgr(file_path), paddle)], ["ocr"]
    if ext != ".pdf":
        return [], []

    import fitz  # type: ignore
    import numpy as np

    pages, methods = [], []
    with fitz.open(file_path) as doc:
        for p_idx in range(min(doc.page_count, _MAX_PAGES)):
            page = doc.load_page(p_idx)
            segs = _text_layer_segments(page)
            if segs:
                pages.append(segs)
                methods.append("text-layer")
                continue
            pix = page.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            bgr = img[:, :, 2::-1].copy() if pix.n >= 3 else np.repeat(img, 3, axis=2)
            pages.append(_ocr_segments(bgr, paddle))
            methods.append("ocr")
    return pages, methods


def locate_fields(file_path: str, fields: Dict[str, Any], paddle=None) -> Dict[str, Any]:
    """{field: {"page", "box", "matched", "method"}} for every field whose
    value could be found on the original document; fields that can't be
    found are left out."""
    if not isinstance(fields, dict) or not fields:
        return {}
    try:
        pages, methods = page_segments(file_path, paddle)
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    for field, value in fields.items():
        hit = _find(pages, str(field), value)
        if hit:
            hit["method"] = methods[hit["page"] - 1]
            out[str(field)] = hit
    return out
