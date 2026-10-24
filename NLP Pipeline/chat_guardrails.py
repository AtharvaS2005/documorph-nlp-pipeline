"""
Guardrails for document question answering (POST /answer).

Each check is deterministic and cheap, so it runs on every request:

- looks_like_injection: text that tries to give the model instructions
  ("ignore the previous instructions", "you are now...", fake role
  markers). Applied to the question and to every retrieved passage --
  documents are untrusted input, so a planted instruction in an uploaded
  file must not steer the answer.
- is_off_topic_request: asks for a task rather than for information in
  the documents (poems, code, translation, role-play). The model is also
  asked to judge this itself; the pattern check catches the obvious cases
  before spending a model call.
- neutralize_delimiters: stops passage text from closing the <document>
  wrapper the prompt puts around it.
- unsupported_numbers: every number in an answer must appear in the
  passages it cites, or be simple arithmetic on them (sum/difference) or a
  small count. Numbers are where a wrong answer does the most damage, and
  this is checkable without a second model call. Non-numeric claims are
  left to the prompt.
- redact_pii: masks high-risk identifiers (payment card numbers passing a
  Luhn check, Aadhaar, PAN, US SSN) in anything returned to the caller.
"""

import re
from typing import Iterable, List, Set, Tuple

_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any|your|these|those)\b[^.\n]{0,25}\b(instructions?|prompts?|rules?|directions?|guidelines?)\b",
    r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,30}\b(instructions?|prompts?|rules?|guidelines?)\b[^.\n]{0,15}\b(above|before|earlier|so far|given to you)\b",
    r"\byou are now\b",
    r"\b(act|behave|respond|answer)\s+as\s+(if\s+you\s+(are|were)|an?\s+(different|new|unrestricted))",
    r"\bpretend\s+(to\s+be|you\s+are)\b",
    r"\b(new|updated|real|actual|hidden)\s+(system\s+)?instructions?\s*:",
    r"\bsystem\s+prompt\b",
    r"\b(developer|jailbreak|dan)\s+mode\b",
    r"^\s*(system|assistant|developer)\s*:",
    r"<\s*/?\s*(system|assistant|instructions?|document)\b",
    r"\b(reveal|print|repeat|show)\b[^.\n]{0,30}\b(system prompt|your instructions|hidden instructions)\b",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)

_OFF_TOPIC_PATTERNS = [
    r"\b(write|compose|create|generate|make)\b[^.\n]{0,30}\b(poem|story|song|essay|joke|limerick|haiku|rap|novel|script|tweet|blog)\b",
    r"\btell\s+me\s+a\s+(joke|story)\b",
    r"\b(write|generate|give me)\b[^.\n]{0,20}\b(code|program|function|sql|script|regex)\b",
    r"\btranslate\b",
    r"\b(role[\s-]?play|roleplay)\b",
]
_OFF_TOPIC_RE = re.compile("|".join(f"(?:{p})" for p in _OFF_TOPIC_PATTERNS), re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    return bool(text) and _INJECTION_RE.search(text) is not None


def strip_injections(text: str) -> Tuple[str, int]:
    """Remove the lines (or, within a line, the sentences) of a passage that
    try to instruct the model, keeping the rest. Returns (text, removed)."""
    kept, removed = [], 0
    for line in (text or "").splitlines():
        if not looks_like_injection(line):
            kept.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        clean = [s for s in parts if not looks_like_injection(s)]
        removed += len(parts) - len(clean)
        if clean:
            kept.append(" ".join(clean))
    return "\n".join(kept), removed


def is_off_topic_request(question: str) -> bool:
    return bool(question) and _OFF_TOPIC_RE.search(question) is not None


def neutralize_delimiters(text: str) -> str:
    return re.sub(r"<(\s*/?\s*)document", r"&lt;\1document", text or "", flags=re.IGNORECASE)


_NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> List[float]:
    out = []
    for m in _NUMBER_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def unsupported_numbers(answer: str, source_texts: Iterable[str]) -> List[float]:
    """Numbers in the answer that the sources don't support. Allowed: a
    number present in a source, a small count (0-10), or the sum or
    difference of two source numbers (e.g. "160 more than" from 250 and
    90). Citation markers like [2] are ignored."""
    cleaned = re.sub(r"\[\d+\]", " ", answer or "")
    known: Set[float] = set()
    for t in source_texts:
        known.update(abs(n) for n in _numbers(t))
    pool = sorted(known)[:300]  # bounds the pairwise check

    def supported(n: float) -> bool:
        n = abs(n)
        if any(abs(n - k) < 0.005 for k in pool):
            return True
        if n <= 10 and float(n).is_integer():
            return True
        for i, a in enumerate(pool):
            for b in pool[i:]:
                if abs((a + b) - n) < 0.005 or abs(abs(a - b) - n) < 0.005:
                    return True
        return False

    return [n for n in _numbers(cleaned) if not supported(n)]


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_AADHAAR_RE = re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")


def redact_pii(text: str) -> Tuple[str, int]:
    """Mask high-risk identifiers. Returns (text, number of redactions)."""
    if not text:
        return text, 0
    count = 0

    def card(m):
        nonlocal count
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            count += 1
            return "[card number hidden]"
        return m.group(0)

    def sub(pattern, label, s):
        nonlocal count
        def repl(_m):
            nonlocal count
            count += 1
            return label
        return pattern.sub(repl, s)

    text = _CARD_RE.sub(card, text)
    text = sub(_AADHAAR_RE, "[Aadhaar number hidden]", text)
    text = sub(_PAN_RE, "[PAN hidden]", text)
    text = sub(_SSN_RE, "[SSN hidden]", text)
    return text, count
