"""Helpers for messy bilingual (Arabic/English) invoice text.

Shared by the invoice checker, the training pipeline and the API, so the
pickled model can always find `normalize_text` on import.
"""
from __future__ import annotations

import re

# Arabic-Indic (U+0660..0669) and Eastern Arabic / Persian (U+06F0..06F9) digits
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
_DIGIT_MAP[ord("٫")] = "."   # Arabic decimal separator
_DIGIT_MAP[ord("٬")] = ","   # Arabic thousands separator
_DIGIT_MAP[ord("،")] = ","   # Arabic comma

_ARABIC_DIGITS_RE = re.compile(r"[٠-٩۰-۹]")
_WESTERN_DIGITS_RE = re.compile(r"[0-9]")
_DIACRITICS_RE = re.compile(r"[ً-ْٰ]")  # harakat
_TATWEEL = "ـ"


def to_western_digits(s: str) -> str:
    return s.translate(_DIGIT_MAP)


def digit_script(s: str | None) -> str:
    """Return 'arabic', 'western', 'mixed' or 'none' for the digits in s."""
    if not s:
        return "none"
    a = bool(_ARABIC_DIGITS_RE.search(s))
    w = bool(_WESTERN_DIGITS_RE.search(s))
    return "mixed" if a and w else "arabic" if a else "western" if w else "none"


def parse_number(value) -> float | None:
    """Parse '1,250.00', '١٬٢٥٠٫٠٠', 'AED 925', '' -> float or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = to_western_digits(str(value)).strip()
    s = re.sub(r"[^\d.,\-]", "", s)  # drop currency, units, spaces
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _norm_ar(w: str) -> str:
    w = _DIACRITICS_RE.sub("", w.replace(_TATWEEL, ""))
    w = re.sub("[إأآا]", "ا", w)
    return w.replace("ى", "ي").replace("ة", "ه")


_AR_NUM = {_norm_ar(k): v for k, v in {
    "واحد": 1, "اثنان": 2, "اثنين": 2, "ثلاثة": 3, "ثلاث": 3, "أربعة": 4, "أربع": 4,
    "خمسة": 5, "خمس": 5, "ستة": 6, "ست": 6, "سبعة": 7, "سبع": 7, "ثمانية": 8, "ثماني": 8,
    "ثمان": 8, "تسعة": 9, "تسع": 9, "عشرة": 10, "عشر": 10, "أحد": 1, "اثنا": 2, "اثني": 2,
    "عشرون": 20, "عشرين": 20, "ثلاثون": 30, "ثلاثين": 30, "أربعون": 40, "أربعين": 40,
    "خمسون": 50, "خمسين": 50, "ستون": 60, "ستين": 60, "سبعون": 70, "سبعين": 70,
    "ثمانون": 80, "ثمانين": 80, "تسعون": 90, "تسعين": 90,
    "مائة": 100, "مئة": 100, "مائتان": 200, "مئتان": 200, "مائتين": 200, "مئتين": 200,
    "ثلاثمائة": 300, "ثلاثمئة": 300, "أربعمائة": 400, "أربعمئة": 400, "خمسمائة": 500,
    "خمسمئة": 500, "ستمائة": 600, "ستمئة": 600, "سبعمائة": 700, "سبعمئة": 700,
    "ثمانمائة": 800, "ثمانمئة": 800, "تسعمائة": 900, "تسعمئة": 900,
}.items()}
_AR_THOUSAND = {_norm_ar(w) for w in ("ألف", "ألفا", "آلاف")}
_AR_TWO_THOUSAND = {_norm_ar(w) for w in ("ألفان", "ألفين")}


def arabic_words_to_number(text: str) -> float | None:
    """'فقط ثلاثة وعشرون ألفاً وستمائة وستون درهماً وثلاثون فلساً لا غير' -> 23660.30"""
    total = current = 0
    dirhams = None
    seen = False
    for raw in text.split():
        w = _norm_ar(raw)
        if w not in _AR_NUM and w.startswith("و") and len(w) > 2:
            w = w[1:]                                   # conjunction "wa-"
        if w in _AR_NUM:
            current += _AR_NUM[w]; seen = True
        elif w in _AR_THOUSAND:
            total += max(current, 1) * 1000; current = 0; seen = True
        elif w in _AR_TWO_THOUSAND:
            total += 2000; seen = True
        elif w.startswith("درهم"):
            dirhams = total + current; total = current = 0
        elif w.startswith("فلس"):
            break
    if not seen:
        return None
    if dirhams is None:                                # no "dirham" word
        return float(total + current)
    return round(dirhams + (total + current) / 100, 2)


def normalize_text(s: str) -> str:
    """Normalise a line-item description before vectorising.

    - Arabic digits -> Western digits (so '٥٠ كجم' and '50 كجم' match)
    - unify alef/yaa/taa-marbuta variants, strip diacritics and tatweel
    - lowercase, collapse whitespace
    """
    s = to_western_digits(s)
    s = s.replace(_TATWEEL, "")
    s = _DIACRITICS_RE.sub("", s)
    s = re.sub("[إأآا]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه")
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()
