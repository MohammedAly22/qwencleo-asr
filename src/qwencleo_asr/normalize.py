"""Egyptian-Arabic-aware text normalization.

Used for cleaning transcripts and for computing fair WER/CER (so the model is
not penalized for orthographic variants that carry no meaning in Egyptian
Arabic). Latin-script tokens (English code-switch terms) are preserved.
"""
import re

# Arabic diacritics (harakat) + tatweel/kashida.
_AR_DIACRITICS = re.compile("[\u064B-\u0652\u0670\u0640\u0653-\u065F\u0610-\u061A]")
# Anything that is not a word char, Arabic block char, or whitespace -> drop.
_PUNCT = re.compile(r"[^\w؀-ۿ\s]", re.UNICODE)
_WS = re.compile(r"\s+")
# Arabic-script punctuation inside the Arabic Unicode block (survives _PUNCT):
# comma, semicolon, question mark, percent/decimal/thousands marks, ellipsis.
_AR_PUNCT = re.compile("[،؛؟٪٫٬…«»]")

_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})


def strip_diacritics(text: str) -> str:
    """Remove harakat and tatweel only."""
    return _AR_DIACRITICS.sub("", text)


def normalize(text: str, *, drop_punct: bool = True, lower_latin: bool = True) -> str:
    """Normalize a transcript for display or scoring.

    - removes diacritics + tatweel
    - unifies alef hamza variants -> bare alef
    - ya/alef-maqsura -> ya, ta-marbuta -> ha, hamza-on-seat -> base
    - optionally strips punctuation and lowercases Latin script
    - collapses whitespace
    """
    if not text:
        return ""
    t = strip_diacritics(text)
    t = t.translate(_ALEF)
    t = (t.replace("ى", "ي").replace("ة", "ه")
           .replace("ؤ", "و").replace("ئ", "ي"))
    if drop_punct:
        t = _AR_PUNCT.sub(" ", t)
        t = _PUNCT.sub(" ", t)
    if lower_latin:
        # only Latin letters are affected by lower(); Arabic is caseless
        t = t.lower()
    t = _WS.sub(" ", t).strip()
    return t
