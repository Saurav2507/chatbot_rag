import re
import unicodedata
from collections import Counter
from typing import Iterable

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:
    sanscript = None
    transliterate = None


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def contains_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))


def normalize_unicode(text: str) -> str:
    """Normalize text without removing Sanskrit diacritics or Devanagari marks."""
    text = unicodedata.normalize("NFC", text or "")
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\u0964", "।").replace("\u0965", "॥")
    return text


def clean_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_sanskrit_text(text: str) -> str:
    return clean_whitespace(normalize_unicode(text))


def _page_edge_lines(text: str, edge_size: int = 2) -> Iterable[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:edge_size] + lines[-edge_size:]:
        if 4 <= len(line) <= 160:
            yield line


def remove_repeated_headers_footers(pages: list[dict]) -> list[dict]:
    """Remove exact repeated top/bottom lines while preserving page content."""
    if len(pages) < 3:
        return pages

    counts = Counter()
    for page in pages:
        counts.update(set(_page_edge_lines(page.get("text", ""))))

    min_count = max(3, len(pages) // 2)
    repeated = {line for line, count in counts.items() if count >= min_count}
    if not repeated:
        return pages

    cleaned_pages = []
    for page in pages:
        lines = [line for line in page["text"].splitlines() if line.strip() not in repeated]
        cleaned = {**page, "text": clean_whitespace("\n".join(lines))}
        if cleaned["text"]:
            cleaned_pages.append(cleaned)
    return cleaned_pages


def transliteration_variants(text: str) -> list[str]:
    """Create conservative Sanskrit script variants for retrieval only."""
    text = normalize_sanskrit_text(text)
    if not text:
        return []

    variants = [text]
    if sanscript is None or transliterate is None:
        return variants

    try:
        if contains_devanagari(text):
            variants.append(transliterate(text, sanscript.DEVANAGARI, sanscript.IAST))
            variants.append(transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS))
        else:
            for scheme in (sanscript.IAST, sanscript.ITRANS, sanscript.HK, sanscript.SLP1):
                try:
                    variants.append(transliterate(text, scheme, sanscript.DEVANAGARI))
                except Exception:
                    continue
    except Exception:
        pass

    deduped = []
    seen = set()
    for variant in variants:
        variant = normalize_sanskrit_text(variant)
        if variant and variant not in seen:
            deduped.append(variant)
            seen.add(variant)
    return deduped


def retrieval_text(text: str) -> str:
    """Text used for embeddings: original plus script/transliteration variants."""
    return "\n".join(transliteration_variants(text))


def lexical_terms(text: str) -> set[str]:
    terms = set()
    for variant in transliteration_variants(text):
        terms.update(token.lower() for token in TOKEN_RE.findall(variant) if len(token) > 1)
    return terms
