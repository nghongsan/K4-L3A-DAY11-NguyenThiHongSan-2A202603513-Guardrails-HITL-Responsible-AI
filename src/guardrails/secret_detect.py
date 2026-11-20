"""
Shared protected-value detector (Blue Team).

Used by input guardrail (confirmation attacks), output guardrail (leaks) and the
egress gate. Catches the exact demo secrets from data/protected/vinbank_secrets.json
plus common obfuscations: separators/spacing, reversed, base64, hex, ROT13.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import re

from core.config import DEMO_SECRETS


def _compact(text: str) -> str:
    """Lowercase and drop everything except letters/digits."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _secret_needles() -> list[str]:
    needles = set()
    for secret in DEMO_SECRETS:
        s = str(secret)
        c = _compact(s)
        if len(c) < 5:
            continue
        needles.update({
            c,
            c[::-1],
            _compact(codecs.encode(s, "rot13")),
            s.encode().hex(),
        })
        # Distinctive prefix (e.g. "vinbanksecret") catches partial leaks
        if len(c) >= 12:
            needles.add(c[: max(8, len(c) * 2 // 3)])
    return sorted(needles)


_NEEDLES = _secret_needles()
_B64_RE = re.compile(r"[A-Za-z0-9+/]{8,}={0,2}")
_HEX_RE = re.compile(r"(?:[0-9a-fA-F]{2}[\s:]?){6,}")


def _decoded_candidates(text: str) -> list[str]:
    out = []
    for m in _B64_RE.findall(text or ""):
        padded = m + "=" * (-len(m) % 4)
        try:
            out.append(base64.b64decode(padded, validate=False).decode("utf-8", "ignore"))
        except (binascii.Error, ValueError):
            pass
    for m in _HEX_RE.findall(text or ""):
        try:
            out.append(bytes.fromhex(re.sub(r"[\s:]", "", m)).decode("utf-8", "ignore"))
        except ValueError:
            pass
    return out


_DOT_WORD_RE = re.compile(r"[\[\(\{<]\s*(?:dot|period|colon|dash|hyphen)\s*[\]\)\}>]|\s+(?:dot|period)\s+", re.I)
_HINT_RE = re.compile(
    r"\b(?:password|passcode|key|token|host|secret|credential)s?\b[^.\n]{0,60}"
    r"(?:\b(?:starts?|begins?|ends?|prefix(?:ed)?|suffix(?:ed)?)\b(?:\s+(?:with|in|is))?\s*['\"“‘`]"
    r"|\b(?:first|last)\s+\d+\s+(?:characters?|letters?|digits?|chars?)\s+(?:are|is|=|:))",
    re.I,
)


def contains_secret_hint(text: str) -> bool:
    """Partial-disclosure hints ("the password starts with 'adm'...")."""
    return bool(_HINT_RE.search(text or ""))


def contains_protected_secret(text: str) -> bool:
    """True if ``text`` contains a protected value, verbatim or obfuscated."""
    if not text:
        return False
    hay = _compact(text)
    hay_nodots = _compact(_DOT_WORD_RE.sub("", text))
    if any(n in hay or n in hay_nodots for n in _NEEDLES):
        return True
    # base64 split by whitespace: also try the whitespace-stripped text
    for decoded in _decoded_candidates(text) + _decoded_candidates(re.sub(r"\s+", "", text)):
        d = _compact(decoded)
        if any(n in d for n in _NEEDLES):
            return True
    return False
