"""Fix visual-order Arabic text produced by PDF extractors that don't run
the Unicode Bidi algorithm (pdfplumber/pypdf return glyphs in physical
left-to-right stream order; Arabic-letter runs come out character-reversed).

This is a real, principled fix (mirroring each RTL letter-run back to
logical reading order) — not a guess at content. It is NOT a full Unicode
BiDi implementation:
- Digit groups (Western or Arabic-Indic) are weak-LTR and are extracted
  already in correct reading order; this module treats them as atomic and
  never reverses their internal digit order.
- A run is only reversed when it is anchored by actual Arabic *letters* on
  both ends. Punctuation/digits sitting between an English word and an
  Arabic word (e.g. a parenthesized number in a bilingual heading line)
  are deliberately left untouched rather than guessed at, to avoid
  corrupting content -- this can leave a rare cosmetic artifact (e.g. a
  mirrored bracket not flipped) in heading lines, a known, documented
  limitation, not a silent "fix".
- Very rare ligature-level glyph reordering in the source PDF (e.g. some
  lam-alef sequences) can still slip through. Any such residual artifact
  must be caught by validation / spot-checking, never silently patched.
"""
from __future__ import annotations

import re

_ARABIC = r"\u0600-\u06FF"
_RTL_RUN_RE = re.compile(rf"[{_ARABIC}](?:[{_ARABIC}\s]*[{_ARABIC}])?")

# Digits are atomic tokens (order preserved); everything else in a matched
# run is a single Arabic-letter/space character (order gets reversed).
_TOKEN_RE = re.compile(r"[0-9\u0660-\u0669]+|.", re.DOTALL)


def _reverse_run(run: str) -> str:
    tokens = _TOKEN_RE.findall(run)
    return "".join(reversed(tokens))


def fix_rtl_line(line: str) -> str:
    """Reverse each Arabic-letter-anchored run in a line to logical
    reading order. Non-Arabic text (English) and digit groups' internal
    order are left untouched.
    """
    return _RTL_RUN_RE.sub(lambda m: _reverse_run(m.group(0)), line)


def fix_rtl_text(text: str) -> str:
    """Apply fix_rtl_line to every line of a multi-line text block."""
    if not text:
        return text
    return "\n".join(fix_rtl_line(line) for line in text.splitlines())
