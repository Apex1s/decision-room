"""Period wording: the engine's text says "quarter"; annual markets relabel it to "year"."""

from __future__ import annotations

import re

_RULES = [
    (r"\bQuarters\b", "Years"), (r"\bquarters\b", "years"),
    (r"\bQuarter\b", "Year"), (r"\bquarter\b", "year"),
    (r"\bMt/q\b", "Mt/yr"), (r"\$M/q\b", "$M/yr"), (r"\b(\d+)q\b", r"\1y"),
]


def relabel(text: str, unit: str) -> str:
    if unit != "year":
        return text
    for pat, rep in _RULES:
        text = re.sub(pat, rep, text)
    return text
