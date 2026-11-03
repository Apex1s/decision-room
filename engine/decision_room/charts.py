"""Small dependency-free SVG charts for the decision memo.

Colours come from CSS custom properties with fallbacks, so the same SVG reads
well inline in the HTML memo (which defines light and dark themes) and as a
standalone file.
"""

from __future__ import annotations

import math
from html import escape

W, H = 640, 300
PAD = dict(l=64, r=20, t=28, b=46)

STYLE = """<style>
.ax{stroke:var(--dr-grid,#d4d4d8);stroke-width:1}
.lbl{fill:var(--dr-muted,#52525b);font:12px ui-sans-serif,system-ui,sans-serif}
.ttl{fill:var(--dr-ink,#18181b);font:600 13px ui-sans-serif,system-ui,sans-serif}
.s1{fill:var(--dr-accent,#2563eb);stroke:var(--dr-accent,#2563eb)}
.s2{fill:var(--dr-muted2,#a1a1aa);stroke:var(--dr-muted2,#a1a1aa)}
.neg{fill:var(--dr-neg,#dc2626)}
.line{fill:none;stroke-width:2.5}
</style>"""


def _frame(title: str, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="{escape(title)}">{STYLE}<text class="ttl" x="{PAD["l"]}" y="18">{escape(title)}</text>'
            f'{body}</svg>')


def _scale(lo, hi, a, b):
    span = (hi - lo) or 1.0
    return lambda v: a + (v - lo) / span * (b - a)


def _nice(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo or 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks


def _yaxis(lo, hi, y, fmt="{:.0f}") -> str:
    out = []
    for t in _nice(lo, hi):
        yy = y(t)
        out.append(f'<line class="ax" x1="{PAD["l"]}" x2="{W - PAD["r"]}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                   f'<text class="lbl" x="{PAD["l"] - 8}" y="{yy + 4:.1f}" text-anchor="end">{fmt.format(t)}</text>')
    return "".join(out)


def line_chart(title: str, xs: list[float], ys: list[float], xlabel: str, ylabel: str,
               annotate: list[str] | None = None, reverse_x: bool = False) -> str:
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.15 or abs(hi) * 0.05 or 1
    lo, hi = lo - pad, hi + pad
    xa, xb = (W - PAD["r"], PAD["l"]) if reverse_x else (PAD["l"], W - PAD["r"])
    x = _scale(min(xs), max(xs), xa, xb)
    y = _scale(lo, hi, H - PAD["b"], PAD["t"])
    pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs, ys))
    body = [_yaxis(lo, hi, y, "{:.1f}"), f'<polyline class="s1 line" points="{pts}"/>']
    for i, (a, b) in enumerate(zip(xs, ys)):
        body.append(f'<circle class="s1" cx="{x(a):.1f}" cy="{y(b):.1f}" r="4"/>')
        body.append(f'<text class="lbl" x="{x(a):.1f}" y="{H - PAD["b"] + 18}" text-anchor="middle">{a:g}</text>')
        if annotate and annotate[i]:
            body.append(f'<text class="lbl" x="{x(a):.1f}" y="{y(b) - 10:.1f}" text-anchor="middle">'
                        f'{escape(annotate[i])}</text>')
    body.append(f'<text class="lbl" x="{(W + PAD["l"]) / 2}" y="{H - 8}" text-anchor="middle">{escape(xlabel)}</text>')
    body.append(f'<text class="lbl" transform="translate(16 {(H) / 2}) rotate(-90)" text-anchor="middle">'
                f'{escape(ylabel)}</text>')
    return _frame(title, "".join(body))


def grouped_bars(title: str, cats: list[str], series: dict[str, list[float]], ylabel: str,
                 zero_based: bool = False, fmt: str = "{:.0f}", highlight: int = 0) -> str:
    vals = [v for s in series.values() for v in s]
    lo = 0 if zero_based else min(vals) * 0.9
    hi = max(vals) * 1.05
    y = _scale(lo, hi, H - PAD["b"], PAD["t"])
    n, k = len(cats), len(series)
    slot = (W - PAD["l"] - PAD["r"]) / n
    bw = slot * 0.7 / k
    body = [_yaxis(lo, hi, y, fmt)]
    for i, c in enumerate(cats):
        for s_i, (name, vs) in enumerate(series.items()):
            x0 = PAD["l"] + i * slot + slot * 0.15 + s_i * bw
            body.append(f'<rect class="{"s1" if s_i == highlight else "s2"}" x="{x0:.1f}" y="{y(vs[i]):.1f}" width="{bw - 2:.1f}" '
                        f'height="{max(y(lo) - y(vs[i]), 0):.1f}" rx="2"><title>{escape(name)} {escape(c)}: '
                        f'{fmt.format(vs[i])}</title></rect>')
        body.append(f'<text class="lbl" x="{PAD["l"] + i * slot + slot / 2:.1f}" y="{H - PAD["b"] + 18}" '
                    f'text-anchor="middle">{escape(c)}</text>')
    lx = W - PAD["r"] - 150 * k
    for s_i, name in enumerate(series):
        body.append(f'<rect class="{"s1" if s_i == highlight else "s2"}" x="{lx + 150 * s_i}" y="8" width="10" height="10" rx="2"/>'
                    f'<text class="lbl" x="{lx + 150 * s_i + 15}" y="17">{escape(name)}</text>')
    body.append(f'<text class="lbl" transform="translate(16 {H / 2}) rotate(-90)" text-anchor="middle">'
                f'{escape(ylabel)}</text>')
    return _frame(title, "".join(body))


def histogram(title: str, counts: list[int], edges: list[float], xlabel: str) -> str:
    hi = max(counts) or 1
    y = _scale(0, hi * 1.1, H - PAD["b"], PAD["t"])
    x = _scale(edges[0], edges[-1], PAD["l"], W - PAD["r"])
    body = []
    for c, a, b in zip(counts, edges, edges[1:]):
        cls = "neg" if b <= 0 else "s1"
        body.append(f'<rect class="{cls}" x="{x(a):.1f}" y="{y(c):.1f}" width="{max(x(b) - x(a) - 1, 1):.1f}" '
                    f'height="{y(0) - y(c):.1f}"/>')
    if edges[0] < 0 < edges[-1]:
        body.append(f'<line class="ax" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{PAD["t"]}" y2="{H - PAD["b"]}" '
                    f'style="stroke-dasharray:4 3"/>')
    for t in _nice(edges[0], edges[-1]):
        body.append(f'<text class="lbl" x="{x(t):.1f}" y="{H - PAD["b"] + 18}" text-anchor="middle">{t:g}</text>')
    body.append(f'<text class="lbl" x="{(W + PAD["l"]) / 2}" y="{H - 8}" text-anchor="middle">{escape(xlabel)}</text>')
    return _frame(title, "".join(body))
