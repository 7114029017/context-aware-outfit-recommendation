"""Small SVG charts for the supplementary analyses.

The reproduction environment has no plotting library, so the figures that the
2025 notebooks drew with matplotlib are redrawn here as plain SVG: bar charts,
line charts with a band, histograms and a box plot. The output contains no
timestamp, so a rerun on the same data writes identical files.
"""
from __future__ import annotations

import math
from html import escape
from pathlib import Path

import numpy as np

FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"
POSITIVE = "#2166AC"   # bar colours of the 2025 programs
NEGATIVE = "#D6604D"
GREY = "#888888"
PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")


def text_width(s: str, size: float) -> float:
    """Approximate rendered width of a DejaVu Sans string (slightly generous)."""
    def em(ch: str) -> float:
        if ch in " .,:;|!il'()[]ftjr":
            return 0.36
        if ch in "mwMW":
            return 0.92
        if ch.isupper() or ch in "%@&":
            return 0.74
        return 0.64
    return sum(em(ch) for ch in str(s)) * size


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return []
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(lo / step - 1e-9) * step
    ticks = []
    value = first
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 12) + 0.0)
        value += step
    return ticks


def tick_format(ticks: list[float]) -> str:
    step = min((b - a for a, b in zip(ticks, ticks[1:])), default=1.0)
    decimals = 0 if step >= 1 else min(4, max(0, -math.floor(math.log10(step) + 1e-9)))
    if decimals and any(abs(t * 10 ** decimals - round(t * 10 ** decimals)) > 1e-6 for t in ticks):
        decimals += 1
    return f"{{:.{decimals}f}}"


def padded(lo: float, hi: float, share: float = 0.05) -> tuple[float, float]:
    span = (hi - lo) or abs(hi) or 1.0
    return lo - span * share, hi + span * share


class Chart:
    def __init__(self, width: float, height: float, title: str | None = None, subtitle: str | None = None):
        self.width, self.height = width, height
        self.items: list[str] = []
        if title:
            self.text(width / 2, 24, title, size=14, anchor="middle", weight="bold")
        if subtitle:
            self.text(width / 2, 42, subtitle, size=10.5, anchor="middle", color="#555555")

    def text(self, x, y, s, size=11, anchor="start", weight=None, color="#222222", rotate=None, baseline=None):
        attrs = [f'x="{x:.2f}"', f'y="{y:.2f}"', f'font-size="{size}"', f'fill="{color}"']
        if anchor != "start":
            attrs.append(f'text-anchor="{anchor}"')
        if weight:
            attrs.append(f'font-weight="{weight}"')
        if baseline:
            attrs.append(f'dominant-baseline="{baseline}"')
        if rotate is not None:
            attrs.append(f'transform="rotate({rotate} {x:.2f} {y:.2f})"')
        self.items.append(f"<text {' '.join(attrs)}>{escape(str(s))}</text>")

    def line(self, x1, y1, x2, y2, color="#333333", width=1.0, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" '
                          f'stroke-width="{width}"{extra}/>')

    def rect(self, x, y, w, h, fill, stroke=None, opacity=None, stroke_width=1.0):
        extra = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
        if opacity is not None:
            extra += f' fill-opacity="{opacity}"'
        self.items.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" '
                          f'fill="{fill}"{extra}/>')

    def polyline(self, points, color, width=1.6, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.items.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

    def polygon(self, points, fill, opacity=0.25):
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.items.append(f'<polygon points="{coords}" fill="{fill}" fill-opacity="{opacity}" stroke="none"/>')

    def circle(self, x, y, r, fill, opacity=1.0):
        self.items.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" fill-opacity="{opacity}"/>')

    def panel(self, left, top, width, height, title=None) -> "Panel":
        if title:
            self.text(left + width / 2, top - 8, title, size=11.5, anchor="middle", weight="bold")
        return Panel(self, left, top, width, height)

    def legend(self, x, y, entries, size=10):
        """entries: (label, colour, kind) with kind 'box', 'line' or 'dash'; drawn on a white box."""
        width = 30 + max(text_width(label, size) for label, _, _ in entries)
        self.rect(x - 6, y - size - 5, width, len(entries) * (size + 7) + 4, "#ffffff", stroke="#cccccc", opacity=0.85)
        for i, (label, color, kind) in enumerate(entries):
            yy = y + i * (size + 7)
            if kind == "box":
                self.rect(x, yy - size + 1, 14, size - 1, color, opacity=0.75)
            else:
                self.line(x, yy - size / 2 + 1, x + 16, yy - size / 2 + 1, color, 2, "5,3" if kind == "dash" else None)
            self.text(x + 21, yy, label, size=size)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.items)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:.0f}" height="{self.height:.0f}" '
            f'viewBox="0 0 {self.width:.0f} {self.height:.0f}" font-family="{FONT}">\n'
            f'<rect width="100%" height="100%" fill="#ffffff"/>\n{body}\n</svg>\n', encoding="utf-8")


class Panel:
    """A plotting area; data coordinates are mapped once set_x / set_y are called."""

    def __init__(self, chart: Chart, left, top, width, height):
        self.c, self.left, self.top, self.width, self.height = chart, left, top, width, height
        self.x0, self.x1, self.y0, self.y1 = 0.0, 1.0, 0.0, 1.0

    def set_x(self, lo, hi):
        self.x0, self.x1 = lo, hi if hi != lo else lo + 1

    def set_y(self, lo, hi):
        self.y0, self.y1 = lo, hi if hi != lo else lo + 1

    def xp(self, v):
        return self.left + (v - self.x0) / (self.x1 - self.x0) * self.width

    def yp(self, v):
        return self.top + self.height - (v - self.y0) / (self.y1 - self.y0) * self.height

    def axes(self, xticks=None, yticks=None, xlabel=None, ylabel=None, xfmt=None, yfmt=None, grid="y",
             xtick_labels=None, size=9.5):
        bottom, right = self.top + self.height, self.left + self.width
        if yticks:
            yfmt = yfmt or tick_format(yticks)
            for t in yticks:
                y = self.yp(t)
                if grid in ("y", "both"):
                    self.c.line(self.left, y, right, y, "#e3e3e3", 0.8)
                self.c.line(self.left - 4, y, self.left, y, "#555555", 0.8)
                self.c.text(self.left - 7, y + size * 0.35, yfmt.format(t), size=size, anchor="end")
        if xticks:
            xfmt = xfmt or tick_format(xticks)
            for t in xticks:
                x = self.xp(t)
                if grid in ("x", "both"):
                    self.c.line(x, self.top, x, bottom, "#e3e3e3", 0.8)
                self.c.line(x, bottom, x, bottom + 4, "#555555", 0.8)
                self.c.text(x, bottom + 6 + size, xfmt.format(t), size=size, anchor="middle")
        for x, label in xtick_labels or []:
            lines = str(label).split("\n")
            for i, part in enumerate(lines):
                self.c.text(self.xp(x), bottom + 6 + size + i * (size + 3), part, size=size, anchor="middle")
        self.c.line(self.left, self.top, self.left, bottom, "#555555", 1)
        self.c.line(self.left, bottom, right, bottom, "#555555", 1)
        if xlabel:
            extra = max((len(str(l).split("\n")) for _, l in xtick_labels or []), default=1)
            self.c.text(self.left + self.width / 2, bottom + 14 + (size + 3) * extra + 8, xlabel, size=10.5,
                        anchor="middle")
        if ylabel:
            x = self.left - 12 - max((text_width((yfmt or "{}").format(t), size) for t in yticks or []), default=20)
            self.c.text(x - 6, self.top + self.height / 2, ylabel, size=10.5, anchor="middle", rotate=-90)

    def bar(self, x_lo, x_hi, y_lo, y_hi, fill, opacity=None, stroke="#ffffff"):
        xa, xb = sorted((self.xp(x_lo), self.xp(x_hi)))
        ya, yb = sorted((self.yp(y_lo), self.yp(y_hi)))
        self.c.rect(xa, ya, xb - xa, yb - ya, fill, stroke=stroke, opacity=opacity, stroke_width=0.6)

    def hline(self, y, color="#999999", dash="4,3", width=1.0):
        self.c.line(self.left, self.yp(y), self.left + self.width, self.yp(y), color, width, dash)

    def vline(self, x, color="#999999", dash="4,3", width=1.0):
        self.c.line(self.xp(x), self.top, self.xp(x), self.top + self.height, color, width, dash)

    def line(self, xs, ys, color, width=1.8, dash=None):
        self.c.polyline([(self.xp(x), self.yp(y)) for x, y in zip(xs, ys)], color, width, dash)

    def band(self, xs, lo, hi, color, opacity=0.22):
        upper = [(self.xp(x), self.yp(y)) for x, y in zip(xs, hi)]
        lower = [(self.xp(x), self.yp(y)) for x, y in zip(reversed(xs), reversed(lo))]
        self.c.polygon(upper + lower, color, opacity)


def fit_width(minimum: float, title: str | None, subtitle: str | None = None) -> float:
    """Chart width that keeps the title and the subtitle inside the image."""
    return max(minimum, text_width(title or "", 14) * 1.08 + 40, text_width(subtitle or "", 10.5) + 40)


def _group_separators(panel: Panel, breaks, vertical=True):
    for b in breaks:
        if vertical:
            panel.vline(b, "#cccccc", "4,3")
        else:
            panel.hline(b, "#cccccc", "4,3")


def hbar_chart(path, title, labels, values, xlabel, value_fmt="{:+.3f}", subtitle=None, note=None, colors=None,
               legend=None):
    """Horizontal bars, top to bottom in the given order; positive blue, negative red unless colors are given."""
    label_w = max(text_width(l, 10) for l in labels) + 18
    width = fit_width(max(640, label_w + 470), title, subtitle)
    top = 62 if subtitle else 48
    height = top + 30 * len(labels) + 70 + (16 if note else 0)
    c = Chart(width, height, title, subtitle)
    p = c.panel(label_w, top, width - label_w - 40, 30 * len(labels))
    lo, hi = min(min(values), 0.0), max(max(values), 0.0)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * (0.04 if lo < 0 else 0), hi + span * 0.20
    p.set_x(lo, hi)
    p.set_y(len(labels), 0)
    ticks = [t for t in nice_ticks(lo, hi, 6) if lo <= t <= hi]
    p.axes(xticks=ticks, xlabel=xlabel, grid="x")
    for i, (label, v) in enumerate(zip(labels, values)):
        p.bar(0, v, i + 0.2, i + 0.8, colors[i] if colors else POSITIVE if v >= 0 else NEGATIVE)
        c.text(label_w - 8, p.yp(i + 0.5) + 3.5, label, size=10, anchor="end")
        c.text(p.xp(max(v, 0)) + 4, p.yp(i + 0.5) + 3.5, value_fmt.format(v), size=9)
    p.vline(0, "#777777", "4,3")
    if legend:
        c.legend(p.left + p.width - 120, p.top + p.height - 12 - 17 * (len(legend) - 1), legend)
    if note:
        c.text(12, height - 10, note, size=9, color="#555555")
    c.save(path)


def vbar_chart(path, title, labels, values, ylabel, value_fmt="{:+.1f}", breaks=(), subtitle=None, note=None):
    """Vertical bars with multi-line category labels; breaks are x positions of group separators."""
    width = fit_width(max(640, 95 * len(labels) + 110), title, subtitle)
    top = 64 if subtitle else 50
    height = top + 300 + 70 + (16 if note else 0)
    c = Chart(width, height, title, subtitle)
    p = c.panel(80, top, width - 110, 300)
    lo, hi = min(min(values), 0.0), max(max(values), 0.0)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * (0.12 if lo < 0 else 0), hi + span * 0.12
    p.set_x(0, len(labels))
    p.set_y(lo, hi)
    ticks = [t for t in nice_ticks(lo, hi, 5) if lo <= t <= hi]
    p.axes(yticks=ticks, ylabel=ylabel, xtick_labels=[(i + 0.5, l) for i, l in enumerate(labels)])
    for i, v in enumerate(values):
        p.bar(i + 0.18, i + 0.82, 0, v, POSITIVE if v >= 0 else NEGATIVE)
        y = p.yp(v) - 5 if v >= 0 else p.yp(v) + 13
        c.text(p.xp(i + 0.5), y, value_fmt.format(v), size=9, anchor="middle")
    p.hline(0, "#777777", "4,3")
    _group_separators(p, breaks)
    if note:
        c.text(12, height - 10, note, size=9, color="#555555")
    c.save(path)


def grouped_vbar_chart(path, title, labels, series, ylabel, value_fmt="{:.3f}", breaks=(), subtitle=None,
                       note=None):
    """series: (name, values, colour) drawn side by side for each label."""
    width = fit_width(max(640, 105 * len(labels) + 110), title, subtitle)
    top = 64 if subtitle else 50
    height = top + 300 + 70 + (16 if note else 0)
    c = Chart(width, height, title, subtitle)
    p = c.panel(80, top, width - 110, 300)
    hi = max(max(v) for _, v, _ in series) * 1.18
    p.set_x(0, len(labels))
    p.set_y(0, hi)
    ticks = [t for t in nice_ticks(0, hi, 5) if t <= hi]
    p.axes(yticks=ticks, ylabel=ylabel, xtick_labels=[(i + 0.5, l) for i, l in enumerate(labels)])
    w = 0.72 / len(series)
    for k, (name, vals, color) in enumerate(series):
        for i, v in enumerate(vals):
            x = i + 0.14 + k * w
            p.bar(x, x + w, 0, v, color)
            c.text(p.xp(x + w / 2), p.yp(v) - 4, value_fmt.format(v), size=8, anchor="middle")
    _group_separators(p, breaks)
    c.legend(p.left + 10, p.top + 14, [(name, color, "box") for name, _, color in series])
    if note:
        c.text(12, height - 10, note, size=9, color="#555555")
    c.save(path)


def line_chart(path, title, x, series, xlabel, ylabel, hline=None, subtitle=None, note=None, xfmt=None,
               legend_at="upper right"):
    """series: dicts with name, y, colour and optionally lo / hi for a shaded band."""
    width, top = fit_width(720, title, subtitle), (64 if subtitle else 50)
    height = top + 320 + 64 + (16 if note else 0)
    c = Chart(width, height, title, subtitle)
    p = c.panel(80, top, width - 110, 320)
    ys = [v for s in series for key in ("y", "lo", "hi") for v in s.get(key) or [] if math.isfinite(v)]
    lo, hi = padded(min(ys + ([hline] if hline is not None else [])), max(ys + ([hline] if hline is not None else [])))
    p.set_x(min(x), max(x))
    p.set_y(lo, hi)
    p.axes(xticks=[t for t in nice_ticks(min(x), max(x), 6) if min(x) - 1e-9 <= t <= max(x) + 1e-9],
           yticks=[t for t in nice_ticks(lo, hi, 5) if lo <= t <= hi], xlabel=xlabel, ylabel=ylabel, xfmt=xfmt,
           grid="both")
    for s in series:
        if s.get("lo") is not None:
            p.band(x, s["lo"], s["hi"], s["color"])
        p.line(x, s["y"], s["color"])
    if hline is not None:
        p.hline(hline, "#555555", "5,4")
    legend_y = p.top + 16 if legend_at == "upper right" else p.top + p.height - 12 - 17 * (len(series) - 1)
    legend_x = p.left + 16 if legend_at == "lower left" else p.left + p.width - 230
    c.legend(legend_x, legend_y, [(s["name"], s["color"], "line") for s in series])
    if note:
        c.text(12, height - 10, note, size=9, color="#555555")
    c.save(path)


def histogram_counts(values, bins, lo=None, hi=None):
    """Counts and bin edges as numpy.histogram (matplotlib's hist) computes them; without lo / hi the range is
    the data's own minimum to maximum, as plt.hist(values, bins=n) does."""
    rng = None if lo is None else (lo, hi)
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins, range=rng)
    return [int(n) for n in counts], [float(e) for e in edges]


def draw_histogram(p: Panel, series, bins, lo=None, hi=None, xlabel=None, ylabel="count", vlines=(), opacity=0.6):
    """series: (name, values, colour), overlaid. vlines: (x, colour, dash). Returns (counts, edges) per series."""
    binned = [histogram_counts(vals, bins, lo, hi) for _, vals, _ in series]
    x0 = min(edges[0] for _, edges in binned) if lo is None else lo
    x1 = max(edges[-1] for _, edges in binned) if hi is None else hi
    top = max(max(c) for c, _ in binned) * 1.08 or 1
    p.set_x(x0, x1)
    p.set_y(0, top)
    p.axes(xticks=[t for t in nice_ticks(x0, x1, 6) if x0 - 1e-9 <= t <= x1 + 1e-9],
           yticks=[t for t in nice_ticks(0, top, 5) if t <= top], xlabel=xlabel, ylabel=ylabel)
    for (name, _, color), (counts, edges) in zip(series, binned):
        for i, n in enumerate(counts):
            if n:
                p.bar(edges[i], edges[i + 1], 0, n, color, opacity=opacity, stroke=None)
    for x, color, dash in vlines:
        p.vline(x, color, dash, 1.6)
    return binned


def histogram_chart(path, title, series, bins, xlabel, lo=None, hi=None, ylabel="count", vlines=(), legend=None,
                    subtitle=None, note=None):
    width, top = fit_width(720, title, subtitle), (64 if subtitle else 50)
    height = top + 320 + 64 + (16 if note else 0)
    c = Chart(width, height, title, subtitle)
    p = c.panel(80, top, width - 110, 320)
    binned = draw_histogram(p, series, bins, lo, hi, xlabel, ylabel, vlines)
    c.legend(p.left + 14, p.top + 16, legend or [(name, color, "box") for name, _, color in series])
    if note:
        c.text(12, height - 10, note, size=9, color="#555555")
    c.save(path)
    return binned


def draw_boxplot(p: Panel, q1, median, q3, low_whisker, high_whisker, outliers, lo, hi, ylabel=None,
                 color=POSITIVE):
    """A vertical box plot (matplotlib conventions: whiskers at the last points within 1.5 IQR)."""
    p.set_x(0, 2)
    p.set_y(lo, hi)
    p.axes(yticks=[t for t in nice_ticks(lo, hi, 5) if lo <= t <= hi], ylabel=ylabel)
    for v in outliers:
        if lo <= v <= hi:
            p.c.circle(p.xp(1), p.yp(v), 1.6, NEGATIVE, 0.3)
    p.c.line(p.xp(1), p.yp(high_whisker), p.xp(1), p.yp(q3), "#2c4a7c", 1.2)
    p.c.line(p.xp(1), p.yp(q1), p.xp(1), p.yp(low_whisker), "#2c4a7c", 1.2)
    for w in (high_whisker, low_whisker):
        p.c.line(p.xp(0.8), p.yp(w), p.xp(1.2), p.yp(w), "#2c4a7c", 1.5)
    p.bar(0.55, 1.45, q1, q3, color, opacity=0.75, stroke="#2c4a7c")
    p.c.line(p.xp(0.55), p.yp(median), p.xp(1.45), p.yp(median), "#DD8452", 2.5)
    for label, v in (("Q1", q1), ("Median", median), ("Q3", q3)):
        p.c.text(p.xp(1.5) + 4, p.yp(v) + 3, f"{label} = {v:.3f}", size=9)
