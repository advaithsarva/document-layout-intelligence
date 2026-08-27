"""Page -> ordered, typed, hierarchical layout elements.

THE INVARIANT
-------------
**Every text span on the page belongs to exactly one output element, and the
elements carry a total reading order.**

Both halves, and both are things that break silently.

*Exactly one* is a partition: no span dropped, no span counted twice. A layout
extractor that quietly loses a column, or a footer, or the second half of every
paragraph, produces output that looks perfect -- correctly typed boxes, clean
JSON, plausible text -- and is missing a third of the document. Nothing raises.
The only way to know is to count characters in against characters out, so
`analyse_page` does exactly that and `coverage` is reported next to every
accuracy number in RESULTS.md.

*A total order* is the other half. Reading order is invisible in a rendered
preview: draw boxes round the elements of a two-column page and the picture is
identical whether the order is correct or interleaved. It only shows up when
the text is concatenated -- and then it is grammatical line by line and
nonsense paragraph by paragraph, which is exactly the failure that survives
review.

THE PIPELINE
------------
    1. spans        PyMuPDF text dict -> lines with font, size, bbox
    2. running      strip header/footer bands (order 0, never inlined)
    3. regions      figures from drawings/images, tables from tables.py
    4. columns      vertical whitespace projection -> column boundaries
    5. order        recursive XY-cut, which handles full-width spanning
                    elements without a special case
    6. classify     font size and weight against the page's own body text,
                    plus positional and lexical cues
    7. hierarchy    headings adopt the elements that follow them

Nothing here is learned. `classify.py` is the learned version of step 6, and
`bench.py` puts the two side by side on the same fixtures.
"""

import re
import statistics
from dataclasses import dataclass, field

import fitz

import tables as tables_mod

# A header/footer lives in a band at the very top or bottom of the page. The
# fraction is of page height, and it is deliberately tight: a first paragraph
# starting high on the page must not be swallowed.
HEADER_BAND = 0.065
FOOTER_BAND = 0.935

# A gap between columns has to be wider than a word space to be a gutter.
MIN_GUTTER_PT = 9.0
# A gutter is defined by what does NOT cross it. This is the fraction of the
# page's lines allowed to span it -- a full-width title on a two-column page is
# one line out of forty, an unruled table's internal whitespace is crossed by
# every paragraph above and below it.
CROSSING_TOLERANCE = 0.12
# ...and each side has to actually be a column of text, not one stray label.
MIN_COLUMN_LINES = 3
# ...and the two sides have to run alongside each other, not merely both exist.
SIDE_OVERLAP = 0.7

BULLET = re.compile(r"^\s*([•●▪\-\*–]|\(?\d{1,2}[.)]|[a-z][.)])\s+")
CAPTION_LEAD = re.compile(r"^\s*(figure|fig\.?|table|tbl\.?|chart|listing|scheme)"
                          r"\s*\d+\s*[.:–-]", re.I)


@dataclass
class Line:
    """One rendered line of text with the geometry needed to classify it."""
    text: str
    bbox: tuple
    size: float
    bold: bool
    font: str

    @property
    def x0(self): return self.bbox[0]

    @property
    def y0(self): return self.bbox[1]


@dataclass
class LayoutElement:
    type: str
    bbox: tuple
    text: str
    order: int = 0
    column: int = 0
    children: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        out = {"type": self.type, "bbox": [round(v, 2) for v in self.bbox],
               "order": self.order, "column": self.column, "text": self.text}
        if self.meta:
            out["meta"] = self.meta
        if self.children:
            out["children"] = [c.to_dict() for c in self.children]
        return out


# --------------------------------------------------------------------------
# 1. spans -> lines
# --------------------------------------------------------------------------

def page_lines(page, clip=None):
    """Every text line on the page, with size and weight resolved per line.

    A line can mix fonts (bold run inside a sentence). The size taken is the
    span-length-weighted one and `bold` requires the majority of characters to
    be bold, because a single bold word does not make a heading and treating it
    as one turns every emphasised term into a section break.
    """
    out = []
    for block in page.get_text("dict", clip=clip)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"])
            weight = sum(len(s["text"].strip()) for s in spans) or 1
            size = sum(s["size"] * len(s["text"].strip()) for s in spans) / weight
            bold_chars = sum(len(s["text"].strip()) for s in spans
                             if (s["flags"] & 16) or "bold" in s["font"].lower())
            out.append(Line(text=text.rstrip(), bbox=tuple(line["bbox"]),
                            size=round(size, 2), bold=bold_chars * 2 > weight,
                            font=spans[0]["font"]))
    out.sort(key=lambda l: (round(l.y0, 1), l.x0))
    return out


def body_size(lines):
    """The page's own body-text size: a character-weighted median over the
    lines that are set to the column measure.

    Absolute font thresholds do not survive contact with real documents -- 11pt
    is a heading in a paper and body text in a memo -- so everything downstream
    is a ratio against this, which makes getting it right load-bearing.

    Captions are excluded, and that exclusion had to be added. Weighting by
    characters alone is right on a dense page and wrong on a sparse one: a page
    holding a figure, a table, a one-line paragraph and two captions has *more
    caption text than body text*, so the median came out at the 8pt caption
    size, the 9.5pt paragraph then scored a ratio of 1.19, and it was
    classified as a heading.

    Filtering by line width was tried first and is worse -- it looks principled
    ("body text is set to the column measure") and it breaks on any page with a
    wide title, because the title becomes the widest line and the real body
    lines fall under the threshold. Excluding lines that announce themselves as
    captions is narrower, uses a signal that is already trusted elsewhere, and
    fails safe: a page with no captions is unaffected.
    """
    if not lines:
        return 10.0
    prose = [l for l in lines if not CAPTION_LEAD.match(l.text)] or lines
    weighted = []
    for line in prose:
        weighted.extend([line.size] * max(1, len(line.text.strip()) // 4))
    return statistics.median(weighted)


# --------------------------------------------------------------------------
# 2. running heads
# --------------------------------------------------------------------------

def split_running(lines, page_height):
    """Peel off the header/footer bands.

    Kept as elements with `order = 0` rather than discarded: a page number is
    real content that a citation may need, and dropping it would break the
    coverage invariant. It is simply not part of the reading flow, and splicing
    it in is what makes concatenated multi-page text read like a fault.
    """
    header, footer, body = [], [], []
    for line in lines:
        if line.bbox[3] <= page_height * HEADER_BAND:
            header.append(line)
        elif line.y0 >= page_height * FOOTER_BAND:
            footer.append(line)
        else:
            body.append(line)
    return header, footer, body


# --------------------------------------------------------------------------
# 3. figures
# --------------------------------------------------------------------------

def figure_regions(page, min_area=1600.0):
    """Vector-drawing clusters and embedded images, merged into figure boxes.

    Individual `draw_line` calls are not figures; a chart is dozens of them
    that overlap. So the rectangles are merged transitively while any two
    overlap or nearly touch, and only the survivors big enough to matter are
    kept. Table rules are drawings too -- `analyse_page` removes anything the
    table detector already claimed, rather than trying to tell them apart here.
    """
    boxes = [fitz.Rect(d["rect"]) for d in page.get_drawings()]
    boxes += [fitz.Rect(page.get_image_bbox(info)) for info in page.get_images(full=True)
              if _safe_bbox(page, info)]

    merged = []
    for box in boxes:
        if box.is_empty or box.is_infinite:
            continue
        grown = fitz.Rect(box) + (-3, -3, 3, 3)
        hits = [m for m in merged if m.intersects(grown)]
        for m in hits:
            merged.remove(m)
            grown |= m
        merged.append(grown)

    # One pass is not a fixed point: merging A into B can bring B within reach
    # of C. Repeat until nothing changes, which is at most a handful of passes.
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if merged[i].intersects(merged[j]):
                    merged[i] |= merged[j]
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break

    return [tuple(m) for m in merged
            if m.width * m.height >= min_area and m.width > 20 and m.height > 20]


def _safe_bbox(page, info):
    try:
        page.get_image_bbox(info)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 4. columns
# --------------------------------------------------------------------------

def column_bounds(lines, page_width):
    """Find vertical gutters: whitespace that almost no line of text crosses.

    Candidates come from the classic x-axis projection, but the projection
    alone is not a test -- and the version that measures "is this slice empty
    for most of the page height" fails in a way worth recording. On a page
    whose only wide element is a table, the whitespace *between the table's own
    columns* is empty for as much of the page as a real gutter is, so the page
    was declared two-column, the table was then analysed one column at a time,
    and it came out as a 5x2 table with its first column emitted as loose
    paragraphs. Every step after that was correct.

    What separates a gutter from any other whitespace is not how tall it is,
    it is that **text does not cross it**. Column text stops at the gutter, by
    definition. Table rows, paragraphs above a table, and captions all cross
    their local whitespace freely. So:

        a gutter is crossed by fewer than CROSSING_TOLERANCE of the lines,
        and has at least MIN_COLUMN_LINES lines wholly on each side

    A full-width title still crosses, which is why the tolerance is a small
    fraction rather than zero.
    """
    if len(lines) < 6:
        return [(0.0, page_width)]

    left = min(l.bbox[0] for l in lines)
    right = max(l.bbox[2] for l in lines)

    step = 2.0
    n = int(page_width / step) + 1
    hits = [0] * n                  # how many lines cover each x slice
    for line in lines:
        lo, hi = int(line.bbox[0] / step), min(int(line.bbox[2] / step) + 1, n)
        for i in range(lo, hi):
            hits[i] += 1

    threshold = max(1, int(len(lines) * CROSSING_TOLERANCE))
    runs, start = [], None
    for i in range(n):
        if hits[i] <= threshold:
            start = i if start is None else start
        else:
            if start is not None and (i - start) * step >= MIN_GUTTER_PT:
                runs.append((start * step, i * step))
            start = None
    if start is not None and (n - start) * step >= MIN_GUTTER_PT:
        runs.append((start * step, n * step))

    inner = []
    for lo, hi in runs:
        if lo <= left + 1 or hi >= right - 1:
            continue
        crossing = sum(1 for l in lines if l.bbox[0] < lo and l.bbox[2] > hi)
        before = [l for l in lines if l.bbox[2] <= lo + 1]
        after = [l for l in lines if l.bbox[0] >= hi - 1]
        if (crossing > len(lines) * CROSSING_TOLERANCE
                or len(before) < MIN_COLUMN_LINES or len(after) < MIN_COLUMN_LINES):
            continue
        # Columns run ALONGSIDE each other. A gap that has text on both sides
        # but at different heights is not a gutter, it is the whitespace inside
        # something -- an unruled table's first column against the rest of it
        # passed the crossing test and split the page in two, after which the
        # table was analysed one column at a time and came back 4x3 with its
        # first column emitted as loose text.
        span = lambda group: (min(l.bbox[1] for l in group),
                              max(l.bbox[3] for l in group))
        (lt, lb), (rt, rb) = span(before), span(after)
        overlap = min(lb, rb) - max(lt, rt)
        shorter = min(lb - lt, rb - rt)
        if overlap < shorter * SIDE_OVERLAP:
            continue
        inner.append((lo, hi))
    if not inner:
        return [(left, right)]

    bounds, cursor = [], left
    for lo, hi in inner:
        bounds.append((cursor, lo))
        cursor = hi
    bounds.append((cursor, right))
    # A "column" narrower than this is an artefact, not a column.
    return [b for b in bounds if b[1] - b[0] > 40] or [(left, right)]


# --------------------------------------------------------------------------
# 5. reading order: recursive XY-cut
# --------------------------------------------------------------------------

def xy_cut(items, depth=0, max_depth=6):
    """Order boxes by recursively splitting on the widest whitespace gap.

    Each item is `(bbox, payload)`. The recursion alternates: try to split
    horizontally (a full-width band separating stacked groups), then
    vertically (a gutter separating columns), and recurse into the parts.

    This is why a full-width title above two columns needs no special case. The
    first horizontal cut separates the title from the body; the vertical cut
    then splits the body into columns. A rule like "sort by column, then by y"
    has to detect the spanning element explicitly and gets it wrong when there
    are two of them.
    """
    if len(items) <= 1 or depth >= max_depth:
        return list(items)

    candidates = {}
    for axis in (0, 1):             # 0 = vertical cut (gutter), 1 = horizontal
        lo_i, hi_i = (1, 3) if axis else (0, 2)
        ordered = sorted(items, key=lambda it: it[0][lo_i])
        threshold = 4.0 if axis else MIN_GUTTER_PT
        reach = ordered[0][0][hi_i]
        # THE FIRST sufficient gap, not the widest. Taking the widest is the
        # textbook phrasing and it reorders two-column pages whenever both
        # columns happen to break a paragraph at a similar height: that shared
        # band is wider than the gap under the title, so the page splits into a
        # top half and a bottom half, each then correctly read in columns --
        # giving title, left-top, right-top, left-bottom, right-bottom. tau
        # 0.81 on exactly the pages this exists for.
        #
        # Cutting at the first gap peels the spanning element off the top and
        # leaves the rest of the group intact for the vertical cut to handle.
        # On a single column it changes nothing: peeling one block at a time
        # top-down produces the same order as splitting in the middle.
        for i in range(1, len(ordered)):
            gap = ordered[i][0][lo_i] - reach
            if gap >= threshold:
                candidates[axis] = (gap, ordered, i)
                break
            reach = max(reach, ordered[i][0][hi_i])

    # A VERTICAL CUT IS TAKEN WHENEVER ONE IS AVAILABLE. Two wrong versions of
    # this were written before the right one:
    #
    #   horizontal first  interleaves every two-column page. Any band of the
    #                     page with whitespace across it becomes a cut, so the
    #                     output reads left, right, left, right down the page.
    #   widest gap wins   almost works, and fails where both columns happen to
    #                     have a paragraph break at the same height. That band
    #                     is then wider than the 18pt gutter, so the page splits
    #                     into two halves and each half is read in columns:
    #                     tau 0.80 on exactly the pages this exists for.
    #
    # A gutter is a global structural fact about the group; a horizontal gap is
    # a local one. So structure first. A full-width title needs no special case
    # -- it spans the gutter, so no vertical gap exists at the top level and the
    # horizontal cut separates it before the columns are ever considered.
    if candidates:
        axis = 0 if 0 in candidates else 1
        _, ordered, best = candidates[axis]
        return (xy_cut(ordered[:best], depth + 1, max_depth)
                + xy_cut(ordered[best:], depth + 1, max_depth))

    # Nothing separates them cleanly; fall back to top-to-bottom, left-to-right.
    return sorted(items, key=lambda it: (round(it[0][1], 1), it[0][0]))


# --------------------------------------------------------------------------
# 6. grouping and classification
# --------------------------------------------------------------------------

def group_lines(lines, body):
    """Consecutive lines into blocks, cutting on a vertical gap or a style change.

    The threshold is derived from the page's OWN line spacing, and that is the
    part that had to be fixed. The first version compared the gap to the line
    *height* (`gap > leading * 1.6`) which sounds reasonable and is not: lines
    inside a paragraph sit about 0-1pt apart, paragraphs about 9pt apart, and
    line height is about 11pt -- so the test demanded 17pt and welded every
    paragraph on the page into one block. Recall on two-column pages was 0.53
    and the output still looked entirely plausible.

    So: take the median gap between consecutive lines, which is by construction
    the *intra*-paragraph gap because most consecutive pairs are inside a
    paragraph, and split on anything meaningfully larger.
    """
    if not lines:
        return []

    # The typical gap is the INTRA-paragraph one, so it is a low quantile, not
    # the median. The median works on a dense page where most consecutive pairs
    # are inside a paragraph, and collapses on a sparse one: a page holding a
    # heading, a one-line paragraph, a figure and two captions has gaps of
    # 7, 185 and 85 points, median 85 -- so the threshold became 85pt and the
    # two captions on opposite sides of the table merged into one element that
    # spanned it. The cap against line height is the second guard, for pages
    # with too few lines for any quantile to mean much.
    gaps = sorted(b.bbox[1] - a.bbox[3] for a, b in zip(lines, lines[1:])
                  if b.bbox[1] >= a.bbox[1])
    leading = statistics.median([l.bbox[3] - l.bbox[1] for l in lines]) or 1.0
    typical = min(gaps[len(gaps) // 4], leading * 0.5) if gaps else 0.0

    # The style-change threshold is derived the same way, and for the same
    # reason. 0.6pt is a fine constant for a born-digital PDF where the
    # reported size is exact. On an OCR'd page the size is *estimated from
    # glyph height* and jitters by a point between consecutive lines of the
    # same paragraph -- so a fixed 0.6pt cut every paragraph into single lines
    # and precision fell to 0.29 while every individual stage stayed correct.
    jitter = [abs(a.size - b.size) for a, b in zip(lines, lines[1:])]
    size_tolerance = max(0.6, 3.0 * statistics.median(jitter)) if jitter else 0.6

    blocks, current = [], []
    for line in lines:
        if not current:
            current = [line]
            continue
        previous = current[-1]
        gap = line.bbox[1] - previous.bbox[3]
        own_leading = max(previous.bbox[3] - previous.bbox[1], 1.0)
        limit = typical + max(2.5, own_leading * 0.3)
        style_change = (abs(line.size - previous.size) > size_tolerance
                        or line.bold != previous.bold)
        # A block also ends where the text stops overlapping horizontally --
        # two side-by-side captions are not one paragraph.
        disjoint = (line.bbox[0] > previous.bbox[2] or line.bbox[2] < previous.bbox[0])
        if gap > limit or style_change or disjoint:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def classify(block, body, page_rect, is_first_block=False):
    """Name a block, using ratios against the page's own body text.

    Order matters: the lexical caption test runs before the size test, because
    a caption is small *and* starts with "Figure 3:" and only the second is
    unambiguous. The size test runs before the list test for the same reason
    in reverse -- a bold 16pt line starting "1." is a numbered heading.
    """
    text = " ".join(l.text for l in block).strip()
    sizes = [l.size for l in block]
    size = statistics.median(sizes)
    bold = sum(l.bold for l in block) * 2 >= len(block)
    ratio = size / body if body else 1.0
    lines = len(block)
    width = max(l.bbox[2] for l in block) - min(l.bbox[0] for l in block)

    if CAPTION_LEAD.match(text) and lines <= 4:
        return "caption"
    if ratio >= 1.55 and lines <= 3 and is_first_block:
        return "title"
    if (ratio >= 1.12 or (bold and ratio >= 0.98)) and lines <= 3 and len(text) < 120:
        return "title" if ratio >= 1.55 and lines <= 2 else "heading"
    if lines >= 2 and sum(bool(BULLET.match(l.text)) for l in block) * 2 >= lines:
        return "list"
    if ratio <= 0.88 and lines <= 3 and width < page_rect[2] * 0.75:
        return "caption"
    return "paragraph"


# --------------------------------------------------------------------------
# 7. the whole page
# --------------------------------------------------------------------------

def analyse_page(page, detect_tables=True, classifier=None):
    """Return `(elements, stats)` for one page, ordered and typed.

    ORDER OF OPERATIONS, AND WHY IT IS THIS ORDER
    ---------------------------------------------
    Columns are found *before* anything groups or detects, and both of the
    steps that follow run **inside a single column**. Doing it the other way
    round -- the obvious way -- produced two distinct failures on the same
    two-column page:

    * lines arrive sorted by (y, x), so a two-column page interleaves left and
      right column lines. Grouping across the page then saw the x jump on every
      single line and split every paragraph into one-line blocks.
    * the unruled-table detector saw two columns of prose as two aligned cells
      repeated down the page and reported three tables on a page with none.

    Both are the same mistake: a column is a reading context, and no
    line-to-line relationship means anything across a gutter.
    """
    page_rect = tuple(page.rect)
    all_lines = page_lines(page)
    total_chars = sum(len(l.text.strip()) for l in all_lines)

    header_lines, footer_lines, body_lines = split_running(all_lines, page_rect[3])
    body_band = (page_rect[3] * HEADER_BAND, page_rect[3] * FOOTER_BAND)

    ruled = tables_mod.ruled_tables(page) if detect_tables else []
    figure_boxes = [b for b in figure_regions(page)
                    if not any(_overlaps(b, t.bbox, 0.35) for t in ruled)]

    claimed = [t.bbox for t in ruled] + list(figure_boxes)
    free_lines = [l for l in body_lines
                  if not any(_overlaps(l.bbox, c, 0.5) for c in claimed)]

    # Body size is measured on prose only. Table cells are typically a point or
    # two smaller, and leaving them in dragged the median down far enough that
    # ordinary paragraphs scored as headings on every two-column page.
    body = body_size(free_lines or body_lines or all_lines)
    columns = column_bounds(free_lines, page_rect[2])

    tables_found = list(ruled)
    items = []
    for index, (lo, hi) in enumerate(columns):
        centre = lambda b: (b[0] + b[2]) / 2
        column_lines = [l for l in free_lines if lo - 1 <= centre(l.bbox) <= hi + 1]
        if detect_tables:
            for table in tables_mod.unruled_tables(page, body_band, (lo, hi), body):
                if any(_overlaps(table.bbox, t.bbox, 0.3) for t in tables_found):
                    continue
                tables_found.append(table)
                items.append((table.bbox, ("table", table, index)))
                column_lines = [l for l in column_lines
                                if not _overlaps(l.bbox, table.bbox, 0.5)]
        for block in group_lines(column_lines, body):
            bbox = (min(l.bbox[0] for l in block), min(l.bbox[1] for l in block),
                    max(l.bbox[2] for l in block), max(l.bbox[3] for l in block))
            items.append((bbox, ("text", block, index)))

    for table in ruled:
        items.append((table.bbox, ("table", table, _column_of(table.bbox, columns))))
    for box in figure_boxes:
        items.append((box, ("figure", None, _column_of(box, columns))))

    first_bbox = min((it[0] for it in items), key=lambda b: (round(b[1]), b[0]),
                     default=None)
    elements = []
    for order, (bbox, (kind, payload, column)) in enumerate(xy_cut(items), start=1):
        if kind == "text":
            type_ = _name(payload, body, page_rect, bbox == first_bbox, classifier)
            text = " ".join(l.text for l in payload).strip()
            elements.append(LayoutElement(type_, bbox, text, order, column))
        elif kind == "table":
            elements.append(LayoutElement(
                "table", bbox, payload.text, order, column,
                meta={"rows": payload.rows, "cols": payload.cols,
                      "cells": payload.cells, "detector": payload.detector}))
        else:
            elements.append(LayoutElement("figure", bbox, "", order, column))

    for line in header_lines:
        elements.append(LayoutElement("header", line.bbox, line.text.strip(), 0, 0))
    for line in footer_lines:
        elements.append(LayoutElement("footer", line.bbox, line.text.strip(), 0, 0))

    # Coverage is counted GEOMETRICALLY -- how many of the page's own text
    # lines landed inside some element's box -- not by comparing string
    # lengths. Comparing lengths lets a table whose cells are re-joined with
    # different whitespace report >100% coverage, which is how a partition
    # check turns into a number that cannot fail.
    boxes = [e.bbox for e in elements]
    covered = sum(len(l.text.strip()) for l in all_lines
                  if any(_overlaps(l.bbox, b, 0.5) for b in boxes))
    stats = {
        "columns": len(columns),
        "body_size": round(body, 2),
        "chars_on_page": total_chars,
        "chars_covered": covered,
        "coverage": round(covered / total_chars, 4) if total_chars else 1.0,
        "tables": len(tables_found),
        "figures": len(figure_boxes),
    }
    return elements, stats


def _name(block, body, page_rect, is_first, classifier):
    """Rules, or the learned classifier when one is supplied and loads.

    The rules are the default and the fallback, in that order. A missing model
    file, a missing scikit-learn, or an unpickling failure all land here and
    the page is still analysed -- `classify.predict` returns None rather than
    raising, so the only visible effect is which path ran.
    """
    if classifier is not None:
        import classify as classify_mod
        predicted = classify_mod.predict(block, body, page_rect, is_first,
                                         model=classifier)
        if predicted is not None:
            return predicted
    return classify(block, body, page_rect, is_first_block=is_first)


def nest(elements):
    """Attach the flow under each heading to that heading, as a tree.

    A flat ordered list is the honest primary output -- this is a convenience
    view for consumers that want sections, and it is derived, never the source
    of truth. `title` opens a document-level section; `heading` opens one
    inside it.
    """
    roots, stack = [], []
    for element in sorted((e for e in elements if e.order), key=lambda e: e.order):
        level = {"title": 0, "heading": 1}.get(element.type)
        if level is None:
            (stack[-1][1].children if stack else roots).append(element)
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        (stack[-1][1].children if stack else roots).append(element)
        stack.append((level, element))
    return roots


def analyse_document(path, detect_tables=True, max_pages=None, classifier=None):
    doc = fitz.open(path)
    pages = []
    try:
        for number, page in enumerate(doc, start=1):
            if max_pages and number > max_pages:
                break
            elements, stats = analyse_page(page, detect_tables=detect_tables,
                                           classifier=classifier)
            pages.append({"page": number, "elements": elements, "stats": stats})
    finally:
        doc.close()
    return pages


def document_json(pages, tree=False):
    return {
        "pages": [{
            "page": p["page"],
            "stats": p["stats"],
            "elements": [e.to_dict() for e in
                         (nest(p["elements"]) if tree else
                          sorted(p["elements"], key=lambda e: (e.order == 0, e.order)))],
        } for p in pages],
        "coverage": round(
            sum(p["stats"]["chars_covered"] for p in pages)
            / max(sum(p["stats"]["chars_on_page"] for p in pages), 1), 4),
    }


def reading_text(pages, include_running=False):
    """Elements concatenated in reading order -- the thing that exposes the bug."""
    out = []
    for page in pages:
        ordered = sorted((e for e in page["elements"]
                          if include_running or e.order), key=lambda e: e.order)
        out.extend(e.text for e in ordered if e.text.strip())
    return "\n\n".join(out)


def _overlaps(a, b, fraction):
    """Does `a` sit inside `b` by at least `fraction` of a's own area."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return False
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return (x1 - x0) * (y1 - y0) / area >= fraction


def _column_of(bbox, columns):
    centre = (bbox[0] + bbox[2]) / 2
    for i, (lo, hi) in enumerate(columns):
        if lo <= centre <= hi:
            return i
    return 0
