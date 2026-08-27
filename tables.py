"""Table detection: use PyMuPDF where it is right, add a detector where it is not.

WHAT WAS MEASURED BEFORE ANY OF THIS WAS WRITTEN
------------------------------------------------
`page.find_tables()` ships with PyMuPDF. On generated tables with known cell
counts:

    strategy="lines"   ruled table    exact bbox, 4 rows x 4 cols   correct
    strategy="lines"   unruled table  nothing found
    strategy="text"    either         one 11x4 "table" spanning the whole page

So the line-based detector is *right*, and reimplementing it would be pure
waste. The text strategy is not usable as shipped, and the reason is
instructive: it is given no candidate region, so it treats the entire page --
headings, paragraphs, captions -- as one grid and finds the column structure of
the page rather than of a table.

This module therefore does two things and nothing else:

    ruled     delegate to find_tables(strategy="lines"). Do not touch it.
    unruled   find candidate REGIONS first, then look for column alignment
              inside them. The region constraint is the whole difference.

Unruled tables are worth the code: roughly half the tables in real documents
have no vertical rules at all, and a "table detector" that only sees boxes is a
line detector wearing a hat.
"""

from dataclasses import dataclass, field

# A table row is a line whose text is split into pieces by horizontal gaps
# bigger than a word space at that font size.
GAP_FACTOR = 1.9
# Column x-positions must agree between rows to within this many points.
COLUMN_TOLERANCE = 6.0
MIN_ROWS = 3
MIN_COLS = 2


@dataclass
class Table:
    bbox: tuple
    rows: int
    cols: int
    cells: list = field(default_factory=list)
    detector: str = "lines"

    @property
    def text(self):
        return "\n".join(" ".join(c or "" for c in row) for row in self.cells)


def ruled_tables(page):
    """Tables with visible rules. Straight through to PyMuPDF; do not touch."""
    out = []
    try:
        finder = page.find_tables(strategy="lines")
    except Exception:
        return out
    for table in finder.tables:
        try:
            cells = [[("" if v is None else str(v)) for v in row]
                     for row in table.extract()]
        except Exception:
            cells = []
        out.append(Table(bbox=tuple(table.bbox), rows=table.row_count,
                         cols=table.col_count, cells=cells, detector="lines"))
    return out


# --------------------------------------------------------------------------
# the unruled path
# --------------------------------------------------------------------------

def _visual_rows(page, body_band, x_range):
    """Words grouped into visual rows by vertical overlap, not by text order.

    This deliberately does not use PyMuPDF's line structure. Each cell of an
    unruled table is inserted separately, so the extractor reports the cells of
    one row as several *different* lines that happen to share a y. Reading the
    page as rows-of-ink instead of lines-of-text is what makes a row a row.
    """
    top, bottom = body_band
    lo, hi = x_range
    words = [w for w in page.get_text("words")
             if w[4].strip() and w[1] >= top and w[3] <= bottom
             and (w[0] + w[2]) / 2 >= lo - 1 and (w[0] + w[2]) / 2 <= hi + 1]
    words.sort(key=lambda w: (round(w[1], 1), w[0]))

    rows, current = [], []
    for word in words:
        if not current:
            current = [word]
            continue
        band_top = min(w[1] for w in current)
        band_bottom = max(w[3] for w in current)
        overlap = min(band_bottom, word[3]) - max(band_top, word[1])
        if overlap > (word[3] - word[1]) * 0.5:
            current.append(word)
        else:
            rows.append(sorted(current, key=lambda w: w[0]))
            current = [word]
    if current:
        rows.append(sorted(current, key=lambda w: w[0]))
    return rows


def _row_cells(words, body_size):
    """Split one visual row into cells at gaps wider than a word space.

    The split is on measured horizontal distance in points, scaled to the font
    size: what makes a cell boundary is *space on the page*, not anything in
    the text stream.
    """
    threshold = max(body_size * GAP_FACTOR, 6.0)
    cells, current = [], [words[0]]
    for previous, word in zip(words, words[1:]):
        if word[0] - previous[2] > threshold:
            cells.append(current)
            current = [word]
        else:
            current.append(word)
    cells.append(current)
    return [(" ".join(w[4] for w in group).strip(),
             (min(w[0] for w in group), min(w[1] for w in group),
              max(w[2] for w in group), max(w[3] for w in group)))
            for group in cells]


def unruled_tables(page, body_band, x_range, body_size):
    """Runs of consecutive rows whose cell x-positions agree, WITHIN ONE COLUMN.

    `x_range` is not an optimisation, it is the correctness condition. Run this
    over a whole two-column page and the left and right columns of ordinary
    prose read as two perfectly aligned cells repeated down the page: three
    "tables" on a page with none. A column is a reading context, and alignment
    only means anything inside one.

    `body_band` excludes the header and footer for the same reason -- a running
    head and a page number are two short strings at fixed x positions.

    Agreement is the whole test, and it is what running prose cannot fake.
    Ragged text produces cell boundaries at arbitrary x on every row; a table
    produces the same boundaries on every row, which is why the tolerance can
    be as tight as a few points.
    """
    rows = []
    for words in _visual_rows(page, body_band, x_range):
        cells = _row_cells(words, body_size)
        rows.append(cells if len(cells) >= MIN_COLS else None)

    tables, run = [], []
    for cells in rows + [None]:
        if cells is not None and (not run or _columns_agree(run[-1], cells)):
            run.append(cells)
            continue
        if len(run) >= MIN_ROWS:
            tables.append(_build(run))
        run = [cells] if cells is not None else []
    return tables


def _columns_agree(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    starts_a = [c[1][0] for c in a]
    starts_b = [c[1][0] for c in b]
    matched = sum(1 for x in starts_b
                  if any(abs(x - y) <= COLUMN_TOLERANCE for y in starts_a))
    return matched >= min(len(starts_a), len(starts_b)) - 0


def _build(run):
    cells = [[c[0] for c in row] for row in run]
    width = max(len(r) for r in cells)
    cells = [r + [""] * (width - len(r)) for r in cells]
    boxes = [c[1] for row in run for c in row]
    bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
    return Table(bbox=bbox, rows=len(cells), cols=width, cells=cells,
                 detector="alignment")


def _overlap_fraction(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / max(min((a[2] - a[0]) * (a[3] - a[1]),
                           (b[2] - b[0]) * (b[3] - b[1])), 1e-6)
