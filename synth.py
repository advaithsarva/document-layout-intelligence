"""Generate PDFs whose layout ground truth is known exactly.

WHY THIS FILE EXISTS
--------------------
Layout analysis has an evaluation problem that quietly ruins most projects
built on it: the output *looks* right. A page rendered with boxes drawn round
the paragraphs is convincing whether the reading order is correct or scrambled,
and whether 100% or 70% of the text survived. There is no error message. The
only honest way to know is labelled ground truth, and the public sets
(PubLayNet, DocBank) are gigabytes of download and licensed for research.

So the documents are *generated*: this file lays out each element at a known
rectangle with a known type and a known position in the reading order, writes
the PDF, and returns the labels alongside it. Everything in `metrics.py` is
measured against those labels.

What that buys, and what it costs, stated plainly:

    buys    element type, bbox, reading order and full text for every element,
            on multi-column pages, deterministically, in milliseconds
    costs   these are clean synthetic documents. They have no scanner skew, no
            JPEG noise, no hyphenation across columns, no ligature soup, and no
            adversarial designer. Numbers here are an UPPER BOUND on real-world
            performance, and RESULTS.md says so next to every one of them.

`ocr.py` is where the messy path lives; the rasterise-and-OCR route runs the
same generated pages through a real scan simulation, so the gap between the
two columns is itself a measurement.
"""

import random

import fitz     # PyMuPDF

# The element vocabulary. Deliberately small: every type here is one a
# downstream consumer treats differently, and nothing is included because a
# taxonomy paper had it.
ELEMENT_TYPES = ("title", "heading", "paragraph", "list", "table", "figure",
                 "caption", "header", "footer")

PAGE_W, PAGE_H = 612.0, 792.0          # US Letter, in points
MARGIN = 54.0

WORDS = ("system", "layout", "document", "structure", "column", "reading",
         "order", "element", "region", "extraction", "baseline", "measured",
         "against", "ground", "truth", "detection", "boundary", "heading",
         "paragraph", "evaluation", "coverage", "geometry", "cluster",
         "threshold", "projection", "recursive", "partition", "hierarchy")

TITLES = ("Recovering Reading Order in Multi-Column Documents",
          "A Geometric Approach to Layout Segmentation",
          "Measuring Text Coverage in Document Extraction",
          "Table Structure Recovery Without a Model",
          "On the Evaluation of Layout Analysis Systems")

HEADINGS = ("Introduction", "Related Work", "Method", "Column Detection",
            "Reading Order", "Table Recovery", "Experiments", "Results",
            "Error Analysis", "Discussion", "Conclusion", "Limitations")


class Element:
    """One labelled region: what it is, where it is, what it says, when it is read."""

    __slots__ = ("type", "bbox", "text", "order", "meta")

    def __init__(self, type, bbox, text, order, meta=None):
        self.type = type
        self.bbox = tuple(round(v, 2) for v in bbox)
        self.text = text
        self.order = order
        self.meta = meta or {}

    def to_dict(self):
        return {"type": self.type, "bbox": list(self.bbox), "text": self.text,
                "order": self.order, **({"meta": self.meta} if self.meta else {})}

    def __repr__(self):
        return f"<{self.type} #{self.order} {self.bbox}>"


def _sentence(rng, n):
    words = [rng.choice(WORDS) for _ in range(n)]
    return " ".join(words).capitalize() + "."


def _paragraph(rng, sentences=None):
    return " ".join(_sentence(rng, rng.randint(6, 14))
                    for _ in range(sentences or rng.randint(2, 5)))


class PageBuilder:
    """Writes elements onto a page and records what it wrote.

    The builder is the only thing that knows the truth, and it knows it because
    it *placed* the text -- not because it inferred anything afterwards. That
    is the entire reason the ground truth can be trusted.
    """

    def __init__(self, page, rng):
        self.page = page
        self.rng = rng
        self.elements = []
        self._order = 0

    def _next_order(self):
        self._order += 1
        return self._order

    def text_block(self, type, rect, text, size=9.5, font="helv", align=0,
                   order=None):
        """Insert text, shrinking the font until it fits rather than clipping.

        Overflowing text that PyMuPDF silently drops would make the ground
        truth wrong -- the label would claim text the page does not contain,
        and every coverage number after that would be measuring the generator's
        bug instead of the extractor.

        The fit is searched on a THROWAWAY page. Retrying on the real one and
        painting a white rectangle between attempts is the obvious way to do
        this and it is wrong: the white rectangles survive into the PDF as
        vector drawings, and the figure detector -- correctly -- reported a
        page-wide figure behind every paragraph. The generator has to leave no
        ink it did not label.
        """
        box = fitz.Rect(*rect)
        scratch = fitz.open()
        try:
            probe = scratch.new_page(width=PAGE_W, height=PAGE_H)
            for _ in range(16):
                if probe.insert_textbox(box, text, fontsize=size,
                                        fontname=font, align=align) >= 0:
                    break
                size -= 0.4
                probe = scratch.new_page(width=PAGE_W, height=PAGE_H)
            else:
                raise RuntimeError(f"{type} did not fit in {rect} even at {size}pt")
        finally:
            scratch.close()
        self.page.insert_textbox(box, text, fontsize=size, fontname=font,
                                 align=align)

        element = Element(type, self._ink(box), text,
                          order or self._next_order())
        self.elements.append(element)
        return element

    def _ink(self, box):
        """The bbox of the ink actually laid down, not the box asked for.

        These differ a lot and it matters: a one-word heading requested in a
        full-width 504x19 rectangle renders as roughly 51x13 of ink. Labelling
        the request means the ground truth claims a region that is 95% empty
        page, every correct detection scores IoU well under 0.5, and the
        evaluation reports the detector missing 28 headings while
        simultaneously inventing 28 headings in the same places.

        Ground truth has to describe the page, not the generator's intent.
        """
        found = None
        for block in self.page.get_text("dict", clip=box)["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    rect = fitz.Rect(span["bbox"])
                    found = rect if found is None else (found | rect)
        return found if found is not None else box

    def figure(self, rect, order=None):
        box = fitz.Rect(*rect)
        self.page.draw_rect(box, color=(0.45, 0.45, 0.45), width=0.8,
                            fill=(0.88, 0.88, 0.9))
        for _ in range(self.rng.randint(3, 7)):
            x0 = self.rng.uniform(box.x0 + 4, box.x1 - 20)
            y0 = self.rng.uniform(box.y0 + 4, box.y1 - 4)
            self.page.draw_line(fitz.Point(x0, y0),
                                fitz.Point(min(x0 + 18, box.x1 - 2), y0),
                                color=(0.3, 0.3, 0.55), width=1.2)
        element = Element("figure", box, "", order or self._next_order())
        self.elements.append(element)
        return element

    def table(self, rect, rows, cols, ruled=True, order=None):
        """A real grid of cells, with the cell text recorded per row.

        `ruled=False` produces a table held together only by column alignment
        and whitespace -- no lines at all. That is the case that separates a
        table detector from a line detector, and roughly half the tables in
        real documents look like it.
        """
        box = fitz.Rect(*rect)
        row_h = box.height / rows
        col_w = box.width / cols

        cells = []
        for r in range(rows):
            row = []
            for c in range(cols):
                value = (f"{self.rng.choice(WORDS)[:8]}" if c == 0 else
                         f"{self.rng.randint(1, 999)}"
                         if r else self.rng.choice(WORDS)[:7].title())
                row.append(value)
                cell = fitz.Rect(box.x0 + c * col_w + 2, box.y0 + r * row_h + 1,
                                 box.x0 + (c + 1) * col_w - 2,
                                 box.y0 + (r + 1) * row_h - 1)
                self.page.insert_textbox(cell, value, fontsize=7.5,
                                         fontname="hebo" if r == 0 else "helv")
            cells.append(row)

        if ruled:
            for r in range(rows + 1):
                y = box.y0 + r * row_h
                self.page.draw_line(fitz.Point(box.x0, y), fitz.Point(box.x1, y),
                                    color=(0.2, 0.2, 0.2), width=0.6)
            for c in range(cols + 1):
                x = box.x0 + c * col_w
                self.page.draw_line(fitz.Point(x, box.y0), fitz.Point(x, box.y1),
                                    color=(0.2, 0.2, 0.2), width=0.6)

        text = "\n".join(" ".join(row) for row in cells)
        element = Element("table", box, text, order or self._next_order(),
                          meta={"rows": rows, "cols": cols, "ruled": ruled,
                                "cells": cells})
        self.elements.append(element)
        return element


def _running_heads(builder, page_number, with_header=True, with_footer=True):
    """Header and footer, placed OUTSIDE the reading order (order = 0).

    They are the standard trap. Geometrically they are ordinary text lines at
    the top and bottom of the page; semantically they repeat on every page and
    must not be spliced into the middle of a sentence when pages are
    concatenated. An extractor that gets everything else right and inlines the
    running head produces text that reads like a fault.
    """
    if with_header:
        builder.elements.append(Element(
            "header", (MARGIN, 30, PAGE_W - MARGIN, 44),
            "Document Layout Intelligence", 0))
        builder.page.insert_textbox(
            fitz.Rect(MARGIN, 30, PAGE_W - MARGIN, 44),
            "Document Layout Intelligence", fontsize=8, fontname="hebo")
    if with_footer:
        builder.elements.append(Element(
            "footer", (MARGIN, PAGE_H - 48, PAGE_W - MARGIN, PAGE_H - 34),
            str(page_number), 0))
        builder.page.insert_textbox(
            fitz.Rect(MARGIN, PAGE_H - 48, PAGE_W - MARGIN, PAGE_H - 34),
            str(page_number), fontsize=8, align=1)


def single_column_page(page, rng, page_number=1):
    b = PageBuilder(page, rng)
    _running_heads(b, page_number)
    y = 70.0
    b.text_block("title", (MARGIN, y, PAGE_W - MARGIN, y + 44),
                 rng.choice(TITLES), size=17, font="hebo")
    y += 54
    for _ in range(rng.randint(2, 3)):
        b.text_block("heading", (MARGIN, y, PAGE_W - MARGIN, y + 19),
                     rng.choice(HEADINGS), size=11.5, font="hebo")
        y += 20
        for _ in range(rng.randint(1, 2)):
            height = rng.randint(58, 82)
            b.text_block("paragraph", (MARGIN, y, PAGE_W - MARGIN, y + height),
                         _paragraph(rng))
            y += height + 10
        if y > PAGE_H - 200:
            break
    return b.elements


def two_column_page(page, rng, page_number=1, with_span=True):
    """The case reading order exists for.

    Two columns plus a full-width element. Sorting blocks top-to-bottom -- what
    a naive extractor does -- interleaves the columns and produces text that is
    grammatical line by line and nonsense paragraph by paragraph.
    """
    b = PageBuilder(page, rng)
    _running_heads(b, page_number)
    gutter = 18.0
    col_w = (PAGE_W - 2 * MARGIN - gutter) / 2
    left_x, right_x = MARGIN, MARGIN + col_w + gutter
    y = 70.0

    if with_span:
        b.text_block("title", (MARGIN, y, PAGE_W - MARGIN, y + 40),
                     rng.choice(TITLES), size=16, font="hebo")
        y += 50

    top = y
    for x in (left_x, right_x):
        y = top
        b.text_block("heading", (x, y, x + col_w, y + 18),
                     rng.choice(HEADINGS), size=11, font="hebo")
        y += 19
        while y < PAGE_H - 130:
            height = rng.randint(60, 90)
            if y + height > PAGE_H - 130:
                break
            b.text_block("paragraph", (x, y, x + col_w, y + height),
                         _paragraph(rng))
            y += height + 9
    return b.elements


def figure_and_table_page(page, rng, page_number=1, ruled=True):
    b = PageBuilder(page, rng)
    _running_heads(b, page_number)
    y = 70.0
    b.text_block("heading", (MARGIN, y, PAGE_W - MARGIN, y + 19),
                 rng.choice(HEADINGS), size=12, font="hebo")
    y += 22
    b.text_block("paragraph", (MARGIN, y, PAGE_W - MARGIN, y + 62), _paragraph(rng))
    y += 72

    b.figure((MARGIN + 40, y, PAGE_W - MARGIN - 40, y + 120))
    y += 126
    b.text_block("caption", (MARGIN + 40, y, PAGE_W - MARGIN - 40, y + 14),
                 f"Figure {rng.randint(1, 6)}: {_sentence(rng, 7)}",
                 size=8, align=1)
    y += 26

    rows, cols = rng.randint(4, 6), rng.randint(3, 4)
    b.table((MARGIN + 30, y, PAGE_W - MARGIN - 30, y + 16 * rows), rows, cols,
            ruled=ruled)
    y += 16 * rows + 6
    b.text_block("caption", (MARGIN + 30, y, PAGE_W - MARGIN - 30, y + 14),
                 f"Table {rng.randint(1, 4)}: {_sentence(rng, 6)}",
                 size=8, align=1)
    return b.elements


def list_page(page, rng, page_number=1):
    b = PageBuilder(page, rng)
    _running_heads(b, page_number)
    y = 70.0
    b.text_block("heading", (MARGIN, y, PAGE_W - MARGIN, y + 19),
                 rng.choice(HEADINGS), size=12, font="hebo")
    y += 22
    b.text_block("paragraph", (MARGIN, y, PAGE_W - MARGIN, y + 50),
                 _paragraph(rng, sentences=2))
    y += 60
    items = "\n".join(f"{i + 1}. {_sentence(rng, rng.randint(5, 9))}"
                      for i in range(rng.randint(3, 6)))
    b.text_block("list", (MARGIN + 16, y, PAGE_W - MARGIN, y + 15 * (items.count("\n") + 1) + 8),
                 items)
    y += 15 * (items.count("\n") + 1) + 20
    b.text_block("paragraph", (MARGIN, y, PAGE_W - MARGIN, y + 60), _paragraph(rng))
    return b.elements


LAYOUTS = {
    "single": single_column_page,
    "two-column": two_column_page,
    "figure-table": figure_and_table_page,
    "unruled-table": lambda p, r, n=1: figure_and_table_page(p, r, n, ruled=False),
    "list": list_page,
}


def build_document(path, layouts=("single", "two-column", "figure-table", "list"),
                   seed=42):
    """Write a PDF and return `[[Element, ...], ...]`, one list per page."""
    rng = random.Random(seed)
    doc = fitz.open()
    truth = []
    for i, name in enumerate(layouts):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        truth.append(LAYOUTS[name](page, rng, i + 1))
    doc.save(path)
    doc.close()
    return truth


def corpus(directory, n=12, seed=42):
    """A seeded set of documents covering every layout, for bench.py."""
    import os
    os.makedirs(directory, exist_ok=True)
    names = list(LAYOUTS)
    out = []
    for i in range(n):
        rng = random.Random(seed + i)
        layouts = [names[i % len(names)]] + rng.sample(names, 2)
        path = os.path.join(directory, f"doc_{i:03d}.pdf")
        out.append((path, build_document(path, layouts, seed=seed + i)))
    return out


if __name__ == "__main__":
    truth = build_document("sample.pdf")
    for i, page in enumerate(truth):
        print(f"page {i + 1}: " + ", ".join(f"{e.type}#{e.order}" for e in page))
    print("wrote sample.pdf")
