# Document Layout Intelligence

Turns a PDF into ordered, typed, hierarchical structure: titles, headings,
paragraphs, lists, tables, figures, captions and running heads — each with a
bounding box, a reading-order index and its text, as JSON.

The hard part is not finding the boxes. It is **reading order through multiple
columns** and **knowing when text was silently dropped**, because both of those
fail invisibly.

**On 36 generated pages with exact ground truth: recall 0.992, precision 0.970,
element-type accuracy 1.000, mean IoU 0.990, reading-order Kendall tau 1.000,
and text coverage 1.000.** PyMuPDF's own reading-order sort scores tau **0.566**
on the two-column pages. Full numbers, the scanned-page degradation, and the
five bugs found on the way in [RESULTS.md](RESULTS.md).

```bash
python cli.py --demo                 # generate a sample PDF and analyse it
python cli.py doc.pdf --json         # the machine interface
python cli.py doc.pdf --tables       # tables as grids
python test_layout.py                # 22/22
python bench.py                      # against PyMuPDF's raw and sorted output
```

---

## The rule the whole thing turns on

> **Every text span on the page belongs to exactly one output element, and the
> elements carry a total reading order.**

Both halves break silently, which is why they are the invariant rather than a
feature.

**"Exactly one" is a partition.** An extractor that quietly loses a column, a
footer, or the second half of every paragraph produces output that looks
perfect: correctly typed boxes, clean JSON, plausible text. Nothing raises. So
`coverage` — characters on the page divided by characters inside some element —
is computed on every page, printed by the CLI, included in the JSON, and
reported beside every accuracy number in RESULTS.md. `coverage < 1.0` is a
machine-checkable signal that something was dropped.

**"A total order" is the other half.** Draw boxes round the elements of a
two-column page and the picture is *identical* whether the order is correct or
interleaved. The failure only appears when the text is concatenated — and then
it is grammatical line by line and nonsense paragraph by paragraph, which is
exactly the failure that survives review.

---

## Ground truth is generated, on purpose

`synth.py` lays out each element at a known rectangle, with a known type and a
known position in the reading order, then writes the PDF and hands back the
labels. That buys element type, bbox, reading order and full text for every
element on multi-column pages, deterministically, in milliseconds — without
downloading PubLayNet.

It costs realism, and the cost is stated wherever a number is: these are clean
documents with no scanner skew, no JPEG noise, no hyphenation across columns
and no adversarial designer. **Every figure on the digital path is an upper
bound.** The scanned path in §OCR below is where the messy case lives.

Two generator bugs had to be fixed before any of its numbers meant anything,
and both are in the test suite now:

- It painted **white rectangles** while searching for a font size that fits.
  Those survive into the PDF as vector drawings, and the figure detector —
  correctly — reported a page-wide figure behind every paragraph.
- The labels were the **rectangles asked for**, not the ink laid down. A
  one-word heading requested in a 504×19 box renders as ~51×13 of ink, so every
  correct detection scored IoU well under 0.5 and the evaluation reported 28
  missed headings and 28 invented ones *in the same places*. Recall 0.31.

---

## Reading order: recursive XY-cut, and two wrong versions of it

Split the page on the widest whitespace, recurse into the halves, alternating
axes. Both mistakes below produce output that looks fine and scores badly, and
both are pinned by tests.

**Wrong version 1 — horizontal cut first.** Every band of the page with
whitespace across it becomes a cut, so a two-column page reads left, right,
left, right down the page.

**Wrong version 2 — take the widest gap, either axis.** Almost works. It fails
whenever both columns happen to break a paragraph at a similar height: that
shared band is wider than the 18pt gutter, so the page splits into a top half
and a bottom half, and each half is then correctly read in columns — giving
title, left-top, right-top, left-bottom, right-bottom. **tau 0.81 on exactly
the pages the algorithm exists for.**

**What works:** take a vertical cut whenever one is available, and when one is
not, cut at the *first* horizontal gap rather than the widest. A gutter is a
global structural fact about the group; a horizontal gap is a local one, so
structure comes first. A full-width title needs no special case at all — it
spans the gutter, so no vertical cut exists at the top level, and the
first-gap horizontal cut peels it off and leaves the rest of the group intact
for the columns.

---

## What a gutter actually is

Not "whitespace that is empty for most of the page height" — that was the first
version, and an unruled table's own internal whitespace satisfies it. The page
went two-column, the table was analysed one column at a time, and came back as
a 4×3 table with its first column emitted as loose paragraphs. Every stage
after the mistake was correct.

A gutter is defined by three things together:

1. **Almost no line crosses it.** Column text stops at the gutter by
   definition. Table rows, paragraphs above a table and captions all cross
   their local whitespace freely. (A full-width title crosses too, which is why
   the tolerance is 12% of lines rather than zero.)
2. **There is a real column of text on each side** — at least three lines.
3. **The two sides run alongside each other.** Their vertical extents must
   overlap over most of the shorter one. Text on both sides at *different
   heights* is the inside of something, not a gutter.

---

## Tables: use PyMuPDF where it is right

Measured before a line of detector code was written:

| | ruled table | unruled table |
|---|---|---|
| `find_tables(strategy="lines")` | exact bbox, correct rows × cols | **nothing found** |
| `find_tables(strategy="text")` | one 11×4 "table" spanning the page | same |

So the line-based detector is right and reimplementing it would be waste — it
is called directly. The text strategy is not usable as shipped, and the reason
is instructive: **it is given no candidate region**, so it finds the column
structure of the *page* rather than of a table.

`tables.py` therefore adds one thing: an alignment detector that runs **inside
a single column**, over rows-of-ink rather than lines-of-text (each cell of an
unruled table is a separate text line that happens to share a y). Roughly half
the tables in real documents have no rules at all, and a table detector that
only sees boxes is a line detector wearing a hat.

The region constraint is not an optimisation, it is the correctness condition:
run the same detector across a whole two-column page and the two columns of
prose read as two perfectly aligned cells repeated down the page — three
tables on a page with none.

---

## The scanned path, and the one learned component

A born-digital PDF hands you the font, the point size and a bold flag. A scan
hands you pixels: tesseract returns words and boxes and **nothing else**. That
removes the rules' primary signal, so the question is whether a learned
classifier — trained only on geometry and text shape, which it never had either
way — recovers it.

Both numbers, same fixtures, held-out documents:

| classifier | born-digital PDF | scanned (render → tesseract) |
|---|---|---|
| rules (`layout.classify`) | 0.9923 | 0.9316 |
| learned (`classify.py`, depth-8 tree, 3.5 KB) | **1.0000** | **0.9737** |

**The model earns its place exactly where the rules lose their signal, and
nowhere else.** On clean PDFs it ties. On scans it removes 61% of the remaining
type errors. It costs one 3.5 KB artefact, ~90 ms/page extra, and a
scikit-learn dependency — and it degrades: no model, no sklearn, or a bad
pickle all fall back to the rules, loudly.

Its most-used features are `size_ratio`, `starts_bullet`, `caption_lead` and
`chars_per_line`. `bold_fraction` scores **0.0** — the tree never uses it,
because it trained on scanned pages too, where bold does not exist.

`ocr.py`'s glyph-height-to-point-size constant is **calibrated, not assumed**
(`python ocr.py`): the textbook 1/0.72 = 1.389 is wrong by 30% here, because
tesseract's line box spans ascender to descender rather than cap height.

---

## Files

| file | what is in it |
|---|---|
| `layout.py` | the pipeline: lines, running heads, figures, columns, XY-cut, grouping, classification, nesting |
| `tables.py` | ruled tables via PyMuPDF, unruled tables via column alignment |
| `synth.py` | PDF generation with exact ground truth |
| `metrics.py` | coverage, detection, type accuracy, Kendall tau — reported together |
| `ocr.py` | render → tesseract → the same stages, plus scan degradation and calibration |
| `classify.py` | the learned classifier, its training set, and its degradation |
| `cli.py` | `--json`, `--tree`, `--text`, `--tables`, `--ocr`, `--classifier` |
| `bench.py` | six extractors on the same corpus, scored identically |
| `test_layout.py` | 22 tests, one per real bug, and a check that they fail on the old code |

```bash
pip install pymupdf                    # required
pip install scikit-learn               # only for --classifier model
pip install pytesseract pillow         # only for --ocr, plus the tesseract binary
```
