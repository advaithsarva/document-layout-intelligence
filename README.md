# Document Layout Intelligence

Turn a PDF into an ordered, typed, hierarchical structure: titles, headings, paragraphs, lists, tables, figures, captions, and running heads. Each element includes its bounding box, reading-order index, and text, returned as JSON.

The difficult part is not detecting the boxes. It is **getting reading order right across multiple columns** and **detecting when text has been silently lost**. Both failures can produce output that looks correct at first glance.

**On 36 generated pages with exact ground truth: recall was 0.992, precision 0.970, element-type accuracy 1.000, mean IoU 0.990, reading-order Kendall tau 1.000, and text coverage 1.000.** PyMuPDF's own reading-order sort scored **0.566 tau** on the two-column pages. Full results, scanned-page degradation, and the five bugs found during development are in [RESULTS.md](RESULTS.md).

```bash id="k4z6q1"
python cli.py --demo                 # generate a sample PDF and analyse it

python cli.py doc.pdf --json         # machine-readable output

python cli.py doc.pdf --tables       # tables as grids

python test_layout.py                # 22/22

python bench.py                      # compare against PyMuPDF raw and sorted output
```

---

## The rule the whole thing turns on

> **Every text span on the page belongs to exactly one output element, and the elements have a total reading order.**

Both parts can fail without producing an error, which is why they are treated as invariants rather than optional features.

**"Exactly one" means the elements form a partition.** An extractor can silently lose a column, a footer, or half of every paragraph while still producing clean JSON with plausible boxes and types. Nothing necessarily raises an exception.

For that reason, `coverage` is calculated on every page: characters found inside output elements divided by the characters present on the page. It is printed by the CLI, included in the JSON, and reported alongside every accuracy number in `RESULTS.md`.

A `coverage < 1.0` result gives a machine-checkable indication that some text was dropped.

**"A total order" is the other half.** On a two-column page, the boxes can look exactly right whether the reading order is correct or interleaved. The problem only becomes obvious when the text is concatenated. Individual lines can still look grammatical while the resulting paragraphs make no sense. That is the kind of error that can survive a visual review.

---

## Ground truth is generated on purpose

`synth.py` creates each element at a known rectangle with a known type and reading-order position. It then writes the PDF and returns the corresponding labels.

This gives exact ground truth for element type, bounding box, reading order, and full text, including multi-column pages. It is deterministic and runs in milliseconds without downloading PubLayNet.

The tradeoff is realism, and that limitation is stated wherever the numbers are reported. These are clean documents with no scanner skew, JPEG noise, column-spanning hyphenation, or deliberately difficult layouts.

**The digital-path results are therefore an upper bound.** The scanned path in the OCR section is where the messier cases appear.

Two bugs in the generator had to be fixed before the benchmark results were meaningful. Both are now covered by tests:

* It drew **white rectangles** while searching for a font size that would fit. Those rectangles remained in the PDF as vector drawings, so the figure detector correctly interpreted them as page-wide figures behind the paragraphs.

* The ground-truth labels represented the **requested rectangles**, rather than the actual ink rendered inside them. A one-word heading requested in a 504×19 box might render as only about 51×13 of ink. As a result, correct detections received IoU scores well below 0.5, producing 28 missed headings and 28 false detections in the same locations. Recall fell to 0.31.

---

## Reading order: recursive XY-cut, including two wrong versions

The page is split on whitespace and the process is applied recursively, alternating between axes.

Two implementations looked reasonable but produced incorrect results. Both are covered by tests.

**Wrong version 1 — horizontal cuts first.**

Any horizontal band with whitespace across the page becomes a cut. On a two-column page, this produces an order like left, right, left, right down the page.

**Wrong version 2 — choose the widest gap on either axis.**

This gets closer, but fails when both columns happen to end a paragraph at roughly the same height. That shared horizontal gap can be wider than the 18pt column gutter, so the algorithm splits the page into a top and bottom half. Each half is then correctly processed as columns, producing:

`title → left-top → right-top → left-bottom → right-bottom`

That version scored **0.81 Kendall tau** on the pages where the algorithm was intended to work.

**The working approach** is to take a vertical cut whenever one exists. If there is no vertical cut, use the **first horizontal gap**, rather than the widest one.

A gutter is a structural property of the group. A horizontal gap is usually local, so the algorithm gives the structural split priority.

A full-width title requires no special case. Because it spans the gutter, there is no vertical cut at the top level. The first horizontal gap then separates the title from the content below while leaving the column structure intact.

---

## What a gutter actually is

A gutter is not simply "whitespace that is empty for most of the page height." That was the first definition, and it broke on unruled tables.

The whitespace inside an unruled table can satisfy that condition. The page was incorrectly split into columns, the table was analysed one column at a time, and the result came back as a 4×3 table with its first column emitted as separate paragraphs. Every later stage was working correctly; the initial gutter detection was wrong.

A gutter is identified from three conditions together:

1. **Almost no line crosses it.** Column text stops at the gutter. Table rows, paragraphs above a table, and captions can cross their local whitespace. Full-width titles can also cross it, so the tolerance is 12% of lines rather than zero.

2. **There is real text on both sides.** Each side needs at least three lines.

3. **The two sides run alongside each other.** Their vertical extents must overlap across most of the shorter side. Text appearing at unrelated heights is more likely to be part of the same internal structure than a column gutter.

---

## Tables: use PyMuPDF where it works

Before writing the table detector, the available PyMuPDF strategies were measured:

|                                 | Ruled table                        | Unruled table     |
| ------------------------------- | ---------------------------------- | ----------------- |
| `find_tables(strategy="lines")` | Exact bbox, correct rows × cols    | **Nothing found** |
| `find_tables(strategy="text")`  | One 11×4 "table" spanning the page | Same              |

The line-based detector handles ruled tables correctly, so reimplementing it would add unnecessary code. It is called directly.

The text strategy has a different problem: it receives no candidate region. As a result, it detects the column structure of the **whole page**, rather than the structure of a table.

`tables.py` adds an alignment detector that works **inside a single column**. It operates on rows of ink rather than lines of text. In an unruled table, each cell can be a separate text line that happens to share the same y-position.

Roughly half of the tables in real documents have no visible rules. A detector that only sees boxes can therefore mistake aligned prose for a table.

The region constraint is part of correctness, not just an optimisation. Running the same detector across an entire two-column page causes the two prose columns to look like two perfectly aligned cells repeated down the page, producing three apparent tables on a page that contains none.

---

## The scanned path, and the one learned component

A born-digital PDF provides font information, point size, and whether text is bold. A scan provides pixels. Tesseract returns words and bounding boxes, but not those font properties.

That removes some of the signals used by the rules. The question is whether a small learned classifier can recover some of that information from geometry and text shape.

Using the same fixtures and held-out documents:

| Classifier                                    | Born-digital PDF | Scanned (render → Tesseract) |
| --------------------------------------------- | ---------------: | ---------------------------: |
| Rules (`layout.classify`)                     |           0.9923 |                       0.9316 |
| Learned (`classify.py`, depth-8 tree, 3.5 KB) |       **1.0000** |                   **0.9737** |

The learned model is used only where the rules lose useful signals. On clean PDFs it ties the rule-based approach. On scans, it removes 61% of the remaining type errors.

The cost is a 3.5 KB model file, about 90 ms of additional processing per page, and a scikit-learn dependency. If the model is missing, scikit-learn is unavailable, or the pickle is invalid, the code falls back to the rules and reports that fallback.

The features used most often are `size_ratio`, `starts_bullet`, `caption_lead`, and `chars_per_line`. `bold_fraction` has an importance of **0.0**. The tree never uses it because the training set also contains scanned pages, where bold information is unavailable.

`ocr.py` also calibrates its glyph-height-to-point-size conversion rather than assuming a standard value. The textbook `1 / 0.72 = 1.389` conversion is about 30% off in this setup because Tesseract's line box extends from the ascender to the descender rather than measuring cap height.

---

## Files

| File             | What it contains                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| `layout.py`      | Main pipeline: lines, running heads, figures, columns, XY-cut, grouping, classification, and nesting |
| `tables.py`      | Ruled tables through PyMuPDF and unruled tables through column alignment                             |
| `synth.py`       | PDF generation with exact ground truth                                                               |
| `metrics.py`     | Coverage, detection, type accuracy, and Kendall tau                                                  |
| `ocr.py`         | Render → Tesseract → the same processing stages, plus scan degradation and calibration               |
| `classify.py`    | Learned classifier, training set, and degradation measurements                                       |
| `cli.py`         | `--json`, `--tree`, `--text`, `--tables`, `--ocr`, and `--classifier`                                |
| `bench.py`       | Six extractors evaluated on the same corpus with the same metrics                                    |
| `test_layout.py` | 22 tests covering real bugs, including checks against the old code                                   |

```bash id="2p9n6v"
pip install pymupdf                    # required

pip install scikit-learn               # only for --classifier

pip install pytesseract pillow          # only for --ocr, plus the Tesseract binary
```
