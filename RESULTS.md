# Results

Every number below has the command that produced it printed above it. All runs
are seeded and the corpus is generated, so they reproduce exactly.

Environment: Windows 11, Python 3.12.1, PyMuPDF 1.26.3, tesseract 5.5.0,
scikit-learn 1.9.0.

> **Read this first.** Every figure here is measured on **generated** documents
> (`synth.py`) with exact ground truth. They are clean: no scanner skew, no
> JPEG artefacts, no hyphenation across columns, no adversarial designer. **The
> born-digital numbers are an upper bound on real-world performance.** §4 is
> the degradation measurement — the same pages, rasterised and OCR'd — and it
> is the more honest guide to what a scan will do.

---

## 1. Against the boring baselines

```bash
python bench.py --docs 12
```

12 documents, 36 pages, all five page layouts.

| | recall | precision | type acc. | mean IoU | order tau | coverage | s/page |
|---|---|---|---|---|---|---|---|
| `raw` — PyMuPDF blocks, stream order | 0.8855 | 0.6170 | 0.0000 | 0.9998 | 1.0000 | – | **0.0021** |
| `sorted` — PyMuPDF `sort=True` | 0.8855 | 0.6170 | 0.0000 | 0.9998 | 0.9397 | – | **0.0013** |
| `layout` — this project, rules | 0.9924 | 0.9701 | 0.9923 | 0.9896 | 1.0000 | 1.0000 | 0.0651 |
| `layout+model` — with the learned classifier | 0.9924 | 0.9701 | **1.0000** | 0.9896 | 1.0000 | 1.0000 | 0.1353 |

Reading order (Kendall tau) broken down by page layout — this is where the
difference actually lives:

| | figure-table | list | single | **two-column** | unruled-table |
|---|---|---|---|---|---|
| `raw` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `sorted` | 1.0000 | 1.0000 | 1.0000 | **0.5657** | 1.0000 |
| `layout` | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |

**Two things in this table cut against the headline, and both stay.**

**The baselines are 30–50× faster.** 1.3 ms/page against 65 ms. If all you need
is text and the documents are single-column, `get_text` is the right answer and
this project is overhead. The recall and precision gap (0.89/0.62 vs 0.99/0.97)
is about *element boundaries* — PyMuPDF's blocks split and merge differently
from the ground truth — not about losing text.

**`raw` scores tau 1.000 on two-column pages, and that is an artefact of the
corpus, not a result.** `synth.py` writes elements to the page in reading
order, so the PDF content stream happens to be in reading order too, and "the
order PyMuPDF returns blocks" inherits it. Real PDFs make no such promise. The
honest comparison is against **`sorted`** — PyMuPDF's own reading-order
heuristic, which is what anyone would actually reach for and which does not
depend on stream order. It scores **0.5657** on two-column pages: right about
roughly half of all element pairs, i.e. it interleaves the columns.

---

## 2. Coverage: the number that can fail

Coverage is characters on the page divided by characters inside some output
element, computed **geometrically** (which of the page's own text lines fall
inside an element box) rather than by comparing string lengths.

```
coverage     1.0000  (worst page 1.0000)
```

across all 36 pages, and across all 30 OCR'd pages in §4.

It matters because it is the one metric here that catches a *silent* failure. A
layout extractor that loses a column produces correctly typed boxes, clean
JSON, plausible text and no error. Precision and type accuracy both stay high —
what it found, it found correctly. Only coverage moves.

Comparing string lengths instead was tried and is wrong: a table whose cells
are re-joined with different whitespace reports **>100%** coverage, which turns
a partition check into a number that cannot fail.

---

## 3. Tables: what PyMuPDF already does, and what it does not

Measured before any detector code was written:

```bash
python test_layout.py    # "PyMuPDF's text strategy is genuinely unusable here"
```

| | ruled table | unruled table |
|---|---|---|
| `find_tables(strategy="lines")` | exact bbox, correct rows × cols | **0 found** |
| `find_tables(strategy="text")` | one **11×4** table spanning the page | one 11×4 table spanning the page |

So `tables.ruled_tables` calls PyMuPDF directly and does nothing else — the
line-based detector is right, and reimplementing it would be pure waste.

The text strategy's failure is informative rather than a defect: it is handed
no candidate region, so it finds the column structure of the *page*. The
alignment detector in `tables.unruled_tables` differs from it in exactly one
way that matters — it runs **inside a single column** — and it recovers unruled
tables at the correct row and column counts (pinned by
`an unruled table is found, which find_tables cannot do`).

Both tests are guarded against becoming vacuous: the unruled test asserts
`find_tables(strategy="lines")` finds nothing first, and the text-strategy test
asserts it still swallows the page.

---

## 4. The scanned path: what a real scan costs

```bash
python bench.py --docs 10 --seed 500 --only ocr,ocr+model
```

The *same* generated pages, rasterised at 200 dpi and passed through tesseract.
Held-out seeds (the classifier trained on seeds 1–24; these are 500–509).

| | recall | precision | type acc. | mean IoU | order tau | coverage | s/page |
|---|---|---|---|---|---|---|---|
| born-digital (§1) | 0.9924 | 0.9701 | 0.9923 | 0.9896 | 1.0000 | 1.0000 | 0.065 |
| **OCR, rules** | 0.8311 | 0.7004 | 0.9572 | 0.7947 | 1.0000 | 1.0000 | 1.503 |
| **OCR, learned classifier** | 0.8311 | 0.7004 | **0.9840** | 0.7947 | 1.0000 | 1.0000 | 1.498 |

**Reading order survives the scan intact (tau 1.000 on every layout including
two-column).** That is the strongest result in this file: column detection and
the XY-cut work on nothing but geometry, and geometry is what OCR preserves.

**Everything that depended on the text layer degrades:**

- **mean IoU 0.990 → 0.795.** Tesseract's line boxes are looser than the PDF's.
- **precision 0.970 → 0.700**, and the error breakdown says why:
  `missed figure x15, table x13`. **The OCR path detects no figures and no
  tables at all** — there is no vector-drawing list on a bitmap and
  `find_tables` needs a text layer. Table cells therefore arrive as ordinary
  text blocks, which is where most of the 36 spurious captions and 23 spurious
  paragraphs come from. This is a stated gap, not a subtle failure: image-based
  region detection is not implemented.
- **23× slower** (1.5 s/page against 65 ms), essentially all of it tesseract.

---

## 5. Phase 3.5 — the learned classifier, and where it earns its place

| classifier | born-digital | scanned |
|---|---|---|
| rules (`layout.classify`) | 0.9923 | 0.9572 |
| learned (`classify.py`) | **1.0000** | **0.9840** |

Trained with:

```bash
python classify.py --train --seeds 24
```

- 350 training blocks, from seeds 1–24, **each page contributed twice**: once
  from the PDF text layer and once from a 150 dpi render through tesseract.
  Training on both is the design — a model trained only on clean pages learns
  to trust the exact font size and is no better than the rules when that size
  is an estimate.
- depth-8 decision tree, `min_samples_leaf=4`, `class_weight="balanced"`,
  **2,503 bytes** on disk.
- **Cost: +70 ms/page on the digital path** (65 → 135 ms), and effectively
  nothing on the OCR path where tesseract dominates (1.503 → 1.498 s, i.e.
  inside the noise). One scikit-learn dependency, one artefact.
- **Degradation:** no model, no scikit-learn, or an unpickling failure all
  return `None` from `predict`, and the rules run instead. `cli.py --classifier
  model` prints a warning to stderr rather than falling back silently.

**The verdict: it earns its place on scanned input and only there.** It removes
63% of the remaining type errors on scans (4.28% → 1.60%) and closes the last
0.77% on clean PDFs. If the input is born-digital, the rules are free, exact
and readable, and they stay the default.

### The feature the model does not use

```bash
python classify.py
```

| feature | importance |
|---|---|
| `size_ratio` | 0.44 |
| `caption_lead` | 0.24 |
| `starts_bullet` | 0.19 |
| `chars_per_line` | 0.12 |
| **`bold_fraction`** | **0.00** |
| everything else | 0.00 |

**The tree never splits on boldness**, despite it being one of the rules'
strongest signals. Because half its training set is OCR'd pages where
`bold = False` unconditionally, a split on bold cannot generalise across the
two halves and the tree finds better ones. That is the mechanism the whole
comparison was built to test, showing up directly in the model's own weights.

### One category error, caught by looking past the headline number

An earlier version had `table` in the classifier's label set. Its headline type
accuracy over matched elements was **0.9733** — fine. Its spurious count was
**52 invented tables across 30 pages.**

Tables are found geometrically, by rules or by column alignment. Letting a text
classifier emit that label is a category error, and *type accuracy over matched
elements cannot see it*, because an invented element never matches anything and
so never enters the average. Removing `table`, `figure`, `header` and `footer`
from the label set fixed it and raised type accuracy to 0.9840 as a side effect.

This is the argument for `metrics.py` reporting four numbers and never one.

---

## 6. Five bugs, and why each was invisible

```bash
python test_layout.py
```

```
22/22 passed

against the pre-fix implementations:
  pre-fix group_lines     2/2 of its tests fail
  pre-fix xy_cut          1/1 of its tests fail
  pre-fix column_bounds   2/2 of its tests fail
```

The second block is the point of the suite: `verify_suite_is_not_decorative()`
re-installs the three pre-fix implementations and re-runs the tests written
against them. All of them fail, as they must.

| # | bug | symptom | why nobody would notice |
|---|---|---|---|
| 1 | XY-cut cut horizontally first | two-column pages read left, right, left, right | boxes are drawn in the right places; only concatenated text is wrong |
| 2 | XY-cut took the *widest* gap | tau **0.81** when both columns broke a paragraph at the same height | the page is split into two halves, each read *correctly* in columns |
| 3 | gutter = "empty for most of the page height" | an unruled table's own whitespace split the page; the table came back **4×3** with its first column as loose paragraphs | every stage after the mistake was correct |
| 4 | line grouping used `gap > line_height × 1.6` | 17pt demanded where paragraphs sit 9pt apart; **recall 0.53** | paragraphs merge into plausible-looking longer paragraphs |
| 5 | ground truth labelled the *requested* rectangle, not the ink | **recall 0.31**; 28 headings reported missed and 28 invented, in the same places | the extractor was working; the evaluation was wrong |

Bug 5 is the one worth dwelling on. A one-word heading requested in a 504×19
box renders as roughly 51×13 of ink, so the label claimed a region that was 95%
empty page and every correct detection scored IoU well under 0.5. **The metric
was broken, not the code**, and the failure mode it produced — simultaneous
missed and spurious elements of the same type at the same coordinates — is the
signature to remember.

Two more, in `synth.py` itself:

- It painted **white rectangles** while searching for a font size that fits.
  Those survive into the PDF as vector drawings and the figure detector
  correctly reported a page-wide figure behind every paragraph. A generator has
  to leave no ink it did not label.
- `body_size` was first filtered by line width ("body text is set to the column
  measure"). It sounds principled and breaks on any page with a wide title,
  which becomes the widest line and pushes the real body lines under the
  threshold — body size 16.0 on a two-column page, and every paragraph
  classified as a caption.

---

## 7. Calibration, not assumption

```bash
python ocr.py
```

```
{'samples': 98, 'height_to_points': 1.0556, 'current': 1.0556}
```

`ocr.py` estimates point size from OCR glyph height. The textbook constant is
1/0.72 = **1.389**, from cap height being ~72% of the point size. Measured
against pages whose true font sizes are known, the right value here is
**1.0556** — 30% out, because tesseract's line box spans ascender to descender
plus slack, not cap height.

Shipping 1.389 would not have crashed anything. Every OCR'd body line would
have read as ~13pt against a 9.5pt truth — a *uniform* bias, so every ratio
downstream stays correct and nothing looks wrong. It is a named constant with a
`calibrate()` function beside it precisely because that class of error does not
surface on its own, and because it will be wrong again for a different typeface
or DPI.

---

## 8. What has NOT been verified

- **No real-world PDF corpus.** Everything is generated. PubLayNet and DocBank
  exist and were not used; the numbers here are an upper bound and the OCR
  section is the only degradation measured.
- **No figure or table detection on the scanned path.** Stated as a gap in §4,
  and it accounts for most of the OCR precision loss.
- **No non-Latin script, no rotated or skewed pages in the benchmark.**
  `ocr.render` takes `rotate`, `blur` and `noise` arguments and they work, but
  no numbers with them switched on are reported here.
- **The `--tree` hierarchy view is not scored.** `metrics.py` measures the flat
  ordered list. Nesting is derived from it and correct by construction, but no
  test compares a tree against a ground-truth tree.
- **The classifier is trained and tested on the same generator.** Held-out
  seeds, not held-out *distributions*. It has never seen a document this
  repository did not write.
