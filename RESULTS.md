# Results

Every number below is tied to the command shown immediately above it. All runs use fixed seeds and a generated corpus, so the results are reproducible.

**Environment:** Windows 11, Python 3.12.1, PyMuPDF 1.26.3, Tesseract 5.5.0, scikit-learn 1.9.0.

> **Read this first.** Every result here comes from **generated documents** produced by `synth.py` with exact ground truth. They are clean: no scanner skew, JPEG artefacts, column-spanning hyphenation, or deliberately difficult layouts. **The born-digital results are therefore an upper bound on real-world performance.** Section 4 measures what happens when the same pages are rasterised and OCR'd, which is a better indication of scan performance.

---

## 1. Against the basic baselines

```bash id="f8r2km"
python bench.py --docs 12
```

The benchmark uses 12 documents and 36 pages covering all five page layouts.

|                                          | Recall | Precision | Type accuracy | Mean IoU | Order tau | Coverage |     s/page |
| ---------------------------------------- | -----: | --------: | ------------: | -------: | --------: | -------: | ---------: |
| `raw` — PyMuPDF blocks, stream order     | 0.8855 |    0.6170 |        0.0000 |   0.9998 |    1.0000 |        – | **0.0021** |
| `sorted` — PyMuPDF `sort=True`           | 0.8855 |    0.6170 |        0.0000 |   0.9998 |    0.9397 |        – | **0.0013** |
| `layout` — this project, rules           | 0.9924 |    0.9701 |        0.9923 |   0.9896 |    1.0000 |   1.0000 |     0.0651 |
| `layout+model` — with learned classifier | 0.9924 |    0.9701 |    **1.0000** |   0.9896 |    1.0000 |   1.0000 |     0.1353 |

The reading-order results by page layout show where the difference matters:

|          | Figure-table |   List | Single | **Two-column** | Unruled-table |
| -------- | -----------: | -----: | -----: | -------------: | ------------: |
| `raw`    |       1.0000 | 1.0000 | 1.0000 |         1.0000 |        1.0000 |
| `sorted` |       1.0000 | 1.0000 | 1.0000 |     **0.5657** |        1.0000 |
| `layout` |       1.0000 | 1.0000 | 1.0000 |     **1.0000** |        1.0000 |

**Two results are worth keeping in mind when reading the headline numbers.**

**The baselines are 30–50× faster.** PyMuPDF takes 1.3 ms/page with `sort=True`, compared with 65 ms/page for the rule-based layout pipeline. If the only requirement is extracting text from clean, single-column documents, `get_text` is the better choice. The recall and precision difference, 0.89/0.62 versus 0.99/0.97, mainly comes from **element boundaries**. PyMuPDF's blocks split and merge differently from the generated ground truth; it is not primarily losing text.

**`raw` scoring tau 1.000 on two-column pages is a corpus artefact.** `synth.py` writes elements to the PDF in reading order, so the PDF content stream also happens to follow that order. PyMuPDF then returns the blocks in the same order.

Real PDFs do not guarantee this. The more useful comparison is **`sorted`**, PyMuPDF's own reading-order heuristic. It does not depend on the content-stream order and scores **0.5657** on the two-column pages, meaning the columns are substantially interleaved.

---

## 2. Coverage: the metric that catches silent loss

Coverage is calculated as characters on the page divided by characters contained in output elements. The calculation is **geometric**: it checks which of the page's own text lines fall inside an element's bounding box rather than comparing string lengths.

```text
coverage    1.0000  (worst page 1.0000)
```

This holds across all 36 pages and all 30 OCR'd pages in Section 4.

Coverage matters because it detects a failure that the other metrics can miss. An extractor can lose an entire column while still producing correctly typed boxes, clean JSON, plausible text, and no error. Precision and type accuracy can remain high because everything that *was* found was classified correctly. Coverage is the metric that exposes the missing text.

String-length comparison was tested and rejected. If table cells are rejoined with different whitespace, the calculated coverage can exceed **100%**, turning what should be a partition check into a metric that cannot reliably fail.

---

## 3. Tables: what PyMuPDF handles, and what it does not

The table strategies were measured before implementing the custom detector:

```bash id="a6n3vx"
python test_layout.py
# "PyMuPDF's text strategy is genuinely unusable here"
```

|                                 | Ruled table                          | Unruled table                    |
| ------------------------------- | ------------------------------------ | -------------------------------- |
| `find_tables(strategy="lines")` | Exact bbox, correct rows × cols      | **0 found**                      |
| `find_tables(strategy="text")`  | One **11×4** table spanning the page | One 11×4 table spanning the page |

`tables.ruled_tables` therefore calls PyMuPDF directly. The line-based detector already works, so there is no reason to reimplement it.

The text strategy fails for a useful reason rather than because the detector itself is broken. It receives no candidate region, so it detects the column structure of the **page** instead of the structure of a table.

The alignment detector in `tables.unruled_tables` changes the relevant part: it operates **inside a single column**. This allows it to recover unruled tables with the correct row and column counts, covered by the test `an unruled table is found, which find_tables cannot do`.

Both tests are designed to avoid becoming meaningless if PyMuPDF changes behaviour. The unruled-table test first confirms that `find_tables(strategy="lines")` finds nothing, while the text-strategy test confirms that it still incorrectly consumes the page.

---

## 4. The scanned path: what a real scan costs

```bash id="c3w7pz"
python bench.py --docs 10 --seed 500 --only ocr,ocr+model
```

These are the **same generated pages**, rendered at 200 DPI and passed through Tesseract. The seeds are held out: the classifier was trained on seeds 1–24, while this benchmark uses seeds 500–509.

|                             | Recall | Precision | Type accuracy | Mean IoU | Order tau | Coverage | s/page |
| --------------------------- | -----: | --------: | ------------: | -------: | --------: | -------: | -----: |
| Born-digital (§1)           | 0.9924 |    0.9701 |        0.9923 |   0.9896 |    1.0000 |   1.0000 |  0.065 |
| **OCR, rules**              | 0.8311 |    0.7004 |        0.9572 |   0.7947 |    1.0000 |   1.0000 |  1.503 |
| **OCR, learned classifier** | 0.8311 |    0.7004 |    **0.9840** |   0.7947 |    1.0000 |   1.0000 |  1.498 |

**Reading order survives the scan.** Kendall tau remains 1.000 across every layout, including two-column pages. Column detection and XY-cut rely on geometry, and that information survives the OCR process well enough for the reading-order stage to keep working.

The parts that depend on the PDF text layer degrade:

* **Mean IoU drops from 0.990 to 0.795.** Tesseract's line boxes are less precise than the PDF's original text geometry.

* **Precision drops from 0.970 to 0.700.** The error breakdown shows why: `missed figure x15, table x13`. The OCR path currently detects neither figures nor tables. A bitmap has no vector-drawing list, and `find_tables` requires a text layer. Table cells therefore arrive as ordinary text blocks. This accounts for much of the 36 false captions and 23 false paragraphs. Image-based region detection is not implemented yet.

* **Processing becomes about 23× slower.** The pipeline takes about 1.5 seconds/page instead of 65 ms, with most of the additional time coming from Tesseract.

---

## 5. Phase 3.5 — the learned classifier

| Classifier                | Born-digital |    Scanned |
| ------------------------- | -----------: | ---------: |
| Rules (`layout.classify`) |       0.9923 |     0.9572 |
| Learned (`classify.py`)   |   **1.0000** | **0.9840** |

The model is trained with:

```bash id="n7k2pd"
python classify.py --train --seeds 24
```

* **350 training blocks** from seeds 1–24. Each page contributes twice: once from the PDF text layer and once from a 150 DPI render passed through Tesseract.

  Training on both representations is intentional. A model trained only on clean PDFs can learn to rely on exact font size, which is not available on scans.

* **Depth-8 decision tree**, `min_samples_leaf=4`, `class_weight="balanced"`, occupying **2,503 bytes** on disk.

* **Cost:** about +70 ms/page on the digital path, increasing latency from 65 ms to 135 ms. On the OCR path, where Tesseract dominates, the difference is effectively zero: 1.503 → 1.498 s/page.

* **Fallback behaviour:** if the model is missing, scikit-learn is unavailable, or unpickling fails, `predict` returns `None` and the rules are used instead. `cli.py --classifier model` prints a warning to stderr rather than silently falling back.

**The classifier earns its place on scanned input.** It removes 63% of the remaining type errors on scans, reducing them from 4.28% to 1.60%, and closes the remaining 0.77% gap on clean PDFs.

For born-digital input, the rule-based classifier is simple, fast, and already accurate, so it remains the default.

### The feature the model does not use

```bash id="r5x1cv"
python classify.py
```

| Feature             | Importance |
| ------------------- | ---------: |
| `size_ratio`        |       0.44 |
| `caption_lead`      |       0.24 |
| `starts_bullet`     |       0.19 |
| `chars_per_line`    |       0.12 |
| **`bold_fraction`** |   **0.00** |
| Everything else     |       0.00 |

**The tree never splits on boldness**, even though boldness is one of the strongest signals used by the rules.

Half of the training data comes from OCR'd pages, where `bold = False` unconditionally. A split based on boldness therefore cannot generalise across both representations, so the tree chooses other features instead. The feature-importance output shows this directly.

### One category error hidden by the headline number

An earlier classifier included `table` in its label set. Its matched-element type accuracy was **0.9733**, which looked reasonable. But it also produced **52 invented tables across 30 pages**.

Tables are detected geometrically, either through rules or column alignment. Allowing a text classifier to emit the `table` label mixes two different detection mechanisms.

Matched-element type accuracy cannot detect this problem. An invented element does not match ground truth, so it never enters that average.

Removing `table`, `figure`, `header`, and `footer` from the classifier's label set fixed the issue and raised type accuracy to **0.9840**.

This is why `metrics.py` reports four separate measurements rather than reducing the evaluation to one headline number.

---

## 6. Five bugs, and why they were easy to miss

```bash id="u3m8qa"
python test_layout.py
```

```text
22/22 passed

against the pre-fix implementations:

  pre-fix group_lines      2/2 of its tests fail
  pre-fix xy_cut           1/1 of its tests fail
  pre-fix column_bounds    2/2 of its tests fail
```

The important part of the suite is `verify_suite_is_not_decorative()`. It reinstalls the three pre-fix implementations and runs the tests written for them. Each one fails as expected.

| # | Bug                                                                            | Symptom                                                                                                           | Why it was easy to miss                                                                |
| - | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1 | XY-cut cut horizontally first                                                  | Two-column pages read left, right, left, right                                                                    | The boxes were in the right places; only concatenated text exposed the error           |
| 2 | XY-cut chose the **widest** gap                                                | Tau **0.81** when both columns broke a paragraph at the same height                                               | The page was divided into two halves, each of which was then correctly read in columns |
| 3 | Gutter = "empty for most of the page height"                                   | An unruled table split the page; it returned as a **4×3** table with its first column emitted as loose paragraphs | Every stage after the initial split was correct                                        |
| 4 | Line grouping used `gap > line_height × 1.6`                                   | Required 17pt separation where paragraphs sit 9pt apart; **recall 0.53**                                          | Paragraphs merged into longer paragraphs that still looked plausible                   |
| 5 | Ground truth labelled the **requested** rectangle rather than the rendered ink | **Recall 0.31**; 28 headings reported as missed and 28 as invented in the same places                             | The extractor was working; the evaluation data was wrong                               |

Bug 5 is particularly useful because it shows how an evaluation problem can look like an extraction problem.

A one-word heading requested in a 504×19 box might render as only about 51×13 of ink. The ground-truth label therefore described a region that was roughly 95% empty. Correct detections then received IoU scores well below 0.5.

**The metric was broken, not the extractor.** The resulting pattern — missed and spurious elements of the same type at the same coordinates — is a useful diagnostic signature.

There were two additional problems in `synth.py`:

* It drew **white rectangles** while searching for a font size that would fit. Those rectangles remained in the PDF as vector drawings, so the figure detector correctly reported a page-wide figure behind every paragraph. A generator should not leave unlabeled ink in the document.

* `body_size` was initially filtered using line width, based on the assumption that body text would match the column measure. A wide title breaks that assumption. On a two-column page, the title became the widest line and pushed the real body text below the threshold. The measured body size became 16.0, causing paragraphs to be classified as captions.

---

## 7. Calibration, not assumption

```bash id="w1d4pe"
python ocr.py
```

```text
{'samples': 98, 'height_to_points': 1.0556, 'current': 1.0556}
```

`ocr.py` estimates point size from OCR glyph height.

The textbook conversion is `1 / 0.72 = 1.389`, based on the assumption that cap height is approximately 72% of the point size. Measurements against pages with known font sizes give a different value here:

**1.0556**, about 30% lower.

The reason is that Tesseract's line box covers the ascender and descender, with additional slack, rather than measuring cap height directly.

Using 1.389 would not cause an obvious failure. OCR body lines would simply be estimated at roughly 13pt instead of the true 9.5pt. Because the bias would be consistent, ratios used later in the pipeline could still look correct.

That is why the constant has an explicit `calibrate()` function beside it. This type of error can remain invisible, and the correct value can change with the typeface or DPI.

---

## 8. What has not been verified

* **No real-world PDF corpus.** All benchmark documents are generated. PubLayNet and DocBank were not used. The reported digital results are therefore an upper bound, while the OCR section is the only measured degradation path.

* **No figure or table detection on scanned input.** This is explicitly listed as a gap in Section 4 and accounts for much of the OCR precision loss.

* **No non-Latin scripts, rotated pages, or skewed pages in the benchmark.** `ocr.render` accepts `rotate`, `blur`, and `noise` arguments and those paths work, but no benchmark numbers are reported with them enabled.

* **The `--tree` hierarchy view is not scored.** `metrics.py` evaluates the flat ordered element list. Nesting is derived from that structure and is correct by construction, but there is no test comparing the generated hierarchy with a ground-truth tree.

* **The classifier is trained and tested on the same generator.** The benchmark uses held-out seeds, but not held-out distributions. The classifier has never been tested on a document produced outside this repository's generator.
