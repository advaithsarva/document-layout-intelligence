"""The learned alternative to `layout.classify`, and the honest comparison.

WHAT THIS IS FOR
----------------
It classifies TEXT blocks only. Figures, tables and running heads are found
geometrically -- by vector-drawing clusters, by rules or column alignment, and
by page band -- and letting a text classifier emit those labels is a category
error with a measurable cost: an earlier version had `table` in its label set
and invented **52 spurious tables** across 30 scanned pages, while its headline
"type accuracy over matched elements" stayed at 0.973 and hid it entirely.

`layout.classify` is a handful of rules over font size, weight and position. On
born-digital PDFs it is right 100% of the time, and no model is going to beat
that. The rules are also *free*: no training, no artefact, no dependency.

So the question this file exists to answer is not "can a model do layout
classification" -- obviously -- it is:

    On input where the rules lose their primary signal, does a learned
    classifier recover enough of it to be worth a dependency and a
    training step?

Scanned pages are exactly that input. Tesseract reports no font, no point size
and no bold flag; `ocr.py` estimates size from glyph height and hard-codes
`bold = False`. The rules' two strongest features are gone. A model trained on
*geometry and text shape* -- indentation, width, line count, how a block starts
and ends, digit and capital fractions -- never had them either, so it should
degrade less.

Both numbers go in RESULTS.md whichever way it comes out. If the model loses,
that is the result, and the rules stay the default.

DEGRADATION
-----------
No scikit-learn, or no trained model on disk: `predict` returns None and
`layout.analyse_page` keeps using the rules. There is no crash and no silent
empty page -- `--classifier model` on the CLI says out loud which path ran.
"""

import argparse
import json
import os
import pickle
import re
import statistics

from layout import BULLET, CAPTION_LEAD

MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier.pkl")

FEATURE_NAMES = (
    "size_ratio", "bold_fraction", "n_lines", "width_fraction", "x_fraction",
    "y_fraction", "chars", "chars_per_line", "starts_bullet", "caption_lead",
    "is_first", "ends_sentence", "digit_fraction", "upper_fraction",
    "size_rank", "line_gap_ratio",
)


def features(block, body, page_rect, is_first_block=False):
    """One feature vector for a block of lines.

    Deliberately contains NOTHING that is unavailable from an OCR'd page except
    `bold_fraction`, which is simply 0 there. The point of the experiment is
    that the model is trained on both kinds of input and has to find signal
    that survives the loss.
    """
    text = " ".join(l.text for l in block).strip()
    sizes = [l.size for l in block]
    size = statistics.median(sizes)
    width = max(l.bbox[2] for l in block) - min(l.bbox[0] for l in block)
    x0 = min(l.bbox[0] for l in block)
    y0 = min(l.bbox[1] for l in block)
    letters = [c for c in text if c.isalpha()]
    gaps = [b.bbox[1] - a.bbox[3] for a, b in zip(block, block[1:])]
    leading = statistics.median([l.bbox[3] - l.bbox[1] for l in block]) or 1.0

    return [
        size / body if body else 1.0,
        sum(l.bold for l in block) / len(block),
        float(len(block)),
        width / max(page_rect[2], 1.0),
        x0 / max(page_rect[2], 1.0),
        y0 / max(page_rect[3], 1.0),
        float(len(text)),
        len(text) / len(block),
        float(bool(BULLET.match(block[0].text))),
        float(bool(CAPTION_LEAD.match(text))),
        float(bool(is_first_block)),
        float(text.endswith((".", "!", "?"))),
        sum(c.isdigit() for c in text) / max(len(text), 1),
        sum(c.isupper() for c in letters) / max(len(letters), 1),
        float(size >= body * 1.1) - float(size <= body * 0.9),
        (statistics.median(gaps) / leading) if gaps else 0.0,
    ]


# --------------------------------------------------------------------------
# training data: blocks paired with the ground-truth type they overlap
# --------------------------------------------------------------------------

def _label_for(bbox, truth):
    """The ground-truth type whose box this block sits inside.

    Uses containment-in-truth rather than IoU: a block that is exactly one
    paragraph of a two-paragraph truth element is still a paragraph, and
    training would rather have that example than discard it.
    """
    best, best_score = None, 0.0
    area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1e-6)
    for element in truth:
        if not element.order:
            continue
        x0, y0 = max(bbox[0], element.bbox[0]), max(bbox[1], element.bbox[1])
        x1, y1 = min(bbox[2], element.bbox[2]), min(bbox[3], element.bbox[3])
        if x1 <= x0 or y1 <= y0:
            continue
        score = (x1 - x0) * (y1 - y0) / area
        if score > best_score:
            best, best_score = element.type, score
    return best if best_score >= 0.5 else None


def collect(seeds, layouts=None, include_ocr=True, dpi=150):
    """Feature vectors and labels from generated documents.

    `include_ocr` doubles every page: once from the PDF text layer, once from a
    render passed through tesseract. Training on both is the whole design -- a
    model trained only on clean pages learns to rely on the exact font size and
    is no better than the rules when that size is an estimate.
    """
    import fitz

    import layout as layout_mod
    import synth

    if layouts is None:
        layouts = list(synth.LAYOUTS)

    rows, labels = [], []
    for seed in seeds:
        import tempfile
        handle, path = tempfile.mkstemp(suffix=".pdf")
        os.close(handle)
        try:
            chosen = [layouts[seed % len(layouts)]]
            truth = synth.build_document(path, chosen, seed=seed)
            doc = fitz.open(path)
            try:
                for index, page in enumerate(doc):
                    sources = [layout_mod.page_lines(page)]
                    if include_ocr:
                        try:
                            import ocr
                            sources.append(ocr.ocr_lines(
                                ocr.render(page, dpi), dpi))
                        except Exception:
                            pass            # no tesseract: train on digital only
                    for lines in sources:
                        rows_, labels_ = _page_rows(page, lines, truth[index],
                                                    layout_mod)
                        rows.extend(rows_)
                        labels.extend(labels_)
            finally:
                doc.close()
        finally:
            os.unlink(path)
    return rows, labels


def _page_rows(page, lines, truth, layout_mod):
    page_rect = tuple(page.rect)
    _, _, body_lines = layout_mod.split_running(lines, page_rect[3])
    if not body_lines:
        return [], []
    body = layout_mod.body_size(body_lines)
    columns = layout_mod.column_bounds(body_lines, page_rect[2])

    blocks = []
    for lo, hi in columns:
        centre = lambda b: (b[0] + b[2]) / 2
        column_lines = [l for l in body_lines if lo - 1 <= centre(l.bbox) <= hi + 1]
        blocks.extend(layout_mod.group_lines(column_lines, body))

    if not blocks:
        return [], []
    boxes = [(min(l.bbox[0] for l in b), min(l.bbox[1] for l in b),
              max(l.bbox[2] for l in b), max(l.bbox[3] for l in b))
             for b in blocks]
    first = min(boxes, key=lambda b: (round(b[1]), b[0]))

    rows, labels = [], []
    for block, bbox in zip(blocks, boxes):
        label = _label_for(bbox, truth)
        if label is None or label in ("figure", "header", "footer", "table"):
            continue        # found geometrically, never by classifying text
        rows.append(features(block, body, page_rect, is_first_block=(bbox == first)))
        labels.append(label)
    return rows, labels


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

def train(seeds=range(1, 25), path=MODEL_PATH, include_ocr=True, depth=8):
    """A small decision tree. Deliberately small, for two reasons.

    It has to be *inspectable* -- a layout classifier that cannot be asked why
    it called something a heading is worse than the rules it replaces, because
    the rules can be read. And it has to be honest about the data: a few
    thousand blocks from five generated layouts does not support a model with
    the capacity to memorise them.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to train "
                           "(pip install scikit-learn)") from exc

    rows, labels = collect(seeds, include_ocr=include_ocr)
    if not rows:
        raise RuntimeError("no training data was produced")
    model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=4,
                                   class_weight="balanced", random_state=42)
    model.fit(rows, labels)
    with open(path, "wb") as fh:
        pickle.dump({"model": model, "features": FEATURE_NAMES}, fh)
    return {"path": path, "samples": len(rows), "depth": depth,
            "classes": sorted(set(labels)),
            "bytes": os.path.getsize(path),
            "train_accuracy": round(model.score(rows, labels), 4)}


_cache = {}


def load(path=MODEL_PATH):
    """The trained model, or None. Never raises for a missing model."""
    if path in _cache:
        return _cache[path]
    model = None
    try:
        import sklearn        # noqa: F401  -- unpickling needs it importable
        with open(path, "rb") as fh:
            model = pickle.load(fh)["model"]
    except Exception:
        model = None
    _cache[path] = model
    return model


def predict(block, body, page_rect, is_first_block=False, model=None):
    """Type for one block, or None if no model is available."""
    model = model if model is not None else load()
    if model is None:
        return None
    return model.predict([features(block, body, page_rect, is_first_block)])[0]


def importances(path=MODEL_PATH):
    model = load(path)
    if model is None:
        return {}
    return dict(sorted(zip(FEATURE_NAMES, (round(float(v), 4) for v in
                                           model.feature_importances_)),
                       key=lambda kv: -kv[1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--no-ocr", action="store_true",
                    help="train on the PDF text layer only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.train:
        report = train(range(1, args.seeds + 1), include_ocr=not args.no_ocr)
        report["importances"] = importances()
        print(json.dumps(report, indent=2) if args.json else
              "\n".join(f"{k}: {v}" for k, v in report.items()))
    else:
        print(json.dumps(importances(), indent=2))
