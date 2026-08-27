"""Three extractors on the same generated corpus, scored the same way.

    python bench.py
    python bench.py --docs 20 --json
    python bench.py --only layout

THE BASELINES CAN WIN, AND ON ONE COLUMN THEY DO
------------------------------------------------
    raw         PyMuPDF `get_text("blocks")` in the order it returns them.
                This is what a "PDF extraction pipeline" usually is: block
                text, concatenated. No types, no columns, no tables.
    sorted      the same, with PyMuPDF's own `sort=True` reading order --
                blocks top-to-bottom, left-to-right.
    layout      this project.

`raw` and `sorted` produce no element types at all, so they are scored on the
two things they *can* be scored on -- did they find the regions, and are they
in the right order -- and their type accuracy is reported as the 0.000 it
honestly is rather than omitted.

The interesting comparison is `sorted` against `layout` on `order_tau`, on
two-column pages specifically. Top-to-bottom-left-to-right is exactly right on
a single column, and there is no reason to expect this project to beat it
there. The `--by-layout` breakdown is where the difference lives.
"""

import argparse
import json
import shutil
import statistics
import tempfile
import time

import fitz

import layout as layout_mod
import metrics
import synth


class Simple:
    """A ground-truth-shaped view of a raw block extractor, so it can be scored."""

    def __init__(self, bbox, order):
        self.bbox = bbox
        self.order = order
        self.type = "untyped"       # it has no opinion; never scores a match
        self.text = ""


def extract_raw(page, sort=False):
    out = []
    for i, block in enumerate(page.get_text("blocks", sort=sort), start=1):
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if not str(text).strip():
            continue
        out.append(Simple((x0, y0, x1, y1), i))
    return out


def _split(result):
    elements, stats = result
    return elements, stats["coverage"]


def _model():
    import classify
    return classify.load()


# Each extractor returns (elements, coverage). Coverage is None for the raw
# baselines because they do not claim a partition of the page and it would be
# dishonest to compute one on their behalf.
EXTRACTORS = {
    "raw": lambda page: (extract_raw(page, sort=False), None),
    "sorted": lambda page: (extract_raw(page, sort=True), None),
    "layout": lambda page: _split(layout_mod.analyse_page(page)),
    # Phase 3.5: the same pipeline with the learned classifier in place of the
    # rules. Loads lazily and falls back to the rules if there is no model.
    "layout+model": lambda page: _split(layout_mod.analyse_page(
        page, classifier=_model())),
    # The scanned path: render -> tesseract -> the same stages. Slow (~2s/page),
    # so it is opt-in via --only.
    "ocr": lambda page: _split(__import__("ocr").analyse_scanned_page(page)),
    "ocr+model": lambda page: _split(__import__("ocr").analyse_scanned_page(
        page, classifier=_model())),
}

# Extractors that are not run unless asked for by name.
OPT_IN = ("ocr", "ocr+model")


def run(corpus, names):
    results = {n: {"pages": [], "seconds": 0.0, "by_layout": {}} for n in names}
    for path, truth_pages, layout_names in corpus:
        doc = fitz.open(path)
        try:
            for index, page in enumerate(doc):
                truth = truth_pages[index]
                kind = layout_names[index]
                for name in names:
                    start = time.perf_counter()
                    predicted, coverage = EXTRACTORS[name](page)
                    results[name]["seconds"] += time.perf_counter() - start
                    score = metrics.score_page(truth, predicted, coverage)
                    results[name]["pages"].append(score)
                    results[name]["by_layout"].setdefault(kind, []).append(score)
        finally:
            doc.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", help="comma-separated extractor names; "
                                   f"available: {', '.join(EXTRACTORS)}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    names = ([n.strip() for n in args.only.split(",")] if args.only
             else [n for n in EXTRACTORS if n not in OPT_IN])
    directory = tempfile.mkdtemp(prefix="dli-bench-")
    try:
        corpus = []
        import random
        kinds = list(synth.LAYOUTS)
        for i in range(args.docs):
            rng = random.Random(args.seed + i)
            chosen = [kinds[i % len(kinds)]] + rng.sample(kinds, 2)
            path = f"{directory}/doc_{i:03d}.pdf"
            truth = synth.build_document(path, chosen, seed=args.seed + i)
            corpus.append((path, truth, chosen))

        results = run(corpus, names)
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    report = {"documents": args.docs, "seed": args.seed, "extractors": {}}
    for name in names:
        summary = metrics.aggregate(results[name]["pages"])
        summary["seconds_per_page"] = round(
            results[name]["seconds"] / max(len(results[name]["pages"]), 1), 5)
        summary["by_layout"] = {
            kind: {"order_tau": round(statistics.mean(
                       [s["order_tau"] for s in scores]), 4),
                   "recall": round(statistics.mean(
                       [s["recall"] for s in scores]), 4),
                   "type_accuracy": round(statistics.mean(
                       [s["type_accuracy"] for s in scores]), 4)}
            for kind, scores in sorted(results[name]["by_layout"].items())}
        report["extractors"][name] = summary

    if args.json:
        print(json.dumps(report, indent=2))
        return

    pages = report["extractors"][names[0]]["pages"]
    print(f"{args.docs} generated documents, {pages} pages, seed {args.seed}\n")
    print(f"{'':<10}{'recall':>9}{'prec':>9}{'type':>9}{'IoU':>9}{'tau':>9}"
          f"{'cover':>9}{'s/page':>9}")
    for name in names:
        s = report["extractors"][name]
        cover = "-" if s["coverage"] is None else f"{s['coverage']:.4f}"
        print(f"{name:<10}{s['recall']:>9.4f}{s['precision']:>9.4f}"
              f"{s['type_accuracy']:>9.4f}{s['mean_iou']:>9.4f}"
              f"{s['order_tau']:>9.4f}{cover:>9}{s['seconds_per_page']:>9.4f}")

    print("\nreading order (tau) by page layout")
    kinds = sorted(report["extractors"][names[0]]["by_layout"])
    print(f"{'':<10}" + "".join(f"{k:>16}" for k in kinds))
    for name in names:
        row = report["extractors"][name]["by_layout"]
        print(f"{name:<10}" + "".join(
            f"{row[k]['order_tau']:>16.4f}" for k in kinds))

    print()
    print(metrics.format_report(report["extractors"][names[-1]],
                                f"{names[-1]} -- error detail"))


if __name__ == "__main__":
    main()
