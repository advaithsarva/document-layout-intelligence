"""Analyse a PDF and print its structure, as text or as JSON.

    python cli.py sample.pdf                    # readable outline
    python cli.py sample.pdf --json             # flat, ordered elements
    python cli.py sample.pdf --json --tree      # nested under headings
    python cli.py sample.pdf --text             # reading order, concatenated
    python cli.py sample.pdf --tables           # just the tables, as grids
    python cli.py scan.pdf --ocr                # rasterise + tesseract path
    python cli.py sample.pdf --classifier model # learned types instead of rules
    python cli.py --demo                        # generate and analyse a sample

THE MACHINE INTERFACE
---------------------
`--json` is the contract, and it is the answer to "where does an agent plug
in?": give it a file, get back every element with a type, a bounding box, a
reading-order index and its text, plus a `coverage` figure saying what fraction
of the page's characters made it into the output. Nothing about the pipeline
needs to be understood to consume it, and `coverage < 1.0` is a machine-checkable
signal that something was dropped.
"""

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", nargs="?", help="path to a PDF")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tree", action="store_true",
                    help="nest elements under their headings")
    ap.add_argument("--text", action="store_true",
                    help="print the text in reading order and nothing else")
    ap.add_argument("--tables", action="store_true",
                    help="print detected tables as grids")
    ap.add_argument("--ocr", action="store_true",
                    help="rasterise and OCR instead of reading the text layer")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--classifier", choices=["rules", "model"], default="rules")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--pages", type=int, help="stop after N pages")
    ap.add_argument("--demo", action="store_true",
                    help="generate sample.pdf with known ground truth first")
    args = ap.parse_args(argv)

    import layout

    path = args.pdf
    if args.demo:
        import synth
        path = path or "sample.pdf"
        synth.build_document(path)
        print(f"generated {path}", file=sys.stderr)
    if not path:
        ap.error("a PDF path is required (or --demo)")

    classifier = None
    if args.classifier == "model":
        import classify
        classifier = classify.load()
        if classifier is None:
            # Degrade loudly. A silent fallback to the rules would make the
            # comparison in RESULTS.md unreproducible by anyone who forgot to
            # run `python classify.py --train`.
            print("no trained classifier found (run: python classify.py "
                  "--train); falling back to rules", file=sys.stderr)

    if args.ocr:
        import fitz
        import ocr as ocr_mod
        doc = fitz.open(path)
        pages = []
        try:
            for number, page in enumerate(doc, start=1):
                if args.pages and number > args.pages:
                    break
                elements, stats = ocr_mod.analyse_scanned_page(
                    page, dpi=args.dpi, classifier=classifier)
                pages.append({"page": number, "elements": elements,
                              "stats": stats})
        finally:
            doc.close()
    else:
        pages = layout.analyse_document(
            path, detect_tables=not args.no_tables, max_pages=args.pages,
            classifier=classifier)

    if args.text:
        print(layout.reading_text(pages))
        return 0

    if args.tables:
        found = 0
        for page in pages:
            for element in page["elements"]:
                if element.type != "table":
                    continue
                found += 1
                meta = element.meta
                print(f"page {page['page']}  {meta.get('rows')}x{meta.get('cols')}"
                      f"  ({meta.get('detector')})")
                for row in meta.get("cells", []):
                    print("  " + " | ".join(f"{c:<12}" for c in row))
                print()
        if not found:
            print("no tables detected")
        return 0

    if args.json:
        print(json.dumps(layout.document_json(pages, tree=args.tree), indent=2))
        return 0

    for page in pages:
        stats = page["stats"]
        print(f"--- page {page['page']}   {stats['columns']} column(s), "
              f"body {stats['body_size']}pt, {stats['tables']} table(s), "
              f"{stats['figures']} figure(s), coverage {stats['coverage']:.4f}")
        for element in sorted(page["elements"],
                              key=lambda e: (e.order == 0, e.order)):
            marker = "  " if element.order else "* "     # * = outside the flow
            preview = element.text[:66].replace("\n", " ")
            print(f"{marker}{element.order:>3} c{element.column} "
                  f"{element.type:<10} {preview}")
    total = layout.document_json(pages)["coverage"]
    print(f"\ndocument coverage {total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
