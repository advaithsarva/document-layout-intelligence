"""The scanned path: render a page to pixels, OCR it, rebuild the same Lines.

WHY THIS IS A SEPARATE FILE AND NOT A FLAG
------------------------------------------
A born-digital PDF hands you the font name, the point size and a bold flag for
every span. `layout.classify` leans on all three, and on clean PDFs it is right
every time. A scan hands you pixels. Tesseract returns words and boxes and
nothing else: **no font, no point size, no bold flag.**

That is not a small degradation, it is the removal of the primary signal. So
this module exists to make the loss *measurable* rather than hypothetical:

    render(page, dpi)   ->  a bitmap, optionally degraded like a real scan
    ocr_lines(image)    ->  layout.Line objects with `size` estimated from
                            glyph height and `bold` permanently False

Everything downstream -- columns, XY-cut, grouping, classification -- is the
same code. The only difference is the quality of what goes in, which is exactly
the comparison worth having, and it is what `classify.py` is measured on: a
learned classifier that never had the font metadata to begin with should lose
on clean PDFs and win here, or the feature is not worth its dependency.

`size` from glyph height is an estimate, not a measurement. Cap height is
roughly 0.7 of the point size for the fonts here, so the conversion is a
constant that was *calibrated against the generated pages* rather than guessed
-- see `calibrate()`. It will be wrong for a different typeface, which is a
real limitation and is stated in RESULTS.md rather than hidden in a constant.
"""

import io
import statistics

import fitz

from layout import Line

# Glyph-height -> point-size constant. CALIBRATED, not assumed.
#
#   python ocr.py     ->  {'samples': 98, 'height_to_points': 1.0556, ...}
#
# The textbook value is 1/0.72 = 1.389, from cap height being ~0.72 of the
# point size. It is wrong here by 30%, because tesseract's line box is not cap
# height: it spans ascender to descender plus a little slack. Shipping the
# textbook constant would have made every OCR'd body line read as ~13pt against
# a 9.5pt truth -- not a failure, just a uniform bias, which is the kind of
# error that never surfaces because every ratio downstream stays correct.
# Re-run `calibrate()` for a different typeface or DPI.
HEIGHT_TO_POINTS = 1.0556

DEFAULT_DPI = 200


class OCRUnavailable(RuntimeError):
    """Tesseract is not installed. Raised, never silently degraded.

    A layout system that quietly returns an empty page when OCR is missing
    reports 0 elements and 100% coverage of 0 characters, which every metric in
    this project would call a perfect score.
    """


def _require():
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OCRUnavailable(
            "pytesseract and Pillow are required for the scanned path "
            "(pip install pytesseract pillow, plus the tesseract binary)") from exc
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise OCRUnavailable(
            "the tesseract binary was not found on PATH; pytesseract is only "
            "a wrapper around it") from exc
    return pytesseract, Image


def render(page, dpi=DEFAULT_DPI, blur=0.0, noise=0.0, rotate=0.0):
    """Rasterise a page, optionally degraded the way a real scan is.

    The degradations are the three that actually change OCR output: a slight
    rotation (a page fed crooked), gaussian blur (a soft or out-of-focus scan)
    and pixel noise (compression and sensor grain). They are applied here
    rather than left to the imagination so the numbers in RESULTS.md have a
    knob attached to them.
    """
    _, Image = _require()
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("L")

    if rotate:
        image = image.rotate(rotate, resample=Image.BICUBIC, expand=False,
                             fillcolor=255)
    if blur:
        from PIL import ImageFilter
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        import random as _random
        rng = _random.Random(0)
        pixels = bytearray(image.tobytes())
        span = int(noise * 255)
        for i in range(0, len(pixels), 7):      # every 7th pixel; enough to bite
            pixels[i] = min(255, max(0, pixels[i] + rng.randint(-span, span)))
        image = Image.frombytes("L", image.size, bytes(pixels))
    return image


def ocr_lines(image, dpi=DEFAULT_DPI, min_confidence=30):
    """Tesseract words -> `layout.Line` objects in PDF points.

    Words are regrouped into lines by tesseract's own block/par/line ids rather
    than by geometry, because tesseract already did that work and doing it
    again from boxes only introduces a second place to be wrong.
    """
    pytesseract, _ = _require()
    data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT)
    scale = 72.0 / dpi

    grouped = {}
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < min_confidence:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append((
            text, data["left"][i], data["top"][i],
            data["width"][i], data["height"][i]))

    lines = []
    for key in sorted(grouped):
        words = sorted(grouped[key], key=lambda w: w[1])
        x0 = min(w[1] for w in words) * scale
        y0 = min(w[2] for w in words) * scale
        x1 = max(w[1] + w[3] for w in words) * scale
        y1 = max(w[2] + w[4] for w in words) * scale
        # Median glyph height over the words, so one tall parenthesis or a
        # descender-free word does not set the estimated point size.
        height = statistics.median([w[4] for w in words]) * scale
        lines.append(Line(text=" ".join(w[0] for w in words),
                          bbox=(x0, y0, x1, y1),
                          size=round(height * HEIGHT_TO_POINTS, 2),
                          bold=False,           # tesseract does not report it
                          font="ocr"))
    lines.sort(key=lambda l: (round(l.bbox[1], 1), l.bbox[0]))
    return lines


def calibrate(pdf_path="sample.pdf", dpi=DEFAULT_DPI):
    """Fit HEIGHT_TO_POINTS by comparing OCR glyph heights to the real sizes.

    Hardware and typefaces are never the ideal on paper: cap height as a
    fraction of point size varies by font, and a scan adds its own bias. This
    prints the constant this corpus actually implies so it can be re-tuned
    rather than trusted, which is the whole reason it is a named constant and
    not an inline number.
    """
    doc = fitz.open(pdf_path)
    ratios = []
    try:
        for page in doc:
            truth = {round(l.bbox[1]): l.size for l in _digital_lines(page)}
            for line in ocr_lines(render(page, dpi), dpi):
                key = min(truth, key=lambda y: abs(y - line.bbox[1]), default=None)
                if key is None or abs(key - line.bbox[1]) > 4:
                    continue
                measured = (line.bbox[3] - line.bbox[1])
                if measured > 0:
                    ratios.append(truth[key] / measured)
    finally:
        doc.close()
    return {"samples": len(ratios),
            "height_to_points": round(statistics.median(ratios), 4) if ratios else None,
            "current": round(HEIGHT_TO_POINTS, 4)}


def _digital_lines(page):
    from layout import page_lines
    return page_lines(page)


def analyse_scanned_page(page, dpi=DEFAULT_DPI, classifier=None, **degradation):
    """The full pipeline over an OCR'd render of `page`.

    Deliberately reuses `layout.analyse_page`'s stages rather than
    reimplementing them -- the point of the experiment is that only the input
    changes.
    """
    import layout as layout_mod
    import tables as tables_mod

    lines = ocr_lines(render(page, dpi, **degradation), dpi)
    page_rect = tuple(page.rect)
    total = sum(len(l.text.strip()) for l in lines)

    header, footer, body = layout_mod.split_running(lines, page_rect[3])
    body_pt = layout_mod.body_size(body or lines)
    columns = layout_mod.column_bounds(body, page_rect[2])

    items = []
    for index, (lo, hi) in enumerate(columns):
        centre = lambda b: (b[0] + b[2]) / 2
        column_lines = [l for l in body if lo - 1 <= centre(l.bbox) <= hi + 1]
        for block in layout_mod.group_lines(column_lines, body_pt):
            bbox = (min(l.bbox[0] for l in block), min(l.bbox[1] for l in block),
                    max(l.bbox[2] for l in block), max(l.bbox[3] for l in block))
            items.append((bbox, block, index))

    first = min((it[0] for it in items), key=lambda b: (round(b[1]), b[0]),
                default=None)
    ordered = layout_mod.xy_cut([(b, (blk, c)) for b, blk, c in items])
    elements = []
    for order, (bbox, (block, column)) in enumerate(ordered, start=1):
        type_ = layout_mod._name(block, body_pt, page_rect, bbox == first,
                                 classifier)
        elements.append(layout_mod.LayoutElement(
            type_, bbox, " ".join(l.text for l in block).strip(), order, column))
    for line in header:
        elements.append(layout_mod.LayoutElement("header", line.bbox,
                                                 line.text.strip(), 0, 0))
    for line in footer:
        elements.append(layout_mod.LayoutElement("footer", line.bbox,
                                                 line.text.strip(), 0, 0))

    boxes = [e.bbox for e in elements]
    covered = sum(len(l.text.strip()) for l in lines
                  if any(layout_mod._overlaps(l.bbox, b, 0.5) for b in boxes))
    stats = {"columns": len(columns), "body_size": round(body_pt, 2),
             "chars_on_page": total, "chars_covered": covered,
             "coverage": round(covered / total, 4) if total else 1.0,
             "tables": 0, "figures": 0, "source": "ocr"}
    return elements, stats


if __name__ == "__main__":
    print(calibrate())
