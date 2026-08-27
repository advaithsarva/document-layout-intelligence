"""Scoring predicted layout against known ground truth.

FOUR NUMBERS, AND WHY NONE OF THEM ALONE IS ENOUGH
--------------------------------------------------
    coverage        did any text get lost? A partition check.
    detection       were the right regions found, at the right places? (IoU)
    type accuracy   were they named correctly, given they were found?
    order tau       does the reading order match?

They are reported separately and always together, because each one hides a
different disaster:

* **Type accuracy alone** is the number everyone quotes and it is measured
  *only over matched elements*. A system that finds three elements on a page of
  thirty and names all three correctly scores 1.000. `detection` is what stops
  that, and `coverage` is what stops both.
* **Detection alone** says nothing about reading order, and reading order is
  invisible in every visual check anyone will actually do.
* **Order tau alone** can be perfect on a page where everything was merged into
  one giant element.

The pairing is deliberate: a single headline number for layout analysis is
almost always the one that is easiest to make look good.
"""

import statistics

IOU_MATCH = 0.5


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def match(truth, predicted, threshold=IOU_MATCH):
    """Greedy highest-IoU matching, best pairs first.

    Greedy rather than Hungarian on purpose: at IoU >= 0.5 two ground-truth
    boxes cannot both match one prediction, so the assignment is very nearly
    forced and the optimal matching would differ only in cases that are already
    being scored as errors.
    """
    pairs = sorted(
        ((iou(t.bbox, p.bbox), ti, pi)
         for ti, t in enumerate(truth) for pi, p in enumerate(predicted)),
        reverse=True)
    used_t, used_p, matched = set(), set(), []
    for score, ti, pi in pairs:
        if score < threshold or ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        matched.append((ti, pi, score))
    return matched, used_t, used_p


def kendall_tau(order_a, order_b):
    """Rank correlation between two orderings of the same items.

    +1 identical, -1 reversed, 0 uncorrelated. This is the right measure for
    reading order because it counts *pairwise* inversions: a two-column page
    read left-right-left-right is not "a bit late", it is wrong about half of
    all pairs, and tau reports approximately 0 rather than the flattering
    number a positional distance would give.
    """
    n = len(order_a)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = order_a[i] - order_a[j]
            sign_b = order_b[i] - order_b[j]
            if sign_a == 0 or sign_b == 0:
                continue
            if (sign_a > 0) == (sign_b > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def score_page(truth, predicted, coverage=None):
    """Score one page. `truth` and `predicted` both need .type/.bbox/.order."""
    flow_truth = [t for t in truth if t.order]
    flow_pred = [p for p in predicted if p.order]

    matched, used_t, used_p = match(flow_truth, flow_pred)
    correct_type = sum(1 for ti, pi, _ in matched
                       if flow_truth[ti].type == flow_pred[pi].type)

    tau = 1.0
    if len(matched) >= 2:
        pairs = sorted(matched, key=lambda m: flow_truth[m[0]].order)
        tau = kendall_tau([flow_truth[ti].order for ti, _, _ in pairs],
                          [flow_pred[pi].order for _, pi, _ in pairs])

    confusion = {}
    for ti, pi, _ in matched:
        key = (flow_truth[ti].type, flow_pred[pi].type)
        confusion[key] = confusion.get(key, 0) + 1

    return {
        "truth_elements": len(flow_truth),
        "predicted_elements": len(flow_pred),
        "matched": len(matched),
        "recall": len(matched) / len(flow_truth) if flow_truth else 1.0,
        "precision": len(matched) / len(flow_pred) if flow_pred else 1.0,
        "type_accuracy": correct_type / len(matched) if matched else 0.0,
        "mean_iou": round(statistics.mean([m[2] for m in matched]), 4)
        if matched else 0.0,
        "order_tau": round(tau, 4),
        "coverage": coverage,
        "missed": [flow_truth[i].type for i in range(len(flow_truth))
                   if i not in used_t],
        "spurious": [flow_pred[i].type for i in range(len(flow_pred))
                     if i not in used_p],
        "confusion": confusion,
    }


def aggregate(page_scores):
    """Micro-averaged over elements, not macro-averaged over pages.

    A page with two elements and a page with forty are not equally informative,
    and averaging per-page rates lets a trivially easy page cancel a hard one.
    """
    truth = sum(s["truth_elements"] for s in page_scores)
    pred = sum(s["predicted_elements"] for s in page_scores)
    matched = sum(s["matched"] for s in page_scores)
    typed = sum(s["type_accuracy"] * s["matched"] for s in page_scores)
    ious = sum(s["mean_iou"] * s["matched"] for s in page_scores)

    confusion = {}
    for s in page_scores:
        for key, count in s["confusion"].items():
            confusion[key] = confusion.get(key, 0) + count

    coverages = [s["coverage"] for s in page_scores if s["coverage"] is not None]
    missed, spurious = {}, {}
    for s in page_scores:
        for t in s["missed"]:
            missed[t] = missed.get(t, 0) + 1
        for t in s["spurious"]:
            spurious[t] = spurious.get(t, 0) + 1

    return {
        "pages": len(page_scores),
        "truth_elements": truth,
        "predicted_elements": pred,
        "matched": matched,
        "recall": round(matched / truth, 4) if truth else 1.0,
        "precision": round(matched / pred, 4) if pred else 1.0,
        "type_accuracy": round(typed / matched, 4) if matched else 0.0,
        "mean_iou": round(ious / matched, 4) if matched else 0.0,
        "order_tau": round(statistics.mean([s["order_tau"] for s in page_scores]), 4)
        if page_scores else 1.0,
        "coverage": round(statistics.mean(coverages), 4) if coverages else None,
        "min_coverage": round(min(coverages), 4) if coverages else None,
        "missed_by_type": dict(sorted(missed.items(), key=lambda kv: -kv[1])),
        "spurious_by_type": dict(sorted(spurious.items(), key=lambda kv: -kv[1])),
        "confusion": {f"{a}->{b}": c for (a, b), c in
                      sorted(confusion.items(), key=lambda kv: -kv[1]) if a != b},
    }


def format_report(summary, title="layout"):
    lines = [f"{title}",
             f"  elements     {summary['matched']}/{summary['truth_elements']} matched "
             f"({summary['predicted_elements']} predicted)",
             f"  recall       {summary['recall']:.4f}",
             f"  precision    {summary['precision']:.4f}",
             f"  type acc.    {summary['type_accuracy']:.4f}  (over matched only)",
             f"  mean IoU     {summary['mean_iou']:.4f}",
             f"  order tau    {summary['order_tau']:.4f}"]
    if summary["coverage"] is not None:
        lines.append(f"  coverage     {summary['coverage']:.4f}  "
                     f"(worst page {summary['min_coverage']:.4f})")
    if summary["confusion"]:
        lines.append("  confusions   " + ", ".join(
            f"{k} x{v}" for k, v in list(summary["confusion"].items())[:6]))
    if summary["missed_by_type"]:
        lines.append("  missed       " + ", ".join(
            f"{k} x{v}" for k, v in summary["missed_by_type"].items()))
    if summary["spurious_by_type"]:
        lines.append("  spurious     " + ", ".join(
            f"{k} x{v}" for k, v in summary["spurious_by_type"].items()))
    return "\n".join(lines)
