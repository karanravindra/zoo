"""Per-field scores for extraction fields, DocVQA (ANLS) and ChartQA (relaxed accuracy)."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

_EDGE = " \t\r\n.,;:!?\"'`()[]{}"
_ARTICLE = re.compile(r"^(a|an|the)\s+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def norm(value) -> str:
    """Case, edge punctuation, a leading article and inner whitespace don't count (as in synthetic-extraction's
    verification)."""
    s = unicodedata.normalize("NFKC", str(value)).casefold().strip(_EDGE)
    s = _ARTICLE.sub("", s)
    return " ".join(s.split())


def is_null(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "") or value == []


def score_scalar(pred, gold: str | None) -> float:
    if gold is None:
        return float(is_null(pred))
    if is_null(pred) or isinstance(pred, (list, dict)):
        return 0.0
    return float(norm(pred) == norm(gold))


def score_list(pred, gold: list[str]) -> float:
    """Set F1 of normalized values; 1 when both are empty."""
    if is_null(pred):
        pred = []
    elif not isinstance(pred, list):
        pred = [pred]
    p = {norm(v) for v in pred if not is_null(v)}
    g = {norm(v) for v in gold}
    if not p and not g:
        return 1.0
    tp = len(p & g)
    return 2 * tp / (len(p) + len(g))


def score_extraction(fields: dict, gold: dict, preds: dict) -> list[dict]:
    """One record per requested field: name, type, score, and whether gold / prediction are null."""
    out = []
    for name, spec in fields.items():
        g, p = gold[name], preds.get(name)
        if spec["type"] == "list":
            s = score_list(p, g)
        else:
            s = score_scalar(p, g)
        out.append({"field": name, "type": spec["type"], "score": s, "gold_null": is_null(g), "pred_null": is_null(p)})
    return out


def anls(pred, golds: list[str], tau: float = 0.5) -> float:
    """DocVQA's ANLS for one question: best 1 - normalized Levenshtein over the gold answers, 0 below `tau`."""
    if is_null(pred) or isinstance(pred, (list, dict)):
        return 0.0
    p = " ".join(str(pred).casefold().split())
    best = max(1 - Levenshtein.normalized_distance(p, " ".join(g.casefold().split())) for g in golds)
    return best if best >= tau else 0.0


def parse_number(value) -> float | None:
    """A float from a number or a string like '1,234.5', '45%' or '$12'; None if there is none."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _NUMBER.search(str(value).replace(",", ""))
    return float(m.group()) if m else None


def gold_number(labels: list[str]) -> float | None:
    """ChartQA gold as a number when it is a single plain numeric answer ('12', '1,234.5', '45%'), else None."""
    if len(labels) != 1 or not re.fullmatch(r"\s*-?[\d,]*\.?\d+\s*%?\s*", labels[0]):
        return None
    return parse_number(labels[0])


def relaxed_accuracy(pred, gold: float, tol: float = 0.05) -> float:
    """ChartQA's relaxed accuracy: within 5% of the gold value (exact when gold is 0)."""
    p = parse_number(pred)
    if p is None:
        return 0.0
    if gold == 0:
        return float(p == 0)
    return float(abs(p - gold) / abs(gold) <= tol)
