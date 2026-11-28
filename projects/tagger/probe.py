"""Linear probes: one logistic regression per task, trained on the labels you have given so far.

A task is a named set of tags used as its classes: "species" = {lion, tiger, jaguar}. An item carrying exactly one of the
task's tags is labeled with it; an item carrying none is unlabeled, and one carrying several is left out (a conflict).
The probe sees the embedding only, so it is as good as the embedding is at that attribute. Tasks are saved in
`<dataset>/tasks.json`.
"""

import json

import numpy as np
import polars as pl

from store import dataset_dir


def load_tasks(name: str) -> dict[str, list[str]]:
    path = dataset_dir(name) / "tasks.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_tasks(name: str, tasks: dict[str, list[str]]) -> None:
    (dataset_dir(name) / "tasks.json").write_text(json.dumps(tasks, indent=2))


def labels(tags: pl.DataFrame, classes: list[str], n_items: int) -> tuple[np.ndarray, int]:
    """(the class of each item, "" where unlabeled; the number of items carrying two or more of the classes)."""
    per_item: dict[int, set[str]] = {}
    for i, tag in tags.select("id", "tag").iter_rows():
        if tag in classes:
            per_item.setdefault(i, set()).add(tag)
    y = np.full(n_items, "", dtype=object)
    conflicts = 0
    for i, have in per_item.items():
        if len(have) == 1:
            y[i] = next(iter(have))
        else:
            conflicts += 1
    return y, conflicts


def train(X: np.ndarray, y: np.ndarray, C: float = 10.0, balanced: bool = True, seed: int = 0) -> dict:
    """Fit on the labeled items (`y != ""`). Returns the class names, class probabilities for every item (N, K),
    cross-validated accuracy and a per-class report (None if some class has a single example)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    mask = y != ""
    Xl, yl = X[mask], y[mask].astype(str)
    classes, counts = np.unique(yl, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"need at least 2 values with examples, have {list(classes)}")

    def make() -> LogisticRegression:
        return LogisticRegression(C=C, max_iter=2000, class_weight="balanced" if balanced else None)

    result: dict = {"classes": list(classes), "counts": dict(zip(classes, counts.tolist())), "n": int(mask.sum()),
                    "accuracy": None, "report": None}  # fmt: skip
    k = min(5, int(counts.min()))
    if k >= 2:
        pred = cross_val_predict(make(), Xl, yl, cv=StratifiedKFold(k, shuffle=True, random_state=seed))
        rep = classification_report(yl, pred, output_dict=True, zero_division=0)
        result["accuracy"] = float((pred == yl).mean())
        result["folds"] = k
        result["report"] = pl.DataFrame(
            [{"value": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"], "examples": int(rep[c]["support"])}
             for c in classes]
        )  # fmt: skip
    clf = make().fit(Xl, yl)
    result["proba"] = clf.predict_proba(X).astype(np.float32)
    return result


def uncertain(proba: np.ndarray, unlabeled: np.ndarray, n: int) -> np.ndarray:
    """The `n` unlabeled items the probe is least sure about (smallest gap between its top two classes): the most
    useful ones to label next."""
    top2 = np.sort(proba, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    rows = np.flatnonzero(unlabeled)
    return rows[np.argsort(margin[rows])[:n]]
