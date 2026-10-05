"""Evaluation metrics for the lesion classifier, in plain numpy.

Accuracy alone hides the errors that matter clinically. A cancer called
benign is far worse than an adenoma called cancer, and a confident wrong
answer is worse than an abstention. So this module reports:

* per-class sensitivity, specificity, PPV, NPV and one-vs-rest AUROC;
* neoplastic (precancerous + cancerous) versus non-neoplastic, including the
  NPV used by the ASGE PIVI "diagnose-and-leave" benchmark;
* sensitivity for cancer;
* calibration (expected calibration error), because the report shows
  probabilities to clinicians;
* the deployed decision (`decide`), abstention included: how many lesions the
  model answers for, how accurate it is on those, the neoplastic NPV of the
  benign calls it actually makes, and where every cancer ended up;
* patient-grouped bootstrap confidence intervals, because lesions from one
  patient are not independent. When no errors are observed every resample gives
  the same value and a percentile interval would read [100%-100%]; proportions
  then get an exact (Clopper-Pearson) interval over patients instead, and the
  other metrics are marked as not estimable.

Everything returned is JSON-serialisable (floats, ints, lists, None). A metric
that cannot be computed, such as AUROC for a class with no positives, is None
rather than NaN. There is no scikit-learn dependency, so the runtime package
can verify a model without the training stack.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .taxonomy import CATEGORIES, DiagnosticCategory

ECE_BINS = 15
NEOPLASTIC_THRESHOLD = 0.5
# Below 0.5 a "benign" call could be made for a lesion the model rates more likely neoplastic
# than not, so abstention thresholds start there (OnnxLesionClassifier enforces the same bound).
MIN_ABSTAIN_BELOW = 0.5
COVERAGE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
ABSTAIN = -1


def _validate(y_true, probs, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise ValueError(f"probs must have shape (N, {n_classes}), got {p.shape}")
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"{y.shape[0]} labels but {p.shape[0]} probability rows")
    if y.size and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError(f"labels must be class indices in [0, {n_classes - 1}]")
    if not np.all(np.isfinite(p)):
        raise ValueError("probs contain NaN or infinite values")
    return y, p


def _ratio(num: float, den: float) -> Optional[float]:
    return float(num) / float(den) if den > 0 else None


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Counts with rows = true class, columns = predicted class."""
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    return np.bincount(y * n_classes + yp, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def binary_rates(tp: int, fp: int, tn: int, fn: int) -> dict:
    return {
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _binary_counts(positive: np.ndarray, predicted_positive: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum(positive & predicted_positive))
    fp = int(np.sum(~positive & predicted_positive))
    tn = int(np.sum(~positive & ~predicted_positive))
    fn = int(np.sum(positive & ~predicted_positive))
    return tp, fp, tn, fn


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values share the mean of the ranks they span."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    n = values.size
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_vals) != 0) + 1]
    ends = np.r_[starts[1:], n]
    mean_rank = (starts + ends + 1) / 2.0  # mean of ranks start+1 .. end
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(mean_rank, ends - starts)
    return ranks


def auroc(scores, positive) -> Optional[float]:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Ties count as half a correct ordering. Returns None when there are no
    positives or no negatives, since the AUROC is then undefined.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    pos = np.asarray(positive, dtype=bool).reshape(-1)
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(s)
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def calibration_bins(y_true, probs, n_bins: int = ECE_BINS) -> list[dict]:
    """Top-label reliability bins over (0, 1]: mean confidence vs accuracy per bin."""
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(probs, dtype=np.float64)
    conf = p.max(axis=1) if p.size else np.zeros(0)
    correct = (p.argmax(axis=1) == y) if p.size else np.zeros(0, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins so a confidence of exactly 1.0 lands in the last one.
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        bins.append({
            "low": float(edges[b]),
            "high": float(edges[b + 1]),
            "count": n,
            "confidence": float(conf[sel].mean()) if n else None,
            "accuracy": float(correct[sel].mean()) if n else None,
        })
    return bins


def expected_calibration_error(y_true, probs, n_bins: int = ECE_BINS) -> Optional[float]:
    """Top-label ECE: count-weighted mean |accuracy - confidence| over equal-width bins."""
    bins = calibration_bins(y_true, probs, n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return None
    return float(sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]) / total)


def _accuracy(y: np.ndarray, pred: np.ndarray) -> Optional[float]:
    return float(np.mean(pred == y)) if y.size else None


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> Optional[float]:
    recalls = [float(np.mean(pred[y == k] == k)) for k in range(n_classes) if np.any(y == k)]
    return float(np.mean(recalls)) if recalls else None


def _macro_auroc(y: np.ndarray, p: np.ndarray, n_classes: int) -> tuple[Optional[float], list[Optional[float]]]:
    per_class = [auroc(p[:, k], y == k) for k in range(n_classes)]
    return _mean_or_none(per_class), per_class


def _neoplastic_index(classes: Sequence[str]) -> Optional[tuple[int, list[int]]]:
    """(benign index, [neoplastic indices]) if `classes` is the diagnostic taxonomy."""
    names = list(classes)
    needed = [c.value for c in DiagnosticCategory]
    if not all(n in names for n in needed):
        return None
    neo = [names.index(DiagnosticCategory.PRECANCEROUS.value), names.index(DiagnosticCategory.CANCEROUS.value)]
    return names.index(DiagnosticCategory.BENIGN.value), neo


def _neoplastic_npv(y: np.ndarray, p: np.ndarray, neo: list[int]) -> Optional[float]:
    positive = np.isin(y, neo)
    predicted = p[:, neo].sum(axis=1) >= NEOPLASTIC_THRESHOLD
    _, _, tn, fn = _binary_counts(positive, predicted)
    return _ratio(tn, tn + fn)


def decide(probs, abstain_below: float, classes: Sequence[str] = CATEGORIES) -> np.ndarray:
    """The deployed decision per row: the index of the top class, or ABSTAIN (-1).

    The model abstains when the top probability is below `abstain_below`, and, for the
    diagnostic taxonomy, when the top class is benign but P(precancerous) + P(cancerous)
    is at least as large: a lesion the model rates as likely neoplastic as not is never
    called benign. OnnxLesionClassifier applies exactly this rule to each finding's
    frame-averaged probabilities, so these metrics describe what the report shows.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(classes):
        raise ValueError(f"probs must have shape (N, {len(classes)}), got {p.shape}")
    if p.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    pred = p.argmax(axis=1)
    out = np.where(p.max(axis=1) >= abstain_below, pred, ABSTAIN)
    neo = _neoplastic_index(classes)
    if neo is not None:
        benign, neo_idx = neo
        out[(pred == benign) & (p[:, neo_idx].sum(axis=1) >= p[:, benign])] = ABSTAIN
    return out.astype(np.int64)


def _deployed_neoplastic(y: np.ndarray, decision: np.ndarray, benign: int, neo: list[int]) -> dict:
    """Neoplastic vs benign on the lesions the model answers for, plus where abstentions fell."""
    answered = decision != ABSTAIN
    positive = np.isin(y, neo)
    tp, fp, tn, fn = _binary_counts(positive[answered], np.isin(decision[answered], neo))
    return {
        "definition": "answered lesions only; predicted neoplastic when the AI category is precancerous "
        "or cancerous, benign when it is benign",
        **binary_rates(tp, fp, tn, fn),
        "neoplastic_abstained": int(np.sum(positive & ~answered)),
        "benign_abstained": int(np.sum(~positive & ~answered)),
    }


def _deployed_cancer(y: np.ndarray, decision: np.ndarray, classes: list[str]) -> dict:
    """Where every cancer ended up under the deployed decision. Abstentions are not detections."""
    c = classes.index(DiagnosticCategory.CANCEROUS.value)
    cancers = decision[y == c]
    out = {"n_positive": int(cancers.size)}
    for name in classes:
        out[f"called_{name}"] = int(np.sum(cancers == classes.index(name)))
    out["abstained"] = int(np.sum(cancers == ABSTAIN))
    out["sensitivity"] = _ratio(out["called_cancerous"], cancers.size)
    out["called_benign_rate"] = _ratio(out["called_benign"], cancers.size)
    return out


def selective_prediction(y_true, probs, threshold: float, classes: Sequence[str] = CATEGORIES) -> dict:
    """Coverage and performance of the deployed decision (`decide`) at abstention threshold `threshold`."""
    classes = list(classes)
    y, p = _validate(y_true, probs, len(classes))
    decision = decide(p, threshold, classes)
    covered = decision != ABSTAIN
    n_cov = int(covered.sum())
    k = len(classes)
    confusion = np.zeros((k, k + 1), dtype=np.int64)  # last column: abstained
    for t, d in zip(y, decision):
        confusion[t, k if d == ABSTAIN else d] += 1
    out = {
        "threshold": float(threshold),
        "rule": "top probability >= threshold"
        + (", and a benign call only when P(benign) > P(precancerous) + P(cancerous)"
           if _neoplastic_index(classes) is not None else ""),
        "n": int(y.size),
        "n_covered": n_cov,
        "n_abstained": int(y.size - n_cov),
        "coverage": _ratio(n_cov, y.size),
        "accuracy_covered": _accuracy(y[covered], decision[covered]),
        "balanced_accuracy_covered": _balanced_accuracy(y[covered], decision[covered], k),
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": "rows = true class, columns = AI category, last column = abstained",
    }
    neo = _neoplastic_index(classes)
    if neo is not None:
        benign, neo_idx = neo
        deployed = _deployed_neoplastic(y, decision, benign, neo_idx)
        out["neoplastic"] = deployed
        # The PIVI benchmark is defined on high-confidence optical diagnoses only.
        out["neoplastic_npv_covered"] = deployed["npv"]
        out["cancer"] = _deployed_cancer(y, decision, classes)
    return out


def classification_report(
    y_true,
    probs,
    classes: Sequence[str] = CATEGORIES,
    abstain_below: Optional[float] = None,
) -> dict:
    """All classifier metrics as one JSON-serialisable dict.

    `y_true` holds class indices into `classes`; `probs` is (N, len(classes)),
    ideally temperature-calibrated. The top-level figures score every lesion (argmax,
    and P(neoplastic) >= 0.5 for the neoplastic summary), including lesions the
    deployed model would abstain on. `selective` scores the deployed decision at
    `abstain_below` (see `decide`), which is what the report shows; headline clinical
    figures should come from there.
    """
    classes = list(classes)
    k = len(classes)
    y, p = _validate(y_true, probs, k)
    n = int(y.size)
    pred = p.argmax(axis=1) if n else np.zeros(0, dtype=np.int64)
    cm = confusion_matrix(y, pred, k)
    macro, per_auc = _macro_auroc(y, p, k)

    per_class = {}
    for i, name in enumerate(classes):
        tp, fp, tn, fn = _binary_counts(y == i, pred == i)
        per_class[name] = {"support": int(np.sum(y == i)), **binary_rates(tp, fp, tn, fn), "auroc": per_auc[i]}

    report = {
        "n": n,
        "classes": classes,
        "class_counts": {name: int(np.sum(y == i)) for i, name in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_axes": "rows = true class, columns = predicted class",
        "accuracy": _accuracy(y, pred),
        "balanced_accuracy": _balanced_accuracy(y, pred, k),
        "macro_auroc": macro,
        "macro_auroc_n_classes": int(sum(a is not None for a in per_auc)),
        "per_class": per_class,
        "ece": expected_calibration_error(y, p),
        "ece_bins": ECE_BINS,
        "calibration": calibration_bins(y, p),
        "neoplastic": None,
        "cancer": None,
        "selective": selective_prediction(y, p, abstain_below, classes) if abstain_below is not None else None,
        "coverage_table": [selective_prediction(y, p, t, classes) for t in COVERAGE_THRESHOLDS],
    }

    neo = _neoplastic_index(classes)
    if neo is not None:
        _, neo_idx = neo
        positive = np.isin(y, neo_idx)
        p_neo = p[:, neo_idx].sum(axis=1)
        tp, fp, tn, fn = _binary_counts(positive, p_neo >= NEOPLASTIC_THRESHOLD)
        report["neoplastic"] = {
            "definition": "precancerous + cancerous vs benign over all lesions, including those the "
            f"deployed model abstains on; predicted neoplastic when P(precancerous) + P(cancerous) >= "
            f"{NEOPLASTIC_THRESHOLD}",
            "threshold": NEOPLASTIC_THRESHOLD,
            "n_positive": int(positive.sum()),
            "n_negative": int((~positive).sum()),
            **binary_rates(tp, fp, tn, fn),
            "auroc": auroc(p_neo, positive),
        }
        c = classes.index(DiagnosticCategory.CANCEROUS.value)
        tp, fp, tn, fn = _binary_counts(y == c, pred == c)
        report["cancer"] = {
            "definition": "cancerous vs rest at the argmax decision over all lesions, including those "
            "the deployed model abstains on",
            "n_positive": int(np.sum(y == c)),
            **binary_rates(tp, fp, tn, fn),
            "auroc": per_auc[c],
        }
    return report


def _percentile_ci(values: list[float], alpha: float) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    lo, hi = np.percentile(np.asarray(values), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def exact_all_or_none_ci(n: int, all_successes: bool, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson interval for n successes out of n (or none out of n)."""
    bound = (alpha / 2) ** (1.0 / n)
    return (bound, 1.0) if all_successes else (0.0, 1.0 - bound)


def bootstrap_ci(
    y_true,
    probs,
    groups,
    n: int = 1000,
    seed: int = 0,
    classes: Sequence[str] = CATEGORIES,
    alpha: float = 0.05,
    abstain_below: Optional[float] = None,
) -> dict:
    """Patient-grouped percentile bootstrap confidence intervals.

    Whole patients (`groups`) are resampled with replacement, so correlated
    lesions or frames from one patient move together. Covers accuracy,
    balanced accuracy, macro AUROC and neoplastic NPV. With `abstain_below`, the
    neoplastic NPV is that of the deployed decision (benign calls actually made,
    abstentions excluded), as in `selective`; without it, P(neoplastic) >= 0.5 over
    all lesions. Replicates in which a metric is undefined (for example no
    negatives) are skipped and counted in `n_valid`.

    With no errors (or no correct results) every resample gives the same value, so the
    percentile interval collapses to a point. Accuracy and neoplastic NPV then get the
    exact Clopper-Pearson interval with each patient that contributes to the metric as
    one trial ("method": "exact"); balanced accuracy and macro AUROC get no interval
    (low and high None) and a "note" saying why.
    """
    classes = list(classes)
    k = len(classes)
    y, p = _validate(y_true, probs, k)
    g = np.asarray(groups).reshape(-1)
    if g.shape[0] != y.shape[0]:
        raise ValueError(f"{y.shape[0]} labels but {g.shape[0]} group ids")
    neo = _neoplastic_index(classes)

    def npv_rows(yy: np.ndarray, pp: np.ndarray) -> np.ndarray:
        """Rows in the NPV denominator: the lesions called (or predicted) benign."""
        if neo is None:
            return np.zeros(yy.shape, bool)
        if abstain_below is None:
            return pp[:, neo[1]].sum(axis=1) < NEOPLASTIC_THRESHOLD
        return decide(pp, abstain_below, classes) == neo[0]

    def npv(yy: np.ndarray, pp: np.ndarray) -> Optional[float]:
        if neo is None:
            return None
        if abstain_below is None:
            return _neoplastic_npv(yy, pp, neo[1])
        return _deployed_neoplastic(yy, decide(pp, abstain_below, classes), *neo)["npv"]

    def compute(yy: np.ndarray, pp: np.ndarray) -> dict[str, Optional[float]]:
        pred = pp.argmax(axis=1)
        return {
            "accuracy": _accuracy(yy, pred),
            "balanced_accuracy": _balanced_accuracy(yy, pred, k),
            "macro_auroc": _macro_auroc(yy, pp, k)[0],
            "neoplastic_npv": npv(yy, pp),
        }

    point = compute(y, p) if y.size else dict.fromkeys(("accuracy", "balanced_accuracy", "macro_auroc", "neoplastic_npv"))
    _, inverse = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)] if y.size else []
    samples: dict[str, list[float]] = {m: [] for m in point}
    rng = np.random.default_rng(seed)
    for _ in range(n if members else 0):
        chosen = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[c] for c in chosen])
        for metric, value in compute(y[idx], p[idx]).items():
            if value is not None:
                samples[metric].append(value)

    out = {"method": "patient-grouped percentile bootstrap", "n_resamples": int(n), "n_groups": len(members),
           "confidence": 1 - alpha, "seed": int(seed),
           "degenerate_rule": "no errors observed: exact Clopper-Pearson interval over patients for accuracy and "
                              "neoplastic NPV, no interval for the other metrics",
           "neoplastic_npv_rule": "P(neoplastic) >= 0.5, all lesions" if abstain_below is None
           else f"deployed decision at abstain_below {abstain_below}, answered lesions"}
    # Patients that contribute to each proportion: one trial each for the exact interval.
    units = {"accuracy": len(members),
             "neoplastic_npv": len(np.unique(g[npv_rows(y, p)])) if y.size else 0}
    for metric, value in point.items():
        values = samples[metric]
        entry = {"estimate": value, "low": None, "high": None, "n_valid": len(values)}
        if values and max(values) - min(values) <= 1e-12:  # every resample identical: the percentile CI is a point
            extreme = value is not None and (value >= 1 - 1e-12 or value <= 1e-12)
            if extreme and units.get(metric):
                all_right = value >= 1 - 1e-12
                lo, hi = exact_all_or_none_ci(units[metric], all_right, alpha)
                entry.update(low=lo, high=hi, method="exact",
                             note=f"no {'errors' if all_right else 'correct results'} observed; Clopper-Pearson "
                                  f"interval over {units[metric]} patient{'s' if units[metric] != 1 else ''}")
            else:
                entry["note"] = ("not estimable: every resample gave the same value"
                                 + (" (no errors observed)" if extreme else ""))
        else:
            entry["low"], entry["high"] = _percentile_ci(values, alpha)
        out[metric] = entry
    return out
