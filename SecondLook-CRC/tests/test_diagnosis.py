"""Optical diagnosis (CADx): label taxonomy, the shared lesion crop, metrics, the ONNX
classifier and the whole train -> export -> analyse -> report chain on synthetic data.

The classifier tests build small ONNX models whose logits are the mean colour of the
crop (red -> benign, green -> precancerous, blue -> cancerous), so the expected output
of every check is known without training anything.
"""

import csv
import html
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from secondlook.diagnosis import metrics
from secondlook.diagnosis.preprocess import bbox_from_mask, crop_lesion, square_crop_box, to_model_input
from secondlook.diagnosis.taxonomy import CATEGORIES, DiagnosticCategory, category_for
from secondlook.models import Detection, Finding

ROOT = Path(__file__).resolve().parents[1]
SIDECAR_KEYS = {
    "name", "version", "task", "classes", "input_size", "mean", "std", "crop_margin", "temperature",
    "abstain_below", "min_frame_agreement", "output", "backbone", "training_data", "split_sizes", "metrics",
    "created_utc", "intended_use",
}


# ----------------------------------------------------------------------------- taxonomy


@pytest.mark.parametrize(
    "label, expected",
    [
        ("hyperplastic", "benign"),
        ("Hyperplastic polyp", "benign"),
        ("inflammatory_polyp", "benign"),
        ("Tubular adenoma", "precancerous"),
        ("tubulo-villous adenoma", "precancerous"),
        ("SSL", "precancerous"),
        ("Sessile serrated lesion", "precancerous"),
        ("high grade dysplasia", "precancerous"),
        ("adenocarcinoma", "cancerous"),
        ("Intramucosal carcinoma", "cancerous"),
        ("CANCEROUS", "cancerous"),
        # Pathology-report phrasing with a dysplasia grade (the report's own placeholder among them).
        ("tubular adenoma, low-grade dysplasia", "precancerous"),
        ("tubular adenoma with low-grade dysplasia", "precancerous"),
        ("Tubulovillous adenoma with high-grade dysplasia", "precancerous"),
        ("adenoma with high-grade dysplasia", "precancerous"),
        ("SSL with dysplasia", "precancerous"),
        ("hyperplastic polyp with dysplasia", "precancerous"),  # dysplasia is never benign
        ("adenocarcinoma with high grade dysplasia", "cancerous"),
    ],
)
def test_histology_labels_map_to_categories(label, expected):
    assert category_for(label).value == expected


# "neoplastic" alone could be a carcinoma, and a lipoma is a (mesenchymal) neoplasm: a person decides.
@pytest.mark.parametrize("label", ["carcinoid", "polyp", "", "adenoma?", "neoplastic", "lipoma",
                                   "hyperplastic polyp, no dysplasia", "lipoma with dysplasia", "dysplasia",
                                   "adenoma, negative for high grade dysplasia"])
def test_unknown_label_raises(label):
    with pytest.raises(KeyError):
        category_for(label)


def test_report_histology_placeholder_is_a_valid_label():
    from secondlook import report

    assert category_for(report.HISTOLOGY_EXAMPLE).value == "precancerous"


def test_category_order_is_the_model_class_order():
    assert CATEGORIES == ["benign", "precancerous", "cancerous"]
    assert [c.value for c in DiagnosticCategory] == CATEGORIES


# ----------------------------------------------------------------------------- preprocessing


def _disc_image(shape=(120, 160), centre=(80, 60), radius=15, colour=(0, 0, 255), background=128):
    img = np.full((*shape, 3), background, np.uint8)
    cv2.circle(img, centre, radius, colour, -1)
    return img


def test_crop_lesion_shape_dtype_and_centring():
    img = _disc_image()
    crop = crop_lesion(img, (65, 45, 30, 30), 64)
    assert crop.shape == (64, 64, 3) and crop.dtype == np.uint8
    red = (crop[..., 2] > 200) & (crop[..., 0] < 60)
    x, y, w, h = bbox_from_mask(red)
    assert abs((x + w / 2) - 32) <= 2 and abs((y + h / 2) - 32) <= 2
    # 25% margin each side: the 30 px lesion fills about 1 / 1.5 of the crop.
    assert w == pytest.approx(64 / 1.5, abs=3)


def test_crop_at_border_is_padded_not_stretched():
    img = _disc_image(centre=(15, 60), radius=10)
    box = square_crop_box((5, 50, 20, 20), img.shape, margin=1.0)
    assert box[0] == 0 and box[2] - box[0] < box[3] - box[1]  # clipped: narrower than tall
    crop = crop_lesion(img, (5, 50, 20, 20), 96, margin=1.0)
    assert crop.shape == (96, 96, 3)
    _, _, w, h = bbox_from_mask(crop[..., 2] > 200)
    assert abs(w - h) <= 2  # the disc stays round
    # Whole-image crops of a non-square image are padded to a square too.
    whole = crop_lesion(_disc_image(shape=(60, 160), centre=(80, 30), radius=20), None, 80)
    _, _, w, h = bbox_from_mask(whole[..., 2] > 200)
    assert whole.shape == (80, 80, 3) and abs(w - h) <= 2


def test_bbox_from_mask():
    mask = np.zeros((20, 30), bool)
    mask[3:7, 10:15] = True
    assert bbox_from_mask(mask) == (10, 3, 5, 4)
    assert bbox_from_mask(np.zeros((5, 5), np.uint8)) is None


def test_to_model_input_is_normalised_nchw_rgb():
    blue = np.zeros((8, 8, 3), np.uint8)
    blue[..., 0] = 255  # BGR blue
    grey = np.full((8, 8, 3), 51, np.uint8)
    x = to_model_input([blue, grey], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    assert x.shape == (2, 3, 8, 8) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert np.allclose(x[0, 0], -1) and np.allclose(x[0, 1], -1) and np.allclose(x[0, 2], 1)  # R, G, B
    assert np.allclose(x[1], (0.2 - 0.5) / 0.5)
    imagenet = to_model_input([grey], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    assert np.allclose(imagenet[0, :, 0, 0], (0.2 - np.array([0.485, 0.456, 0.406])) / [0.229, 0.224, 0.225], atol=1e-6)


# ----------------------------------------------------------------------------- metrics


def test_auroc_hand_computed_cases():
    pos = np.array([0, 0, 1, 1], bool)
    assert metrics.auroc([0.1, 0.2, 0.8, 0.9], pos) == 1.0
    assert metrics.auroc([0.9, 0.8, 0.2, 0.1], pos) == 0.0
    assert metrics.auroc([0.5, 0.5, 0.5, 0.5], pos) == 0.5
    # Pairs (pos, neg): (0.4, 0.1) 1, (0.4, 0.4) 0.5, (0.8, 0.1) 1, (0.8, 0.4) 1 -> 3.5 / 4.
    assert metrics.auroc([0.1, 0.4, 0.4, 0.8], pos) == pytest.approx(0.875)
    assert metrics.auroc([0.3, 0.7], [False, False]) is None
    assert metrics.auroc([0.3, 0.7], [True, True]) is None


# Six lesions: true classes and calibrated probabilities chosen so every number below can be
# checked by hand. Argmax predictions: [0, 0, 1, 1, 0, 2].
Y = np.array([0, 0, 0, 1, 1, 2])
P = np.array([
    [0.82, 0.13, 0.05],  # benign, correct, P(neoplastic) 0.18
    [0.65, 0.25, 0.10],  # benign, correct, 0.35
    [0.30, 0.65, 0.05],  # benign called precancerous, 0.70
    [0.10, 0.70, 0.20],  # precancerous, correct, 0.90
    [0.55, 0.40, 0.05],  # precancerous called benign, 0.45 (and below the abstention threshold)
    [0.05, 0.13, 0.82],  # cancerous, correct, 0.95
])


def test_classification_report_hand_computed():
    r = metrics.classification_report(Y, P, abstain_below=0.6)
    assert r["n"] == 6 and r["classes"] == CATEGORIES
    assert r["confusion_matrix"] == [[2, 1, 0], [1, 1, 0], [0, 0, 1]]
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)

    benign, pre, can = (r["per_class"][c] for c in CATEGORIES)
    assert (benign["tp"], benign["fp"], benign["tn"], benign["fn"]) == (2, 1, 2, 1)
    for key in ("sensitivity", "specificity", "ppv", "npv"):
        assert benign[key] == pytest.approx(2 / 3)
    assert pre["sensitivity"] == pytest.approx(0.5) and pre["specificity"] == pytest.approx(0.75)
    assert pre["ppv"] == pytest.approx(0.5) and pre["npv"] == pytest.approx(0.75)
    assert can["sensitivity"] == 1.0 and can["specificity"] == 1.0 and can["auroc"] == 1.0

    neo = r["neoplastic"]  # positive when P(precancerous) + P(cancerous) >= 0.5
    assert (neo["tp"], neo["fp"], neo["tn"], neo["fn"]) == (2, 1, 2, 1)
    assert neo["npv"] == pytest.approx(2 / 3) and neo["sensitivity"] == pytest.approx(2 / 3)
    assert r["cancer"]["sensitivity"] == 1.0 and r["cancer"]["n_positive"] == 1

    sel = r["selective"]  # lesion 5 (top probability 0.55) is not answered
    assert sel["n_covered"] == 5 and sel["coverage"] == pytest.approx(5 / 6)
    assert sel["accuracy_covered"] == pytest.approx(4 / 5)
    assert sel["neoplastic_npv_covered"] == 1.0  # the confident 'benign' calls were all benign
    assert sel["confusion_matrix"] == [[2, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0]]  # last column: abstained
    dn = sel["neoplastic"]
    assert (dn["tp"], dn["fp"], dn["tn"], dn["fn"]) == (2, 1, 2, 0) and dn["neoplastic_abstained"] == 1
    assert sel["cancer"] == {"n_positive": 1, "called_benign": 0, "called_precancerous": 0, "called_cancerous": 1,
                             "abstained": 0, "sensitivity": 1.0, "called_benign_rate": 0.0}

    # Top-label ECE, 15 bins: {0.82, 0.82} both right, {0.65, 0.65} one right, 0.70 right, 0.55 wrong.
    assert r["ece"] == pytest.approx((2 * 0.18 + 2 * 0.15 + 0.30 + 0.55) / 6)
    json.dumps(r, allow_nan=False)


def test_deployed_decision_never_calls_likely_neoplastic_lesions_benign():
    probs = [[0.50, 0.30, 0.20], [0.51, 0.30, 0.19], [0.30, 0.60, 0.10], [0.45, 0.30, 0.25], [0.05, 0.05, 0.90]]
    assert metrics.decide(probs, 0.5).tolist() == [metrics.ABSTAIN, 0, 1, metrics.ABSTAIN, 2]
    assert metrics.decide(probs, 0.7).tolist() == [metrics.ABSTAIN] * 4 + [2]
    # The deployed figures follow the same rule: the 0.50 / 0.50 lesion is withheld, not called benign.
    sel = metrics.classification_report([1, 0, 1, 2, 2], probs, abstain_below=0.5)["selective"]
    assert sel["neoplastic"]["fn"] == 0 and sel["neoplastic"]["neoplastic_abstained"] == 2
    assert sel["cancer"]["abstained"] == 1 and sel["cancer"]["called_cancerous"] == 1
    assert min(metrics.COVERAGE_THRESHOLDS) >= metrics.MIN_ABSTAIN_BELOW == 0.5


def test_undefined_metrics_are_none_and_json_safe():
    only_benign = metrics.classification_report([0, 0, 0], [[0.9, 0.05, 0.05], [0.6, 0.3, 0.1], [0.4, 0.5, 0.1]])
    assert only_benign["macro_auroc"] is None and only_benign["macro_auroc_n_classes"] == 0
    assert only_benign["per_class"]["cancerous"]["sensitivity"] is None
    assert only_benign["cancer"]["sensitivity"] is None and only_benign["neoplastic"]["npv"] == pytest.approx(1.0)
    assert only_benign["neoplastic"]["sensitivity"] is None and only_benign["selective"] is None
    empty = metrics.classification_report(np.zeros(0, int), np.zeros((0, 3)), abstain_below=0.6)
    assert empty["accuracy"] is None and empty["ece"] is None and empty["selective"]["coverage"] is None
    for report in (only_benign, empty):
        json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError):
        metrics.classification_report([0, 1], [[0.5, 0.5, 0.0]])


def test_patient_bootstrap_ci():
    y = np.repeat([0, 1, 2], 4)
    perfect = np.eye(3)[y] * 0.9 + 0.1 / 3
    ci = metrics.bootstrap_ci(y, perfect, groups=np.arange(12) // 2, n=200, seed=1)
    # No errors: the percentile interval would be [100%, 100%]; an exact interval over the 6 patients is used.
    acc = ci["accuracy"]
    assert ci["n_groups"] == 6 and acc["method"] == "exact" and acc["high"] == 1.0
    assert acc["low"] == pytest.approx(0.025 ** (1 / 6)) == pytest.approx(0.5407, abs=1e-4)
    assert "no errors observed" in acc["note"] and "6 patients" in acc["note"]
    for metric in ("balanced_accuracy", "macro_auroc"):
        assert ci[metric]["low"] is None and ci[metric]["high"] is None and "not estimable" in ci[metric]["note"]
    deployed = metrics.bootstrap_ci(y, perfect, groups=np.arange(12) // 2, n=200, seed=1, abstain_below=0.6)
    npv = deployed["neoplastic_npv"]  # benign calls come from 2 patients: [0.158, 1]
    assert npv["estimate"] == 1.0 and npv["method"] == "exact" and npv["low"] == pytest.approx(0.025 ** (1 / 2))
    wrong = metrics.bootstrap_ci(y, np.roll(perfect, 1, axis=1), groups=np.arange(12) // 2, n=50, seed=1)
    assert wrong["accuracy"]["estimate"] == 0.0 and wrong["accuracy"]["low"] == 0.0
    assert wrong["accuracy"]["high"] == pytest.approx(1 - 0.025 ** (1 / 6))
    json.dumps(ci, allow_nan=False)
    noisy = metrics.bootstrap_ci(Y, P, groups=["a", "a", "b", "c", "c", "d"], n=300, seed=1)
    acc = noisy["accuracy"]
    assert acc["low"] <= acc["estimate"] <= acc["high"] and acc["n_valid"] == 300
    json.dumps(noisy, allow_nan=False)


# ----------------------------------------------------------------------------- ONNX classifier


def _train_module():
    if "train_classifier" in sys.modules:
        return sys.modules["train_classifier"]
    spec = importlib.util.spec_from_file_location("train_classifier", ROOT / "training" / "train_classifier.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up while being defined
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def colour_model(tmp_path_factory):
    """make(name, order=..., gain=..., **sidecar) -> path of an ONNX model whose logits are
    `gain` x the crop's mean normalised R, G and B, output in the category order `order`."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    export_onnx = _train_module().export_onnx
    root = tmp_path_factory.mktemp("colour_models")

    class MeanColour(torch.nn.Module):
        def __init__(self, order, gain):
            super().__init__()
            rows = [CATEGORIES.index(c) for c in order]  # benign = R, precancerous = G, cancerous = B
            self.register_buffer("w", torch.eye(3)[rows] * gain)

        def forward(self, x):
            return x.mean(dim=(2, 3)) @ self.w.T

    def make(name, order=CATEGORIES, gain=4.0, size=64, module=None, **sidecar):
        path = root / f"{name}.onnx"
        assert export_onnx((module or MeanColour(order, gain)).eval(), size, path) >= 17
        meta = {
            "name": f"colour-{name}", "version": "test-1", "task": "lesion-classification", "classes": list(order),
            "input_size": size, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "crop_margin": 0.25,
            "temperature": 1.0, "abstain_below": 0.6, "min_frame_agreement": 0.5, "output": "logits",
            **sidecar,
        }
        path.with_suffix(".json").write_text(json.dumps(meta))
        return path

    make.MeanColour = MeanColour
    return make


BOX = (65, 45, 30, 30)  # around the disc drawn by _disc_image


def _finding(colours, fid="F001", box=BOX):
    """A finding seen in len(colours) frames, frame i showing a disc of colours[i] (BGR)."""
    frames, dets = {}, []
    for i, colour in enumerate(colours):
        frames[10 + i] = _disc_image(colour=colour)
        dets.append(Detection(10 + i, (10 + i) / 5, box, 0.6 + 0.01 * (i % 7), "lesion", None, None))
    return Finding(fid, dets), frames


RED, GREEN, BLUE, GREY = (0, 0, 255), (0, 255, 0), (255, 0, 0), (128, 128, 128)


def test_classifier_aggregates_frames_and_explains(colour_model):
    from secondlook.diagnosis import load_classifier
    from secondlook.diagnosis.classifier import SAFETY_STATEMENT

    clf = load_classifier(colour_model("base"))
    assert (clf.name, clf.version) == ("colour-base", "test-1")
    finding, frames = _finding([RED] * 12)
    c = clf.characterise(finding, frames)
    assert c.category == "benign" and not c.abstained and c.abstain_reason is None
    assert (c.model, c.model_version) == ("colour-base", "test-1")
    assert c.frames_used == 8 and c.frame_agreement == 1.0  # max_frames
    assert list(c.probabilities) == CATEGORIES and sum(c.probabilities.values()) == pytest.approx(1.0)
    assert c.confidence == pytest.approx(max(c.probabilities.values())) and c.confidence > 0.8
    assert c.malignancy_risk == pytest.approx(c.probabilities["cancerous"])
    assert c.neoplasia_risk == pytest.approx(c.probabilities["precancerous"] + c.probabilities["cancerous"])
    assert SAFETY_STATEMENT in c.note and "8 frames" in c.note

    assert clf.characterise(*_finding([BLUE] * 3)).category == "cancerous"
    assert clf.characterise(*_finding([GREEN] * 3)).category == "precancerous"

    crop, heat = c.explanation_crop, c.explanation_heatmap
    assert crop.dtype == np.uint8 and crop.ndim == 3 and crop.shape[2] == 3
    assert heat.dtype == np.float32 and heat.shape == crop.shape[:2]
    assert heat.min() >= 0 and 0.95 < heat.max() <= 1.0
    s = heat.shape[0]
    centre = heat[s // 2 - s // 8 : s // 2 + s // 8, s // 2 - s // 8 : s // 2 + s // 8].mean()
    corners = np.mean([heat[:s // 8, :s // 8].mean(), heat[-s // 8:, -s // 8:].mean()])
    assert centre > 0.5 > corners  # hiding the red disc, not the grey mucosa, lowers P(benign)

    quiet = load_classifier(colour_model("base"), explain=False).characterise(finding, frames)
    assert quiet.explanation_crop is None and quiet.explanation_heatmap is None


def test_classifier_abstains_on_low_confidence(colour_model):
    from secondlook.diagnosis import load_classifier

    c = load_classifier(colour_model("base")).characterise(*_finding([GREY] * 5))
    assert c.abstained and c.category is None
    assert c.abstain_reason.startswith("low confidence (0.3")
    assert sum(c.probabilities.values()) == pytest.approx(1.0) and c.frames_used == 5
    assert c.malignancy_risk is not None  # still reported, for the priority list
    assert c.explanation_crop is None  # no map for a category the model declined to give
    assert "No AI category given" in c.note


def test_classifier_abstains_when_frames_disagree(colour_model):
    from secondlook.diagnosis import load_classifier

    # Saturated model: each frame is certain, but only 3 of 7 frames say benign.
    colours = [RED, RED, RED, GREEN, GREEN, BLUE, BLUE]
    strict = load_classifier(colour_model("sharp", gain=60.0)).characterise(*_finding(colours))
    assert strict.abstained and strict.frame_agreement == pytest.approx(3 / 7)
    assert "low confidence" in strict.abstain_reason and "frames disagree (3/7 agree)" in strict.abstain_reason

    # Confident on average (0.66 benign), but only 3 of 7 frames have benign on top.
    clf = load_classifier(colour_model("base"), explain=False)
    clf.predict = lambda x: np.array([[0.95, 0.03, 0.02]] * 3 + [[0.45, 0.50, 0.05]] * 4)[: len(x)]
    c = clf.characterise(*_finding([RED] * 7))
    assert c.abstained and c.abstain_reason == "frames disagree (3/7 agree)"
    assert c.confidence == pytest.approx((3 * 0.95 + 4 * 0.45) / 7)


def _stubbed(colour_model, rows, **sidecar):
    """Classifier on the colour model whose per-frame probabilities are `rows` (one per frame)."""
    from secondlook.diagnosis import load_classifier

    clf = load_classifier(colour_model(sidecar.pop("name", "base"), **sidecar), explain=False)
    rows = np.asarray(rows, np.float64)
    clf.predict = lambda x: rows[: len(x)]
    return clf.characterise(*_finding([RED] * len(rows)))


def test_classifier_never_calls_a_likely_neoplastic_lesion_benign(colour_model):
    c = _stubbed(colour_model, [[0.50, 0.30, 0.20]] * 4, name="half", abstain_below=0.5)
    assert c.abstained and c.category is None and "not called benign" in c.abstain_reason
    assert c.neoplasia_risk == pytest.approx(0.5)
    ok = _stubbed(colour_model, [[0.55, 0.30, 0.15]] * 4, name="half", abstain_below=0.5)
    assert ok.category == "benign" and not ok.abstained


def test_classifier_abstains_when_some_frames_confidently_suggest_cancer(colour_model):
    # A depressed area seen from two of eight angles: the average alone says precancerous at 0.71.
    c = _stubbed(colour_model, [[0.03, 0.92, 0.05]] * 6 + [[0.02, 0.08, 0.90]] * 2)
    assert c.abstained and c.category is None
    assert "2 of 8 frames confidently suggest a higher-risk category (Suspected cancer)" in c.abstain_reason
    assert c.peak_malignancy_risk == pytest.approx(0.90) and c.malignancy_risk == pytest.approx(0.2625)
    one = _stubbed(colour_model, [[0.03, 0.92, 0.05]] * 7 + [[0.02, 0.08, 0.90]])  # one frame is not enough
    assert one.category == "precancerous" and not one.abstained


@pytest.mark.parametrize("n_benign", [2, 3, 5, 7])
def test_benign_call_is_withheld_when_any_frame_confidently_suggests_neoplasia(colour_model, n_benign):
    # Short tracks (colonoscopy keeps tracks of 3 frames, capsule of 2) must be protected too:
    # a benign call is what leads to "leave in place".
    benign, cancer, adenoma = [0.98, 0.01, 0.01], [0.01, 0.01, 0.98], [0.10, 0.85, 0.05]
    c = _stubbed(colour_model, [benign] * n_benign + [cancer])
    assert c.abstained and c.category is None and c.peak_malignancy_risk == pytest.approx(0.98)
    assert f"1 of {n_benign + 1} frames confidently suggest a higher-risk category (Suspected cancer)" in c.abstain_reason
    pre = _stubbed(colour_model, [benign] * n_benign + [adenoma])
    assert pre.abstained and "(Precancerous" in pre.abstain_reason
    unsure = _stubbed(colour_model, [benign] * n_benign + [[0.40, 0.35, 0.25]])  # not confident: no veto
    assert unsure.category == "benign" and not unsure.abstained


def test_classifier_abstains_on_non_finite_model_output(colour_model):
    nan = [np.nan, np.nan, np.nan]
    c = _stubbed(colour_model, [nan] * 4)
    assert c.abstained and c.category is None and "non-finite" in c.abstain_reason and not c.probabilities
    part = _stubbed(colour_model, [[0.9, 0.05, 0.05]] * 3 + [[np.inf, 0.0, 0.0]])
    assert part.abstained and "non-finite output for 1 of 4 frames" in part.abstain_reason
    assert part.frames_used == 3 and np.isfinite(list(part.probabilities.values())).all()
    assert np.isfinite([part.confidence, part.malignancy_risk, part.neoplasia_risk]).all()


@pytest.mark.parametrize("sidecar, message", [
    ({"temperature": float("nan")}, "temperature"),
    ({"temperature": 0.0}, "temperature"),
    ({"abstain_below": float("nan")}, "abstain_below"),
    ({"abstain_below": 0.4}, "abstain_below"),
    ({"abstain_below": 1.0}, "abstain_below"),
    ({"min_frame_agreement": 1.5}, "min_frame_agreement"),
    ({"input_size": 96}, "input has shape"),  # the ONNX model's input is fixed at 64 x 64
])
def test_classifier_rejects_invalid_sidecar_values(colour_model, sidecar, message):
    from secondlook.diagnosis import load_classifier

    path = colour_model("base")
    meta = json.loads(path.with_suffix(".json").read_text())
    bad = path.with_name(f"bad_{message.split()[0]}.onnx")
    bad.write_bytes(path.read_bytes())
    bad.with_suffix(".json").write_text(json.dumps({**meta, **sidecar}))  # json writes NaN, as a hand edit could
    with pytest.raises(ValueError, match=message):
        load_classifier(bad)


@pytest.mark.parametrize("output", ["Logits", " LOGITS "])
def test_sidecar_output_is_case_insensitive(colour_model, output):
    from secondlook.diagnosis import load_classifier

    exact = load_classifier(colour_model("base"), explain=False)
    loose = load_classifier(colour_model(f"out_{output.strip()}", output=output), explain=False)
    assert loose.output_is_logits
    a, b = (clf.characterise(*_finding([(1, 1, 13)] * 3)) for clf in (exact, loose))
    assert a.probabilities == pytest.approx(b.probabilities) and a.abstained == b.abstained


@pytest.mark.parametrize("output, message", [
    ("logit", "output must be one of"),
    ("softmax", "output must be one of"),
    (["logits"], "output must be one of"),
    ("probabilities", "not a probability per class"),  # the model returns logits
])
def test_sidecar_output_kind_is_checked(colour_model, output, message):
    from secondlook.diagnosis import load_classifier

    with pytest.raises(ValueError, match=message):
        load_classifier(colour_model(f"out_{output}", output=output))


def test_probability_output_models_are_accepted(colour_model):
    import torch

    from secondlook.diagnosis import load_classifier

    class Softmaxed(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x.mean(dim=(2, 3)) * 4.0, dim=1)

    probs = load_classifier(colour_model("softmaxed", module=Softmaxed(), output="probabilities"), explain=False)
    logits = load_classifier(colour_model("base"), explain=False)
    for colours in ([RED] * 3, [BLUE] * 3, [GREY] * 3):
        a, b = (clf.characterise(*_finding(colours)) for clf in (logits, probs))
        assert a.category == b.category and a.probabilities == pytest.approx(b.probabilities, abs=1e-5)


def test_fixed_batch_models_run_on_any_number_of_frames(colour_model, tmp_path):
    import torch

    from secondlook.diagnosis import load_classifier

    base = colour_model("base")
    fixed = tmp_path / "fixed4.onnx"
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        torch.onnx.export(colour_model.MeanColour(CATEGORIES, 4.0).eval(), (torch.zeros(4, 3, 64, 64),), str(fixed),
                          input_names=["image"], output_names=["logits"], opset_version=17,
                          **({"dynamo": False} if "dynamo" in __import__("inspect").signature(torch.onnx.export).parameters else {}))
    fixed.with_suffix(".json").write_text(base.with_suffix(".json").read_text())
    reference = load_classifier(base)
    clf = load_classifier(fixed)
    assert clf.batch_limit == 4
    for n in (1, 3, 4, 5, 8):  # partial chunks, and the one-crop and 64-crop occlusion batches
        a, b = reference.characterise(*_finding([RED] * n)), clf.characterise(*_finding([RED] * n))
        assert b.category == "benign" and not b.abstained, b.abstain_reason
        assert b.probabilities == pytest.approx(a.probabilities, abs=1e-6)
        assert np.allclose(a.explanation_heatmap, b.explanation_heatmap, atol=1e-4)


def test_classifier_rejects_models_that_do_not_fit_the_sidecar(colour_model):
    import torch

    from secondlook.diagnosis import load_classifier

    class FourOutputs(torch.nn.Module):  # an extra column that a 3-class sidecar would silently drop
        def forward(self, x):
            m = x.mean(dim=(2, 3))
            return torch.cat([m, m[:, 2:3] * 5], dim=1)

    class Broken(torch.nn.Module):  # log of a negative number: NaN for every input
        def forward(self, x):
            return torch.log(x.mean(dim=(2, 3)) - 10)

    with pytest.raises(ValueError, match="output"):
        load_classifier(colour_model("four", module=FourOutputs()))
    with pytest.raises(ValueError, match="non-finite"):
        load_classifier(colour_model("nan", module=Broken()))


def test_classifier_reports_provenance_and_calibration(colour_model):
    from secondlook.diagnosis import load_classifier
    from secondlook.ingest import file_sha256

    path = colour_model("base")
    clf = load_classifier(path, explain=False)
    assert clf.info["sha256"] == file_sha256(path) and clf.info["sidecar_sha256"] == file_sha256(path.with_suffix(".json"))
    assert clf.calibrated is False  # no "calibrated": true in the sidecar
    assert "not calibrated" in clf.info["validation_status"] and "No external validation" in clf.info["validation_status"]
    assert clf.characterise(*_finding([RED] * 3)).calibrated is False
    cal = load_classifier(colour_model("calibrated", calibrated=True, synthetic_training_data=True), explain=False)
    assert cal.calibrated and cal.characterise(*_finding([RED] * 3)).calibrated is True
    assert "synthetic images only" in cal.info["validation_status"]


def test_classifier_abstains_without_usable_frames(colour_model):
    from secondlook.diagnosis import load_classifier

    clf = load_classifier(colour_model("base"))
    finding, _ = _finding([RED] * 3)
    c = clf.characterise(finding, {})  # none of the finding's frames supplied
    assert c.abstained and c.abstain_reason == "no usable frames" and c.frames_used == 0 and not c.probabilities
    broken = {d.frame_index: np.zeros((0, 0, 3), np.uint8) for d in finding.detections}
    c = clf.characterise(finding, broken)
    assert c.abstained and c.abstain_reason == "no usable frames"
    # One unreadable frame among good ones is skipped; a greyscale frame is converted.
    finding, frames = _finding([RED] * 4)
    frames[10] = np.zeros((0, 0, 3), np.uint8)
    frames[11] = cv2.cvtColor(frames[11], cv2.COLOR_BGR2GRAY)
    c = clf.characterise(finding, frames)
    assert c.frames_used == 3 and c.category is not None


def test_classifier_remaps_class_order(colour_model):
    from secondlook.diagnosis import load_classifier

    base = load_classifier(colour_model("base"), explain=False)
    order = ["cancerous", "benign", "precancerous"]
    permuted = load_classifier(colour_model("permuted", order=order), explain=False)
    for colours in ([RED] * 3, [BLUE] * 3, [GREEN, RED, BLUE]):
        a, b = base.characterise(*_finding(colours)), permuted.characterise(*_finding(colours))
        assert a.category == b.category
        for k in CATEGORIES:
            assert a.probabilities[k] == pytest.approx(b.probabilities[k], abs=1e-6)


def test_classifier_temperature_softens_confidence(colour_model):
    from secondlook.diagnosis import load_classifier

    sharp = load_classifier(colour_model("base"), explain=False).characterise(*_finding([RED] * 3))
    soft = load_classifier(colour_model("warm", temperature=3.0), explain=False).characterise(*_finding([RED] * 3))
    assert soft.category == sharp.category == "benign" or soft.abstained
    assert soft.confidence < sharp.confidence


def test_classifier_error_in_one_finding_does_not_stop_the_rest(colour_model):
    from secondlook.diagnosis import load_classifier

    clf = load_classifier(colour_model("base"), explain=False)
    real_predict, calls = clf.predict, []

    def flaky(x):
        calls.append(len(x))
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real_predict(x)

    clf.predict = flaky
    bad = clf.characterise(*_finding([RED] * 3, fid="F001"))
    good = clf.characterise(*_finding([RED] * 3, fid="F002"))
    assert bad.abstained and bad.category is None and bad.model == "colour-base"
    assert bad.abstain_reason == "classifier error (RuntimeError: boom)"
    assert good.category == "benign" and not good.abstained


def test_classifier_rejects_bad_model_files(colour_model, tmp_path):
    from secondlook.diagnosis import load_classifier

    with pytest.raises(ValueError, match="not a lesion classifier"):
        load_classifier(colour_model("segmenter", task="segmentation"))
    with pytest.raises(ValueError, match="permutation"):
        load_classifier(colour_model("two_class", classes=["benign", "precancerous", "adenoma"]))
    lonely = tmp_path / "lonely.onnx"
    lonely.write_bytes(colour_model("base").read_bytes())
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_classifier(lonely)
    with pytest.raises(FileNotFoundError):
        load_classifier(tmp_path / "missing.onnx")


def test_frame_selection_spreads_over_track_and_keeps_best():
    from secondlook.diagnosis.classifier import select_detections

    dets = [Detection(i, i / 10, BOX, 0.5 + 0.001 * i, "lesion", None, None) for i in range(40)]
    dets[3].score = 0.99
    picks = select_detections(dets, set(range(40)) - {0, 1}, 8)
    assert len(picks) == 8 and dets[3] in picks
    assert [d.frame_index for d in picks] == sorted(d.frame_index for d in picks)
    assert picks[0].frame_index < 8 and picks[-1].frame_index > 32 and 0 not in [d.frame_index for d in picks]
    assert select_detections(dets[:5], set(range(5)), 8) == dets[:5]


class _StubDetector:
    """Reports the same box in every frame."""

    name, version = "stub", "1"

    def __init__(self, box):
        self.box = box

    def detect(self, frame):
        h, w = frame.image.shape[:2]
        mask = np.zeros((h, w), bool)
        x, y, bw, bh = self.box
        mask[y : y + bh, x : x + bw] = True
        return [Detection(frame.index, frame.timestamp_s, self.box, 0.9, "lesion", mask, mask.astype(np.float32))]


def _noisy_frames(folder, n=4, size=(240, 320)):
    rng = np.random.default_rng(0)
    folder.mkdir()
    for i in range(n):
        img = rng.integers(60, 200, (*size, 3), dtype=np.uint8)  # sharp, well exposed
        cv2.circle(img, (160, 120), 30, RED, -1)
        cv2.imwrite(str(folder / f"frame_{i:03d}.png"), img)
    return folder


def test_classifier_is_not_applied_to_another_modality(colour_model, tmp_path):
    from secondlook import report
    from secondlook.config import get_config
    from secondlook.diagnosis import load_classifier
    from secondlook.pipeline import analyse

    frames = _noisy_frames(tmp_path / "frames")
    clf = load_classifier(colour_model("base"))  # no modality in the sidecar: a colonoscopy model
    detector = _StubDetector((130, 90, 60, 60))
    capsule = analyse(frames, get_config("capsule"), detector, characteriser=clf, image_sequence_fps=5)
    assert capsule.findings
    for f in capsule.findings:
        c = f.characterisation
        assert c.abstained and c.category is None and "not validated for capsule" in c.abstain_reason
    assert "not validated for capsule" in capsule.characteriser_info["unsupported_reason"]
    data = report.write(capsule, tmp_path / "capsule_out")
    assert data["summary"]["ai_categories"]["indeterminate"] == len(capsule.findings)
    assert "Not applied to this recording" in (tmp_path / "capsule_out" / "report.html").read_text()
    # Strict JSON although capsule analysis_fps is infinite.
    json.loads((tmp_path / "capsule_out" / "result.json").read_text(),
               parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))

    colon = analyse(frames, get_config("colonoscopy"), detector, characteriser=clf, image_sequence_fps=5)
    assert colon.findings and all(f.characterisation.frames_used > 0 for f in colon.findings)
    assert "unsupported_reason" not in colon.characteriser_info
    small = clf.unsupported_reason(get_config("colonoscopy", working_size=256))
    assert small is None  # the sidecar gives no working size, so there is nothing to compare
    sized = load_classifier(colour_model("sized", working_size=512))
    assert "less detail" in sized.unsupported_reason(get_config("colonoscopy", working_size=256))
    assert sized.unsupported_reason(get_config("colonoscopy")) is None


def _laplacian_var(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def test_training_and_inference_crops_match_on_hd_frames(colour_model, tmp_path):
    """A 1920 x 1080 frame with a 120 px lesion carrying a 3 px surface pattern. Analysis shrinks the
    frame to 512 px first, so the pattern is gone by the time the classifier sees it; training must
    shrink the same way, or the model is validated on detail it never gets in use."""
    from secondlook.diagnosis import load_classifier
    from secondlook.ingest import resize_to_working_size

    tc = _train_module()
    hd = np.full((1080, 1920, 3), (120, 130, 200), np.uint8)
    x, y, side = 900, 480, 120
    lesion = np.zeros((side, side, 3), np.uint8)
    lesion[:, :] = (60, 60, 170)
    lesion[::6] = lesion[1::6] = lesion[2::6] = (200, 200, 240)  # 3 px stripes
    hd[y : y + side, x : x + side] = lesion
    path = tmp_path / "hd.png"
    cv2.imwrite(str(path), hd)

    clf = load_classifier(colour_model("hd", size=224, working_size=512), explain=False)
    sample = tc.Sample(path=path, image="hd.png", dataset=str(tmp_path), label="adenoma", category=1, patient_id="P1",
                       group="P1", bbox=(x, y, side, side), bbox_shape=None, split=None)
    train_crop = tc.LesionDataset([sample], 224, 0.25, augment=False, working_size=clf.working_size).crop(0)

    frame = resize_to_working_size(hd, 512)  # what ingest hands the pipeline
    s = frame.shape[1] / hd.shape[1]
    detected = tuple(int(round(v * s)) for v in (x, y, side, side))  # the detector's box on that frame
    serve_crop = clf.lesion_crop(frame, detected)
    native = tc.crop_lesion(hd, (x, y, side, side), 224, 0.25)

    assert np.abs(train_crop.astype(int) - serve_crop.astype(int)).mean() < 4
    assert _laplacian_var(native) > 20 * _laplacian_var(train_crop)  # the old training crop kept the pattern
    assert _laplacian_var(train_crop) == pytest.approx(_laplacian_var(serve_crop), rel=0.5, abs=5)


def test_class_weight_prior_shift_is_removed_before_calibration():
    """Inverse-frequency loss weights inflate rare-class probabilities; one temperature cannot undo
    that, the logit adjustment can. Prevalence as in real data: benign 0.30, precancerous 0.62,
    cancerous 0.08."""
    torch = pytest.importorskip("torch")
    tc = _train_module()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    prevalence = np.array([0.30, 0.62, 0.08])
    centres = np.array([[0.0, 0.0], [1.2, 0.3], [2.0, 1.4]])

    def sample(n):
        y = rng.choice(3, size=n, p=prevalence)
        return torch.tensor(centres[y] + rng.normal(size=(n, 2)), dtype=torch.float32), y

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = sample(3000), sample(2000), sample(4000)
    weights = tc.inverse_frequency_weights(np.bincount(y_train, minlength=3))
    model = torch.nn.Linear(2, 3)
    criterion = tc.make_criterion(weights, "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        criterion(model(x_train), torch.tensor(y_train)).backward()
        opt.step()
    exported = tc.LogitAdjusted(model, tc.logit_adjustment(weights)).eval()
    with torch.no_grad():
        raw_val, raw_test = model(x_val).double().numpy(), model(x_test).double().numpy()
        adj_val, adj_test = exported(x_val).double().numpy(), exported(x_test).double().numpy()
    cancer_rate = np.mean(y_test == 2)
    naive = tc.softmax(raw_test, tc.fit_temperature(raw_val, y_val))[:, 2].mean()
    fixed = tc.softmax(adj_test, tc.fit_temperature(adj_val, y_val))[:, 2].mean()
    assert naive > cancer_rate + 0.1  # weighted loss + temperature only: P(cancer) inflated
    assert abs(fixed - cancer_rate) < 0.03


def _split_samples(prefix, n, seed):
    tc = _train_module()
    rng = np.random.default_rng(seed)
    cats = rng.choice(3, size=n, p=[0.3, 0.6, 0.1])
    return [tc.Sample(path=Path(f"{prefix}{i}.png"), image=f"{prefix}{i}.png", dataset="d", label=CATEGORIES[k],
                      category=int(k), patient_id=f"{prefix}{i:03d}", group=f"{prefix}{i:03d}", bbox=None,
                      bbox_shape=None, split=None)
            for i, k in enumerate(cats)]


def test_init_training_patients_stay_in_train_when_data_are_added():
    tc = _train_module()
    first = _split_samples("P", 80, 0)
    parts1, _, _ = tc.split_samples(first, 0.15, 0.15, 0)
    trained = {s.group for s in parts1["train"]}
    more = first + _split_samples("Q", 40, 1)

    def held_out(parts):
        return {s.group for sp in ("val", "test") for s in parts[sp]}

    leaky, _, _ = tc.split_samples(more, 0.15, 0.15, 0)
    assert held_out(leaky) & trained  # the problem: a reshuffle moves fitted patients into val/test
    salt = "abc"
    hashes = {tc.group_hash(salt, g) for g in trained}
    pinned = {s.group for s in more if tc.group_hash(salt, s.group) in hashes}
    assert pinned == trained
    parts2, info, _ = tc.split_samples(more, 0.15, 0.15, 0, pinned)
    assert not held_out(parts2) & trained and info["patients_kept_in_train_from_init"] == len(trained)
    assert len(held_out(parts2)) >= 0.25 * 120  # new patients fill val and test
    explicit = [tc.Sample(**{**s.__dict__, "split": "test"}) if s.group == sorted(trained)[0] else s for s in more]
    with pytest.raises(SystemExit, match="--init checkpoint was trained on"):
        tc.split_samples(explicit, 0.15, 0.15, 0, pinned)


def test_labels_problems_are_found_before_training(tmp_path):
    tc = _train_module()
    (tmp_path / "images").mkdir()
    for i in range(3):
        cv2.imwrite(str(tmp_path / "images" / f"{i}.png"), np.full((96, 96, 3), 128, np.uint8))
    (tmp_path / "images" / "broken.png").write_bytes(b"not an image")
    (tmp_path / "labels.csv").write_text(
        "image,label,patient_id,x,y,w,h\n"
        "images/0.png,adenoma,P1,10,10,20,20\n"
        "images/1.png,adenoma,P2,900,500,120,120\n"
        "images/2.png,hyperplastic,P3,,,,\n"
        "images/broken.png,adenoma,P4,,,,\n"
        "images/0.png,lipoma,P5,,,,\n"
        "images/missing.png,adenoma,P6,,,,\n"
        "images/also_missing.png,neoplastic,P7,,,,\n"
    )
    with pytest.raises(SystemExit) as e:
        tc.load_samples([tmp_path])
    message = str(e.value)  # every kind of problem in one round, not one kind per run
    assert "2 problems" in message and "lies outside image" in message and "cannot read image" in message
    assert "Unknown labels: 'lipoma' (1 row), 'neoplastic' (1 row)" in message
    assert "2 referenced files not found" in message and "missing.png" in message and "also_missing.png" in message


def _write_rows(folder, rows, header="image,label,patient_id"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "labels.csv").write_text(header + "\n" + "".join(",".join(r) + "\n" for r in rows))
    return folder


def test_groups_follow_image_content_and_normalised_patient_ids(tmp_path):
    tc = _train_module()
    src = tmp_path / "orig"
    (src / "images").mkdir(parents=True)
    for i in range(4):
        cv2.imwrite(str(src / "images" / f"{i}.png"), np.full((32, 32, 3), 40 * i, np.uint8))
    _write_rows(src, [("images/0.png", "adenoma", ""), ("images/1.png", "adenoma", ""),
                      ("images/2.png", "hyperplastic", " P01 "), ("images/3.png", "hyperplastic", "p01")])
    before, _ = tc.load_samples([src])
    assert before[2].group == before[3].group == "p01"  # one patient, whatever the case
    assert before[0].group.startswith("image:") and before[0].group != before[1].group
    moved = tmp_path / "moved"
    src.rename(moved)
    after, _ = tc.load_samples([moved])  # a moved or copied data set keeps its groups (and hashes)
    assert [s.group for s in after] == [s.group for s in before]
    assert all(s.image_sha256 for s in after)


def test_development_overlap_compares_images_and_case_insensitive_patients(tmp_path):
    tc = _train_module()
    (tmp_path / "images").mkdir()
    for i in range(3):
        cv2.imwrite(str(tmp_path / "images" / f"{i}.png"), np.full((32, 32, 3), 60 * i, np.uint8))
    dev = _write_rows(tmp_path / "dev", [(f"../images/{i}.png", "adenoma", f"SYN{i}") for i in range(2)])
    dev_samples, _ = tc.load_samples([dev])
    salt = "s"
    manifest = {"salt": salt, "sha256_16": sorted({tc.group_hash(salt, k) for s in dev_samples for k in tc.identity_keys(s)})}
    meta = {"training_data": ["/elsewhere/a"], "training_datasets": [], "init_training_datasets": []}

    def check(folder, rows, header="image,label,patient_id", manifest=manifest, meta=meta):
        samples, _ = tc.load_samples([_write_rows(folder, rows, header)])
        return tc.development_overlap(meta, [folder], samples, manifest)

    lower = check(tmp_path / "lower", [("../images/0.png", "adenoma", "syn0")])
    assert lower["shared_patients"] == 1 and lower["shared_images"] == 1 and lower["overlap"]
    no_ids = check(tmp_path / "noid", [("../images/1.png", "adenoma")], header="image,label")
    assert no_ids["shared_images"] == 1 and no_ids["overlap"]
    assert no_ids["patients_checked"] is False and no_ids["shared_patients"] is None and no_ids["rows_without_patient_id"] == 1
    new = check(tmp_path / "new", [("../images/2.png", "adenoma", "Q9")])
    assert not new["overlap"] and new["patients_checked"] and new["images_checked"] and new["shared_images"] == 0
    unchecked = check(tmp_path / "unchecked", [("../images/0.png", "adenoma", "SYN0")], manifest=None)
    assert not unchecked["patients_checked"] and not unchecked["images_checked"] and not unchecked["overlap"]
    # A fine-tuned model also knows the data its --init checkpoint was trained on.
    fine_tuned = {**meta, "init_training_datasets": [{"path": "/gone/dev", "labels_sha256": tc.file_sha256(dev / "labels.csv")}]}
    import shutil as _sh
    _sh.copytree(dev, tmp_path / "dev_copy")
    copied = tc.development_overlap(fine_tuned, [tmp_path / "dev_copy"], [], None)
    assert copied["same_labels_csv"] == [str(tmp_path / "dev_copy")] and copied["overlap"]
    assert tc.init_lineage({"training_data": ["/old/path"]}) == [{"path": "/old/path"}]


@pytest.mark.parametrize("argv, message", [
    (["--crop-margin", "-0.4"], "--crop-margin"),
    (["--crop-margin", "nan"], "--crop-margin"),
    (["--crop-margin", "11"], "--crop-margin"),
    (["--size", "4"], "--size"),
])
def test_trainer_refuses_settings_the_runtime_would_refuse(tmp_path, capsys, argv, message):
    tc = _train_module()
    with pytest.raises(SystemExit):
        tc.main(["--data", str(tmp_path), "--out", str(tmp_path / "out"), *argv])
    assert message in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_model_is_not_marked_calibrated_when_validation_lacks_a_class(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from secondlook import synthetic

    tc = _train_module()
    data = tmp_path / "lesions"
    synthetic.generate_lesion_dataset(data, n_per_class=6, views_per_lesion=1, size=(64, 64), seed=0)
    with (data / "labels.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    seen: dict[str, int] = {}
    for r in rows:  # cancers only in train and test; two patients of each other class in val
        k = seen[r["label"]] = seen.get(r["label"], 0) + 1
        r["split"] = ("train" if k % 2 else "test") if r["label"] == "cancerous" else \
            "train" if k <= 3 else "val" if k <= 5 else "test"
    with (data / "labels.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "out"
    tc.main(["--data", str(data), "--out", str(out), "--backbone", "tiny", "--size", "32", "--epochs", "1",
             "--batch", "8", "--workers", "0", "--bootstrap", "0"])
    meta = json.loads((out / "model.json").read_text())
    assert meta["calibrated"] is False
    assert any("no cancerous examples" in w and "not calibrated" in w for w in meta["validation_warnings"])
    assert "development_patients" not in meta  # hashes of patient IDs stay out of the distributed sidecar
    manifest = json.loads((out / "model.development.json").read_text())
    assert len(manifest["sha256_16"]) == 18 + 18 and manifest["salt"]  # 18 patients and their 18 images


def test_synthetic_sets_with_different_seeds_share_no_patients(tmp_path):
    from secondlook import synthetic

    tc = _train_module()
    ids = {}
    for seed in (0, 123):
        folder = tmp_path / f"s{seed}"
        synthetic.generate_lesion_dataset(folder, n_per_class=2, views_per_lesion=1, size=(64, 64), seed=seed)
        with (folder / "labels.csv").open() as fh:
            ids[seed] = {row["patient_id"] for row in csv.DictReader(fh)}
        assert all(i.startswith(f"SYN{seed}-") for i in ids[seed])
    assert not ids[0] & ids[123]
    # So a set drawn with another seed is not reported as the development data of a model trained on the first.
    first, _ = tc.load_samples([tmp_path / "s0"])
    salt = "x"
    manifest = {"salt": salt, "sha256_16": [tc.group_hash(salt, k) for s in first for k in tc.identity_keys(s)]}
    second, _ = tc.load_samples([tmp_path / "s123"])
    assert not tc.development_overlap({}, [tmp_path / "s123"], second, manifest)["overlap"]
    custom = synthetic.generate_lesion_dataset(tmp_path / "p", n_per_class=1, views_per_lesion=1, size=(64, 64),
                                               patient_prefix="SITEB-")
    assert custom["patient_prefix"] == "SITEB-"
    with pytest.raises(ValueError, match="patient_prefix"):
        synthetic.generate_lesion_dataset(tmp_path / "bad", n_per_class=1, patient_prefix="../x")


def test_diverging_training_stops_with_a_clear_message(tmp_path):
    pytest.importorskip("torch")
    from secondlook import synthetic

    tc = _train_module()
    data = tmp_path / "lesions"
    synthetic.generate_lesion_dataset(data, n_per_class=4, views_per_lesion=1, size=(64, 64), seed=0)
    with pytest.raises(SystemExit, match="lower --lr"):
        tc.main(["--data", str(data), "--out", str(tmp_path / "out"), "--backbone", "tiny", "--size", "32",
                 "--epochs", "2", "--batch", "4", "--workers", "0", "--lr", "1e30"])


def test_server_and_cli_load_the_classifier_up_front(colour_model, tmp_path):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from secondlook.cli import main
    from secondlook.server import make_handler

    handler = make_handler(tmp_path / "runs", colour_model("base"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/").read().decode()
    finally:
        server.shutdown()
    assert "Lesion classifier: colour-base test-1" in page and "confirm with histopathology" in page
    with pytest.raises(FileNotFoundError):
        make_handler(tmp_path / "runs", tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError):  # before any recording is rendered
        main(["demo", "--out", str(tmp_path / "demo"), "--classifier", str(tmp_path / "missing.onnx")])
    assert not (tmp_path / "demo").exists()


def test_package_imports_without_onnxruntime(monkeypatch):
    # A fresh interpreter, so no module imported by other tests hides an eager import.
    code = ("import sys; sys.modules['onnxruntime'] = None\n"
            "import secondlook.cli, secondlook.server, secondlook.report, secondlook.diagnosis.metrics")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # makes `import onnxruntime` fail
    from secondlook.diagnosis import load_classifier

    with pytest.raises(ImportError, match="model"):
        load_classifier(ROOT / "no_such_model.onnx")


# ----------------------------------------------------------------------------- end to end


def _overlaps(finding, lesion, slack=1.0):
    return finding["first_s"] <= lesion["visible_to_s"] + slack and finding["last_s"] >= lesion["visible_from_s"] - slack


@pytest.mark.slow
def test_train_export_and_analyse_synthetic_lesions(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from secondlook import synthetic
    from secondlook.cli import run_analysis
    from secondlook.diagnosis.classifier import SAFETY_STATEMENT
    from secondlook.report import AI_DIAGNOSIS_NOTICE

    data = tmp_path / "lesions"
    summary = synthetic.generate_lesion_dataset(data, n_per_class=30, views_per_lesion=3, seed=0)
    assert summary["images"] == 270 and (data / "labels.csv").exists()

    out = tmp_path / "model"
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    cmd = [sys.executable, str(ROOT / "training" / "train_classifier.py"), "--data", str(data), "--out", str(out),
           "--backbone", "tiny", "--size", "64", "--epochs", "12", "--batch", "16", "--lr", "3e-3",
           "--workers", "0", "--bootstrap", "0", "--seed", "0", "--name", "e2e-tiny", "--version", "test"]
    run = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    for name in ("model.onnx", "model.json", "best.pt", "model.development.json", "metrics.json",
                 "predictions_test.csv", "split.json"):
        assert (out / name).exists(), name

    meta = json.loads((out / "model.json").read_text())
    assert SIDECAR_KEYS <= set(meta)
    assert meta["task"] == "lesion-classification" and meta["classes"] == CATEGORIES and meta["output"] == "logits"
    assert meta["input_size"] == 64 and meta["temperature"] > 0 and meta["name"] == "e2e-tiny"
    assert "not a medical device" in meta["intended_use"].lower()
    assert {"val", "test"} <= set(meta["metrics"])
    assert meta["modality"] == "colonoscopy" and meta["working_size"] == 512 and isinstance(meta["calibrated"], bool)
    assert meta["synthetic_training_data"] is True and meta["training_datasets"][0]["synthetic"] is True
    assert meta["training_data"] == [str(data.resolve())] and "development_patients" not in meta
    from secondlook.ingest import file_sha256

    manifest = json.loads((out / "model.development.json").read_text())
    assert len(manifest["sha256_16"]) == 90 + 270 and manifest["model_sha256"] == file_sha256(out / "model.onnx")
    assert meta["class_weights"] == pytest.approx([1.0, 1.0, 1.0])  # balanced classes: no logit adjustment

    report = json.loads((out / "metrics.json").read_text())
    test_accuracy = report["test"]["accuracy"]
    print(f"\nSynthetic test-set accuracy of the tiny classifier: {test_accuracy:.3f}")
    assert test_accuracy > 0.7  # easy synthetic classes; a pipeline check, not a performance claim

    split = json.loads((out / "split.json").read_text())
    ids = {sp: set(split["patient_ids"][sp]) for sp in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    with (data / "labels.csv").open() as fh:
        assert set().union(*ids.values()) == {row["patient_id"] for row in csv.DictReader(fh)}

    # A copy of the training data, given by a relative path from another directory, is still recognised.
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(data, elsewhere / "copy")
    run = subprocess.run([sys.executable, str(ROOT / "training" / "train_classifier.py"), "--evaluate-only",
                          str(out / "model.onnx"), "--data", "copy", "--out", "evaluation", "--bootstrap", "0"],
                         capture_output=True, text=True, env=env, timeout=300, cwd=elsewhere)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "NOT an external validation" in run.stdout
    ev = json.loads((elsewhere / "evaluation" / "metrics.json").read_text())
    overlap = ev["overlap_with_development_data"]
    assert overlap["overlap"] and overlap["same_labels_csv"] == ["copy"] and overlap["shared_patients"] == 90
    assert overlap["shared_images"] == 270 and overlap["patients_checked"] and overlap["images_checked"]
    assert "external" not in ev and ev["evaluation"]["n"] == 270

    # One lesion of each category in a recording; the default (heuristic) detector finds them.
    video = tmp_path / "three_lesions.avi"
    truth = synthetic.generate(video, duration_s=60, blur_interval_s=(30, 37), seed=1,
                               lesion_kinds=["benign", "precancerous", "cancerous"])
    assert [p["category"] for p in truth["polyps"]] == CATEGORIES
    result = run_analysis(video, tmp_path / "review", "colonoscopy", tmp_path / "three_lesions_report.json",
                          audit_log=tmp_path / "audit.jsonl", classifier_path=out / "model.onnx")
    assert result["characteriser"]["name"] == "e2e-tiny" and result["characteriser"]["version"] == "test"
    assert result["characteriser"]["sha256"] == file_sha256(out / "model.onnx")
    assert result["findings"]
    for f in result["findings"]:
        c = f["characterisation"]
        assert c["model"] == "e2e-tiny" and c["model_version"] == "test"
        assert c["frames_used"] > 0 and sum(c["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
        assert SAFETY_STATEMENT in c["note"]
        assert f["ai_category"] in (*CATEGORIES, "indeterminate")
        if f["ai_category"] != "indeterminate":
            assert (tmp_path / "review" / "findings" / f"{f['id']}_diagnosis.png").exists()

    matched = []
    for lesion in truth["polyps"]:
        hits = [f for f in result["findings"] if _overlaps(f, lesion)]
        if hits:
            matched.append((lesion["category"], max(hits, key=lambda f: f["frames"])["ai_category"]))
    correct = sum(t == p for t, p in matched)
    print(f"Video lesions found: {len(matched)}/3; AI category = drawn category for {correct}: {matched}")
    assert len(matched) >= 2
    # Over 2 training seeds x 8 recordings this model labelled every detected lesion as drawn,
    # so a wrong category here means the chain is broken. Abstaining is always allowed.
    assert all(p in (t, "indeterminate") for t, p in matched) and correct >= 2

    page = (tmp_path / "review" / "report.html").read_text()
    assert "AI optical diagnosis" in page and html.escape(AI_DIAGNOSIS_NOTICE) in page
    assert "Priority list" in page and "e2e-tiny" in page
    assert "AI lesion classifier" in page and "synthetic images only" in page
    entry = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])
    assert entry["characteriser"] == "e2e-tiny" and entry["characteriser_version"] == "test"
    assert entry["characteriser_sha256"] == file_sha256(out / "model.onnx")
