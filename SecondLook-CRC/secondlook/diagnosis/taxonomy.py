"""Diagnostic categories and how histology labels map onto them.

Colonoscopy shows lesions, not individual cells. What an AI can do from
endoscopic images is *optical diagnosis*: predict which histology a lesion will
turn out to have, grouped here into three clinically actionable categories:

* BENIGN        - non-neoplastic: hyperplastic polyp, inflammatory / pseudo-polyp,
                  normal mucosa, lymphoid aggregate, lipoma.
* PRECANCEROUS  - neoplastic but not invasive: conventional adenomas (tubular,
                  tubulovillous, villous; low- or high-grade dysplasia), sessile
                  serrated lesions, traditional serrated adenomas.
* CANCEROUS     - adenocarcinoma, including suspected submucosal invasion.

Cell-level diagnosis (dysplasia grade, invasion depth) is made by a
pathologist on the resected tissue. Training labels should come from that
histopathology report, mapped through `category_for()`.

Edge case: "intramucosal carcinoma" (pTis) is mapped to CANCEROUS. This is the
conservative choice for a triage tool, because the AI then errs towards the
higher-risk category. Pass a custom mapping if your pathology service reports
it as high-grade dysplasia.
"""

from __future__ import annotations

import re
from enum import Enum


class DiagnosticCategory(str, Enum):
    BENIGN = "benign"
    PRECANCEROUS = "precancerous"
    CANCEROUS = "cancerous"


# Order matters: it is the class index order of every model trained by this project.
CATEGORIES: list[str] = [c.value for c in DiagnosticCategory]

DISPLAY_NAME = {
    DiagnosticCategory.BENIGN.value: "Benign (non-neoplastic)",
    DiagnosticCategory.PRECANCEROUS.value: "Precancerous (adenoma / serrated lesion)",
    DiagnosticCategory.CANCEROUS.value: "Suspected cancer",
}

_B, _P, _C = DiagnosticCategory.BENIGN, DiagnosticCategory.PRECANCEROUS, DiagnosticCategory.CANCEROUS

HISTOLOGY_TO_CATEGORY: dict[str, DiagnosticCategory] = {
    # Category names themselves are accepted as labels.
    "benign": _B,
    "precancerous": _P,
    "cancerous": _C,
    "cancer": _C,
    # Non-neoplastic
    "normal": _B,
    "normal mucosa": _B,
    "hyperplastic": _B,
    "hyperplastic polyp": _B,
    "hp": _B,
    "inflammatory": _B,
    "inflammatory polyp": _B,
    "pseudopolyp": _B,
    "lymphoid aggregate": _B,
    "lipoma": _B,
    "non neoplastic": _B,
    "nonneoplastic": _B,
    # Neoplastic, non-invasive
    "adenoma": _P,
    "adenomatous": _P,
    "tubular adenoma": _P,
    "ta": _P,
    "tubulovillous adenoma": _P,
    "tva": _P,
    "villous adenoma": _P,
    "va": _P,
    "low grade dysplasia": _P,
    "lgd": _P,
    "high grade dysplasia": _P,
    "hgd": _P,
    "sessile serrated lesion": _P,
    "sessile serrated adenoma": _P,
    "sessile serrated polyp": _P,
    "ssl": _P,
    "ssa": _P,
    "ssa p": _P,
    "traditional serrated adenoma": _P,
    "tsa": _P,
    "neoplastic": _P,
    # Invasive
    "adenocarcinoma": _C,
    "carcinoma": _C,
    "invasive carcinoma": _C,
    "intramucosal carcinoma": _C,
    "submucosal invasive cancer": _C,
    "t1 cancer": _C,
}


def normalise_label(label: str) -> str:
    return re.sub(r"[\s_\-/]+", " ", str(label).strip().lower()).strip()


def category_for(label: str, mapping: dict[str, DiagnosticCategory] | None = None) -> DiagnosticCategory:
    """Map a histology or category label to a DiagnosticCategory. Raises KeyError if unknown."""
    table = mapping or HISTOLOGY_TO_CATEGORY
    key = normalise_label(label)
    if key not in table:
        raise KeyError(
            f"Unknown histology label {label!r}. Add it to HISTOLOGY_TO_CATEGORY or pass a custom mapping."
        )
    return table[key]
