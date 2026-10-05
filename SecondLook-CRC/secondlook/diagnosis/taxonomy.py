"""Diagnostic categories and how histology labels map onto them.

Colonoscopy shows lesions, not individual cells. What an AI can do from
endoscopic images is *optical diagnosis*: predict which histology a lesion will
turn out to have, grouped here into three clinically actionable categories:

* BENIGN        - non-neoplastic: hyperplastic polyp, inflammatory / pseudo-polyp,
                  normal mucosa, lymphoid aggregate.
* PRECANCEROUS  - neoplastic but not invasive: conventional adenomas (tubular,
                  tubulovillous, villous; low- or high-grade dysplasia), sessile
                  serrated lesions, traditional serrated adenomas.
* CANCEROUS     - adenocarcinoma, including suspected submucosal invasion.

"Benign" means non-neoplastic histology, not "of no consequence": UK surveillance
guidance (BSG/ACPGBI/PHE 2020) counts hyperplastic polyps, apart from diminutive
rectal ones, among the premalignant serrated polyps.

Labels that do not fit these epithelial categories, or that are too vague to
place, are deliberately absent so category_for() raises and a person decides:
"neoplastic" alone could be an adenoma or a carcinoma, and a lipoma is a benign
mesenchymal neoplasm, neither non-neoplastic nor precancerous.

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
    DiagnosticCategory.PRECANCEROUS.value: "Precancerous (adenoma / SSL / TSA)",
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
    "non neoplastic": _B,
    "nonneoplastic": _B,
    # Neoplastic, non-invasive
    "adenoma": _P,
    "adenomatous": _P,
    "tubular adenoma": _P,
    "ta": _P,
    "tubulovillous adenoma": _P,
    "tubulo villous adenoma": _P,
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
    # Invasive
    "adenocarcinoma": _C,
    "carcinoma": _C,
    "invasive carcinoma": _C,
    "intramucosal carcinoma": _C,
    "submucosal invasive cancer": _C,
    "t1 cancer": _C,
}


# A known histology followed by a dysplasia grade, as pathology reports write it:
# "tubular adenoma, low-grade dysplasia", "adenoma with high grade dysplasia", "SSL with dysplasia".
# Negations ("no dysplasia", "negative for ...") leave an unknown base and still raise.
_DYSPLASIA_SUFFIX = re.compile(r"^(?P<base>.+?)[\s,;]*(?:with\s+)?(?:(?:low|high)\s+grade\s+)?dysplasia$")
_RISK_ORDER = list(DiagnosticCategory)


def normalise_label(label: str) -> str:
    return re.sub(r"[\s_\-/]+", " ", str(label).strip().lower()).strip()


def category_for(label: str, mapping: dict[str, DiagnosticCategory] | None = None) -> DiagnosticCategory:
    """Map a histology or category label to a DiagnosticCategory. Raises KeyError if unknown.

    A known label followed by a dysplasia grade maps to the label's category, but never below
    precancerous (dysplasia is neoplasia), so an adenocarcinoma stays cancerous."""
    table = mapping or HISTOLOGY_TO_CATEGORY
    key = normalise_label(label)
    if key in table:
        return table[key]
    m = _DYSPLASIA_SUFFIX.match(key)
    base = m.group("base").strip(" ,;") if m else None
    if base in table:
        return max(table[base], _P, key=_RISK_ORDER.index)
    raise KeyError(
        f"Unknown histology label {label!r}. Add it to HISTOLOGY_TO_CATEGORY or pass a custom mapping."
    )
