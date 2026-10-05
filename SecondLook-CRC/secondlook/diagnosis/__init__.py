"""AI optical diagnosis: classify detected lesions as benign, precancerous or cancerous."""

from .taxonomy import CATEGORIES, DISPLAY_NAME, DiagnosticCategory, category_for

__all__ = ["CATEGORIES", "DISPLAY_NAME", "DiagnosticCategory", "category_for"]
