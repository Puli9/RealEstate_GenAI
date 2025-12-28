from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data: Path = root / "data"
    structured: Path = data / "structured"
    docs: Path = data / "docs"
    regulatory_pdfs: Path = docs / "regulatory_pdfs"
    unstructured_text: Path = docs / "unstructured_text"
    chroma: Path = data / "chroma"
    models: Path = root / "models"
    outputs: Path = root / "outputs"
    outputs_json: Path = outputs / "json"
    outputs_reports: Path = outputs / "reports"

PATHS = Paths()

def gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
