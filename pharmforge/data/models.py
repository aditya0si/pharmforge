"""Chemical data models."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Molecule(BaseModel):
    id: str = Field(description="Unique molecule ID, e.g. PF0001")
    name: str
    smiles: str
    formula: Optional[str] = None
    mw: Optional[float] = None
    description: str = Field(description="Assay notes / provenance text")
    source: str = Field(default="PharmForge Curated (ChEMBL-inspired)")
    tags: list[str] = Field(default_factory=list)
    assay_notes: Optional[str] = None


class MoleculeRecord(BaseModel):
    molecule: Molecule
    doc_text: str = Field(description="Full text used for RAG ingestion")
    provenance: dict = Field(default_factory=dict)
