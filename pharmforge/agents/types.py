"""Typed contracts for agentic DAG."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from pharmforge.chem import PropertyPrediction, SimilarityResult
from pharmforge.rag.store import RetrievedDoc


class PlannerOutput(BaseModel):
    original_query: str
    intent: Literal["similarity", "property", "codegen", "general", "multi"] = "general"
    subtasks: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    needs_chem: bool = True
    needs_codegen: bool = False
    target_smiles: Optional[str] = None

class RetrieverOutput(BaseModel):
    docs: List[RetrievedDoc] = Field(default_factory=list)
    query: str
    recall_note: str = ""

class ChemistOutput(BaseModel):
    properties: List[PropertyPrediction] = Field(default_factory=list)
    similarities: List[SimilarityResult] = Field(default_factory=list)
    summary: str = ""
    raw_smiles_validated: List[str] = Field(default_factory=list)

class CriticVerdict(BaseModel):
    passed: bool = True
    issues: List[str] = Field(default_factory=list)
    hallucination_risk: Literal["low", "medium", "high"] = "low"
    adme_flags: List[str] = Field(default_factory=list)
    validated_docs: List[str] = Field(default_factory=list)  # doc ids cited

class ReporterOutput(BaseModel):
    markdown: str
    provenance: List[dict] = Field(default_factory=list)
    generated_code_path: Optional[str] = None
    code_validated: bool = False

class DagTrace(BaseModel):
    query: str
    planner: PlannerOutput
    retriever: RetrieverOutput
    chemist: ChemistOutput
    critic: CriticVerdict
    reporter: ReporterOutput
    latency_ms: float = 0.0
    total_docs: int = 0

class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_codegen: bool = False
    target_smiles: Optional[str] = None

class QueryResponse(BaseModel):
    trace: DagTrace
    answer: str
