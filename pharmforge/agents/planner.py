"""Planner agent — rule-based decomposition (LLM fallback optional)."""
from __future__ import annotations

import re
from typing import Optional

from pharmforge.agents.types import PlannerOutput

# Keywords that signal intent
SIM_KEYWORDS = ["similar", "analog", "related", "like", "scaffold", "substructure"]
PROP_KEYWORDS = ["solubility", "logp", "qed", "adme", "property", "properties", "tpsa", "mw", "weight"]
CODE_KEYWORDS = ["script", "code", "generate", "plot", "python", "docking", "prep", "analyze"]

SMILES_RE = re.compile(r"\b([A-Za-z0-9@+\-\[\]\(\)\\\/=#%:.\*]{5,})\b")

def _extract_smiles(query: str) -> Optional[str]:
    # Heuristic: look for SMILES-like long token with chemistry chars
    # Prioritize explicit SMILES param
    tokens = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)\\\/=#%:\.]+", query)
    for tok in tokens:
        if len(tok) > 10 and any(c in tok for c in ["C", "O", "N", "c", "="]):
            # quick validate length and composition
            if tok.count("C") + tok.count("c") >= 2:
                # crude but catches many SMILES
                return tok
    return None

def plan(query: str, target_smiles: Optional[str] = None) -> PlannerOutput:
    q = query.lower()
    intent = "general"
    needs_chem = True
    needs_code = False
    subtasks: list[str] = []
    keywords: list[str] = []

    has_sim = any(k in q for k in SIM_KEYWORDS)
    has_prop = any(k in q for k in PROP_KEYWORDS)
    has_code = any(k in q for k in CODE_KEYWORDS)

    if has_sim and has_prop:
        intent = "multi"
    elif has_sim:
        intent = "similarity"
    elif has_prop:
        intent = "property"
    elif has_code:
        intent = "codegen"
    
    needs_code = has_code or "generate" in q or "analyze" in q

    # Extract SMILES if not provided
    smiles = target_smiles or _extract_smiles(query)

    if has_sim or intent == "multi":
        subtasks.append("Retrieve molecules similar to query/target SMILES")
        subtasks.append("Compute Tanimoto similarity via Morgan fingerprints")
    if has_prop or intent in ("general", "multi"):
        subtasks.append("Predict ADME properties (LogP, QED, TPSA, solubility)")
        keywords.extend([k for k in ["solubility", "logp", "qed", "adme"] if k in q])
    subtasks.append("Critique for hallucinations and ADME thresholds")
    subtasks.append("Generate provenance-traced report")

    if needs_code:
        subtasks.append("Generate runnable analysis script")

    if not keywords:
        keywords = re.findall(r"[a-z]{3,}", q)[:8]

    return PlannerOutput(
        original_query=query,
        intent=intent,
        subtasks=subtasks,
        keywords=keywords,
        needs_chem=needs_chem,
        needs_codegen=needs_code,
        target_smiles=smiles,
    )
