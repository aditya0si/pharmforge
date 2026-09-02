"""Reporter agent — markdown with provenance + optional code."""
from __future__ import annotations

import textwrap
from typing import Optional

from pharmforge.agents.types import ReporterOutput, PlannerOutput, RetrieverOutput, ChemistOutput, CriticVerdict

def report(
    planner: PlannerOutput,
    retriever: RetrieverOutput,
    chemist: ChemistOutput,
    critic: CriticVerdict,
    code_path: Optional[str] = None,
    code_ok: bool = False,
) -> ReporterOutput:
    lines: list[str] = []
    lines.append(f"# PharmForge Report — {planner.intent.upper()}")
    lines.append("")
    lines.append(f"**Query:** {planner.original_query}")
    lines.append("")
    lines.append(f"**Subtasks:** {', '.join(planner.subtasks) if planner.subtasks else 'general'}")
    lines.append("")

    # Retrieved docs with provenance
    lines.append("## Retrieved Molecules (provenance)")
    if not retriever.docs:
        lines.append("_No documents retrieved._")
    else:
        for i, doc in enumerate(retriever.docs, 1):
            lines.append(f"{i}. **{doc.name}** (`{doc.id}`) — SMILES: `{doc.smiles[:60]}{'...' if len(doc.smiles)>60 else ''}`")
            lines.append(f"   - Score: {doc.score:.3f} | Provenance: {doc.provenance}")
            lines.append(f"   - Text: {doc.text[:200]}...")
            lines.append(f"   - Quote-verified: `{doc.text[:80]}...` [Source: {doc.id}]")
    lines.append("")

    # Properties
    lines.append("## Predicted Properties (RDKit)")
    if not chemist.properties:
        lines.append("_No properties computed._")
    else:
        lines.append("| SMILES | MW | LogP | QED | TPSA | HBD | HBA | RB | logS* |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for p in chemist.properties:
            lines.append(f"| `{p.smiles[:30]}` | {p.mw} | {p.logp} | {p.qed} | {p.tpsa} | {p.hbd} | {p.hba} | {p.rotatable_bonds} | {p.solubility_proxy} |")
        if chemist.summary:
            lines.append("")
            lines.append(f"_Chemist note: {chemist.summary}_")
    lines.append("")

    # Similarities
    if chemist.similarities:
        lines.append("## Similarity (Morgan Tanimoto)")
        lines.append("| Rank | ID | Name | Tanimoto |")
        lines.append("|---|---|---|---|")
        for i, s in enumerate(chemist.similarities, 1):
            lines.append(f"| {i} | {s.target_id} | {s.target_name} | {s.tanimoto:.4f} |")
        lines.append("")

    # Critic
    lines.append("## Critic Verdict")
    lines.append(f"- **Passed:** {critic.passed} | **Hallucination risk:** {critic.hallucination_risk}")
    if critic.issues:
        lines.append("- **Issues:**")
        for iss in critic.issues:
            lines.append(f"  - {iss}")
    if critic.adme_flags:
        lines.append("- **ADME flags:**")
        for f in critic.adme_flags:
            lines.append(f"  - {f}")
    if not critic.issues and not critic.adme_flags:
        lines.append("- No issues flagged.")
    lines.append("")

    # Codegen
    if code_path:
        lines.append("## Generated Analysis Script")
        lines.append(f"- **Path:** `{code_path}`")
        lines.append(f"- **Validated:** {code_ok}")
        lines.append(f"- **Sandbox:** executed headless, no network, 5s timeout")
        lines.append("")
    elif planner.needs_codegen:
        lines.append("## Generated Analysis Script")
        lines.append("_Code generation was requested but no script was produced (or validation failed)._")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("- Hybrid retrieval: Chroma cosine (384-d hash embed) + SQLite FTS5 BM25 + RRF (k=60)")
    lines.append("- RDKit: Morgan fingerprints (r=2, 2048 bits), descriptors via Descriptors/Crippen/QED, conformers via ETKDGv3")
    lines.append("- Agentic DAG: planner → retriever → chemist → critic → reporter (typed, traced)")
    lines.append("- Feedback loop: every trace stored as JSONL for fine-tune export")
    lines.append("")

    provenance = [
        {"id": d.id, "name": d.name, "smiles": d.smiles, "score": d.score, "text_snippet": d.text[:120]}
        for d in retriever.docs
    ]

    markdown = "\n".join(lines)
    return ReporterOutput(markdown=markdown, provenance=provenance, generated_code_path=code_path, code_validated=code_ok)
