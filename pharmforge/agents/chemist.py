"""Chemist agent — RDKit property prediction + similarity."""
from __future__ import annotations

from pharmforge.agents.types import ChemistOutput, PlannerOutput, RetrieverOutput
from pharmforge.chem import batch_similarity_search, predict_properties, validate_smiles


def chemist(planner: PlannerOutput, retriever: RetrieverOutput) -> ChemistOutput:
    props = []
    sims = []
    validated = []
    summary_parts = []

    # If query contains a SMILES, validate and predict
    if planner.target_smiles:
        ok, err = validate_smiles(planner.target_smiles)
        if ok:
            validated.append(planner.target_smiles)
            p = predict_properties(planner.target_smiles)
            props.append(p)
            summary_parts.append(f"Target SMILES valid; LogP={p.logp} QED={p.qed} solubility_proxy={p.solubility_proxy}")
            # Similarity vs retrieved docs
            targets = [(d.id, d.name, d.smiles) for d in retriever.docs if d.smiles]
            if targets:
                sims = batch_similarity_search(planner.target_smiles, targets, top_k=5)
                if sims:
                    summary_parts.append(f"Top similar: {sims[0].target_name} ({sims[0].target_id}) Tanimoto={sims[0].tanimoto}")
        else:
            summary_parts.append(f"Target SMILES invalid: {err}")

    # Also predict for top retrieved docs (up to 3)
    for doc in retriever.docs[:3]:
        if doc.smiles:
            ok, _ = validate_smiles(doc.smiles)
            if ok and doc.smiles not in validated:
                p = predict_properties(doc.smiles)
                props.append(p)
                validated.append(doc.smiles)

    # If no target SMILES, similarity based on first doc's SMILES as pseudo-query
    if not sims and retriever.docs and retriever.docs[0].smiles:
        q_smi = retriever.docs[0].smiles
        targets = [(d.id, d.name, d.smiles) for d in retriever.docs[1:] if d.smiles]
        if targets:
            # Include first doc itself as query vs others
            sims = batch_similarity_search(q_smi, targets, top_k=5)

    if not summary_parts:
        summary_parts.append(f"Computed properties for {len(props)} molecules; {len(sims)} similarity pairs.")

    return ChemistOutput(
        properties=props,
        similarities=sims,
        summary=" | ".join(summary_parts),
        raw_smiles_validated=validated,
    )
