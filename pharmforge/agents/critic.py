"""Critic agent — validates SMILES, checks ADME, hallucination guard."""
from __future__ import annotations

from pharmforge.agents.types import ChemistOutput, CriticVerdict, RetrieverOutput
from pharmforge.chem import validate_smiles

# ADME thresholds (Lipinski + Veber inspired)
ADME_THRESHOLDS = {
    "logp_max": 5.0,
    "mw_max": 500,
    "tpsa_max": 140,
    "qed_min": 0.3,
    "solubility_min": -6.0,  # logS proxy, higher is better
}

def criticize(chemist: ChemistOutput, retriever: RetrieverOutput) -> CriticVerdict:
    issues: list[str] = []
    adme_flags: list[str] = []
    hallucination_risk = "low"

    # Validate each validated SMILES actually came from chemist
    for smi in chemist.raw_smiles_validated:
        ok, err = validate_smiles(smi)
        if not ok:
            issues.append(f"Invalid SMILES slipped through: {err}")

    # ADME checks
    for prop in chemist.properties:
        if not prop.valid:
            issues.append(f"Property prediction failed for {prop.smiles[:30]}: {prop.error}")
            continue
        if prop.logp is not None and prop.logp > ADME_THRESHOLDS["logp_max"]:
            adme_flags.append(f"High LogP {prop.logp} for {prop.smiles[:20]} (Lipinski >5)")
        if prop.mw is not None and prop.mw > ADME_THRESHOLDS["mw_max"]:
            adme_flags.append(f"High MW {prop.mw} (Lipinski >500)")
        if prop.tpsa is not None and prop.tpsa > ADME_THRESHOLDS["tpsa_max"]:
            adme_flags.append(f"High TPSA {prop.tpsa} (>140, poor absorption)")
        if prop.qed is not None and prop.qed < ADME_THRESHOLDS["qed_min"]:
            adme_flags.append(f"Low QED {prop.qed} (<0.3, poor drug-likeness)")
        if prop.solubility_proxy is not None and prop.solubility_proxy < ADME_THRESHOLDS["solubility_min"]:
            adme_flags.append(f"Very low solubility proxy {prop.solubility_proxy}")

    # Hallucination check: are docs actually cited? If chemist says something without provenance, medium risk.
    doc_ids = {d.id for d in retriever.docs}
    if not retriever.docs:
        issues.append("No retrieved docs — answer would be ungrounded")
        hallucination_risk = "high"
    elif len(retriever.docs) < 2:
        hallucination_risk = "medium"
    # If properties say invalid but retriever has many docs, risk low
    if adme_flags and len(adme_flags) > 3:
        hallucination_risk = "medium"

    # Cross-check: if chemist similarities exist but no docs, inconsistency
    if chemist.similarities and not doc_ids:
        issues.append("Similarity results without retrieved docs — inconsistent")

    passed = len([i for i in issues if "Invalid" in i or "ungrounded" in i.lower()]) == 0

    return CriticVerdict(
        passed=passed,
        issues=issues,
        hallucination_risk=hallucination_risk,
        adme_flags=adme_flags,
        validated_docs=list(doc_ids)[:10],
    )
