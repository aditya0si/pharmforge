"""Feedback export — curates high-quality training examples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pharmforge.feedback.store import load_traces


def export_high_quality(min_qed: float = 0.4, require_passed: bool = True) -> List[dict]:
    """Export filtered high-quality examples for fine-tuning."""
    traces = load_traces()
    out = []
    for rec in traces:
        label = rec.get("_label", {})
        if require_passed and not label.get("passed", False):
            continue
        if label.get("hallucination") == "high":
            continue
        # Check chemist properties QED threshold if present
        chemist = rec.get("chemist", {})
        props = chemist.get("properties", [])
        # Allow if no props or at least one passes QED
        if props:
            qs = [p.get("qed", 0) for p in props if p.get("qed") is not None]
            if qs and max(qs) < min_qed:
                continue
        # This is a high-quality example
        out.append({
            "query": rec.get("query", ""),
            "planner": rec.get("planner", {}),
            "retriever_docs": [d["id"] for d in rec.get("retriever", {}).get("docs", [])],
            "chemist_summary": chemist.get("summary", ""),
            "critic": rec.get("critic", {}),
            "markdown": rec.get("reporter", {}).get("markdown", "")[:2000],
            "label": label,
        })
    return out

def export_to_file(out_path: Path | str = "data/training_export.jsonl", min_qed: float = 0.4) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples = export_high_quality(min_qed=min_qed)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return out_path
