"""Feedback store — JSONL per-trace persistence."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from pharmforge.agents.types import DagTrace

FEEDBACK_PATH = Path("data/feedback.jsonl")

def append_trace(trace: DagTrace) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = trace.model_dump()
    record["_ts"] = datetime.now(timezone.utc).isoformat()
    # Add training-label heuristics
    record["_label"] = {
        "passed": trace.critic.passed,
        "hallucination": trace.critic.hallucination_risk,
        "adme_flag_count": len(trace.critic.adme_flags),
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def load_traces(limit: int | None = None) -> List[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    out = []
    for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out

def count_traces() -> int:
    if not FEEDBACK_PATH.exists():
        return 0
    return sum(1 for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
