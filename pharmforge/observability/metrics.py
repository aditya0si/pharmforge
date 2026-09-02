"""Prometheus metrics (in-process)."""
from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, Gauge

    QUERY_COUNTER = Counter("pharmforge_queries_total", "Total queries", ["passed"])
    QUERY_LATENCY = Histogram("pharmforge_query_latency_ms", "Query latency ms", buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000])
    RETRIEVAL_PRECISION = Gauge("pharmforge_retrieval_precision", "Last retrieval precision proxy")
    CHEM_VALIDATION = Counter("pharmforge_chem_validations_total", "Chem validations", ["valid"])
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False
    QUERY_COUNTER = None  # type: ignore
    QUERY_LATENCY = None  # type: ignore


def record_query(latency_ms: float, passed: bool):
    if not PROM_AVAILABLE:
        return
    try:
        QUERY_COUNTER.labels(passed=str(passed)).inc()  # type: ignore
        QUERY_LATENCY.observe(latency_ms)  # type: ignore
    except Exception:
        pass

def record_validation(valid: bool):
    if not PROM_AVAILABLE:
        return
    try:
        CHEM_VALIDATION.labels(valid=str(valid)).inc()  # type: ignore
    except Exception:
        pass
