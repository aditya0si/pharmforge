"""Observability — OTel tracing stubs (no collector required)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    _provider = TracerProvider(resource=Resource.create({"service.name": "pharmforge"}))
    # Console exporter for local dev; if OTEL_EXPORTER_OTLP_ENDPOINT set, user can override
    _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("pharmforge")
    OTEL_AVAILABLE = True
except Exception:
    OTEL_AVAILABLE = False
    _tracer = None  # type: ignore

@contextmanager
def trace_span(name: str, attributes: Optional[dict] = None) -> Generator[None, None, None]:
    if OTEL_AVAILABLE and _tracer:
        with _tracer.start_as_current_span(name, attributes=attributes or {}):  # type: ignore
            yield
    else:
        yield
