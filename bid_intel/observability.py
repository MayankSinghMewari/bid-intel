"""Observability: optional Langfuse tracing on top of the local JSON trace.

Enabled when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (environment or the git-ignored
.env; LANGFUSE_HOST selects the region). Without keys every function here is a no-op, so tests,
CI and offline runs are unaffected. BID_INTEL_TRACING=off disables it explicitly.

What is sent:
- one trace per extraction run ("extract <bid>") and per question ("ask"),
- one span per LangGraph node (LangChain callback handler),
- one generation per LLM call (model, prompt, output, token usage including thinking tokens),
- every local trace step (retrieval hits, validator verdicts, retries, routing) as an event.
Tracing failures never break the pipeline.
"""
from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from . import llm  # noqa: F401  (loads .env into the environment)

log = logging.getLogger("bid_intel.observability")
_trace_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("bid_intel_trace_ctx", default=None)
_client = None


def enabled() -> bool:
    if os.environ.get("BID_INTEL_TRACING", "").lower() in ("0", "off", "false", "no"):
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def client():
    global _client
    if not enabled():
        return None
    if _client is None:
        try:
            from langfuse import get_client
            _client = get_client()
        except Exception as exc:  # misconfiguration must not break runs
            log.warning("Langfuse disabled: %s", exc)
            return None
    return _client


def check() -> dict:
    """Connectivity check for the CLI: are keys set and accepted by the server?"""
    c = client()
    if c is None:
        return {"enabled": False, "reason": "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set"}
    try:
        return {"enabled": True, "auth_ok": bool(c.auth_check()),
                "host": os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "default"}
    except Exception as exc:
        return {"enabled": True, "auth_ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


@contextmanager
def trace(name: str, input: Any = None, metadata: dict | None = None, as_type: str = "agent") -> Iterator[Any]:
    """Root observation for one run. Yields the observation (or None when tracing is off)."""
    c = client()
    if c is None:
        yield None
        return
    try:
        cm = c.start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata)
        obs = cm.__enter__()
    except Exception as exc:
        log.warning("Langfuse trace start failed: %s", exc)
        yield None
        return
    token = _trace_ctx.set({"trace_id": obs.trace_id, "parent_span_id": obs.id})
    try:
        yield obs
    finally:
        _trace_ctx.reset(token)
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


@contextmanager
def span(name: str, as_type: str = "span", input: Any = None) -> Iterator[Any]:
    """Child observation of the current trace (e.g. retrieval inside a question)."""
    c = client()
    if c is None or _trace_ctx.get() is None:
        yield None
        return
    try:
        cm = c.start_as_current_observation(name=name, as_type=as_type, input=input)
        obs = cm.__enter__()
    except Exception:
        yield None
        return
    try:
        yield obs
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


def set_output(obs: Any, output: Any) -> None:
    if obs is not None:
        try:
            obs.update(output=output)
        except Exception:
            pass


def trace_context() -> dict | None:
    """Current run's trace context; captured by callers that log from worker threads."""
    return _trace_ctx.get()


def langgraph_callbacks() -> list:
    """LangChain callback handler that turns every LangGraph node into a span of the current trace."""
    if client() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception as exc:  # langchain-core missing or incompatible: node spans are skipped
        log.warning("LangGraph callback handler unavailable: %s", exc)
        return []


def event(name: str, input: Any = None, output: Any = None, metadata: dict | None = None,
          ctx: dict | None = None, level: str = "DEFAULT") -> None:
    c = client()
    ctx = ctx or _trace_ctx.get()
    if c is None or ctx is None:
        return
    try:
        c.create_event(trace_context=ctx, name=name, input=input, output=output, metadata=metadata, level=level)
    except Exception:
        pass


@contextmanager
def generation(name: str, model: str, input: Any, metadata: dict | None = None) -> Iterator[Any]:
    """One LLM call. The caller reports output and usage with record_generation()."""
    c = client()
    ctx = _trace_ctx.get()
    if c is None:
        yield None
        return
    try:
        cm = c.start_as_current_observation(name=name, as_type="generation", model=model, input=input,
                                            metadata=metadata, **({"trace_context": ctx} if ctx else {}))
        obs = cm.__enter__()
    except Exception:
        yield None
        return
    try:
        yield obs
    except Exception as exc:
        try:
            obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {str(exc)[:300]}")
        except Exception:
            pass
        raise
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


def record_generation(obs: Any, output: Any, input_tokens: int | None, output_tokens: int | None,
                      thinking_tokens: int | None = None) -> None:
    if obs is None:
        return
    usage = {k: v for k, v in (("input", input_tokens), ("output", output_tokens),
                               ("reasoning", thinking_tokens)) if v is not None}
    try:
        obs.update(output=output, usage_details=usage or None)
    except Exception:
        pass


def trace_url(obs: Any) -> str | None:
    c = client()
    if c is None or obs is None:
        return None
    try:
        return c.get_trace_url(trace_id=obs.trace_id)
    except Exception:
        return None


def flush() -> None:
    c = client()
    if c is not None:
        try:
            c.flush()
        except Exception:
            pass
