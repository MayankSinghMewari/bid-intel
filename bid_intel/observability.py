"""Observability: optional Langfuse tracing on top of the local JSON trace.

Enabled when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (environment or the git-ignored .env;
LANGFUSE_BASE_URL / LANGFUSE_HOST select the region). Without keys every function is a no-op, so tests,
CI and offline runs are unaffected. BID_INTEL_TRACING=off disables it explicitly.

Trace design (audited against https://langfuse.com/docs/observability/best-practices):
- One trace per unit of work: "extract-bid" (one bid folder) and "answer-question" (one question),
  tagged by feature ("extraction" / "qa"); the bid id and options go to metadata, not the name.
- Every graph node is a typed observation, and the agent work inside it nests under it:
  extract-<group>-fields (agent) > search-evidence (retriever) + apply-field-rules (tool);
  validate-fields > validate-field (evaluator); llm-extract-<group>-fields (agent) > gather-evidence
  (retriever) + extract-fields (generation) + check-quote (evaluator); reconcile-addenda (chain);
  summarize-bid (chain) > generate-text (generation).
- Generations carry the model, role-labelled input messages, output, and token usage including
  thinking tokens. Orchestrator decisions (plan, retry, route, merge) are events.
- Environment (BID_INTEL_ENV, default "development") and release (git commit) are set on the client.
- Short-lived CLI runs call shutdown() at exit; nothing flushes per request.
Tracing failures never break the pipeline.
"""
from __future__ import annotations

import contextvars
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import llm  # noqa: F401  (loads .env into the environment before the Langfuse client is created)

log = logging.getLogger("bid_intel.observability")
_trace_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("bid_intel_trace_ctx", default=None)
_client = None


def enabled() -> bool:
    if os.environ.get("BID_INTEL_TRACING", "").lower() in ("0", "off", "false", "no"):
        return False
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def _release() -> str:
    if os.environ.get("BID_INTEL_RELEASE"):
        return os.environ["BID_INTEL_RELEASE"]
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def client():
    global _client
    if not enabled():
        return None
    if _client is None:
        try:
            from langfuse import Langfuse
            host = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
            _client = Langfuse(environment=os.environ.get("BID_INTEL_ENV", "development"), release=_release(),
                               **({"base_url": host} if host else {}))
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
                "host": os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "default",
                "environment": os.environ.get("BID_INTEL_ENV", "development"), "release": _release()}
    except Exception as exc:
        return {"enabled": True, "auth_ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _enter(cm) -> Any:
    try:
        return cm.__enter__()
    except Exception as exc:
        log.debug("Langfuse observation start failed: %s", exc)
        return None


def _exit(cm) -> None:
    try:
        cm.__exit__(None, None, None)
    except Exception:
        pass


@contextmanager
def trace(name: str, input: Any = None, metadata: dict | None = None, tags: list[str] | None = None,
          as_type: str = "agent") -> Iterator[Any]:
    """Root observation of one unit of work. Yields the observation (or None when tracing is off)."""
    c = client()
    if c is None:
        yield None
        return
    try:
        from langfuse import propagate_attributes
        attrs = propagate_attributes(trace_name=name, tags=tags or [],
                                     metadata={k: str(v) for k, v in (metadata or {}).items()})
        attrs.__enter__()
    except Exception:
        attrs = None
    cm = c.start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata)
    obs = _enter(cm)
    token = _trace_ctx.set({"trace_id": obs.trace_id, "parent_span_id": obs.id} if obs else None)
    try:
        yield obs
    finally:
        _trace_ctx.reset(token)
        if obs is not None:
            _exit(cm)
        if attrs is not None:
            try:
                attrs.__exit__(None, None, None)
            except Exception:
                pass


@contextmanager
def span(name: str, as_type: str = "span", input: Any = None, metadata: dict | None = None) -> Iterator[Any]:
    """Typed child observation of whatever observation is current (graph node, agent, ...).
    No-op outside a trace, so agents can be instrumented unconditionally."""
    c = client()
    if c is None or _trace_ctx.get() is None:
        yield None
        return
    cm = c.start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata)
    obs = _enter(cm)
    try:
        yield obs
    except Exception as exc:
        if obs is not None:
            try:
                obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {str(exc)[:300]}")
            except Exception:
                pass
        raise
    finally:
        if obs is not None:
            _exit(cm)


def set_output(obs: Any, output: Any = None, level: str | None = None, status_message: str | None = None,
               metadata: dict | None = None) -> None:
    if obs is None:
        return
    kw = {k: v for k, v in (("output", output), ("level", level), ("status_message", status_message),
                            ("metadata", metadata)) if v is not None}
    try:
        obs.update(**kw)
    except Exception:
        pass


def trace_context() -> dict | None:
    return _trace_ctx.get()


def event(name: str, input: Any = None, output: Any = None, metadata: dict | None = None,
          level: str = "DEFAULT") -> None:
    """Point-in-time event under the current observation (orchestrator decisions)."""
    c = client()
    if c is None or _trace_ctx.get() is None:
        return
    try:
        c.create_event(name=name, input=input, output=output, metadata=metadata, level=level)
    except Exception:
        pass


@contextmanager
def generation(name: str, model: str, input: Any, metadata: dict | None = None) -> Iterator[Any]:
    """One LLM call, nested under the current observation; report results with record_generation()."""
    c = client()
    if c is None or _trace_ctx.get() is None:
        yield None
        return
    cm = c.start_as_current_observation(name=name, as_type="generation", model=model, input=input,
                                        metadata=metadata)
    obs = _enter(cm)
    try:
        yield obs
    except Exception as exc:
        if obs is not None:
            try:
                obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {str(exc)[:300]}")
            except Exception:
                pass
        raise
    finally:
        if obs is not None:
            _exit(cm)


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


def shutdown() -> None:
    """Send everything queued; call once when a short-lived process (CLI) exits."""
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:
            pass
