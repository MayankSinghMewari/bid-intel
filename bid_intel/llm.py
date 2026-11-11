"""LLM layer: the generation step of RAG and the LLM extraction specialists' calls.

Providers: Gemini (GEMINI_API_KEY, model GEMINI_MODEL, default gemini-3.5-flash) or Anthropic
(ANTHROPIC_API_KEY or an `ant auth login` profile, model BID_INTEL_LLM_MODEL). Keys come from the
environment or the git-ignored .env file. Without a key, with BID_INTEL_LLM=off, or while the circuit
breaker is open after a quota error, callers fall back to rules and extractive answers, so the CLI, API,
UI, tests and CI work offline.

Every call goes through _complete(): optional Pydantic structured output, retries with backoff on
503/504, one wait on per-minute 429s, truncation detection, and a Langfuse generation per call.
answer_from_citations() answers a question from numbered passages and rejects answers that do not cite
them.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from a project .env file (git-ignored) without overriding real env vars."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
MODEL = os.environ.get("BID_INTEL_LLM_MODEL", "claude-opus-5-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


_breaker = {"open_until": 0.0, "reason": ""}  # circuit breaker after quota errors


def trip_breaker(reason: str, seconds: float = 600) -> None:
    """Stop calling the LLM for a while (e.g. daily quota exhausted); callers use their fallbacks."""
    _breaker.update(open_until=time.time() + seconds, reason=reason)


def breaker_reason() -> str | None:
    return _breaker["reason"] if time.time() < _breaker["open_until"] else None


def provider() -> str | None:
    """'gemini' if GEMINI_API_KEY is set, else 'anthropic' if Anthropic credentials exist, else None.
    BID_INTEL_LLM=off disables both. BID_INTEL_LLM_PROVIDER forces one."""
    if os.environ.get("BID_INTEL_LLM", "").lower() in ("0", "off", "false", "no"):
        return None
    forced = os.environ.get("BID_INTEL_LLM_PROVIDER", "").lower()
    if forced in ("gemini", "anthropic"):
        return forced
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic" if _anthropic_credentials() else None


def _complete(system: str, user: str, max_tokens: int = 2048, schema: type | None = None,
              timeout_s: int = 30) -> dict:
    """LLM call wrapped in a Langfuse generation (model, prompt, output, tokens); see _call()."""
    from . import observability
    model = GEMINI_MODEL if provider() == "gemini" else MODEL
    name = "extract-fields" if schema else "generate-text"  # stable, verb-first; context comes from the parent
    with observability.generation(name, model, [{"role": "system", "content": system},
                                                {"role": "user", "content": user}],
                                  metadata={"max_tokens": max_tokens}) as gen:
        out = _call(system, user, max_tokens, schema, timeout_s)
        observability.record_generation(
            gen, out["parsed"].model_dump() if out.get("parsed") is not None else out["text"],
            out.get("input_tokens"), out.get("output_tokens"), out.get("thinking_tokens"))
        return out


def _call(system: str, user: str, max_tokens: int = 2048, schema: type | None = None,
          timeout_s: int = 30) -> dict:
    """One completion from the configured provider:
    {"text", "model", "input_tokens", "output_tokens", "thinking_tokens", "parsed"}.
    With schema (a Pydantic model class) the reply is constrained to that JSON schema and returned
    parsed in "parsed". Raises on API errors or invalid JSON; callers fall back without crashing."""
    if provider() == "gemini":
        from google import genai
        from google.genai import errors, types
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                              http_options=types.HttpOptions(timeout=timeout_s * 1000))
        # Gemini 3.x models think before answering and thinking tokens count against
        # max_output_tokens, so leave generous headroom over the visible answer length
        config = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max(max_tokens, 8192), temperature=0,
            **({"response_mime_type": "application/json", "response_schema": schema} if schema else {}))
        for attempt in range(3):
            try:
                r = client.models.generate_content(model=GEMINI_MODEL, contents=user, config=config)
                break
            except errors.ServerError:  # 503 overloaded / 504 deadline: back off 2 s, then 6 s
                if attempt == 2:
                    raise
                time.sleep(2 * 3 ** attempt)
            except errors.ClientError as exc:
                if getattr(exc, "code", None) != 429:
                    raise
                if "quota" in str(exc).lower() and "per minute" not in str(exc).lower() or attempt > 0:
                    trip_breaker("LLM quota exhausted (429)")  # daily quota: stop calling for this run
                    raise
                time.sleep(30)  # per-minute rate limit: wait once
        finish = r.candidates[0].finish_reason if r.candidates else None
        if finish is not None and getattr(finish, "name", str(finish)) == "MAX_TOKENS":
            raise RuntimeError("Gemini response truncated at max_output_tokens")
        u = r.usage_metadata
        text = (r.text or "").strip()
        parsed = None
        if schema:
            parsed = r.parsed if isinstance(r.parsed, schema) else schema.model_validate_json(text)
        return {"text": text, "model": GEMINI_MODEL, "parsed": parsed,
                "input_tokens": getattr(u, "prompt_token_count", None),
                "output_tokens": getattr(u, "candidates_token_count", None),
                "thinking_tokens": getattr(u, "thoughts_token_count", None)}
    import anthropic
    if schema:
        system += ("\n\nReturn only a JSON object, with no prose and no code fences, that validates against "
                   f"this JSON schema:\n{json.dumps(schema.model_json_schema())}")
    r = anthropic.Anthropic().beta.messages.create(
        model=MODEL, max_tokens=max(max_tokens, 8192), system=system, output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": user}])
    if r.stop_reason == "refusal":
        raise RuntimeError("refusal")
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    parsed = schema.model_validate_json(re.sub(r"^```(?:json)?|```$", "", text).strip()) if schema else None
    return {"text": text, "model": r.model, "parsed": parsed, "thinking_tokens": None,
            "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}

def enabled() -> bool:
    return provider() is not None and breaker_reason() is None


def _anthropic_credentials() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") \
            or os.environ.get("ANTHROPIC_PROFILE"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def answer_from_citations(question: str, hits: list[dict]) -> dict | None:
    """Bid-folder Q&A (the generation step of RAG): answer only from the retrieved passages, cite [n]."""
    if not enabled() or not hits:
        return None
    passages = "\n\n".join(f"<passage n=\"{i + 1}\" bid=\"{h['bid_id']}\" file=\"{h['file']}\" page=\"{h['page']}\">"
                           f"\n{h['text']}\n</passage>" for i, h in enumerate(hits))
    system = ("You answer questions about public-sector bid documents using only the numbered passages. "
              "Addendums override the original RFP; say so when relevant. Cite every claim as [n]. "
              "If the passages do not contain the answer, reply exactly: Not found in documents. "
              "Answer in at most 4 sentences.")
    start = time.perf_counter()
    try:
        out = _complete(system, f"{passages}\n\nQuestion: {question}")
    except Exception as exc:  # timeout, auth, quota, bad response: fall back without crashing
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    text = out["text"]
    # accepts [1], [1, 4] and [1][2]
    cited = {int(n) for grp in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text) for n in re.findall(r"\d+", grp)}
    if not text.startswith("Not found") and (not cited or max(cited) > len(hits)):
        return {"error": "ungrounded answer rejected (no valid [n] citations)"}
    return {"answer": text, "model": out["model"], "latency_s": round(time.perf_counter() - start, 2),
            "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"]}
