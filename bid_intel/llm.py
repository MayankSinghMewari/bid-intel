"""LLM answer generation over retrieved passages (the "G" in RAG), using Claude.

Enabled when Anthropic credentials are available (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
`ant auth login` profile). Without credentials, or with BID_INTEL_LLM=off, callers fall back to the
extractive answer, so the CLI, API, tests and CI all work offline.

Grounding check: the answer must cite at least one question number, and every cited number must
belong to a passage that was actually provided. Otherwise the answer is rejected and the caller
falls back to the extractive passage.
"""
from __future__ import annotations

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
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


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


def _complete(system: str, user: str, max_tokens: int = 2048) -> dict:
    """One completion from the configured provider: {"text", "model", "input_tokens", "output_tokens"}.
    Raises on API errors; callers fall back to extractive answers."""
    if provider() == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        r = client.models.generate_content(
            model=GEMINI_MODEL, contents=user,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens,
                                               temperature=0))
        u = r.usage_metadata
        return {"text": (r.text or "").strip(), "model": GEMINI_MODEL,
                "input_tokens": getattr(u, "prompt_token_count", None),
                "output_tokens": getattr(u, "candidates_token_count", None)}
    import anthropic
    r = anthropic.Anthropic().beta.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system, output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": user}])
    if r.stop_reason == "refusal":
        raise RuntimeError("refusal")
    return {"text": "".join(b.text for b in r.content if b.type == "text").strip(), "model": r.model,
            "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}

SYSTEM_PROMPT = """You answer questions about a public-sector bid document (an RFP addendum of vendor \
questions and District answers).

Rules:
- Use only the numbered passages provided. Do not use outside knowledge.
- Prefer the District's answer over the vendor's question. If a fact appears only in the vendor's \
question, say that the vendor stated it.
- Cite every claim with the question number in square brackets, for example [Q9].
- If the passages do not answer the question, reply exactly: Not stated in the document.
- Answer in one to three sentences. No preamble."""


def enabled() -> bool:
    return provider() is not None


def _anthropic_credentials() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") \
            or os.environ.get("ANTHROPIC_PROFILE"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def _passages(hits) -> str:
    blocks = []
    for h in hits:
        label = ", ".join(f"Q{q}" for q in h.chunk.qnums)
        blocks.append(f"<passage cite=\"{label}\" page=\"{h.chunk.page}\">\n{h.chunk.text}\n</passage>")
    return "\n\n".join(blocks)


def generate_answer(question: str, hits) -> dict | None:
    """Return {"answer", "model", "latency_s", "input_tokens", "output_tokens", "cited_qnums"},
    or None when the LLM is disabled, fails, or produces an ungrounded answer."""
    if provider() != "anthropic" or not hits:
        return None
    import anthropic

    client = anthropic.Anthropic()
    start = time.perf_counter()
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},  # short grounded lookups do not need deep reasoning
            # on a safety decline, the API re-runs the request on a fallback model in the same call
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content":
                       f"{_passages(hits)}\n\nQuestion: {question}"}],
        )
    except anthropic.APIError as exc:  # auth, rate limit, network, server errors: fall back
        return {"error": f"{type(exc).__name__}: {exc}"}
    latency = time.perf_counter() - start
    if response.stop_reason == "refusal":
        return {"error": "refusal"}

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    cited = sorted({int(n) for n in re.findall(r"\[Q(\d+)", text)})
    provided = {q for h in hits for q in h.chunk.qnums}
    not_stated = text.startswith("Not stated in the document")
    if not not_stated and (not cited or not set(cited) <= provided):
        return {"error": f"ungrounded answer rejected (cited {cited}, provided {sorted(provided)})"}
    return {
        "answer": text,
        "model": response.model,
        "latency_s": round(latency, 2),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cited_qnums": cited,
    }


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
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    if not text.startswith("Not found") and (not cited or max(cited) > len(hits)):
        return {"error": "ungrounded answer rejected (no valid [n] citations)"}
    return {"answer": text, "model": out["model"], "latency_s": round(time.perf_counter() - start, 2),
            "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"]}
