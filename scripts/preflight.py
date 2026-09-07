"""Preflight checks: fail fast with actionable errors, not mid-run crashes.

Run standalone:

    python -m scripts.preflight

Or it runs automatically before a non-mock eval:

    python -m eval.run_eval_v2            # preflight is part of startup

Checks (in order, stop at first failure):
  1. settings  - LLM_BASE_URL / LLM_MODEL resolve to something sane
  2. api key   - remote endpoints must not carry a placeholder key
  3. llm probe - one 1-token chat completion against the real endpoint
  4. vector db - the chroma collection is non-empty (ingest was run)

Exit code 0 = all green, 1 = something the user must fix first.
"""
from __future__ import annotations

import sys

# Localhost endpoints (Ollama, llama.cpp server, vLLM dev box) don't need
# a real key; anything else is assumed to be a hosted provider.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")

# Keys that are obviously placeholders used for local serving.
_PLACEHOLDER_KEYS = {"ollama", "missing", "changeme", "none", "test", "dummy"}


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]} (len={len(key)})"


def _is_local(base_url: str) -> bool:
    return any(h in base_url for h in _LOCAL_HOSTS)


def check_settings() -> tuple[bool, str]:
    from app.config import get_settings

    s = get_settings()
    if not s.llm_base_url:
        return False, (
            "LLM_BASE_URL is not set. Add it to .env, e.g.\n"
            "  LLM_BASE_URL=https://api.deepseek.com   (hosted)\n"
            "  LLM_BASE_URL=http://localhost:11434/v1  (local Ollama)"
        )
    key = s.llm_api_key or ""
    return True, (
        f"base_url={s.llm_base_url}  model={s.llm_model}  "
        f"key={_mask(key) if key else '(none)'}"
    )


def check_api_key() -> tuple[bool, str]:
    from app.config import get_settings

    s = get_settings()
    base, key = s.llm_base_url or "", s.llm_api_key or ""

    if _is_local(base):
        return True, f"local endpoint ({base}); placeholder key is fine"

    if not key:
        return False, (
            f"LLM_API_KEY is empty but the endpoint is remote ({base}).\n"
            "Export it or put it in .env:\n"
            "  export LLM_API_KEY=sk-...   # your real key"
        )
    if key.strip().lower() in _PLACEHOLDER_KEYS or len(key) < 20:
        return False, (
            f"LLM_API_KEY looks like a placeholder ({_mask(key)}) but the "
            f"endpoint is remote ({base}).\n"
            "The .env default 'ollama' leaked through: export the real key "
            "in the SAME terminal before running, or edit .env directly."
        )
    if key != key.strip():
        return False, "LLM_API_KEY has leading/trailing whitespace - strip it."
    return True, f"key looks plausible: {_mask(key)}"


def check_llm_probe() -> tuple[bool, str]:
    """One 1-token chat completion. Catches 401/403/404/model-name errors."""
    from rag.generator import _get_client
    from app.config import get_settings

    s = get_settings()
    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=s.llm_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        model = getattr(completion, "model", s.llm_model)
        return True, f"LLM responded (resolved model: {model})"
    except Exception as e:  # noqa: BLE001 - report, don't crash
        msg = str(e)[:400]
        hint = ""
        if "401" in msg or "authentication" in msg.lower():
            hint = "\n  -> the API key was rejected by the provider."
        if "404" in msg or "not found" in msg.lower():
            hint = "\n  -> check LLM_MODEL matches the provider's model id."
        if "connection" in msg.lower() or "refused" in msg.lower():
            hint = (
                "\n  -> endpoint unreachable. Local Ollama not running? "
                "Start it with: ollama serve"
            )
        return False, f"LLM probe failed: {msg}{hint}"


def check_vector_db() -> tuple[bool, str]:
    try:
        from rag.vector_store import _get_collection

        count = _get_collection().count()
    except Exception as e:  # noqa: BLE001
        return False, f"cannot open chroma collection: {e}"
    if count == 0:
        return False, (
            "chroma collection is EMPTY. Run ingest first:\n"
            "  python -m scripts.ingest --reset"
        )
    return True, f"chroma collection has {count} chunks"


CHECKS = [
    ("settings", check_settings),
    ("api key", check_api_key),
    ("llm probe", check_llm_probe),
    ("vector db", check_vector_db),
]


def run_preflight() -> int:
    """Run every check; print a report; return a process exit code."""
    failed = False
    print("=== PREFLIGHT ===")
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"check crashed: {e}"
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        failed = failed or not ok
    print("=================")
    if failed:
        print("Preflight FAILED - fix the items above, then re-run.")
        return 1
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_preflight())
