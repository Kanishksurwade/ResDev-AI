"""
ResDev AI - Gemini API Configuration & Shared Call Helper

Resolution order for GEMINI_API_KEY:
  1. GEMINI_API_KEY environment variable (local development).
  2. st.secrets["GEMINI_API_KEY"] (Streamlit Community Cloud).

Provides call_gemini_with_retry():
  - Timeout: configurable (default 120 s per request).
  - Retry: up to MAX_RETRIES attempts on transient 429 / 503 errors.
  - Backoff: 2 s, then 4 s between attempts.
"""

import os
import time
from typing import Optional

from google import genai
from google.genai import types as genai_types


# ─── constants ────────────────────────────────────────────────
DEFAULT_MODEL    = os.environ.get("RESDEV_MODEL", "gemini-3.5-flash-lite")
DEFAULT_TIMEOUT  = int(os.environ.get("RESDEV_TIMEOUT", "120"))   # seconds per request
MAX_RETRIES      = 2                                               # extra attempts after first failure
RETRY_BACKOFF    = (2, 4)                                          # seconds between retries

# Error substrings that indicate a transient server-side failure worth retrying
_RETRYABLE_SUBSTRINGS = (
    "429",
    "503",
    "quota",
    "rate limit",
    "resource exhausted",
    "internal error",
    "server error",
    "overloaded",
)


def get_gemini_api_key() -> str:
    """
    Retrieve the Gemini API key from the environment or Streamlit secrets.

    Resolution order:
    1. GEMINI_API_KEY environment variable (standard / local development).
    2. Streamlit secrets (st.secrets["GEMINI_API_KEY"]) (Streamlit Community Cloud).

    Returns:
        str: The Gemini API key string.

    Raises:
        RuntimeError: If the API key is not found in either source.
    """
    # 1. Check environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip()

    # 2. Safely check Streamlit secrets if running in Streamlit
    try:
        import streamlit as st  # Safe optional import

        if hasattr(st, "secrets") and st.secrets is not None:
            secret_key = st.secrets.get("GEMINI_API_KEY")
            if secret_key and str(secret_key).strip():
                return str(secret_key).strip()
    except Exception:
        # Streamlit not available or secrets file not present
        pass

    # 3. Raise a clear configuration error without exposing any secret values
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Please set the GEMINI_API_KEY environment variable or configure it in Streamlit secrets."
    )


def _is_retryable(error: Exception) -> bool:
    """Return True if the error looks like a transient Gemini server issue."""
    msg = str(error).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


def call_gemini_with_retry(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Send a prompt to Gemini with timeout and retry-on-transient-error.

    Parameters
    ----------
    prompt      : Text prompt to send.
    model       : Gemini model name.
    timeout     : Per-request timeout in seconds (passed to httpx via google-genai).
    max_retries : Number of additional attempts after the first failure (default 2).

    Returns
    -------
    str
        The model's text response.

    Raises
    ------
    RuntimeError
        After all retries are exhausted or on non-retryable errors.
    """
    api_key = get_gemini_api_key()

    # google-genai passes timeout down to httpx; configure via http_options
    http_options = genai_types.HttpOptions(timeout=timeout * 1000)  # milliseconds
    client = genai.Client(api_key=api_key, http_options=http_options)

    last_error: Exception | None = None
    attempts = max_retries + 1

    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = response.text
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            return text

        except Exception as exc:
            last_error = exc
            if attempt < attempts and _is_retryable(exc):
                wait = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                print(
                    f"[GEMINI] Transient error on attempt {attempt}/{attempts}: {exc!r}. "
                    f"Retrying in {wait}s…",
                    flush=True,
                )
                time.sleep(wait)
            else:
                break

    raise RuntimeError(
        f"Gemini request failed after {attempts} attempt(s): {last_error}"
    ) from last_error
