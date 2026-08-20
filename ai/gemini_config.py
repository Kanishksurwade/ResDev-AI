import os
from typing import Optional


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
