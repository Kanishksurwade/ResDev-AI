"""
ResDev AI - Resume File Parser
-------------------------------
Parses uploaded resume files (PDF, DOCX, TXT, JSON) into a normalized
master-resume dict that the existing AI pipeline can consume.

Supported formats:
    - .json  : loaded directly if it conforms to the master-resume schema,
               otherwise wrapped into a minimal schema from flat text.
    - .txt   : raw text; converted into a minimal master-resume structure.
    - .pdf   : text extracted via pypdf (pure-Python, no C dependency).
    - .docx  : text extracted via python-docx.
    - .doc   : not supported; user receives a clear message.

No LLM calls. Fully offline. The text extraction is deterministic.
The resulting dict is evidence-grounded: only what is in the file is returned.
"""

import io
import json
import re
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract plain text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to read PDF files. "
            "Add 'pypdf>=3.0.0' to requirements.txt."
        ) from exc

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return _clean_text("\n\n".join(pages))


def _extract_docx_text(file_bytes: bytes) -> str:
    """Extract plain text from a DOCX using python-docx."""
    try:
        import docx as python_docx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "python-docx is required to read DOCX files. "
            "Add 'python-docx>=0.8.11' to requirements.txt."
        ) from exc

    doc = python_docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return _clean_text("\n".join(paragraphs))


# ─────────────────────────────────────────────────────────────
#  TEXT → MASTER-RESUME WRAPPER
# ─────────────────────────────────────────────────────────────

def _text_to_master_resume(raw_text: str, source_filename: str = "") -> dict[str, Any]:
    """
    Wrap extracted plain text into the master-resume JSON schema that the
    AI pipeline expects.  Every piece of information is taken verbatim from
    the uploaded file — nothing is invented.

    The AI modules (jd_analyzer, resume_generator, etc.) accept the raw
    profile text and use it as the grounding source.  This wrapper makes the
    text compatible with the existing schema.
    """
    # Try to extract a name from the first non-empty line
    first_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    candidate_name = first_lines[0] if first_lines else "Uploaded Candidate"

    # Try to extract email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}", raw_text)
    candidate_email = email_match.group(0) if email_match else ""

    # Try to extract phone
    phone_match = re.search(
        r"(\+?\d[\d\s\-().]{7,}\d)", raw_text
    )
    candidate_phone = phone_match.group(0).strip() if phone_match else ""

    # Try to extract LinkedIn
    linkedin_match = re.search(
        r"linkedin\.com/in/[\w\-]+", raw_text, re.IGNORECASE
    )
    candidate_linkedin = linkedin_match.group(0) if linkedin_match else ""

    # Try to extract GitHub
    github_match = re.search(
        r"github\.com/[\w\-]+", raw_text, re.IGNORECASE
    )
    candidate_github = github_match.group(0) if github_match else ""

    # Build a master-resume dict from the raw text.
    # The full raw text is stored in `profile` so the AI can read everything.
    return {
        "candidate": {
            "personal_info": {
                "name": candidate_name,
                "email": candidate_email,
                "phone": candidate_phone,
                "location": "",
                "linkedin": candidate_linkedin,
                "github": candidate_github,
            },
            "professional_identity": {
                "current_profile": candidate_name,
                "profile": raw_text,
            },
        },
        "capabilities": {
            "skills": {
                "ai_llm": [],
                "tools_platforms": [],
                "core": [],
            },
            "certifications": [],
        },
        "experience": [],
        "projects": [],
        "education": [],
        "_source": f"parsed_from_upload:{source_filename}",
        "_raw_text": raw_text,
    }


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".json"}
UNSUPPORTED_EXTENSIONS = {".doc"}


class ResumeParseError(ValueError):
    """Raised when a file cannot be parsed into a usable resume structure."""


def parse_uploaded_resume(
    file_bytes: bytes,
    filename: str,
) -> dict[str, Any]:
    """
    Parse an uploaded resume file into the master-resume dict used by the AI
    pipeline.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded file.
    filename : str
        Original filename (used to determine format and for error messages).

    Returns
    -------
    dict
        Master-resume dict compatible with ai/resume_optimizer.py.

    Raises
    ------
    ResumeParseError
        If the format is unsupported or the file cannot be parsed.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in UNSUPPORTED_EXTENSIONS:
        raise ResumeParseError(
            f"'.doc' (legacy Word) files are not supported because they require "
            f"a proprietary COM/LibreOffice parser unavailable on Streamlit Cloud.  \n"
            f"Please save your resume as **DOCX, PDF, or TXT** and upload again."
        )

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ResumeParseError(
            f"Unsupported file type '{suffix}'.  \n"
            f"Supported formats: PDF, DOCX, TXT, JSON."
        )

    # ── JSON ──────────────────────────────────────────────────
    if suffix == ".json":
        try:
            data = json.loads(file_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ResumeParseError(f"Invalid JSON file: {exc}") from exc

        if not isinstance(data, dict):
            raise ResumeParseError(
                "JSON file must be a JSON object (dict), not a list or other type."
            )

        # If it already looks like a master-resume schema, use it directly
        if "candidate" in data and "capabilities" in data:
            data["_source"] = f"parsed_from_upload:{filename}"
            return data

        # Otherwise, treat the JSON's string values as raw text
        raw = json.dumps(data, indent=2)
        return _text_to_master_resume(raw, source_filename=filename)

    # ── TXT ───────────────────────────────────────────────────
    if suffix == ".txt":
        try:
            raw_text = file_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            raise ResumeParseError(f"Cannot read text file: {exc}") from exc
        if not raw_text.strip():
            raise ResumeParseError("The uploaded TXT file appears to be empty.")
        return _text_to_master_resume(raw_text, source_filename=filename)

    # ── PDF ───────────────────────────────────────────────────
    if suffix == ".pdf":
        try:
            raw_text = _extract_pdf_text(file_bytes)
        except ImportError as exc:
            raise ResumeParseError(str(exc)) from exc
        except Exception as exc:
            raise ResumeParseError(
                f"Could not extract text from PDF '{filename}': {exc}"
            ) from exc
        if not raw_text.strip():
            raise ResumeParseError(
                "No readable text found in the PDF.  \n"
                "If it is a scanned image PDF, please convert it to DOCX or TXT first."
            )
        return _text_to_master_resume(raw_text, source_filename=filename)

    # ── DOCX ──────────────────────────────────────────────────
    if suffix == ".docx":
        try:
            raw_text = _extract_docx_text(file_bytes)
        except ImportError as exc:
            raise ResumeParseError(str(exc)) from exc
        except Exception as exc:
            raise ResumeParseError(
                f"Could not extract text from DOCX '{filename}': {exc}"
            ) from exc
        if not raw_text.strip():
            raise ResumeParseError(
                "No readable text found in the DOCX file."
            )
        return _text_to_master_resume(raw_text, source_filename=filename)

    # Should never reach here
    raise ResumeParseError(f"Unhandled extension '{suffix}'.")
