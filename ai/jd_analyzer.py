import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TIMEOUT_SECONDS = 300

# Expected schema fields and their default types
EXPECTED_SCHEMA: dict[str, type] = {
    "job_title": str,
    "seniority": str,
    "required_skills": list,
    "preferred_skills": list,
    "responsibilities": list,
    "experience_requirements": list,
    "education_requirements": list,
    "certifications": list,
    "soft_skills": list,
    "keywords": list,
    "raw_job_description": str,
}

# Fallback prompt template in case prompt file is not found
FALLBACK_PROMPT_TEMPLATE = """You are an expert ATS (Applicant Tracking System) job description analyzer.
Analyze the following job description and extract key information into a clean, structured JSON object.

Job Description:
\"\"\"{job_description}\"\"\"

Extraction Guidelines:
- job_title: The official role or job title (e.g., "Data Analyst", "Senior Software Engineer").
- seniority: Seniority level (e.g., "Entry-level", "Junior", "Mid-level", "Senior", "Lead", "Principal", "Manager", or "" if not mentioned).
- required_skills: Mandatory hard technical skills, programming languages, platforms, frameworks, and tools explicitly required.
- preferred_skills: Nice-to-have, bonus, or preferred technical skills, tools, and experience.
- responsibilities: Core duties, key tasks, deliverables, and operational responsibilities.
- experience_requirements: Required years of experience or specific domain background requirements.
- education_requirements: Required or preferred academic degrees, majors, and educational qualifications.
- certifications: Required or preferred professional certifications and licenses.
- soft_skills: Interpersonal, communication, leadership, and organizational competencies.
- keywords: High-priority ATS matching keywords, technical terms, acronyms, and domain terminology for resume alignment.
- raw_job_description: The exact raw job description provided.

Rules:
1. Do NOT fabricate or hallucinate information that is not mentioned in the job description.
2. If any field or category is not mentioned in the job description, return an empty string "" or empty list [].
3. Return ONLY a single valid JSON object adhering strictly to the schema below.
4. Do not include markdown codeblocks, explanations, notes, or conversational text outside the JSON.

Expected JSON Structure:
{
  "job_title": "",
  "seniority": "",
  "required_skills": [],
  "preferred_skills": [],
  "responsibilities": [],
  "experience_requirements": [],
  "education_requirements": [],
  "certifications": [],
  "soft_skills": [],
  "keywords": [],
  "raw_job_description": ""
}"""


def load_prompt_template(prompt_path: str | Path | None = None) -> str:
    """
    Load the job description analysis prompt template from disk or use the fallback.
    """
    if prompt_path is None:
        # Default to prompts/jd_analysis_prompt.txt relative to repository root
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "jd_analysis_prompt.txt"
    else:
        prompt_path = Path(prompt_path)

    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8")
        except Exception:
            return FALLBACK_PROMPT_TEMPLATE

    return FALLBACK_PROMPT_TEMPLATE


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """
    Send prompt to the local Ollama API and return the raw model text response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
            return data.get("response", "")
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Failed to connect to Ollama at {api_url}. "
            f"Ensure Ollama is running locally and model '{model}' is available. Details: {error}"
        ) from error
    except Exception as error:
        raise RuntimeError(f"Error communicating with Ollama API: {error}") from error


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """
    Safely extract and parse a JSON dictionary from the model response text.
    Handles thinking blocks, markdown code blocks, and whitespace.
    """
    cleaned = raw_text.strip()

    # Strip thinking blocks (<think>...</think>) if present
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Direct JSON parse attempt
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback regex search for the outermost JSON object
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON from model response:\n{raw_text}")


def validate_and_normalize_jd_structure(
    extracted_data: dict[str, Any], raw_job_description: str
) -> dict[str, Any]:
    """
    Ensure all required schema fields are present with correct types and preserve raw JD.
    """
    normalized: dict[str, Any] = {}

    for field, expected_type in EXPECTED_SCHEMA.items():
        if field == "raw_job_description":
            # Always ensure the original raw JD is faithfully preserved
            normalized[field] = raw_job_description
            continue

        value = extracted_data.get(field)

        if expected_type is str:
            if isinstance(value, str):
                normalized[field] = value.strip()
            elif value is None:
                normalized[field] = ""
            else:
                normalized[field] = str(value).strip()

        elif expected_type is list:
            if isinstance(value, list):
                # Ensure elements are clean strings
                normalized[field] = [str(item).strip() for item in value if item is not None and str(item).strip()]
            elif isinstance(value, str) and value.strip():
                # If the model returned a comma-separated string instead of a list
                normalized[field] = [item.strip() for item in value.split(",") if item.strip()]
            else:
                normalized[field] = []

    return normalized


def analyze_job_description(
    job_description: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    prompt_path: str | Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Analyze a raw job description string using local Ollama and return structured JSON.

    Parameters:
        job_description: Raw job description text.
        model: Ollama model name (default: "qwen3.5:4b").
        ollama_url: Ollama API endpoint (default: "http://localhost:11434/api/generate").
        prompt_path: Optional custom path to prompt template.
        timeout: Timeout in seconds for API request.

    Returns:
        A dictionary containing all structured JD fields conforming to the schema.
    """
    if not job_description or not job_description.strip():
        # Return empty structured schema if input is empty
        return {
            field: "" if expected_type is str else []
            for field, expected_type in EXPECTED_SCHEMA.items()
        }

    # 1. Load prompt template and insert raw job description safely
    template = load_prompt_template(prompt_path)
    prompt = template.replace("{job_description}", job_description.strip())

    # 2. Call local Ollama LLM
    raw_response = call_ollama(
        prompt=prompt,
        model=model,
        api_url=ollama_url,
        timeout=timeout,
    )

    # 3. Safely parse JSON output
    parsed_json = parse_json_response(raw_response)

    # 4. Validate and normalize structure against schema
    structured_jd = validate_and_normalize_jd_structure(
        extracted_data=parsed_json,
        raw_job_description=job_description,
    )

    return structured_jd


if __name__ == "__main__":
    sample_jd = """
    Data Annotator

    Required Skills:
    - Data annotation
    - Attention to detail

    About the Role:
    micro1 is engaging Data Annotators to contribute expertise to a
    client-driven project focused on high-quality data labeling.

    Scope of Work:
    1. Precisely annotate video and audio samples according to detailed
       project guidelines and protocols.
    2. Review, validate, and enhance existing annotations to maximize
       data accuracy and consistency.
    3. Identify, flag, and document edge cases and ambiguous data points.
    4. Collaborate with project coordinators to clarify annotation criteria
       and resolve uncertainties.
    5. Maintain records of completed annotations and submit deliverables
       within established milestones.
    6. Provide feedback on annotation tools and workflow for continuous
       process improvement.
    7. Ensure all annotated data meets quality assurance standards.

    Preferred Qualifications:
    - Experience in data annotation, especially video and audio content.
    - Exceptional attention to detail.
    - Strong written and verbal English communication.
    - Ability to interpret and execute specific annotation instructions.
    - Ability to work independently with minimal supervision.
    - Strong analytical skills and objective decision-making.
    - Familiarity with data labeling platforms or annotation tools.
    - Remote work experience.

    Role Type: Contractor
    Location: Remote
    """

    print("Analyzing real Job Description with Ollama (qwen3.5:4b)...")

    result = analyze_job_description(sample_jd)

    print("\nStructured Job Description Analysis Result:")
    print(json.dumps(result, indent=2))