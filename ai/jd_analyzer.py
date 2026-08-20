import json
import re
import os
from pathlib import Path
from typing import Any

from google import genai


DEFAULT_MODEL = os.environ.get(
    "RESDEV_MODEL",
    "gemini-3.5-flash-lite",
)

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


# Fallback prompt template
FALLBACK_PROMPT_TEMPLATE = """You are an expert ATS (Applicant Tracking System) job description analyzer.

Analyze the following job description and extract key information into a clean, structured JSON object.

Job Description:
\"\"\"{job_description}\"\"\"

Extraction Guidelines:

- job_title: The official role or job title.
- seniority: Seniority level such as Entry-level, Junior, Mid-level, Senior, Lead, Principal, Manager, or "" if not mentioned.
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
2. If a field is not mentioned, return an empty string "" or empty list [].
3. Return ONLY one valid JSON object.
4. Do not include markdown code blocks or explanations outside the JSON.

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
}
"""


def load_prompt_template(
    prompt_path: str | Path | None = None,
) -> str:
    """
    Load the JD analysis prompt from disk.
    Use the fallback prompt if the file is unavailable.
    """

    if prompt_path is None:
        prompt_path = (
            Path(__file__).resolve().parent.parent
            / "prompts"
            / "jd_analysis_prompt.txt"
        )
    else:
        prompt_path = Path(prompt_path)

    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8")
        except Exception:
            return FALLBACK_PROMPT_TEMPLATE

    return FALLBACK_PROMPT_TEMPLATE


def call_gemini(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send the prompt to Gemini and return the model's text response.
    """

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Please configure your Gemini API key."
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        text = response.text

        if not text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return text

    except Exception as error:
        raise RuntimeError(
            f"Error communicating with Gemini: {error}"
        ) from error


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """
    Safely extract and parse a JSON dictionary from the model response.
    """

    cleaned = raw_text.strip()

    # Remove thinking blocks if present
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL,
    ).strip()

    # Remove markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        cleaned = cleaned.strip()

    # Direct JSON parse
    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Fallback: find JSON object inside response
    match = re.search(
        r"(\{.*\})",
        cleaned,
        re.DOTALL,
    )

    if match:
        try:
            parsed = json.loads(match.group(1))

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse valid JSON from Gemini response:\n"
        f"{raw_text}"
    )


def validate_and_normalize_jd_structure(
    extracted_data: dict[str, Any],
    raw_job_description: str,
) -> dict[str, Any]:
    """
    Ensure all required schema fields exist
    and have the correct data types.
    """

    normalized: dict[str, Any] = {}

    for field, expected_type in EXPECTED_SCHEMA.items():

        if field == "raw_job_description":
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

                normalized[field] = [
                    str(item).strip()
                    for item in value
                    if item is not None
                    and str(item).strip()
                ]

            elif isinstance(value, str) and value.strip():

                normalized[field] = [
                    item.strip()
                    for item in value.split(",")
                    if item.strip()
                ]

            else:
                normalized[field] = []

    return normalized


def analyze_job_description(
    job_description: str,
    model: str = DEFAULT_MODEL,
    prompt_path: str | Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Analyze a job description using Gemini
    and return structured JSON.
    """

    if not job_description or not job_description.strip():

        return {
            field: ""
            if expected_type is str
            else []
            for field, expected_type in EXPECTED_SCHEMA.items()
        }

    # 1. Load prompt
    template = load_prompt_template(prompt_path)

    # 2. Insert the job description
    prompt = template.replace(
        "{job_description}",
        job_description.strip(),
    )

    # 3. Call Gemini
    raw_response = call_gemini(
        prompt=prompt,
        model=model,
    )

    # 4. Parse JSON
    parsed_json = parse_json_response(raw_response)

    # 5. Validate and normalize
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
    micro1 is engaging Data Annotators to contribute expertise
    to a client-driven project focused on high-quality data labeling.

    Responsibilities:
    - Precisely annotate video and audio samples.
    - Review and validate existing annotations.
    - Identify and document edge cases.
    - Collaborate with project coordinators.
    - Maintain annotation quality standards.

    Preferred Qualifications:
    - Experience in data annotation.
    - Strong written and verbal English communication.
    - Ability to work independently.
    - Strong analytical skills.

    Role Type: Contractor
    Location: Remote
    """

    print(
        "Analyzing Job Description with Gemini..."
    )

    result = analyze_job_description(sample_jd)

    print(
        "\nStructured Job Description Analysis Result:"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )