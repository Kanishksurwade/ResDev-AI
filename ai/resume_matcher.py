import json
import re
import os
from pathlib import Path
from typing import Any

from google import genai

try:
    from ai.gemini_config import get_gemini_api_key
except ImportError:
    from gemini_config import get_gemini_api_key


DEFAULT_MODEL = os.environ.get(
    "RESDEV_MODEL",
    "gemini-3.5-flash-lite",
)

DEFAULT_TIMEOUT_SECONDS = 600


EXPECTED_MATCH_SCHEMA: dict[str, type] = {
    "overall_match_score": int,
    "matched_required_skills": list,
    "missing_required_skills": list,
    "matched_preferred_skills": list,
    "missing_preferred_skills": list,
    "matched_keywords": list,
    "missing_keywords": list,
    "relevant_experience": list,
    "relevant_projects": list,
    "transferable_skills": list,
    "evidence_gaps": list,
    "explanation": str,
}


FALLBACK_PROMPT_TEMPLATE = """You are an expert ATS (Applicant Tracking System) and Resume Matching Analyst.

Your task is to compare a structured Job Description with a structured Master Resume and produce a rigorous, evidence-based matching analysis in JSON format.

Master Resume (Source of Truth):
\"\"\"{master_resume_json}\"\"\"

Target Structured Job Description:
\"\"\"{structured_jd_json}\"\"\"

Evaluation Guidelines & Rules:

1. SOURCE OF TRUTH:
   The Master Resume is the ONLY source of candidate truth.
   Never fabricate, invent, or assume skills, tools, experiences,
   projects, or metrics not explicitly supported by the Master Resume.

2. MATCH TYPES:
   - "strong": Clear, explicit evidence exists in the candidate's
     skills, work experience bullets, or projects.
   - "transferable": Candidate has closely related or underlying
     competence.
   - "missing": No reasonable evidence exists in the Master Resume.

3. EVIDENCE TRACEABILITY:
   For matched items, keep evidence concise and cite specific
   resume experience, skills, or projects.

4. MISSING SKILLS & KEYWORDS:
   If a required or preferred skill/keyword in the JD has no backing
   evidence in the Master Resume, place it in the corresponding
   missing list with a brief reason.

5. OVERALL MATCH SCORE:
   Compute an explainable integer score between 0 and 100 based on:
   - Required skills alignment (highest weight ~45%)
   - Relevant experience and projects (~25%)
   - Preferred skills & keywords (~20%)
   - Soft skills & general alignment (~10%)

   Do NOT claim this is a universal ATS algorithm score.
   It is a heuristic semantic match score.

6. CONCISENESS & FORMAT:
   Return ONLY one valid JSON object.
   Do not include markdown code blocks.
   Do not include conversational text.

Expected JSON Structure:

{
  "overall_match_score": 0,
  "matched_required_skills": [
    {
      "skill": "Skill name",
      "evidence": "Concise evidence from Master Resume",
      "match_type": "strong"
    }
  ],
  "missing_required_skills": [
    {
      "skill": "Missing skill name",
      "reason": "Brief reason"
    }
  ],
  "matched_preferred_skills": [
    {
      "skill": "Preferred skill name",
      "evidence": "Concise evidence from Master Resume",
      "match_type": "strong"
    }
  ],
  "missing_preferred_skills": [
    {
      "skill": "Missing preferred skill name",
      "reason": "Brief reason"
    }
  ],
  "matched_keywords": [
    {
      "keyword": "Keyword",
      "evidence": "Brief evidence",
      "match_type": "strong"
    }
  ],
  "missing_keywords": [
    {
      "keyword": "Missing keyword",
      "reason": "Brief reason"
    }
  ],
  "relevant_experience": [
    {
      "company": "Company Name",
      "role": "Role Title",
      "relevance": "Concise relevance statement",
      "matching_points": ["Key matching point"]
    }
  ],
  "relevant_projects": [
    {
      "name": "Project Name",
      "relevance": "Concise relevance statement",
      "matching_points": ["Key matching point"]
    }
  ],
  "transferable_skills": [
    {
      "jd_requirement": "JD requirement name",
      "candidate_evidence": "Candidate experience / tool",
      "reason": "Why this is a transferable equivalent",
      "match_type": "transferable"
    }
  ],
  "evidence_gaps": [
    {
      "gap": "Specific gap description",
      "impact": "Low / Moderate / High"
    }
  ],
  "explanation": "Clear, concise summary paragraph explaining the match evaluation."
}"""


def load_prompt_template(
    prompt_path: str | Path | None = None,
) -> str:
    """Load the resume matching prompt template."""

    if prompt_path is None:
        prompt_path = (
            Path(__file__).resolve().parent.parent
            / "prompts"
            / "resume_matching_prompt.txt"
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
    """Send the prompt to Gemini and return its text response."""

    api_key = get_gemini_api_key()

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


def parse_json_response(
    raw_text: str,
) -> dict[str, Any]:
    """Safely extract and parse a JSON dictionary."""

    cleaned = raw_text.strip()

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL,
    ).strip()

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
        ).strip()

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

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
        "Could not parse valid JSON from model response:\n"
        f"{raw_text}"
    )


def validate_and_normalize_match_structure(
    extracted_data: dict[str, Any],
) -> dict[str, Any]:
    """Ensure all required schema fields exist."""

    normalized: dict[str, Any] = {}

    for field, expected_type in EXPECTED_MATCH_SCHEMA.items():

        value = extracted_data.get(field)

        if expected_type is int:

            try:
                score = (
                    int(float(value))
                    if value is not None
                    else 0
                )

                normalized[field] = max(
                    0,
                    min(100, score),
                )

            except (ValueError, TypeError):
                normalized[field] = 0

        elif expected_type is str:

            if isinstance(value, str):
                normalized[field] = value.strip()

            elif value is None:
                normalized[field] = ""

            else:
                normalized[field] = str(value).strip()

        elif expected_type is list:

            if isinstance(value, list):
                normalized[field] = value

            elif value is None:
                normalized[field] = []

            else:
                normalized[field] = [value]

    return normalized


def _clean_resume_for_prompt(
    resume_data: dict[str, Any],
) -> dict[str, Any]:
    """Remove internal generation rules while keeping candidate facts."""

    cleaned = dict(resume_data)

    cleaned.pop(
        "generation_rules",
        None,
    )

    return cleaned


def match_resume_to_jd(
    master_resume: dict[str, Any] | str | Path,
    structured_jd: dict[str, Any] | str | Path,
    model: str = DEFAULT_MODEL,
    prompt_path: str | Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Compare a structured Job Description with the Master Resume
    using Gemini.
    """

    # 1. Resolve Master Resume
    if (
        isinstance(master_resume, (str, Path))
        and Path(str(master_resume)).exists()
    ):

        with open(
            master_resume,
            "r",
            encoding="utf-8",
        ) as f:
            master_resume_dict = json.load(f)

    elif isinstance(master_resume, dict):
        master_resume_dict = master_resume

    elif isinstance(master_resume, str):
        master_resume_dict = json.loads(master_resume)

    else:
        raise ValueError(
            f"Invalid master_resume input type: "
            f"{type(master_resume)}"
        )

    clean_resume = _clean_resume_for_prompt(
        master_resume_dict
    )

    master_resume_str = json.dumps(
        clean_resume,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    # 2. Resolve Structured JD
    if (
        isinstance(structured_jd, (str, Path))
        and Path(str(structured_jd)).exists()
    ):

        with open(
            structured_jd,
            "r",
            encoding="utf-8",
        ) as f:
            structured_jd_dict = json.load(f)

    elif isinstance(structured_jd, dict):
        structured_jd_dict = structured_jd

    elif isinstance(structured_jd, str):
        structured_jd_dict = json.loads(structured_jd)

    else:
        raise ValueError(
            f"Invalid structured_jd input type: "
            f"{type(structured_jd)}"
        )

    structured_jd_str = json.dumps(
        structured_jd_dict,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    # 3. Load prompt
    template = load_prompt_template(prompt_path)

    prompt = template.replace(
        "{master_resume_json}",
        master_resume_str,
    )

    prompt = prompt.replace(
        "{structured_jd_json}",
        structured_jd_str,
    )

    # 4. Call Gemini
    raw_response = call_gemini(
        prompt=prompt,
        model=model,
    )

    # 5. Parse response
    parsed_json = parse_json_response(
        raw_response
    )

    # 6. Validate and normalize
    return validate_and_normalize_match_structure(
        parsed_json
    )


if __name__ == "__main__":

    master_resume_file = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "master_resume.json"
    )

    if not master_resume_file.exists():
        raise FileNotFoundError(
            f"Master resume file not found at "
            f"{master_resume_file}"
        )

    with open(
        master_resume_file,
        "r",
        encoding="utf-8",
    ) as f:
        master_resume_data = json.load(f)

    target_structured_jd = {
        "job_title": "Data Annotator",
        "seniority": "",
        "required_skills": [
            "Data annotation",
            "Attention to detail",
            "Video annotation",
            "Audio annotation",
            "Data labeling platforms",
            "Annotation tools",
        ],
        "preferred_skills": [
            "Strong written and verbal English communication",
            "Ability to work independently with minimal supervision",
            "Strong analytical skills",
            "Remote work experience",
        ],
        "responsibilities": [
            "Precisely annotate video and audio samples",
            "Review and validate annotations",
            "Identify and document edge cases",
            "Maintain data accuracy and consistency",
            "Follow detailed annotation guidelines",
        ],
        "experience_requirements": [],
        "education_requirements": [],
        "certifications": [],
        "soft_skills": [
            "Attention to detail",
            "Communication",
            "Analytical skills",
        ],
        "keywords": [
            "Data Annotator",
            "Data annotation",
            "Video annotation",
            "Audio annotation",
            "Data labeling",
            "Quality assurance",
            "Edge cases",
            "Remote work",
            "Annotation tools",
        ],
    }

    print("=" * 70)
    print("RESDEV AI - JD <-> MASTER RESUME MATCHING ENGINE")
    print("=" * 70)

    print(
        "Candidate:",
        master_resume_data
        .get("candidate", {})
        .get("personal_info", {})
        .get("name"),
    )

    print(
        "Target Role:",
        target_structured_jd.get("job_title"),
    )

    print(
        "Running matching analysis with "
        "Gemini (gemini-3.5-flash-lite)..."
    )

    print("-" * 70)

    match_result = match_resume_to_jd(
        master_resume=master_resume_data,
        structured_jd=target_structured_jd,
    )

    print(
        "\n[FULL STRUCTURED MATCHING RESULT]"
    )

    print(
        json.dumps(
            match_result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MATCHING SUMMARY & INSIGHTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Overall Match Score: "
        f"{match_result.get('overall_match_score')}/100"
    )

    print(
        f"Explanation: "
        f"{match_result.get('explanation')}\n"
    )

    print(
        f"--- MATCHED REQUIRED SKILLS "
        f"({len(match_result.get('matched_required_skills', []))}) ---"
    )

    for item in match_result.get(
        "matched_required_skills",
        [],
    ):

        skill = (
            item.get("skill")
            if isinstance(item, dict)
            else str(item)
        )

        evidence = (
            item.get("evidence", "")
            if isinstance(item, dict)
            else ""
        )

        match_type = (
            item.get("match_type", "")
            if isinstance(item, dict)
            else ""
        )

        print(
            f"  [+] {skill} "
            f"({match_type}): {evidence}"
        )

    print(
        f"\n--- MISSING REQUIRED SKILLS "
        f"({len(match_result.get('missing_required_skills', []))}) ---"
    )

    for item in match_result.get(
        "missing_required_skills",
        [],
    ):

        skill = (
            item.get("skill")
            if isinstance(item, dict)
            else str(item)
        )

        reason = (
            item.get("reason", "")
            if isinstance(item, dict)
            else ""
        )

        print(
            f"  [-] {skill}: {reason}"
        )

    print(
        f"\n--- TRANSFERABLE SKILLS "
        f"({len(match_result.get('transferable_skills', []))}) ---"
    )

    for item in match_result.get(
        "transferable_skills",
        [],
    ):

        req = (
            item.get("jd_requirement", "")
            if isinstance(item, dict)
            else ""
        )

        cand = (
            item.get("candidate_evidence", "")
            if isinstance(item, dict)
            else ""
        )

        reason = (
            item.get("reason", "")
            if isinstance(item, dict)
            else ""
        )

        print(
            f"  [~] Requirement: {req} "
            f"-> Evidence: {cand} "
            f"({reason})"
        )

    print(
        f"\n--- EVIDENCE GAPS "
        f"({len(match_result.get('evidence_gaps', []))}) ---"
    )

    for item in match_result.get(
        "evidence_gaps",
        [],
    ):

        gap = (
            item.get("gap", "")
            if isinstance(item, dict)
            else str(item)
        )

        impact = (
            item.get("impact", "")
            if isinstance(item, dict)
            else ""
        )

        print(
            f"  [!] {gap} "
            f"(Impact: {impact})"
        )

    print("=" * 70)