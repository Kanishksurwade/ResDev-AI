import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TIMEOUT_SECONDS = 600

# Expected schema fields and their default types
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

# Fallback prompt template in case prompt file is not found
FALLBACK_PROMPT_TEMPLATE = """You are an expert ATS (Applicant Tracking System) and Resume Matching Analyst.
Your task is to compare a structured Job Description with a structured Master Resume and produce a rigorous, evidence-based matching analysis in JSON format.

Master Resume (Source of Truth):
\"\"\"{master_resume_json}\"\"\"

Target Structured Job Description:
\"\"\"{structured_jd_json}\"\"\"

Evaluation Guidelines & Rules:
1. SOURCE OF TRUTH: The Master Resume is the ONLY source of candidate truth. Never fabricate, invent, or assume skills, tools, experiences, projects, or metrics not explicitly supported by the Master Resume.
2. MATCH TYPES:
   - "strong": Clear, explicit evidence exists in the candidate's skills, work experience bullets, or projects.
   - "transferable": Candidate has closely related or underlying competence (e.g., CVAT tool experience transferable to generic data labeling platforms).
   - "missing": No reasonable evidence exists in the Master Resume.
3. EVIDENCE TRACEABILITY: For matched items, keep evidence concise (1 brief sentence citing specific resume experience, skills, or projects).
4. MISSING SKILLS & KEYWORDS: If a required or preferred skill/keyword in the JD has no backing evidence in the Master Resume, place it in the corresponding missing list with a brief reason.
5. OVERALL MATCH SCORE: Compute an explainable integer score between 0 and 100 based on:
   - Required skills alignment (highest weight ~45%)
   - Relevant experience and projects (~25%)
   - Preferred skills & keywords (~20%)
   - Soft skills & general alignment (~10%)
   Do NOT claim this is a universal ATS algorithm score; it is a heuristic semantic match score.
6. CONCISENESS & FORMAT: Be concise. Return ONLY a single valid JSON object adhering strictly to the schema below without markdown backticks or conversational text.

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


def load_prompt_template(prompt_path: str | Path | None = None) -> str:
    """
    Load the resume matching prompt template from disk or use the fallback.
    """
    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "resume_matching_prompt.txt"
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
            "num_predict": 4096,
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

    # Secondary fallback: attempt to repair truncated JSON if closing brackets are missing
    if cleaned.startswith("{"):
        for suffix in ["}", "]}", "]}}", "\"\n}", "\"\n]}", "\"\n]}}"]:
            try:
                parsed = json.loads(cleaned + suffix)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse valid JSON from model response:\n{raw_text}")


def validate_and_normalize_match_structure(extracted_data: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure all required schema fields are present with correct types.
    """
    normalized: dict[str, Any] = {}

    for field, expected_type in EXPECTED_MATCH_SCHEMA.items():
        value = extracted_data.get(field)

        if expected_type is int:
            try:
                score = int(float(value)) if value is not None else 0
                normalized[field] = max(0, min(100, score))
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


def _clean_resume_for_prompt(resume_data: dict[str, Any]) -> dict[str, Any]:
    """
    Strip internal generation meta-rules while preserving all candidate factual data.
    """
    cleaned = dict(resume_data)
    cleaned.pop("generation_rules", None)
    return cleaned


def match_resume_to_jd(
    master_resume: dict[str, Any] | str | Path,
    structured_jd: dict[str, Any] | str | Path,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    prompt_path: str | Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Compare structured Job Description with Master Resume using local Ollama.

    Parameters:
        master_resume: Master resume dict or path to JSON file.
        structured_jd: Structured JD dict or path to JSON file / JSON string.
        model: Ollama model name (default: "qwen3.5:4b").
        ollama_url: Ollama API endpoint.
        prompt_path: Optional custom prompt template path.
        timeout: Timeout in seconds.

    Returns:
        Structured matching analysis dictionary conforming to schema.
    """
    # 1. Resolve Master Resume into clean JSON string
    if isinstance(master_resume, (str, Path)) and Path(str(master_resume)).exists():
        with open(master_resume, "r", encoding="utf-8") as f:
            master_resume_dict = json.load(f)
    elif isinstance(master_resume, dict):
        master_resume_dict = master_resume
    elif isinstance(master_resume, str):
        master_resume_dict = json.loads(master_resume)
    else:
        raise ValueError(f"Invalid master_resume input type: {type(master_resume)}")

    clean_resume = _clean_resume_for_prompt(master_resume_dict)
    master_resume_str = json.dumps(clean_resume, separators=(',', ':'))

    # 2. Resolve Structured JD into clean JSON string
    if isinstance(structured_jd, (str, Path)) and Path(str(structured_jd)).exists():
        with open(structured_jd, "r", encoding="utf-8") as f:
            structured_jd_dict = json.load(f)
    elif isinstance(structured_jd, dict):
        structured_jd_dict = structured_jd
    elif isinstance(structured_jd, str):
        structured_jd_dict = json.loads(structured_jd)
    else:
        raise ValueError(f"Invalid structured_jd input type: {type(structured_jd)}")

    structured_jd_str = json.dumps(structured_jd_dict, separators=(',', ':'))

    # 3. Load prompt template and safely inject inputs
    template = load_prompt_template(prompt_path)
    prompt = template.replace("{master_resume_json}", master_resume_str)
    prompt = prompt.replace("{structured_jd_json}", structured_jd_str)

    # 4. Call Ollama
    raw_response = call_ollama(
        prompt=prompt,
        model=model,
        api_url=ollama_url,
        timeout=timeout,
    )

    # 5. Parse response safely
    parsed_json = parse_json_response(raw_response)

    # 6. Validate and normalize structure
    matching_result = validate_and_normalize_match_structure(parsed_json)

    return matching_result


if __name__ == "__main__":
    master_resume_file = Path(__file__).resolve().parent.parent / "data" / "master_resume.json"

    if not master_resume_file.exists():
        raise FileNotFoundError(f"Master resume file not found at {master_resume_file}")

    with open(master_resume_file, "r", encoding="utf-8") as f:
        master_resume_data = json.load(f)

    # Test Structured Job Description
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
    print("Candidate:", master_resume_data.get("candidate", {}).get("personal_info", {}).get("name"))
    print("Target Role:", target_structured_jd.get("job_title"))
    print("Running matching analysis with local Ollama (qwen3.5:4b)...")
    print("-" * 70)

    match_result = match_resume_to_jd(
        master_resume=master_resume_data,
        structured_jd=target_structured_jd,
    )

    print("\n[FULL STRUCTURED MATCHING RESULT]")
    print(json.dumps(match_result, indent=2))

    print("\n" + "=" * 70)
    print("MATCHING SUMMARY & INSIGHTS")
    print("=" * 70)
    print(f"Overall Match Score: {match_result.get('overall_match_score')}/100")
    print(f"Explanation: {match_result.get('explanation')}\n")

    print(f"--- MATCHED REQUIRED SKILLS ({len(match_result.get('matched_required_skills', []))}) ---")
    for item in match_result.get("matched_required_skills", []):
        skill = item.get("skill") if isinstance(item, dict) else str(item)
        evidence = item.get("evidence", "") if isinstance(item, dict) else ""
        match_type = item.get("match_type", "") if isinstance(item, dict) else ""
        print(f"  [+] {skill} ({match_type}): {evidence}")

    print(f"\n--- MISSING REQUIRED SKILLS ({len(match_result.get('missing_required_skills', []))}) ---")
    for item in match_result.get("missing_required_skills", []):
        skill = item.get("skill") if isinstance(item, dict) else str(item)
        reason = item.get("reason", "") if isinstance(item, dict) else ""
        print(f"  [-] {skill}: {reason}")

    print(f"\n--- TRANSFERABLE SKILLS ({len(match_result.get('transferable_skills', []))}) ---")
    for item in match_result.get("transferable_skills", []):
        req = item.get("jd_requirement", "") if isinstance(item, dict) else ""
        cand = item.get("candidate_evidence", "") if isinstance(item, dict) else ""
        reason = item.get("reason", "") if isinstance(item, dict) else ""
        print(f"  [~] Requirement: {req} -> Evidence: {cand} ({reason})")

    print(f"\n--- EVIDENCE GAPS ({len(match_result.get('evidence_gaps', []))}) ---")
    for item in match_result.get("evidence_gaps", []):
        gap = item.get("gap", "") if isinstance(item, dict) else str(item)
        impact = item.get("impact", "") if isinstance(item, dict) else ""
        print(f"  [!] {gap} (Impact: {impact})")
    print("=" * 70)
