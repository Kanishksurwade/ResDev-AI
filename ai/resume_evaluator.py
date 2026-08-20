import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.gemini_config import call_gemini_with_retry, DEFAULT_MODEL, DEFAULT_TIMEOUT

DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT
DEFAULT_TARGET_SCORE = 85

# Expected dimensions in evaluation rubric
EVALUATION_DIMENSIONS = [
    "keyword_match",
    "skills_match",
    "experience_relevance",
    "grammar",
    "ats_structure",
    "readability",
    "resume_length",
    "action_verbs",
    "missing_keywords",
]

# Fallback prompt template in case prompt file is not found
FALLBACK_PROMPT_TEMPLATE = """You are an expert ATS (Applicant Tracking System) Quality Evaluator and Senior Technical Resume Reviewer.
Your task is to objectively evaluate a generated tailored resume against a target Job Description across standardized rubric dimensions.

Target Structured Job Description:
\"\"\"{structured_jd_json}\"\"\"

Generated Tailored Resume to Evaluate:
\"\"\"{tailored_resume_json}\"\"\"

Evaluation Dimensions & Scoring Rubric (0 to 100 per dimension):
1. keyword_match: Alignment and density of key domain keywords from the JD in the resume.
2. skills_match: Completeness and coverage of mandatory and preferred technical skills.
3. experience_relevance: Direct relevance of work history, bullet points, and deliverables to the target role duties.
4. grammar: Professional tone, grammatical correctness, syntax, and clarity.
5. ats_structure: Logical organization, section completeness, formatting cleanliness, and parseability.
6. readability: Conciseness, scannability, clarity of expression, and avoidance of fluff.
7. resume_length: Appropriate density, well-balanced content volume without being too brief or overly verbose.
8. action_verbs: Strong, diverse, active verb usage at the beginning of bullet points (e.g., "Engineered", "Annotated", "Evaluated").
9. missing_keywords: Score reflecting the absence of critical JD keywords (100 = zero critical missing keywords, lower if important keywords are absent).
10. overall_score: Weighted composite score (0-100) reflecting overall alignment, quality, and ATS readiness.

Rules & Guidelines:
- Base your evaluation strictly on the provided Generated Resume and Job Description.
- CRITICAL FACTUAL BOUNDARIES: Only recommend changes that rephrase, prioritize, or emphasize factual candidate evidence already present in the resume. NEVER recommend adding fabricated metrics, unmentioned tools/platforms, fake employment history, or assumed native language proficiency.
- Identify matched and missing keywords clearly.
- Provide structured `improvement_actions` specifying the exact target section (`summary`, `skills`, `experience`, `projects`, etc.), the concrete change, and the rationale.
- Return ONLY a single valid JSON object adhering strictly to the schema below without markdown backticks or conversational text.

Expected JSON Structure:
{
  "overall_score": 0,
  "dimension_scores": {
    "keyword_match": 0,
    "skills_match": 0,
    "experience_relevance": 0,
    "grammar": 0,
    "ats_structure": 0,
    "readability": 0,
    "resume_length": 0,
    "action_verbs": 0,
    "missing_keywords": 0
  },
  "matched_keywords": ["Keywords from JD present in the resume"],
  "missing_keywords": ["Keywords from JD that are absent or poorly represented"],
  "strengths": ["Key strength or strong alignment point"],
  "weaknesses": ["Key weakness or gap in alignment"],
  "improvement_actions": [
    {
      "target": "summary",
      "change": "Specific phrasing or keyword emphasis to adjust in summary",
      "reason": "Why this improves alignment with target JD"
    }
  ],
  "improvement_instructions": ["Brief summary of actionable improvements"],
  "explanation": "Comprehensive summary paragraph explaining the evaluation score and overall assessment."
}"""


def load_prompt_template(prompt_path: str | Path | None = None) -> str:
    """
    Load the resume evaluation prompt template from disk or use the fallback.
    """
    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "resume_evaluation_prompt.txt"
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
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """
    Send the evaluation prompt to Gemini and return the raw text response.
    Uses call_gemini_with_retry from gemini_config for timeout + retry handling.
    """
    return call_gemini_with_retry(
        prompt=prompt,
        model=model,
        timeout=timeout,
    )


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

    # 1. Direct JSON parse attempt
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Fallback regex search for the outermost JSON object
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Robust progressive repair for truncated JSON
    if cleaned.startswith("{"):
        for i in range(len(cleaned), 0, -1):
            sub = cleaned[:i].strip()
            sub = re.sub(r",\s*$", "", sub)

            # Fix unclosed string quote
            quotes = len(re.findall(r'(?<!\\)"', sub))
            if quotes % 2 != 0:
                sub += '"'

            # Balance braces and brackets
            stack = []
            in_string = False
            escape = False
            for char in sub:
                if char == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if char in '{[':
                        stack.append(char)
                    elif char in '}]':
                        if stack:
                            if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                                stack.pop()
                if char == '\\' and not escape:
                    escape = True
                else:
                    escape = False

            closing = ""
            for open_bracket in reversed(stack):
                if open_bracket == '{':
                    closing += "}"
                elif open_bracket == '[':
                    closing += "]"

            try:
                candidate = sub + closing
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and len(parsed) >= 2:
                    return parsed
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse valid JSON from model response:\n{raw_text}")


def validate_and_normalize_evaluation_structure(
    extracted_data: dict[str, Any], target_score: int = DEFAULT_TARGET_SCORE
) -> dict[str, Any]:
    """
    Ensure all required schema fields are present with correct types and compute pass/fail.
    """
    normalized: dict[str, Any] = {}

    # Overall score
    raw_overall = extracted_data.get("overall_score")
    try:
        overall_score = int(float(raw_overall)) if raw_overall is not None else 0
        overall_score = max(0, min(100, overall_score))
    except (ValueError, TypeError):
        overall_score = 0

    normalized["overall_score"] = overall_score
    normalized["target_score"] = target_score
    normalized["pass_status"] = bool(overall_score >= target_score)

    # Dimension scores
    raw_dims = extracted_data.get("dimension_scores", {})
    if not isinstance(raw_dims, dict):
        raw_dims = {}

    dimension_scores: dict[str, int] = {}
    for dim in EVALUATION_DIMENSIONS:
        val = raw_dims.get(dim)
        try:
            score = int(float(val)) if val is not None else overall_score
            dimension_scores[dim] = max(0, min(100, score))
        except (ValueError, TypeError):
            dimension_scores[dim] = overall_score

    normalized["dimension_scores"] = dimension_scores

    # List fields
    for list_field in [
        "matched_keywords",
        "missing_keywords",
        "strengths",
        "weaknesses",
        "improvement_instructions",
    ]:
        val = extracted_data.get(list_field)
        if isinstance(val, list):
            normalized[list_field] = [str(item).strip() for item in val if item is not None and str(item).strip()]
        elif isinstance(val, str) and val.strip():
            normalized[list_field] = [val.strip()]
        else:
            normalized[list_field] = []

    # Structured improvement_actions
    raw_actions = extracted_data.get("improvement_actions")
    normalized_actions: list[dict[str, str]] = []
    if isinstance(raw_actions, list):
        for act in raw_actions:
            if isinstance(act, dict):
                target = str(act.get("target", "general")).strip().lower()
                change = str(act.get("change", "")).strip()
                reason = str(act.get("reason", "")).strip()
                if change:
                    normalized_actions.append({
                        "target": target,
                        "change": change,
                        "reason": reason,
                    })
            elif isinstance(act, str) and act.strip():
                normalized_actions.append({
                    "target": "general",
                    "change": act.strip(),
                    "reason": "Improves job description alignment",
                })
    normalized["improvement_actions"] = normalized_actions

    # Ensure improvement_instructions exists (populate from actions if empty)
    if not normalized["improvement_instructions"] and normalized_actions:
        normalized["improvement_instructions"] = [
            f"[{act['target'].upper()}] {act['change']}" for act in normalized_actions
        ]

    # Explanation
    explanation = extracted_data.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        normalized["explanation"] = explanation.strip()
    else:
        normalized["explanation"] = (
            f"Overall quality score is {overall_score}/100 with a target of {target_score}/100. "
            f"Pass status: {'PASS' if normalized['pass_status'] else 'NEEDS_REVISION'}."
        )

    return normalized


def evaluate_resume(
    tailored_resume: dict[str, Any] | str | Path,
    structured_jd: dict[str, Any] | str | Path,
    target_score: int = DEFAULT_TARGET_SCORE,
    model: str = DEFAULT_MODEL,
    prompt_path: str | Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    # Legacy compatibility: ollama_url kwarg silently ignored
    **_kwargs: object,
) -> dict[str, Any]:
    """
    Evaluate a tailored resume against a target Job Description using Gemini.

    Parameters:
        tailored_resume: Tailored resume dict or path to JSON file / JSON string.
        structured_jd: Structured JD dict or path to JSON file / JSON string.
        target_score: Minimum overall score required to pass (default: 85).
        model: Gemini model name (default: "gemini-3.5-flash-lite").
        prompt_path: Optional custom prompt template path.
        timeout: Per-request timeout in seconds.

    Returns:
        Structured evaluation dictionary conforming to schema.
    """
    # 1. Resolve Tailored Resume
    if isinstance(tailored_resume, (str, Path)) and Path(str(tailored_resume)).exists():
        with open(tailored_resume, "r", encoding="utf-8") as f:
            tailored_resume_dict = json.load(f)
    elif isinstance(tailored_resume, dict):
        tailored_resume_dict = tailored_resume
    elif isinstance(tailored_resume, str):
        tailored_resume_dict = json.loads(tailored_resume)
    else:
        raise ValueError(f"Invalid tailored_resume input type: {type(tailored_resume)}")

    tailored_resume_str = json.dumps(tailored_resume_dict, separators=(',', ':'))

    # 2. Resolve Structured JD
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
    prompt = template.replace("{structured_jd_json}", structured_jd_str)
    prompt = prompt.replace("{tailored_resume_json}", tailored_resume_str)

    # 4. Call Gemini
    raw_response = call_gemini(
        prompt=prompt,
        model=model,
        timeout=timeout,
    )

    # 5. Parse response safely
    parsed_json = parse_json_response(raw_response)

    # 6. Validate and normalize structure against schema
    evaluation_result = validate_and_normalize_evaluation_structure(
        extracted_data=parsed_json,
        target_score=target_score,
    )

    return evaluation_result


if __name__ == "__main__":
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

    # Sample tailored resume
    sample_tailored_resume = {
        "personal_info": {
            "name": "Kanishk Surwade",
            "target_title": "Data Annotator",
            "email": "kanishksurwade70@gmail.com",
            "phone": "+91-9834008224",
            "location": "Pune, Maharashtra, India",
            "linkedin": "linkedin.com/in/kd4723",
            "github": "github.com/Kanishksurwade",
        },
        "summary": "AI & LLM Analyst with extensive experience in precise data annotation across text, audio, video, and image modalities on confidential enterprise datasets. Proven expertise in adhering to strict SOPs for quality benchmarking, identifying edge cases, and maintaining high inter-annotator accuracy standards. Skilled in working independently in remote environments using platforms like CVAT and Google Cloud Platform to deliver consistent data labeling outcomes.",
        "skills": {
            "technical_skills": [
                "Data annotation",
                "Video annotation",
                "Audio annotation",
                "Data labeling",
                "Quality assurance",
                "Edge case identification",
            ],
            "tools_and_technologies": [
                "CVAT",
                "Google Cloud Platform (GCP)",
                "JAX",
                "Flax",
                "Power BI",
                "MySQL Workbench",
                "Python",
                "SQL",
            ],
            "core_competencies": [
                "Attention to detail",
                "SOP adherence",
                "Remote work experience",
                "Cross-functional collaboration",
                "Confidential data handling",
                "Strong analytical skills",
            ],
        },
        "experience": [
            {
                "company": "Innodata Inc.",
                "role": "AI & LLM Analyst",
                "location": "Noida, India (Remote)",
                "start_date": "Dec 2025",
                "end_date": "Jul 2026",
                "bullets": [
                    "Executed precise multimodal data annotation for text, audio, video, and image samples, ensuring strict adherence to SOPs and maintaining high inter-annotator accuracy.",
                    "Performed comprehensive quality assurance by reviewing annotations, validating outputs, and documenting complex edge cases to improve dataset integrity.",
                    "Collaborated cross-functionally to resolve annotation ambiguities during calibration sessions, supporting downstream data consistency for production AI workflows.",
                ],
            },
            {
                "company": "Deloitte",
                "role": "Data Analytics Intern",
                "location": "Virtual Internship",
                "start_date": "Sep 2025",
                "end_date": "Sep 2025",
                "bullets": [
                    "Analyzed operational datasets to identify performance gaps, applying strong analytical skills to support data-driven decision-making.",
                    "Developed KPI dashboards and calculated metrics using Excel to enhance reporting efficiency and accuracy.",
                ],
            },
        ],
        "projects": [
            {
                "name": "Google Tunix — Structured Reasoning Fine-Tuning with GRPO on Gemma 3",
                "technologies": ["Gemma 3 (1B)", "GRPO", "LoRA", "JAX", "Flax", "Google Cloud", "TPU"],
                "start_date": "Dec 2025",
                "end_date": "Jan 2026",
                "bullets": [
                    "Designed reward functions and prompt templates to enforce output consistency, demonstrating advanced understanding of structured data requirements.",
                    "Built and executed a TPU-based training and evaluation pipeline using JAX and Flax on Google Cloud, ensuring robust model performance.",
                ],
            },
            {
                "name": "Northwind Sales Analysis & Dashboard",
                "technologies": ["SQL", "Excel", "Power BI", "DAX", "ETL"],
                "start_date": "Oct 2025",
                "end_date": "Oct 2025",
                "bullets": [
                    "Cleaned and analyzed 5,000+ sales records using SQL and Excel to prepare high-quality data for business intelligence.",
                    "Constructed interactive Power BI dashboards with DAX-driven KPIs to visualize trends and improve reporting efficiency.",
                ],
            },
        ],
        "education": [
            {
                "degree": "B.Tech in Automation and Robotics",
                "institution": "JSPM Rajarshi Shahu College of Engineering",
                "location": "Pune, India",
                "start_year": "2021",
                "end_year": "2025",
                "details": "CGPA: 7.75",
            }
        ],
        "certifications": [
            {
                "name": "Career Essentials in Generative AI",
                "issuer": "Microsoft & LinkedIn Learning",
            }
        ],
        "tailoring_metadata": {
            "target_role": "Data Annotator",
            "primary_keywords_integrated": [
                "Data annotation",
                "Video annotation",
                "Audio annotation",
                "Data labeling",
                "Quality assurance",
                "Edge cases",
                "Remote work",
            ],
            "tailoring_summary": "Resume tailored to highlight multimodal annotation experience (video, audio, text) and strict SOP adherence from Innodata role.",
        },
    }

    print("=" * 70)
    print("RESDEV AI - RESUME QUALITY EVALUATION ENGINE")
    print("=" * 70)
    print("Candidate:", sample_tailored_resume.get("personal_info", {}).get("name"))
    print("Target Role:", target_structured_jd.get("job_title"))
    print("Running quality evaluation with Gemini (gemini-3.5-flash-lite)...")
    print("-" * 70)

    evaluation = evaluate_resume(
        tailored_resume=sample_tailored_resume,
        structured_jd=target_structured_jd,
        target_score=85,
    )

    print("\n[FULL STRUCTURED EVALUATION RESULT]")
    print(json.dumps(evaluation, indent=2))

    print("\n" + "=" * 70)
    print("EVALUATION SCORECARD & SUMMARY")
    print("=" * 70)
    status_icon = "[PASS]" if evaluation.get("pass_status") else "[NEEDS REVISION]"
    print(f"Overall Score: {evaluation.get('overall_score')}/100  {status_icon} (Target: {evaluation.get('target_score')})")
    print(f"Explanation: {evaluation.get('explanation')}\n")

    print("--- DIMENSION SCORES (0-100) ---")
    for dim, score in evaluation.get("dimension_scores", {}).items():
        print(f"  * {dim.replace('_', ' ').title():<25}: {score}/100")

    print(f"\n--- KEYWORD ALIGNMENT ---")
    print(f"  Matched ({len(evaluation.get('matched_keywords', []))}): {', '.join(evaluation.get('matched_keywords', []))}")
    if evaluation.get("missing_keywords"):
        print(f"  Missing ({len(evaluation.get('missing_keywords', []))}): {', '.join(evaluation.get('missing_keywords', []))}")
    else:
        print("  Missing: None (all critical keywords represented)")

    print(f"\n--- KEY STRENGTHS ({len(evaluation.get('strengths', []))}) ---")
    for s in evaluation.get("strengths", []):
        print(f"  [+] {s}")

    print(f"\n--- WEAKNESSES & GAPS ({len(evaluation.get('weaknesses', []))}) ---")
    for w in evaluation.get("weaknesses", []):
        print(f"  [-] {w}")

    print(f"\n--- ACTIONABLE IMPROVEMENT INSTRUCTIONS ({len(evaluation.get('improvement_instructions', []))}) ---")
    for inst in evaluation.get("improvement_instructions", []):
        print(f"  [>] {inst}")
    print("=" * 70)
