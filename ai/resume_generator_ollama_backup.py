import json
import re
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = os.environ.get("RESDEV_MODEL", "qwen3.5:4b")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TIMEOUT_SECONDS = 600

# Expected schema fields
EXPECTED_RESUME_SCHEMA: dict[str, type] = {
    "personal_info": dict,
    "summary": str,
    "skills": dict,
    "experience": list,
    "projects": list,
    "education": list,
    "certifications": list,
    "tailoring_metadata": dict,
}

# Fallback prompt template in case prompt file is not found
FALLBACK_PROMPT_TEMPLATE = """You are an expert ATS Resume Tailoring and Generation Specialist.
Your task is to generate a tailored, professional, ATS-optimized structured resume in JSON format by aligning candidate evidence from the Master Resume to the target Job Description and matching analysis.

Master Resume (Source of Truth):
\"\"\"{master_resume_json}\"\"\"

Target Job Description:
\"\"\"{structured_jd_json}\"\"\"

Matching Analysis & Keyword Alignment:
\"\"\"{matching_analysis_json}\"\"\"

{revision_feedback_section}

Tailoring Rules & Strict Guidelines:
1. FACTUAL TRUTH: The Master Resume is the ONLY source of candidate facts. NEVER invent or fabricate employers, job titles, dates, education, certifications, metrics, projects, or technologies.
2. RELEVANCE & PRIORITIZATION: Prioritize and emphasize experiences, skills, tools, and projects that directly match the target role requirements.
3. KEYWORD ALIGNMENT: Integrate matched JD keywords naturally into the summary, skill categories, and experience bullet points where supported by evidence.
4. PROFESSIONAL SUMMARY: Craft a compelling 3-4 sentence professional summary tailored specifically to the target job title, showcasing relevant expertise and proven strengths.
5. BULLET POINTS: Formulate concise, achievement-focused bullet points using strong action verbs (e.g., "Annotated", "Evaluated", "Streamlined", "Engineered"). Maintain the candidate's authentic metrics.
6. UNSUPPORTED REQUIREMENTS: Do NOT claim unsupported skills or qualifications.
7. OUTPUT FORMAT: Return ONLY a single valid JSON object strictly matching the structure below. No markdown backticks, no conversational preamble.

Expected JSON Structure:
{
  "personal_info": {
    "name": "Candidate Name",
    "target_title": "Target Role Title (e.g., Data Annotator)",
    "email": "email@example.com",
    "phone": "Phone Number",
    "location": "City, State/Country",
    "linkedin": "linkedin URL or handle",
    "github": "github URL or handle"
  },
  "summary": "Tailored professional summary highlighting relevant experience and alignment with target role.",
  "skills": {
    "technical_skills": ["Prioritized technical skills matching JD"],
    "tools_and_technologies": ["Prioritized tools, platforms, and software"],
    "core_competencies": ["Relevant domain skills, quality standards, and soft skills"]
  },
  "experience": [
    {
      "company": "Company Name",
      "role": "Role Title",
      "location": "Location / Remote",
      "start_date": "Start Date",
      "end_date": "End Date",
      "bullets": [
        "Action-oriented bullet point emphasizing relevant achievements and keywords"
      ]
    }
  ],
  "projects": [
    {
      "name": "Project Name",
      "technologies": ["Relevant tech stack"],
      "start_date": "Start Date",
      "end_date": "End Date",
      "bullets": [
        "Key achievement or technical outcome"
      ]
    }
  ],
  "education": [
    {
      "degree": "Degree and Major",
      "institution": "University / College",
      "location": "City, Country",
      "start_year": "Start Year",
      "end_year": "End Year",
      "details": "CGPA, honors, or relevant details"
    }
  ],
  "certifications": [
    {
      "name": "Certification Name",
      "issuer": "Issuing Organization"
    }
  ],
  "tailoring_metadata": {
    "target_role": "Target Role",
    "primary_keywords_integrated": ["Keywords naturally integrated"],
    "tailoring_summary": "Brief explanation of how the resume was tailored for this role"
  }
}"""


def load_prompt_template(prompt_path: str | Path | None = None) -> str:
    """
    Load the resume generation prompt template from disk or use the fallback.
    """
    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "resume_generation_prompt.txt"
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


def validate_and_normalize_resume_structure(
    extracted_data: dict[str, Any], master_resume_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Ensure all required schema fields are present with correct types,
    filling in defaults from the master resume where appropriate.
    """
    normalized: dict[str, Any] = {}

    # 1. Personal Info
    p_info = extracted_data.get("personal_info", {})
    master_cand = master_resume_data.get("candidate", {})
    master_p_info = master_cand.get("personal_info", {})
    master_prof = master_cand.get("professional_identity", {})

    if not isinstance(p_info, dict):
        p_info = {}

    normalized["personal_info"] = {
        "name": p_info.get("name") or master_p_info.get("name", ""),
        "target_title": p_info.get("target_title") or master_prof.get("current_profile", ""),
        "email": p_info.get("email") or master_p_info.get("email", ""),
        "phone": p_info.get("phone") or master_p_info.get("phone", ""),
        "location": p_info.get("location") or master_p_info.get("location", ""),
        "linkedin": p_info.get("linkedin") or master_p_info.get("linkedin", ""),
        "github": p_info.get("github") or master_p_info.get("github", ""),
    }

    # 2. Summary
    summary = extracted_data.get("summary")
    if isinstance(summary, str) and summary.strip():
        normalized["summary"] = summary.strip()
    else:
        normalized["summary"] = master_prof.get("profile", "")

    # 3. Skills
    skills = extracted_data.get("skills", {})
    if not isinstance(skills, dict):
        skills = {}
    master_skills = master_resume_data.get("capabilities", {}).get("skills", {})
    normalized["skills"] = {
        "technical_skills": skills.get("technical_skills") or master_skills.get("ai_llm", []),
        "tools_and_technologies": skills.get("tools_and_technologies") or master_skills.get("tools_platforms", []),
        "core_competencies": skills.get("core_competencies") or master_skills.get("core", []),
    }

    # 4. Experience
    exp_list = extracted_data.get("experience")
    if isinstance(exp_list, list) and exp_list:
        normalized["experience"] = exp_list
    else:
        # Fallback to master resume experience format
        normalized["experience"] = []
        for exp in master_resume_data.get("experience", []):
            bullets = [
                item.get("text", "")
                for item in exp.get("responsibilities_and_achievements", [])
                if item.get("text")
            ]
            normalized["experience"].append({
                "company": exp.get("company", ""),
                "role": exp.get("role", ""),
                "location": exp.get("location", ""),
                "start_date": exp.get("start_date", ""),
                "end_date": exp.get("end_date", ""),
                "bullets": bullets,
            })

    # 5. Projects
    proj_list = extracted_data.get("projects")
    if isinstance(proj_list, list) and proj_list:
        normalized["projects"] = proj_list
    else:
        normalized["projects"] = []
        for proj in master_resume_data.get("projects", []):
            bullets = [
                item.get("text", "")
                for item in proj.get("responsibilities_and_achievements", [])
                if item.get("text")
            ]
            normalized["projects"].append({
                "name": proj.get("name", ""),
                "technologies": proj.get("technologies", []),
                "start_date": proj.get("start_date", ""),
                "end_date": proj.get("end_date", ""),
                "bullets": bullets,
            })

    # 6. Education
    edu_list = extracted_data.get("education")
    if isinstance(edu_list, list) and edu_list:
        normalized["education"] = edu_list
    else:
        normalized["education"] = []
        for edu in master_resume_data.get("education", []):
            normalized["education"].append({
                "degree": edu.get("degree", ""),
                "institution": edu.get("institution", ""),
                "location": edu.get("location", ""),
                "start_year": str(edu.get("start_year", "")),
                "end_year": str(edu.get("end_year", "")),
                "details": f"CGPA: {edu.get('cgpa')}" if edu.get("cgpa") else "",
            })

    # 7. Certifications
    cert_list = extracted_data.get("certifications")
    if isinstance(cert_list, list) and cert_list:
        normalized["certifications"] = cert_list
    else:
        normalized["certifications"] = master_resume_data.get("capabilities", {}).get("certifications", [])

    # 8. Tailoring Metadata
    meta = extracted_data.get("tailoring_metadata")
    if isinstance(meta, dict):
        normalized["tailoring_metadata"] = meta
    else:
        normalized["tailoring_metadata"] = {
            "target_role": normalized["personal_info"]["target_title"],
            "primary_keywords_integrated": [],
            "tailoring_summary": "Tailored from master resume",
        }

    return normalized


def _clean_resume_for_prompt(resume_data: dict[str, Any]) -> dict[str, Any]:
    """
    Strip internal generation meta-rules while preserving all candidate factual data.
    """
    cleaned = dict(resume_data)
    cleaned.pop("generation_rules", None)
    return cleaned


def generate_tailored_resume(
    master_resume: dict[str, Any] | str | Path,
    structured_jd: dict[str, Any] | str | Path,
    matching_analysis: dict[str, Any] | str | Path | None = None,
    revision_feedback: str | None = None,
    improvement_actions: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    prompt_path: str | Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Generate a tailored structured resume JSON aligned with the target JD, matching analysis, and improvement actions.

    Parameters:
        master_resume: Master resume dict or path to JSON file.
        structured_jd: Structured JD dict or path to JSON file / JSON string.
        matching_analysis: Optional matching analysis dict or JSON string.
        revision_feedback: Optional feedback from previous evaluation loop for revisions.
        improvement_actions: Optional structured list of evaluator improvement action dictionaries.
        model: Ollama model name (default: "qwen3.5:4b").
        ollama_url: Ollama API endpoint.
        prompt_path: Optional custom prompt template path.
        timeout: Timeout in seconds.

    Returns:
        Structured tailored resume dictionary conforming to schema.
    """
    # 1. Resolve Master Resume
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

    # 3. Resolve Matching Analysis
    if matching_analysis is None:
        matching_analysis_str = "No prior matching analysis provided."
    elif isinstance(matching_analysis, (str, Path)) and Path(str(matching_analysis)).exists():
        with open(matching_analysis, "r", encoding="utf-8") as f:
            matching_analysis_dict = json.load(f)
        matching_analysis_str = json.dumps(matching_analysis_dict, separators=(',', ':'))
    elif isinstance(matching_analysis, dict):
        matching_analysis_str = json.dumps(matching_analysis, separators=(',', ':'))
    elif isinstance(matching_analysis, str):
        matching_analysis_str = matching_analysis
    else:
        matching_analysis_str = str(matching_analysis)

    # 4. Format Revision Feedback & Improvement Actions Section
    feedback_blocks = []
    if improvement_actions:
        action_lines = ["Structured Improvement Actions from Evaluator (Apply each action where factually supported by candidate evidence):"]
        for idx, act in enumerate(improvement_actions, 1):
            if isinstance(act, dict):
                target = str(act.get("target", "general")).upper()
                change = str(act.get("change", ""))
                reason = str(act.get("reason", ""))
                action_lines.append(f"{idx}. [{target}] Action: {change} (Rationale: {reason})")
            elif isinstance(act, str) and act.strip():
                action_lines.append(f"{idx}. {act.strip()}")
        feedback_blocks.append("\n".join(action_lines))

    if revision_feedback and revision_feedback.strip():
        feedback_blocks.append(f"Additional Evaluator Feedback:\n{revision_feedback.strip()}")

    if feedback_blocks:
        feedback_section = (
            "Revision Instructions & Improvement Actions:\n\"\"\"\n"
            + "\n\n".join(feedback_blocks)
            + "\n\"\"\"\nCRITICAL: Apply each of the improvement actions above directly to the relevant resume sections, but preserve strict factual truth (never invent unsupported metrics, experiences, or tools)."
        )
    else:
        feedback_section = ""

    # 5. Load prompt template and safely inject inputs
    template = load_prompt_template(prompt_path)
    prompt = template.replace("{master_resume_json}", master_resume_str)
    prompt = prompt.replace("{structured_jd_json}", structured_jd_str)
    prompt = prompt.replace("{matching_analysis_json}", matching_analysis_str)
    prompt = prompt.replace("{revision_feedback_section}", feedback_section)

    # 6. Call Ollama
    raw_response = call_ollama(
        prompt=prompt,
        model=model,
        api_url=ollama_url,
        timeout=timeout,
    )

    # 7. Parse response safely with graceful fallback
    try:
        parsed_json = parse_json_response(raw_response)
    except Exception:
        parsed_json = {}

    # 8. Validate and normalize structure against schema (fills in defaults from master resume)
    tailored_resume = validate_and_normalize_resume_structure(
        extracted_data=parsed_json,
        master_resume_data=master_resume_dict,
    )

    return tailored_resume


if __name__ == "__main__":
    master_resume_file = Path(__file__).resolve().parent.parent / "data" / "master_resume.json"

    if not master_resume_file.exists():
        raise FileNotFoundError(f"Master resume file not found at {master_resume_file}")

    with open(master_resume_file, "r", encoding="utf-8") as f:
        master_resume_data = json.load(f)

    # Target Structured Job Description
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

    # Sample matching analysis context
    sample_matching_analysis = {
        "overall_match_score": 82,
        "matched_required_skills": [
            {"skill": "Data annotation", "evidence": "Multimodal AI annotation experience (Text, Audio, Video, Image) at Innodata.", "match_type": "strong"},
            {"skill": "Attention to detail", "evidence": "Quality benchmarking, SOP adherence, confidential data handling.", "match_type": "strong"},
            {"skill": "Video annotation", "evidence": "Multimodal AI annotation including video samples.", "match_type": "strong"},
            {"skill": "Audio annotation", "evidence": "Multimodal AI annotation including audio samples.", "match_type": "strong"},
            {"skill": "Annotation tools", "evidence": "Experience with CVAT, JAX, Flax, Power BI.", "match_type": "strong"},
        ],
        "transferable_skills": [
            {"jd_requirement": "Data labeling platforms", "candidate_evidence": "CVAT, GCP", "reason": "Directly transferable toolset.", "match_type": "transferable"}
        ],
        "keywords_to_emphasize": [
            "Data annotation", "Video annotation", "Audio annotation", "Data labeling", "Quality assurance", "Edge cases", "Remote work"
        ]
    }

    print("=" * 70)
    print("RESDEV AI - TAILORED RESUME GENERATION ENGINE")
    print("=" * 70)
    print("Candidate:", master_resume_data.get("candidate", {}).get("personal_info", {}).get("name"))
    print("Target Role:", target_structured_jd.get("job_title"))
    print("Generating tailored resume with local Ollama (qwen3.5:4b)...")
    print("-" * 70)

    tailored_resume = generate_tailored_resume(
        master_resume=master_resume_data,
        structured_jd=target_structured_jd,
        matching_analysis=sample_matching_analysis,
    )

    print("\n[TAILORED STRUCTURED RESUME JSON]")
    print(json.dumps(tailored_resume, indent=2))

    print("\n" + "=" * 70)
    print("GENERATED RESUME SUMMARY")
    print("=" * 70)
    print("Target Title:", tailored_resume.get("personal_info", {}).get("target_title"))
    print("\nProfessional Summary:\n", tailored_resume.get("summary"))
    print("\nTechnical Skills:", ", ".join(tailored_resume.get("skills", {}).get("technical_skills", [])))
    print("Tools & Technologies:", ", ".join(tailored_resume.get("skills", {}).get("tools_and_technologies", [])))
    print("Core Competencies:", ", ".join(tailored_resume.get("skills", {}).get("core_competencies", [])))

    print("\nExperience Highlights:")
    for exp in tailored_resume.get("experience", []):
        print(f"  * {exp.get('role')} at {exp.get('company')} ({exp.get('start_date')} - {exp.get('end_date')}):")
        for b in exp.get("bullets", [])[:2]:
            print(f"    - {b}")

    print("\nTailoring Metadata:")
    meta = tailored_resume.get("tailoring_metadata", {})
    print(f"  Target Role: {meta.get('target_role')}")
    print(f"  Integrated Keywords: {', '.join(meta.get('primary_keywords_integrated', []))}")
    print(f"  Summary: {meta.get('tailoring_summary')}")
    print("=" * 70)
