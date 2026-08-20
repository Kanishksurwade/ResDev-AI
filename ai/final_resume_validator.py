"""
ResDev AI - Final Structured Resume Output & Schema Validator

Defines the strict output contract and validator for the final accepted resume candidate.
This structured JSON is the deterministic single source of truth for the downstream LaTeX/PDF generators.

Schema Contract:
{
  "personal_info": {
    "name": str,
    "email": str,
    "phone": str,
    "location": str,
    "linkedin": str,
    "github": str
  },
  "target_role": str,
  "summary": str,
  "skills": {
    "technical": list[str],
    "tools": list[str],
    "soft": list[str]
  },
  "experience": [
    {
      "company": str,
      "role": str,
      "location": str,
      "start_date": str,
      "end_date": str,
      "bullets": list[str]
    }
  ],
  "education": [
    {
      "institution": str,
      "degree": str,
      "field": str,
      "start_date": str,
      "end_date": str
    }
  ],
  "projects": [
    {
      "name": str,
      "description": str,
      "technologies": list[str],
      "bullets": list[str]
    }
  ],
  "certifications": list[Any],
  "achievements": list[str],
  "additional_sections": list[Any]
}
"""

import copy
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = REPO_ROOT / "generated" / "final_resume.json"

REQUIRED_TOP_LEVEL_FIELDS = [
    "personal_info",
    "target_role",
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
    "achievements",
    "additional_sections",
]

REQUIRED_PERSONAL_INFO_FIELDS = [
    "name",
    "email",
    "phone",
    "location",
    "linkedin",
    "github",
]

REQUIRED_SKILLS_FIELDS = [
    "technical",
    "tools",
    "soft",
]

REQUIRED_EXPERIENCE_FIELDS = [
    "company",
    "role",
    "location",
    "start_date",
    "end_date",
    "bullets",
]

REQUIRED_EDUCATION_FIELDS = [
    "institution",
    "degree",
    "field",
    "start_date",
    "end_date",
]

REQUIRED_PROJECT_FIELDS = [
    "name",
    "description",
    "technologies",
    "bullets",
]


def _clean_whitespace(text: str) -> str:
    """
    Safely normalize whitespace in a string while preserving meaningful punctuation.
    """
    if not isinstance(text, str):
        return ""
    # Collapse multiple inline spaces/tabs without stripping newlines if multi-line
    lines = text.splitlines()
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    return "\n".join(cleaned_lines).strip()


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    """
    Remove duplicate strings while preserving insertion order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = _clean_whitespace(item)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def normalize_resume_data(resume: dict[str, Any]) -> dict[str, Any]:
    """
    Apply safe, deterministic normalizations to a resume dictionary:
      - Strip leading/trailing whitespace.
      - Remove empty strings from list fields.
      - Deduplicate identical bullets within experiences and projects.
      - Preserve factual dates and punctuation verbatim.
    """
    if not isinstance(resume, dict):
        return resume

    normalized = copy.deepcopy(resume)

    # 1. Personal info
    if isinstance(normalized.get("personal_info"), dict):
        p_info = normalized["personal_info"]
        for k in REQUIRED_PERSONAL_INFO_FIELDS:
            if k in p_info and isinstance(p_info[k], str):
                p_info[k] = _clean_whitespace(p_info[k])

    # 2. Target role & summary
    if isinstance(normalized.get("target_role"), str):
        normalized["target_role"] = _clean_whitespace(normalized["target_role"])

    if isinstance(normalized.get("summary"), str):
        normalized["summary"] = _clean_whitespace(normalized["summary"])

    # 3. Skills
    if isinstance(normalized.get("skills"), dict):
        s_dict = normalized["skills"]
        for cat in REQUIRED_SKILLS_FIELDS:
            if cat in s_dict and isinstance(s_dict[cat], list):
                s_dict[cat] = _deduplicate_preserve_order(
                    [item for item in s_dict[cat] if isinstance(item, str)]
                )

    # 4. Experience
    if isinstance(normalized.get("experience"), list):
        clean_exp = []
        for exp in normalized["experience"]:
            if isinstance(exp, dict):
                exp_entry = copy.deepcopy(exp)
                for field in ["company", "role", "location", "start_date", "end_date"]:
                    if field in exp_entry and isinstance(exp_entry[field], str):
                        exp_entry[field] = _clean_whitespace(exp_entry[field])
                if "bullets" in exp_entry and isinstance(exp_entry["bullets"], list):
                    exp_entry["bullets"] = _deduplicate_preserve_order(
                        [b for b in exp_entry["bullets"] if isinstance(b, str)]
                    )
                clean_exp.append(exp_entry)
            else:
                clean_exp.append(exp)
        normalized["experience"] = clean_exp

    # 5. Education
    if isinstance(normalized.get("education"), list):
        clean_edu = []
        for edu in normalized["education"]:
            if isinstance(edu, dict):
                edu_entry = copy.deepcopy(edu)
                for field in ["institution", "degree", "field", "start_date", "end_date"]:
                    if field in edu_entry and isinstance(edu_entry[field], str):
                        edu_entry[field] = _clean_whitespace(edu_entry[field])
                clean_edu.append(edu_entry)
            else:
                clean_edu.append(edu)
        normalized["education"] = clean_edu

    # 6. Projects
    if isinstance(normalized.get("projects"), list):
        clean_proj = []
        for proj in normalized["projects"]:
            if isinstance(proj, dict):
                proj_entry = copy.deepcopy(proj)
                if "name" in proj_entry and isinstance(proj_entry["name"], str):
                    proj_entry["name"] = _clean_whitespace(proj_entry["name"])
                if "description" in proj_entry and isinstance(proj_entry["description"], str):
                    proj_entry["description"] = _clean_whitespace(proj_entry["description"])
                if "technologies" in proj_entry and isinstance(proj_entry["technologies"], list):
                    proj_entry["technologies"] = _deduplicate_preserve_order(
                        [t for t in proj_entry["technologies"] if isinstance(t, str)]
                    )
                if "bullets" in proj_entry and isinstance(proj_entry["bullets"], list):
                    proj_entry["bullets"] = _deduplicate_preserve_order(
                        [b for b in proj_entry["bullets"] if isinstance(b, str)]
                    )
                clean_proj.append(proj_entry)
            else:
                clean_proj.append(proj)
        normalized["projects"] = clean_proj

    # 7. Certifications, Achievements, Additional
    for list_sec in ["certifications", "achievements", "additional_sections"]:
        if isinstance(normalized.get(list_sec), list):
            cleaned_list = []
            for item in normalized[list_sec]:
                if isinstance(item, str):
                    cleaned_str = _clean_whitespace(item)
                    if cleaned_str:
                        cleaned_list.append(cleaned_str)
                elif isinstance(item, dict):
                    cleaned_dict = {}
                    for dk, dv in item.items():
                        cleaned_dict[dk] = _clean_whitespace(dv) if isinstance(dv, str) else dv
                    cleaned_list.append(cleaned_dict)
                else:
                    cleaned_list.append(item)
            normalized[list_sec] = cleaned_list

    return normalized


def build_final_resume(candidate_resume: dict[str, Any], target_role: str = "") -> dict[str, Any]:
    """
    Map an optimizer/generator candidate resume into the exact Final Structured Resume contract.
    Preserves all authentic factual content without inventing new items or hallucinating data.
    """
    if not isinstance(candidate_resume, dict):
        return {}

    p_info_raw = candidate_resume.get("personal_info", {})
    if not isinstance(p_info_raw, dict):
        p_info_raw = {}

    # Determine target role from candidate metadata or argument
    resolved_target_role = (
        target_role
        or candidate_resume.get("target_role")
        or p_info_raw.get("target_title")
        or candidate_resume.get("tailoring_metadata", {}).get("target_role", "")
        or ""
    )

    personal_info = {
        "name": p_info_raw.get("name", ""),
        "email": p_info_raw.get("email", ""),
        "phone": p_info_raw.get("phone", ""),
        "location": p_info_raw.get("location", ""),
        "linkedin": p_info_raw.get("linkedin", ""),
        "github": p_info_raw.get("github", ""),
    }

    # Map skills
    raw_skills = candidate_resume.get("skills", {})
    if not isinstance(raw_skills, dict):
        raw_skills = {}

    technical_skills = (
        raw_skills.get("technical")
        or raw_skills.get("technical_skills")
        or raw_skills.get("ai_llm")
        or []
    )
    tools_skills = (
        raw_skills.get("tools")
        or raw_skills.get("tools_and_technologies")
        or raw_skills.get("tools_platforms")
        or []
    )
    soft_skills = (
        raw_skills.get("soft")
        or raw_skills.get("core_competencies")
        or raw_skills.get("core")
        or []
    )

    skills = {
        "technical": [s for s in technical_skills if isinstance(s, str)],
        "tools": [s for s in tools_skills if isinstance(s, str)],
        "soft": [s for s in soft_skills if isinstance(s, str)],
    }

    # Map experience
    raw_exp = candidate_resume.get("experience", [])
    experience = []
    if isinstance(raw_exp, list):
        for item in raw_exp:
            if isinstance(item, dict):
                bullets = []
                if "bullets" in item and isinstance(item["bullets"], list):
                    bullets = [b for b in item["bullets"] if isinstance(b, str)]
                elif "responsibilities_and_achievements" in item and isinstance(item["responsibilities_and_achievements"], list):
                    for r in item["responsibilities_and_achievements"]:
                        if isinstance(r, dict) and "text" in r:
                            bullets.append(r["text"])
                        elif isinstance(r, str):
                            bullets.append(r)

                experience.append({
                    "company": item.get("company", ""),
                    "role": item.get("role", ""),
                    "location": item.get("location", ""),
                    "start_date": item.get("start_date", ""),
                    "end_date": item.get("end_date", ""),
                    "bullets": bullets,
                })

    # Map education
    raw_edu = candidate_resume.get("education", [])
    education = []
    if isinstance(raw_edu, list):
        for item in raw_edu:
            if isinstance(item, dict):
                education.append({
                    "institution": item.get("institution", ""),
                    "degree": item.get("degree", ""),
                    "field": item.get("field", "") or item.get("major", ""),
                    "start_date": item.get("start_date", "") or str(item.get("start_year", "")),
                    "end_date": item.get("end_date", "") or str(item.get("end_year", "")),
                })

    # Map projects
    raw_proj = candidate_resume.get("projects", [])
    projects = []
    if isinstance(raw_proj, list):
        for item in raw_proj:
            if isinstance(item, dict):
                tech = item.get("technologies", [])
                if not isinstance(tech, list):
                    tech = []
                bullets = item.get("bullets", [])
                if not isinstance(bullets, list):
                    bullets = []
                projects.append({
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "technologies": [t for t in tech if isinstance(t, str)],
                    "bullets": [b for b in bullets if isinstance(b, str)],
                })

    # Map certifications
    raw_certs = candidate_resume.get("certifications", [])
    certifications = []
    if isinstance(raw_certs, list):
        certifications = copy.deepcopy(raw_certs)

    # Map achievements
    raw_achievements = candidate_resume.get("achievements", [])
    achievements = []
    if isinstance(raw_achievements, list):
        achievements = copy.deepcopy(raw_achievements)

    # Map additional sections
    raw_additional = candidate_resume.get("additional_sections", [])
    additional_sections = []
    if isinstance(raw_additional, list):
        additional_sections = copy.deepcopy(raw_additional)

    structured_resume = {
        "personal_info": personal_info,
        "target_role": resolved_target_role,
        "summary": candidate_resume.get("summary", ""),
        "skills": skills,
        "experience": experience,
        "education": education,
        "projects": projects,
        "certifications": certifications,
        "achievements": achievements,
        "additional_sections": additional_sections,
    }

    return normalize_resume_data(structured_resume)


def validate_final_resume(resume: dict[str, Any], auto_normalize: bool = True) -> dict[str, Any]:
    """
    Strictly validate a structured resume dictionary against the Final Resume Output Contract.

    Returns:
        {
            "valid": bool,
            "errors": list[str],
            "warnings": list[str],
            "resume": dict
        }
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(resume, dict):
        return {
            "valid": False,
            "errors": ["Resume root must be a valid JSON dictionary / object."],
            "warnings": [],
            "resume": {},
        }

    # Normalize if requested
    validated_resume = normalize_resume_data(resume) if auto_normalize else copy.deepcopy(resume)

    # 1. Verify Top-Level Fields
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in validated_resume:
            errors.append(f"Missing required top-level field: '{field}'")

    # If top-level keys are present, validate them
    if "personal_info" in validated_resume:
        p_info = validated_resume["personal_info"]
        if not isinstance(p_info, dict):
            errors.append("Field 'personal_info' must be a dictionary object.")
        else:
            for pf in REQUIRED_PERSONAL_INFO_FIELDS:
                if pf not in p_info:
                    errors.append(f"Missing required field in personal_info: '{pf}'")
                elif not isinstance(p_info[pf], str):
                    errors.append(f"Field personal_info.'{pf}' must be a string.")
            if isinstance(p_info.get("name"), str) and not p_info["name"].strip():
                errors.append("Candidate name in personal_info must not be empty.")

    if "target_role" in validated_resume:
        if not isinstance(validated_resume["target_role"], str):
            errors.append("Field 'target_role' must be a string.")
        elif not validated_resume["target_role"].strip():
            warnings.append("Field 'target_role' is an empty string.")

    if "summary" in validated_resume:
        if not isinstance(validated_resume["summary"], str):
            errors.append("Field 'summary' must be a string.")
        elif not validated_resume["summary"].strip():
            warnings.append("Professional summary is empty.")

    if "skills" in validated_resume:
        skills = validated_resume["skills"]
        if not isinstance(skills, dict):
            errors.append("Field 'skills' must be a dictionary object.")
        else:
            for sk_cat in REQUIRED_SKILLS_FIELDS:
                if sk_cat not in skills:
                    errors.append(f"Missing required skill category: '{sk_cat}'")
                elif not isinstance(skills[sk_cat], list):
                    errors.append(f"Skill category '{sk_cat}' must be a list of strings.")
                else:
                    for item in skills[sk_cat]:
                        if not isinstance(item, str):
                            errors.append(f"Skill item '{item}' in '{sk_cat}' must be a string.")

    if "experience" in validated_resume:
        exp_list = validated_resume["experience"]
        if not isinstance(exp_list, list):
            errors.append("Field 'experience' must be a list of experience entries.")
        else:
            for idx, exp in enumerate(exp_list):
                if not isinstance(exp, dict):
                    errors.append(f"Experience entry #{idx+1} must be a dictionary object.")
                    continue
                for ef in REQUIRED_EXPERIENCE_FIELDS:
                    if ef not in exp:
                        errors.append(f"Experience entry #{idx+1} is missing required field: '{ef}'")
                    elif ef == "bullets":
                        if not isinstance(exp[ef], list):
                            errors.append(f"Experience entry #{idx+1} 'bullets' must be a list of strings.")
                        else:
                            for b_idx, bullet in enumerate(exp[ef]):
                                if not isinstance(bullet, str):
                                    errors.append(f"Experience entry #{idx+1}, bullet #{b_idx+1} must be a string.")
                    elif not isinstance(exp[ef], str):
                        errors.append(f"Experience entry #{idx+1} '{ef}' must be a string.")

    if "education" in validated_resume:
        edu_list = validated_resume["education"]
        if not isinstance(edu_list, list):
            errors.append("Field 'education' must be a list of education entries.")
        else:
            for idx, edu in enumerate(edu_list):
                if not isinstance(edu, dict):
                    errors.append(f"Education entry #{idx+1} must be a dictionary object.")
                    continue
                for edf in REQUIRED_EDUCATION_FIELDS:
                    if edf not in edu:
                        errors.append(f"Education entry #{idx+1} is missing required field: '{edf}'")
                    elif not isinstance(edu[edf], str):
                        errors.append(f"Education entry #{idx+1} '{edf}' must be a string.")

    if "projects" in validated_resume:
        proj_list = validated_resume["projects"]
        if not isinstance(proj_list, list):
            errors.append("Field 'projects' must be a list of project entries.")
        else:
            for idx, proj in enumerate(proj_list):
                if not isinstance(proj, dict):
                    errors.append(f"Project entry #{idx+1} must be a dictionary object.")
                    continue
                for prf in REQUIRED_PROJECT_FIELDS:
                    if prf not in proj:
                        errors.append(f"Project entry #{idx+1} is missing required field: '{prf}'")
                    elif prf in ["technologies", "bullets"]:
                        if not isinstance(proj[prf], list):
                            errors.append(f"Project entry #{idx+1} '{prf}' must be a list of strings.")
                        else:
                            for item in proj[prf]:
                                if not isinstance(item, str):
                                    errors.append(f"Project entry #{idx+1} '{prf}' item must be a string.")
                    elif not isinstance(proj[prf], str):
                        errors.append(f"Project entry #{idx+1} '{prf}' must be a string.")

    for opt_sec in ["certifications", "achievements", "additional_sections"]:
        if opt_sec in validated_resume:
            if not isinstance(validated_resume[opt_sec], list):
                errors.append(f"Field '{opt_sec}' must be a list.")

    is_valid = len(errors) == 0

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "resume": validated_resume,
    }


def save_final_resume(resume: dict[str, Any], output_path: str | Path | None = None) -> Path:
    """
    Save the validated final resume JSON structure to disk.
    """
    if output_path is None:
        target_file = DEFAULT_OUTPUT_PATH
    else:
        target_file = Path(output_path)

    # Ensure parent folder exists
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Save formatted JSON
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)

    return target_file


def load_final_resume(file_path: str | Path) -> dict[str, Any]:
    """
    Load a final resume JSON file from disk.
    """
    target_file = Path(file_path)
    if not target_file.exists():
        raise FileNotFoundError(f"Final resume JSON not found at {target_file}")

    with open(target_file, "r", encoding="utf-8") as f:
        return json.load(f)
