"""
ResDev AI - Controlled Edit Plan & Requirement Matrix

Converts the optimization loop from unconstrained full-resume regeneration
into a targeted, evidence-grounded edit plan.

Pipeline:
    Structured JD + Master Resume
        ↓
    Requirement Matrix (links JD requirements to Master Resume evidence)
        ↓
    Identify Unmet Valid Gaps (ONLY requirements supported by Master Resume)
        ↓
    Structured Edit Plan (targeted edits with source evidence IDs)
        ↓
    Deterministic Validation & Application of Edits
"""

import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.ats_analyzer import _normalize_text, _extract_all_resume_text, _is_phrase_present
from ai.evidence_validator import (
    extract_master_evidence,
    _is_tool_supported,
    validate_resume_evidence,
    _extract_numbers_and_metrics,
)
from ai.gemini_config import call_gemini_with_retry, DEFAULT_MODEL, DEFAULT_TIMEOUT

DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT


# JSON Schema for Ollama structured output
EDIT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "section": {"type": "string"},
                    "target_id": {"type": "string"},
                    "action": {"type": "string"},
                    "target_requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "source_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "original_text": {"type": "string"},
                    "proposed_text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["section", "action", "proposed_text"],
            },
        },
    },
    "required": ["edits"],
}


def build_requirement_matrix(
    structured_jd: dict[str, Any],
    master_resume: dict[str, Any],
    current_resume: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build a comprehensive requirement matrix linking each JD requirement to:
      - Master Resume evidence (supported vs unsupported)
      - Current resume presence (present vs missing)
      - Allowed action ('rewrite_existing_evidence' vs 'do_not_add')
    """
    master_evidence = extract_master_evidence(master_resume)
    all_allowed_terms = (
        master_evidence["tools_and_technologies"]
        | master_evidence["skills"]
        | master_evidence["roles"]
    )
    for text in master_evidence["verifiable_texts"]:
        for token in _normalize_text(text).split():
            if len(token) > 3:
                all_allowed_terms.add(token)

    current_resume_text = _normalize_text(_extract_all_resume_text(current_resume))
    matrix: list[dict[str, Any]] = []
    seen_reqs: set[str] = set()

    req_index = 1

    def _process_item(item_text: str, category: str, importance: str):
        nonlocal req_index
        text_str = str(item_text).strip()
        norm = _normalize_text(text_str)
        if not text_str or norm in seen_reqs:
            return
        seen_reqs.add(norm)

        # 1. Determine Evidence Support in Master Resume
        is_supported = False
        evidence_ids: list[str] = []

        if norm in master_evidence["tools_and_technologies"] or _is_tool_supported(text_str, master_evidence["tools_and_technologies"]):
            is_supported = True
            evidence_ids.append("master_tools")
        elif norm in master_evidence["skills"]:
            is_supported = True
            evidence_ids.append("master_skills")
        elif any(norm in term or term in norm for term in all_allowed_terms):
            is_supported = True
            evidence_ids.append("master_capabilities")
        elif any(_is_phrase_present(text_str, _normalize_text(vt)) for vt in master_evidence["verifiable_texts"]):
            is_supported = True
            evidence_ids.append("master_experience_texts")

        # 2. Determine Current Resume Status
        is_present = _is_phrase_present(text_str, current_resume_text)

        # 3. Determine Allowed Action
        if is_supported:
            allowed_action = "rewrite_existing_evidence"
            evidence_status = "supported"
        else:
            allowed_action = "do_not_add"
            evidence_status = "unsupported"

        entry = {
            "id": f"REQ_{req_index:03d}",
            "text": text_str,
            "category": category,
            "importance": importance,
            "evidence_status": evidence_status,
            "evidence_ids": evidence_ids,
            "current_resume_status": "present" if is_present else "missing",
            "allowed_action": allowed_action,
        }
        matrix.append(entry)
        req_index += 1

    # Add Required Skills
    for sk in structured_jd.get("required_skills", []):
        _process_item(sk, category="required_skill", importance="required")

    # Add Keywords
    for kw in structured_jd.get("keywords") or structured_jd.get("required_keywords") or []:
        _process_item(kw, category="keyword", importance="required")

    # Add Preferred Skills
    for ps in structured_jd.get("preferred_skills", []):
        _process_item(ps, category="preferred_skill", importance="preferred")

    return matrix


def get_unmet_valid_gaps(requirement_matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter requirement matrix to ONLY gaps that are:
      1. Supported by Master Resume evidence, AND
      2. Currently missing in the active resume.
    """
    return [
        r for r in requirement_matrix
        if r.get("evidence_status") == "supported"
        and r.get("current_resume_status") == "missing"
    ]


def get_unsupported_gaps(requirement_matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return requirements that cannot be added because no Master Resume evidence exists.
    """
    return [
        r for r in requirement_matrix
        if r.get("evidence_status") == "unsupported"
        and r.get("current_resume_status") == "missing"
    ]


def generate_targeted_edit_plan(
    current_resume: dict[str, Any],
    master_resume: dict[str, Any],
    unmet_valid_gaps: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    # Legacy Ollama params silently ignored
    **_kwargs: object,
) -> dict[str, Any]:
    """
    Call Gemini to produce a targeted edit plan for valid unmet gaps.
    Falls back to a deterministic plan if Gemini is unavailable.
    """
    if not unmet_valid_gaps:
        return {"edits": []}

    master_evidence = extract_master_evidence(master_resume)

    # Build prompt instructions
    gaps_summary = "\n".join(
        f"- [{g['id']}] ({g['category']}): '{g['text']}' -> Supported by evidence: {', '.join(g['evidence_ids'])}"
        for g in unmet_valid_gaps[:4]
    )

    allowed_tools = ", ".join(list(master_evidence["tools_and_technologies"])[:10])
    allowed_skills = ", ".join(list(master_evidence["skills"])[:10])

    prompt = f"""You are a Precise Resume Edit Plan Specialist.
Your task is to generate a structured JSON edit plan to integrate ONLY the following supported missing requirements into the current resume.

UNMET SUPPORTED REQUIREMENTS TO INTEGRATE:
{gaps_summary}

VERIFIED ALLOWED EVIDENCE (DO NOT INVENT ANYTHING OUTSIDE THIS):
Allowed Tools: {allowed_tools}
Allowed Skills: {allowed_skills}

CURRENT RESUME:
Summary: {current_resume.get('summary', '')}
Skills: {json.dumps(current_resume.get('skills', {}))}
Experience: {json.dumps(current_resume.get('experience', []))}

STRICT RULES:
1. Target ONLY specific sections: 'summary', 'skills', 'experience'.
2. Action can be 'rewrite', 'add_skill', or 'replace_bullet'.
3. Use exact requirement keywords ONLY where candidate background proves them.
4. NEVER invent tools (e.g. Scale AI, Labelbox), employers, dates, or metrics.
5. Return ONLY a valid JSON object matching this EXACT structure (no markdown, no prose):
{{"edits":[{{"id":"EDIT_001","section":"skills","target_id":"skills.technical_skills","action":"add_skill","target_requirements":["REQ_001"],"source_evidence_ids":["master_skills"],"original_text":"","proposed_text":"Keyword here","reason":"Reason here"}}]}}
"""

    try:
        raw_response = call_gemini_with_retry(
            prompt=prompt,
            model=model,
            timeout=timeout,
        )
        # Strip markdown fences if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            import re as _re
            cleaned = _re.sub(r"^```(?:json)?\s*", "", cleaned, flags=_re.IGNORECASE)
            cleaned = _re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "edits" in parsed:
            return parsed
    except Exception:
        pass

    # Deterministic fallback edit plan if Gemini is unavailable or returns bad JSON
    fallback_edits: list[dict[str, Any]] = []
    for g in unmet_valid_gaps[:3]:
        req_text = g["text"]
        if "skill" in g["category"]:
            fallback_edits.append({
                "id": f"EDIT_{g['id']}",
                "section": "skills",
                "target_id": "skills.technical_skills",
                "action": "add_skill",
                "target_requirements": [g["id"]],
                "source_evidence_ids": g["evidence_ids"],
                "original_text": "",
                "proposed_text": req_text,
                "reason": f"Integrate verified requirement '{req_text}'",
            })
        elif "keyword" in g["category"]:
            fallback_edits.append({
                "id": f"EDIT_{g['id']}",
                "section": "skills",
                "target_id": "skills.tools_and_technologies" if _is_tool_supported(req_text, master_evidence["tools_and_technologies"]) else "skills.technical_skills",
                "action": "add_skill",
                "target_requirements": [g["id"]],
                "source_evidence_ids": g["evidence_ids"],
                "original_text": "",
                "proposed_text": req_text,
                "reason": f"Integrate verified keyword '{req_text}'",
            })

    return {"edits": fallback_edits}



def apply_edit_plan(
    current_resume: dict[str, Any],
    edit_plan: dict[str, Any],
    master_resume: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Deterministically validate each edit against evidence before applying it.
    Rejects any edit that introduces unsupported tools, employers, or metrics.

    Returns:
        (updated_resume, applied_edits_log, rejected_edits_log)
    """
    updated = copy.deepcopy(current_resume)
    applied_log: list[str] = []
    rejected_log: list[str] = []

    master_evidence = extract_master_evidence(master_resume)
    edits = edit_plan.get("edits", [])
    if not isinstance(edits, list):
        return updated, applied_log, rejected_log

    for edit in edits:
        if not isinstance(edit, dict):
            continue

        section = edit.get("section", "").lower()
        action = edit.get("action", "").lower()
        target_id = edit.get("target_id", "")
        proposed = str(edit.get("proposed_text", "")).strip()

        if action == "do_not_add" or not proposed:
            continue

        # 1. Evidence validation on proposed text
        # If adding a tool/skill, check that it is supported
        if section == "skills" or "skill" in action:
            if not _is_tool_supported(proposed, master_evidence["tools_and_technologies"]) and proposed.lower() in {
                "scale ai", "labelbox", "amazon mechanical turk", "mturk", "superannotate", "v7 labs"
            }:
                rejected_log.append(f"Rejected unsupported tool addition: '{proposed}'")
                continue

        # Check for fabricated metrics in bullet additions
        if "bullet" in action or section in ("experience", "projects"):
            new_metrics = _extract_numbers_and_metrics(proposed)
            has_bad_metric = False
            for m in new_metrics:
                if m not in ("1", "2", "3") and m not in master_evidence["metrics"]:
                    if any(char in m for char in ("%", "$", "+", "k", "m")) or len(re.sub(r"\D", "", m)) >= 4:
                        has_bad_metric = True
                        rejected_log.append(f"Rejected bullet with unsupported metric '{m}': '{proposed}'")
                        break
            if has_bad_metric:
                continue

        # 2. Apply verified edit
        if section == "summary":
            updated["summary"] = proposed
            applied_log.append("Updated professional summary")

        elif section == "skills":
            skills_dict = updated.setdefault("skills", {})
            if isinstance(skills_dict, dict):
                target_category = "technical_skills"
                if "tool" in target_id or _is_tool_supported(proposed, master_evidence["tools_and_technologies"]):
                    target_category = "tools_and_technologies"
                elif "core" in target_id or "competenc" in target_id:
                    target_category = "core_competencies"

                current_list = skills_dict.setdefault(target_category, [])
                if isinstance(current_list, list) and proposed not in current_list:
                    current_list.append(proposed)
                    applied_log.append(f"Added verified skill to {target_category}: '{proposed}'")

        elif section == "experience":
            # Target experience bullet update
            experiences = updated.setdefault("experience", [])
            if isinstance(experiences, list) and experiences:
                # Add or update bullet in first experience role
                first_exp = experiences[0]
                if isinstance(first_exp, dict):
                    bullets = first_exp.setdefault("bullets", [])
                    if isinstance(bullets, list):
                        if action == "replace_bullet" and bullets:
                            bullets[0] = proposed
                            applied_log.append(f"Replaced experience bullet: '{proposed[:50]}...'")
                        else:
                            bullets.append(proposed)
                            applied_log.append(f"Added experience bullet: '{proposed[:50]}...'")

    # Post-validation check on entire resume
    full_val = validate_resume_evidence(updated, master_resume)
    if not full_val["passed"]:
        # Revert entirely if global validation fails
        rejected_log.append(f"Global validation failed with {full_val['violation_count']} violations -> Reverted")
        return copy.deepcopy(current_resume), [], rejected_log

    return updated, applied_log, rejected_log


if __name__ == "__main__":
    sample_master_path = Path(__file__).resolve().parent.parent / "data" / "master_resume.json"
    if sample_master_path.exists():
        with open(sample_master_path, "r", encoding="utf-8") as f:
            m_res = json.load(f)

        test_jd = {
            "job_title": "Data Annotator",
            "required_skills": ["Data annotation", "CVAT", "Scale AI", "Video annotation"],
            "preferred_skills": ["Remote work experience", "Labelbox"],
            "keywords": ["Quality assurance", "Prompt Engineering"],
        }

        test_resume = {
            "personal_info": {"name": "Kanishk Surwade", "target_title": "Data Annotator"},
            "summary": "AI & LLM Analyst with experience in annotation.",
            "skills": {
                "technical_skills": ["Data Annotation"],
                "tools_and_technologies": ["Python", "SQL"],
            },
            "experience": [{"company": "Innodata Inc.", "role": "AI & LLM Analyst", "bullets": ["Annotated data."]}],
        }

        matrix = build_requirement_matrix(test_jd, m_res, test_resume)
        valid_gaps = get_unmet_valid_gaps(matrix)
        unsupported = get_unsupported_gaps(matrix)

        print("=== REQUIREMENT MATRIX ===")
        for r in matrix:
            print(f"[{r['id']}] {r['text']:<22} | Support: {r['evidence_status']:<11} | Current: {r['current_resume_status']:<7} | Action: {r['allowed_action']}")

        print(f"\nUnmet Valid Gaps: {len(valid_gaps)}")
        for vg in valid_gaps:
            print(f"  [+] {vg['text']}")

        print(f"\nUnsupported JD Gaps (Correctly blocked): {len(unsupported)}")
        for ug in unsupported:
            print(f"  [-] {ug['text']} -> Action: {ug['allowed_action']}")
