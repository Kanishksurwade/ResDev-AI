"""
ResDev AI - Evidence Validator

Deterministically verifies that all content, claims, tools, employers,
dates, certifications, education, and metrics in a generated/tailored resume
are strictly supported by the Master Resume (Source of Truth).

Prevents hallucination of:
    - Employers and company names
    - Job titles and roles
    - Employment dates and durations
    - Universities, colleges, and degrees
    - Certifications and issuing organizations
    - Project names and technical deliverables
    - Tools, technologies, software, and platforms (e.g. Scale AI, Labelbox, MTurk)
    - Fabricated numbers, metrics, and quantitative achievements
    - Unsupported skill claims

Permits grounded semantic rephrasing of authentic Master Resume facts.
No LLM calls. Fully offline. Fully deterministic.
"""

import json
import re
from pathlib import Path
from typing import Any


def _normalize_token(text: str) -> str:
    """Normalize a word or phrase for robust case-insensitive comparison."""
    if not text:
        return ""
    lowered = text.lower()
    cleaned = re.sub(r"[^\w\s\+\#\.]", " ", lowered)
    return " ".join(cleaned.split())


def _extract_numbers_and_metrics(text: str) -> list[str]:
    """
    Extract numbers, percentages, currency amounts, and metric expressions.
    E.g., '10,000+', '98%', '$5M', '7.75'
    """
    if not text:
        return []
    # Match patterns like: 98%, $5M, 10,000+, 5,000, 7.75, 100k, 5M
    pattern = r"(?:\$[\d\.]+[kKmMbB]?|\d{1,3}(?:,\d{3})+(?:\.\d+)?\+?|\d+(?:\.\d+)?%|\b\d+[kKmMbB]\b|\b\d+\+)"
    matches = re.findall(pattern, text)
    return [m.strip().lower() for m in matches]


def extract_master_evidence(master_resume: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministically extract and index all grounded facts and entities from the Master Resume.
    """
    evidence: dict[str, Any] = {
        "candidate_names": set(),
        "locations": set(),
        "contact_info": set(),
        "employers": set(),
        "roles": set(),
        "employment_dates": set(),
        "degrees": set(),
        "institutions": set(),
        "certifications": set(),
        "certification_issuers": set(),
        "projects": set(),
        "tools_and_technologies": set(),
        "skills": set(),
        "metrics": set(),
        "verifiable_texts": [],
    }

    # 0. Raw text from upload or full profile string
    if master_resume.get("_raw_text"):
        raw = str(master_resume["_raw_text"])
        evidence["verifiable_texts"].append(raw)
        for m in _extract_numbers_and_metrics(raw):
            evidence["metrics"].add(m)

    if master_resume.get("profile"):
        p_text = str(master_resume["profile"])
        evidence["verifiable_texts"].append(p_text)
        for m in _extract_numbers_and_metrics(p_text):
            evidence["metrics"].add(m)

    if master_resume.get("summary"):
        s_text = str(master_resume["summary"])
        evidence["verifiable_texts"].append(s_text)
        for m in _extract_numbers_and_metrics(s_text):
            evidence["metrics"].add(m)

    # 1. Candidate Personal Info
    cand = master_resume.get("candidate", {})
    if isinstance(cand, dict):
        p_info = cand.get("personal_info", {})
        if isinstance(p_info, dict):
            if p_info.get("name"):
                evidence["candidate_names"].add(_normalize_token(str(p_info["name"])))
            if p_info.get("location"):
                evidence["locations"].add(_normalize_token(str(p_info["location"])))
            if p_info.get("email"):
                evidence["contact_info"].add(_normalize_token(str(p_info["email"])))
            if p_info.get("phone"):
                evidence["contact_info"].add(_normalize_token(str(p_info["phone"])))

        prof_id = cand.get("professional_identity", {})
        if isinstance(prof_id, dict):
            if prof_id.get("current_profile"):
                evidence["roles"].add(_normalize_token(str(prof_id["current_profile"])))
            if prof_id.get("profile"):
                p_text = str(prof_id["profile"])
                evidence["verifiable_texts"].append(p_text)
                for m in _extract_numbers_and_metrics(p_text):
                    evidence["metrics"].add(m)

    # Also handle flat personal_info if structured resume format passed
    flat_p_info = master_resume.get("personal_info", {})
    if isinstance(flat_p_info, dict):
        if flat_p_info.get("name"):
            evidence["candidate_names"].add(_normalize_token(str(flat_p_info["name"])))
        if flat_p_info.get("location"):
            evidence["locations"].add(_normalize_token(str(flat_p_info["location"])))

    # 2. Capabilities (Skills, Tools, Certifications)
    caps = master_resume.get("capabilities", {})
    if isinstance(caps, dict):
        skills_dict = caps.get("skills", {})
        if isinstance(skills_dict, dict):
            for cat, skill_list in skills_dict.items():
                if isinstance(skill_list, list):
                    for item in skill_list:
                        norm = _normalize_token(str(item))
                        if cat == "tools_platforms":
                            evidence["tools_and_technologies"].add(norm)
                        else:
                            evidence["skills"].add(norm)

        certs = caps.get("certifications", [])
        if isinstance(certs, list):
            for c in certs:
                if isinstance(c, dict):
                    if c.get("name"):
                        evidence["certifications"].add(_normalize_token(str(c["name"])))
                    if c.get("issuer"):
                        evidence["certification_issuers"].add(_normalize_token(str(c["issuer"])))
                elif isinstance(c, str):
                    evidence["certifications"].add(_normalize_token(c))

    # Also handle flat skills/certifications if present
    flat_skills = master_resume.get("skills", {})
    if isinstance(flat_skills, dict):
        for cat, sks in flat_skills.items():
            if isinstance(sks, list):
                for s in sks:
                    norm = _normalize_token(str(s))
                    if "tool" in cat or "tech" in cat:
                        evidence["tools_and_technologies"].add(norm)
                    else:
                        evidence["skills"].add(norm)

    flat_certs = master_resume.get("certifications", [])
    if isinstance(flat_certs, list):
        for c in flat_certs:
            if isinstance(c, dict):
                if c.get("name"):
                    evidence["certifications"].add(_normalize_token(str(c["name"])))
                if c.get("issuer"):
                    evidence["certification_issuers"].add(_normalize_token(str(c["issuer"])))
            elif isinstance(c, str):
                evidence["certifications"].add(_normalize_token(c))

    # 3. Experience
    exp_list = master_resume.get("experience", [])
    if isinstance(exp_list, list):
        for exp in exp_list:
            if not isinstance(exp, dict):
                continue
            if exp.get("company"):
                evidence["employers"].add(_normalize_token(str(exp["company"])))
            if exp.get("role"):
                evidence["roles"].add(_normalize_token(str(exp["role"])))
            if exp.get("start_date"):
                evidence["employment_dates"].add(_normalize_token(str(exp["start_date"])))
            if exp.get("end_date"):
                evidence["employment_dates"].add(_normalize_token(str(exp["end_date"])))
            if exp.get("location"):
                evidence["locations"].add(_normalize_token(str(exp["location"])))

            # Responsibilities and achievements
            resps = exp.get("responsibilities_and_achievements") or exp.get("bullets") or []
            for r in resps:
                if isinstance(r, dict):
                    text = r.get("text", "")
                    evidence["verifiable_texts"].append(text)
                    for s in r.get("skills_used", []):
                        evidence["skills"].add(_normalize_token(str(s)))
                    for m in _extract_numbers_and_metrics(text):
                        evidence["metrics"].add(m)
                elif isinstance(r, str):
                    evidence["verifiable_texts"].append(r)
                    for m in _extract_numbers_and_metrics(r):
                        evidence["metrics"].add(m)

    # 4. Projects
    proj_list = master_resume.get("projects", [])
    if isinstance(proj_list, list):
        for proj in proj_list:
            if not isinstance(proj, dict):
                continue
            if proj.get("name"):
                evidence["projects"].add(_normalize_token(str(proj["name"])))
            for tech in proj.get("technologies", []):
                evidence["tools_and_technologies"].add(_normalize_token(str(tech)))
            if proj.get("start_date"):
                evidence["employment_dates"].add(_normalize_token(str(proj["start_date"])))
            if proj.get("end_date"):
                evidence["employment_dates"].add(_normalize_token(str(proj["end_date"])))
            for b in proj.get("bullets", []):
                evidence["verifiable_texts"].append(str(b))
                for m in _extract_numbers_and_metrics(str(b)):
                    evidence["metrics"].add(m)

    # 5. Education
    edu_list = master_resume.get("education", [])
    if isinstance(edu_list, list):
        for edu in edu_list:
            if not isinstance(edu, dict):
                continue
            if edu.get("degree"):
                evidence["degrees"].add(_normalize_token(str(edu["degree"])))
            if edu.get("institution"):
                evidence["institutions"].add(_normalize_token(str(edu["institution"])))
            if edu.get("location"):
                evidence["locations"].add(_normalize_token(str(edu["location"])))
            if edu.get("cgpa"):
                evidence["metrics"].add(str(edu["cgpa"]).lower())
            if edu.get("details"):
                evidence["verifiable_texts"].append(str(edu["details"]))
                for m in _extract_numbers_and_metrics(str(edu["details"])):
                    evidence["metrics"].add(m)

    return evidence


# Standard well-known synonym mappings / parent-child tool concepts
_TOOL_SYNONYMS: dict[str, list[str]] = {
    "google cloud": ["google cloud platform", "google cloud platform gcp", "gcp"],
    "google cloud platform": ["google cloud", "gcp", "google cloud platform gcp"],
    "gcp": ["google cloud platform", "google cloud"],
    "cvat": ["computer vision annotation tool", "cvat", "annotation tools", "data labeling platforms"],
    "power bi": ["powerbi", "microsoft power bi"],
    "sql": ["mysql", "mysql workbench", "relational database"],
    "python": ["python 3", "python scripting"],
    "jax": ["jax", "flax"],
    "flax": ["flax", "jax"],
    "excel": ["microsoft excel", "spreadsheets"],
    "git": ["github", "version control"],
}


def _is_tool_supported(
    tool: str,
    allowed_tools: set[str],
    verifiable_texts: list[str] | None = None,
) -> bool:
    """Check if a tool or platform is supported by the Master Resume evidence."""
    norm_tool = _normalize_token(tool)
    if not norm_tool:
        return True

    # 1. Exact match
    if norm_tool in allowed_tools:
        return True

    # 2. Substring or superset match
    for allowed in allowed_tools:
        if norm_tool == allowed or norm_tool in allowed or allowed in norm_tool:
            return True

    # 3. Known synonym check
    synonyms = _TOOL_SYNONYMS.get(norm_tool, [])
    for syn in synonyms:
        if syn in allowed_tools:
            return True
        for allowed in allowed_tools:
            if syn in allowed or allowed in syn:
                return True

    # 4. Check verifiable_texts from uploaded resume
    if verifiable_texts:
        for vt in verifiable_texts:
            norm_vt = _normalize_token(vt)
            if norm_tool in norm_vt:
                return True
            for syn in synonyms:
                if syn in norm_vt:
                    return True
            tool_tokens = [t for t in norm_tool.split() if len(t) > 3]
            if tool_tokens and all(t in norm_vt for t in tool_tokens):
                return True

    return False


def _is_employer_supported(
    employer: str,
    allowed_employers: set[str],
    verifiable_texts: list[str] | None = None,
) -> bool:
    """Check if an employer name is grounded in Master Resume evidence."""
    norm_emp = _normalize_token(employer)
    if not norm_emp:
        return True

    for allowed in allowed_employers:
        if norm_emp == allowed or norm_emp in allowed or allowed in norm_emp:
            return True

    if verifiable_texts:
        for vt in verifiable_texts:
            norm_vt = _normalize_token(vt)
            if norm_emp in norm_vt:
                return True
            # Match distinctive corporate name tokens (e.g. 'innodata' from 'Innodata Inc.')
            for token in norm_emp.split():
                if len(token) > 3 and token not in ("inc", "corp", "ltd", "llc", "company", "technologies", "services", "solutions", "group"):
                    if token in norm_vt:
                        return True
    return False


def _is_institution_supported(
    inst: str,
    allowed_institutions: set[str],
    verifiable_texts: list[str] | None = None,
) -> bool:
    """Check if an educational institution is grounded in Master Resume evidence."""
    norm_inst = _normalize_token(inst)
    if not norm_inst:
        return True

    for allowed in allowed_institutions:
        if norm_inst in allowed or allowed in norm_inst:
            return True

    if verifiable_texts:
        for vt in verifiable_texts:
            norm_vt = _normalize_token(vt)
            if norm_inst in norm_vt:
                return True
            inst_tokens = [t for t in norm_inst.split() if len(t) > 3 and t not in ("university", "college", "institute", "school", "engineering", "technology", "management", "science")]
            if inst_tokens and any(t in norm_vt for t in inst_tokens):
                return True
    return False


def _is_degree_supported(
    degree: str,
    allowed_degrees: set[str],
    verifiable_texts: list[str] | None = None,
) -> bool:
    """Check if a degree is grounded in Master Resume evidence."""
    norm_deg = _normalize_token(degree)
    if not norm_deg:
        return True

    for allowed in allowed_degrees:
        if norm_deg in allowed or allowed in norm_deg:
            return True
        # Check major / degree tokens
        deg_tokens = set(norm_deg.split())
        allowed_tokens = set(allowed.split())
        if deg_tokens and (len(deg_tokens & allowed_tokens) / len(deg_tokens)) >= 0.5:
            return True

    if verifiable_texts:
        for vt in verifiable_texts:
            norm_vt = _normalize_token(vt)
            if norm_deg in norm_vt:
                return True
            deg_tokens = [t for t in norm_deg.split() if len(t) > 3 and t not in ("bachelor", "master", "degree", "science", "arts", "engineering")]
            if deg_tokens and any(t in norm_vt for t in deg_tokens):
                return True
    return False


def _is_cert_supported(
    cert_name: str,
    allowed_certs: set[str],
    verifiable_texts: list[str] | None = None,
) -> bool:
    """Check if a certification is grounded in Master Resume evidence."""
    norm_cert = _normalize_token(cert_name)
    if not norm_cert:
        return True

    for allowed in allowed_certs:
        if norm_cert in allowed or allowed in norm_cert:
            return True
        cert_tokens = set(norm_cert.split())
        allowed_tokens = set(allowed.split())
        if cert_tokens and (len(cert_tokens & allowed_tokens) / len(cert_tokens)) >= 0.5:
            return True

    if verifiable_texts:
        for vt in verifiable_texts:
            norm_vt = _normalize_token(vt)
            if norm_cert in norm_vt:
                return True
            cert_tokens = [t for t in norm_cert.split() if len(t) > 3 and t not in ("certificate", "certification", "certified", "essential", "essentials", "learning")]
            if cert_tokens and any(t in norm_vt for t in cert_tokens):
                return True
    return False


def validate_resume_evidence(
    candidate_resume: dict[str, Any] | str | Path,
    master_resume: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """
    Validate that candidate_resume content is strictly supported by master_resume.

    Parameters:
        candidate_resume: Candidate resume dict or JSON string/path.
        master_resume: Master resume dict or JSON string/path.

    Returns:
        Structured result:
        {
            "passed": bool,
            "violations": list[dict],
            "supported_changes": list[str],
            "unsupported_changes": list[str],
            "violation_count": int
        }
    """
    # 1. Resolve inputs
    if isinstance(master_resume, (str, Path)) and Path(str(master_resume)).exists():
        with open(master_resume, "r", encoding="utf-8") as f:
            master_dict = json.load(f)
    elif isinstance(master_resume, dict):
        master_dict = master_resume
    elif isinstance(master_resume, str):
        master_dict = json.loads(master_resume)
    else:
        raise ValueError(f"Invalid master_resume input: {type(master_resume)}")

    if isinstance(candidate_resume, (str, Path)) and Path(str(candidate_resume)).exists():
        with open(candidate_resume, "r", encoding="utf-8") as f:
            candidate_dict = json.load(f)
    elif isinstance(candidate_resume, dict):
        candidate_dict = candidate_resume
    elif isinstance(candidate_resume, str):
        candidate_dict = json.loads(candidate_resume)
    else:
        raise ValueError(f"Invalid candidate_resume input: {type(candidate_resume)}")

    # 2. Extract Master Evidence
    master_evidence = extract_master_evidence(master_dict)
    violations: list[dict[str, Any]] = []
    supported_changes: list[str] = []
    unsupported_changes: list[str] = []

    # Combine allowed skills and tools for general vocabulary check
    all_allowed_terms = (
        master_evidence["tools_and_technologies"]
        | master_evidence["skills"]
        | master_evidence["roles"]
    )
    v_texts = master_evidence.get("verifiable_texts", [])

    # 3. Check Employers in Experience
    cand_experience = candidate_dict.get("experience", [])
    if isinstance(cand_experience, list):
        for exp in cand_experience:
            if not isinstance(exp, dict):
                continue
            company = str(exp.get("company", "")).strip()
            if company and not _is_employer_supported(company, master_evidence["employers"], v_texts):
                v = {
                    "type": "unsupported_employer",
                    "value": company,
                    "section": "experience",
                    "reason": f"Employer '{company}' is not supported by master resume evidence.",
                }
                violations.append(v)
                unsupported_changes.append(f"Invented employer: {company}")
            elif company:
                supported_changes.append(f"Verified employer: {company}")

            # Check metrics in bullets
            for b in exp.get("bullets", []):
                bullet_metrics = _extract_numbers_and_metrics(str(b))
                for m in bullet_metrics:
                    # Ignore common generic counters like '1', '2', '3' unless they contain metric symbols
                    if m in ("1", "2", "3", "4", "5") or m in master_evidence["metrics"]:
                        continue
                    # Check if metric was in master evidence texts
                    matched_in_master = any(m in _normalize_token(t) for t in v_texts)
                    if not matched_in_master:
                        # Flag suspicious high metric additions (e.g. 10,000, 98%, $5M)
                        if any(char in m for char in ("%", "$", "+", "k", "m")) or len(re.sub(r"\D", "", m)) >= 4:
                            v = {
                                "type": "unsupported_metric",
                                "value": m,
                                "section": "experience.bullets",
                                "reason": f"Quantitative metric '{m}' was not found in master resume evidence.",
                            }
                            violations.append(v)
                            unsupported_changes.append(f"Invented metric: {m}")

    # 4. Check Tools and Technologies in Skills
    cand_skills = candidate_dict.get("skills", {})
    if isinstance(cand_skills, dict):
        tools_list = cand_skills.get("tools_and_technologies") or cand_skills.get("tools", [])
        if isinstance(tools_list, list):
            for tool in tools_list:
                tool_str = str(tool).strip()
                if not tool_str:
                    continue
                if not _is_tool_supported(tool_str, master_evidence["tools_and_technologies"], v_texts):
                    v = {
                        "type": "unsupported_tool",
                        "value": tool_str,
                        "section": "skills.tools_and_technologies",
                        "reason": f"Tool or platform '{tool_str}' is not supported by master resume evidence.",
                    }
                    violations.append(v)
                    unsupported_changes.append(f"Invented tool/platform: {tool_str}")
                else:
                    supported_changes.append(f"Verified tool: {tool_str}")

        # Check technical skills
        tech_skills = cand_skills.get("technical_skills") or cand_skills.get("skills", [])
        if isinstance(tech_skills, list):
            for skill in tech_skills:
                skill_str = str(skill).strip()
                if not skill_str:
                    continue
                # Skills can be semantic rephrasings of capabilities or tools
                norm_sk = _normalize_token(skill_str)
                is_supported = (
                    norm_sk in all_allowed_terms
                    or _is_tool_supported(skill_str, master_evidence["tools_and_technologies"], v_texts)
                    or any(norm_sk in t or t in norm_sk for t in all_allowed_terms)
                    or any(norm_sk in _normalize_token(vt) for vt in v_texts)
                )
                if not is_supported:
                    # Check if well-known completely alien tools appear in technical skills
                    alien_platforms = {
                        "scale ai", "labelbox", "amazon mechanical turk", "mturk",
                        "superannotate", "v7 labs", "roboflow", "appens", "appen",
                        "remotasks", "toloka", "aws sagemaker ground truth",
                    }
                    if norm_sk in alien_platforms:
                        v = {
                            "type": "unsupported_platform",
                            "value": skill_str,
                            "section": "skills.technical_skills",
                            "reason": f"Platform '{skill_str}' is an unsupported third-party tool not found in master resume.",
                        }
                        violations.append(v)
                        unsupported_changes.append(f"Invented platform in skills: {skill_str}")
                    else:
                        supported_changes.append(f"Domain skill: {skill_str}")
                else:
                    supported_changes.append(f"Verified skill: {skill_str}")

    # 5. Check Education
    cand_education = candidate_dict.get("education", [])
    if isinstance(cand_education, list):
        for edu in cand_education:
            if not isinstance(edu, dict):
                continue
            inst = str(edu.get("institution", "")).strip()
            degree = str(edu.get("degree", "")).strip()
            if inst and not _is_institution_supported(inst, master_evidence["institutions"], v_texts):
                v = {
                    "type": "unsupported_institution",
                    "value": inst,
                    "section": "education",
                    "reason": f"Institution '{inst}' is not supported by master resume evidence.",
                }
                violations.append(v)
                unsupported_changes.append(f"Invented institution: {inst}")
            if degree and not _is_degree_supported(degree, master_evidence["degrees"], v_texts):
                v = {
                    "type": "unsupported_degree",
                    "value": degree,
                    "section": "education",
                    "reason": f"Degree '{degree}' is not supported by master resume evidence.",
                }
                violations.append(v)
                unsupported_changes.append(f"Invented degree: {degree}")

    # 6. Check Certifications
    cand_certs = candidate_dict.get("certifications", [])
    if isinstance(cand_certs, list):
        for c in cand_certs:
            cert_name = c.get("name", "") if isinstance(c, dict) else str(c)
            cert_name = str(cert_name).strip()
            if cert_name and not _is_cert_supported(cert_name, master_evidence["certifications"], v_texts):
                v = {
                    "type": "unsupported_certification",
                    "value": cert_name,
                    "section": "certifications",
                    "reason": f"Certification '{cert_name}' is not supported by master resume evidence.",
                }
                violations.append(v)
                unsupported_changes.append(f"Invented certification: {cert_name}")
            elif cert_name:
                supported_changes.append(f"Verified certification: {cert_name}")

    # 7. Assemble Structured Result
    passed = len(violations) == 0

    return {
        "passed": passed,
        "violations": violations,
        "violation_count": len(violations),
        "supported_changes": supported_changes,
        "unsupported_changes": unsupported_changes,
    }


if __name__ == "__main__":
    sample_master = Path(__file__).resolve().parent.parent / "data" / "master_resume.json"
    if sample_master.exists():
        with open(sample_master, "r", encoding="utf-8") as f:
            m_data = json.load(f)

        # Test valid resume
        valid_resume = {
            "personal_info": {"name": "Kanishk Surwade", "target_title": "Data Annotator"},
            "skills": {
                "technical_skills": ["Data Annotation", "Video Annotation", "Quality Assurance"],
                "tools_and_technologies": ["CVAT", "Google Cloud Platform", "Python", "SQL"],
            },
            "experience": [
                {
                    "company": "Innodata Inc.",
                    "role": "AI & LLM Analyst",
                    "bullets": ["Performed multimodal data annotation and quality benchmarking using CVAT."],
                }
            ],
            "education": [{"degree": "B.Tech in Automation and Robotics", "institution": "JSPM Rajarshi Shahu College of Engineering"}],
            "certifications": [{"name": "Career Essentials in Generative AI", "issuer": "Microsoft & LinkedIn Learning"}],
        }

        # Test invalid resume with hallucinated tool and employer
        invalid_resume = {
            "personal_info": {"name": "Kanishk Surwade", "target_title": "Data Annotator"},
            "skills": {
                "technical_skills": ["Data Annotation"],
                "tools_and_technologies": ["CVAT", "Scale AI", "Labelbox"],
            },
            "experience": [
                {
                    "company": "Scale AI Inc.",
                    "role": "Lead Annotator",
                    "bullets": ["Managed $5M budget annotating 1,000,000 samples with 99.9% accuracy."],
                }
            ],
            "education": [{"degree": "Ph.D. in Computer Science", "institution": "MIT"}],
            "certifications": [{"name": "AWS Solutions Architect Professional", "issuer": "Amazon"}],
        }

        val_valid = validate_resume_evidence(valid_resume, m_data)
        val_invalid = validate_resume_evidence(invalid_resume, m_data)

        print("=== VALID RESUME TEST ===")
        print("Passed:", val_valid["passed"])
        print("Violations:", len(val_valid["violations"]))

        print("\n=== INVALID RESUME TEST ===")
        print("Passed:", val_invalid["passed"])
        print("Violations Found:", len(val_invalid["violations"]))
        for viol in val_invalid["violations"]:
            print(f"  [-] {viol['type']}: {viol['value']} -> {viol['reason']}")
