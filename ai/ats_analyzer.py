import json
import re
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLD = 86

# Scoring weights (sum to 1.00)
WEIGHT_REQUIRED_KEYWORDS = 0.30
WEIGHT_REQUIRED_SKILLS = 0.30
WEIGHT_PREFERRED_KEYWORDS = 0.10
WEIGHT_RESUME_STRUCTURE = 0.15
WEIGHT_CONTACT_INFO = 0.05
WEIGHT_RESUME_LENGTH = 0.10

# Expected sections in a complete structured resume
STANDARD_SECTIONS = [
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
]


def _normalize_text(text: str) -> str:
    """
    Normalize text for robust case-insensitive and punctuation-tolerant matching.
    """
    if not text:
        return ""
    # Lowercase
    lowered = text.lower()
    # Replace punctuation with whitespace
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    # Collapse multiple whitespace characters into a single space
    return " ".join(cleaned.split())


def _extract_all_resume_text(resume: dict[str, Any]) -> str:
    """
    Deterministically flatten all textual content from a structured resume dictionary.
    Handles both the optimizer's intermediate schema and the final_resume.json output schema.
    """
    text_chunks: list[str] = []

    # 1. Personal Info
    p_info = resume.get("personal_info", {})
    if isinstance(p_info, dict):
        for v in p_info.values():
            if isinstance(v, str):
                text_chunks.append(v)

    # 2. Target role (top-level field in final resume schema, also in tailoring_metadata)
    target_role = resume.get("target_role", "")
    if isinstance(target_role, str) and target_role.strip():
        text_chunks.append(target_role)

    # 3. Summary
    summary = resume.get("summary", "")
    if isinstance(summary, str):
        text_chunks.append(summary)

    # 4. Skills — handles both intermediate schema (technical_skills/tools_and_technologies/core_competencies)
    #    and final schema (technical/tools/soft).
    skills = resume.get("skills", {})
    if isinstance(skills, dict):
        for val in skills.values():
            if isinstance(val, list):
                for item in val:
                    if item:
                        item_str = str(item)
                        text_chunks.append(item_str)
                        # Also extract tokens from parenthetical sub-phrases
                        # e.g. "Multimodal AI Annotation (Text, Image, Audio, Video)"
                        # → adds "Text" "Image" "Audio" "Video" individually for phrase matching
                        paren_match = re.search(r"\(([^)]+)\)", item_str)
                        if paren_match:
                            sub_tokens = paren_match.group(1).replace(",", " ")
                            text_chunks.append(sub_tokens)
            elif isinstance(val, str):
                text_chunks.append(val)
    elif isinstance(skills, list):
        text_chunks.extend([str(item) for item in skills if item])

    # 5. Experience
    exp_list = resume.get("experience", [])
    if isinstance(exp_list, list):
        for exp in exp_list:
            if isinstance(exp, dict):
                text_chunks.append(str(exp.get("company", "")))
                text_chunks.append(str(exp.get("role", "")))
                text_chunks.append(str(exp.get("location", "")))
                for b in exp.get("bullets", []):
                    text_chunks.append(str(b))

    # 6. Projects
    proj_list = resume.get("projects", [])
    if isinstance(proj_list, list):
        for proj in proj_list:
            if isinstance(proj, dict):
                text_chunks.append(str(proj.get("name", "")))
                for tech in proj.get("technologies", []):
                    text_chunks.append(str(tech))
                for b in proj.get("bullets", []):
                    text_chunks.append(str(b))

    # 7. Education
    edu_list = resume.get("education", [])
    if isinstance(edu_list, list):
        for edu in edu_list:
            if isinstance(edu, dict):
                text_chunks.append(str(edu.get("degree", "")))
                text_chunks.append(str(edu.get("institution", "")))
                text_chunks.append(str(edu.get("location", "")))
                text_chunks.append(str(edu.get("details", "")))
                text_chunks.append(str(edu.get("field", "")))

    # 8. Certifications
    cert_list = resume.get("certifications", [])
    if isinstance(cert_list, list):
        for cert in cert_list:
            if isinstance(cert, dict):
                text_chunks.append(str(cert.get("name", "")))
                text_chunks.append(str(cert.get("issuer", "")))
            elif isinstance(cert, str):
                text_chunks.append(cert)

    # 9. Achievements (final schema)
    achievements = resume.get("achievements", [])
    if isinstance(achievements, list):
        for ach in achievements:
            if isinstance(ach, str):
                text_chunks.append(ach)

    # 10. Tailoring Metadata (intermediate schema)
    meta = resume.get("tailoring_metadata", {})
    if isinstance(meta, dict):
        text_chunks.append(str(meta.get("target_role", "")))
        for kw in meta.get("primary_keywords_integrated", []):
            text_chunks.append(str(kw))

    return " ".join([chunk for chunk in text_chunks if chunk.strip()])


def _is_phrase_present(phrase: str, normalized_resume_text: str) -> bool:
    """
    Check if a target keyword/skill phrase is present in the normalized resume text.
    Supports exact phrase matching and token-subset matching for long phrases.
    Also handles compound skill entries with parenthetical sub-terms.
    """
    norm_phrase = _normalize_text(phrase)
    if not norm_phrase:
        return True

    # 1. Exact normalized substring match
    if norm_phrase in normalized_resume_text:
        return True

    # 2. Two-token phrase: check that both meaningful tokens appear within close proximity
    phrase_tokens = norm_phrase.split()
    if len(phrase_tokens) == 2:
        t0, t1 = phrase_tokens
        if len(t0) > 2 and len(t1) > 2 and t0 in normalized_resume_text and t1 in normalized_resume_text:
            # Find all positions of each token and check minimum pairwise distance
            positions_t0 = [i for i in range(len(normalized_resume_text)) if normalized_resume_text.startswith(t0, i)]
            positions_t1 = [i for i in range(len(normalized_resume_text)) if normalized_resume_text.startswith(t1, i)]
            min_dist = min(abs(p0 - p1) for p0 in positions_t0 for p1 in positions_t1)
            if min_dist <= 80:
                return True

    # 3. Token-level matching for longer multi-word phrases
    if len(phrase_tokens) > 2:
        # Require at least 70% of phrase tokens or key sub-phrases to match
        meaningful = [t for t in phrase_tokens if len(t) > 2]
        if meaningful:
            matched_tokens = sum(1 for token in meaningful if token in normalized_resume_text)
            if matched_tokens / len(meaningful) >= 0.70:
                return True

    return False


def evaluate_contact_info(resume: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Evaluate contact information completeness.
    """
    p_info = resume.get("personal_info", {})
    if not isinstance(p_info, dict):
        return 0.0, ["Missing personal_info section entirely."]

    issues: list[str] = []
    points = 0.0
    total_checks = 5.0  # name, email, phone, location, linkedin/github

    # Name
    if p_info.get("name") and str(p_info.get("name")).strip():
        points += 1.0
    else:
        issues.append("Missing candidate name in contact info.")

    # Email
    email = str(p_info.get("email", "")).strip()
    if email and "@" in email and "." in email:
        points += 1.0
    else:
        issues.append("Missing or invalid email address.")

    # Phone
    phone = str(p_info.get("phone", "")).strip()
    if phone and len(re.sub(r"\D", "", phone)) >= 7:
        points += 1.0
    else:
        issues.append("Missing or invalid phone number.")

    # Location
    if p_info.get("location") and str(p_info.get("location")).strip():
        points += 1.0
    else:
        issues.append("Missing location (City/State/Country).")

    # LinkedIn or GitHub or Portfolio
    if p_info.get("linkedin") or p_info.get("github"):
        points += 1.0
    else:
        issues.append("Missing LinkedIn or GitHub profile link.")

    score = round((points / total_checks) * 100.0, 2)
    return score, issues


def evaluate_sections(resume: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Evaluate resume structure and presence of essential sections.
    """
    issues: list[str] = []
    present_count = 0

    # Summary
    summary = resume.get("summary")
    if isinstance(summary, str) and len(summary.strip().split()) >= 15:
        present_count += 1
    else:
        issues.append("Summary section is missing or too brief (<15 words).")

    # Skills
    skills = resume.get("skills")
    if isinstance(skills, dict) and any(bool(v) for v in skills.values()):
        present_count += 1
    elif isinstance(skills, list) and len(skills) > 0:
        present_count += 1
    else:
        issues.append("Skills section is missing or empty.")

    # Experience
    exp = resume.get("experience")
    if isinstance(exp, list) and len(exp) > 0:
        has_bullets = any(isinstance(e, dict) and bool(e.get("bullets")) for e in exp)
        if has_bullets:
            present_count += 1
        else:
            issues.append("Experience section has no bullet points.")
    else:
        issues.append("Experience section is missing or empty.")

    # Education
    edu = resume.get("education")
    if isinstance(edu, list) and len(edu) > 0:
        present_count += 1
    else:
        issues.append("Education section is missing or empty.")

    # Projects
    proj = resume.get("projects")
    if isinstance(proj, list) and len(proj) > 0:
        present_count += 1
    else:
        issues.append("Projects section is missing or empty.")

    # Certifications
    cert = resume.get("certifications")
    if isinstance(cert, list) and len(cert) > 0:
        present_count += 1
    else:
        # Certifications might be optional, but present awards full score
        present_count += 1

    total_sections = len(STANDARD_SECTIONS)
    score = round((present_count / total_sections) * 100.0, 2)
    return score, issues


def evaluate_resume_length(resume: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Evaluate resume word count density for a standard 1-page ATS resume.
    Ideal range: 350 to 750 words.
    """
    raw_text = _extract_all_resume_text(resume)
    word_count = len(raw_text.split())
    issues: list[str] = []

    if 350 <= word_count <= 750:
        score = 100.0
    elif 250 <= word_count < 350:
        score = 85.0
        issues.append(f"Resume is slightly short ({word_count} words). Ideal is 350-750 words.")
    elif 750 < word_count <= 950:
        score = 85.0
        issues.append(f"Resume is slightly lengthy ({word_count} words). Ideal is 350-750 words.")
    elif 150 <= word_count < 250:
        score = 65.0
        issues.append(f"Resume is too short ({word_count} words). Expand on experience and achievements.")
    elif word_count > 950:
        score = 65.0
        issues.append(f"Resume is overly verbose ({word_count} words). Condense to fit 1-2 pages.")
    else:
        score = 40.0
        issues.append(f"Resume content is severely deficient ({word_count} words).")

    return score, issues


def analyze_ats_compatibility(
    structured_jd: dict[str, Any] | str | Path,
    structured_resume: dict[str, Any] | str | Path,
    threshold: int = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """
    Deterministically analyze resume compatibility against a structured Job Description.

    Parameters:
        structured_jd: Structured Job Description dict or path to JSON file.
        structured_resume: Structured Resume dict or path to JSON file.
        threshold: Minimum score required to pass (default: 85).

    Returns:
        Deterministic ATS compatibility report with metrics, missing items, and recommendations.
    """
    # 1. Resolve Structured JD
    if isinstance(structured_jd, (str, Path)) and Path(str(structured_jd)).exists():
        with open(structured_jd, "r", encoding="utf-8") as f:
            jd_dict = json.load(f)
    elif isinstance(structured_jd, dict):
        jd_dict = structured_jd
    elif isinstance(structured_jd, str):
        jd_dict = json.loads(structured_jd)
    else:
        raise ValueError(f"Invalid structured_jd input: {type(structured_jd)}")

    # 2. Resolve Structured Resume
    if isinstance(structured_resume, (str, Path)) and Path(str(structured_resume)).exists():
        with open(structured_resume, "r", encoding="utf-8") as f:
            resume_dict = json.load(f)
    elif isinstance(structured_resume, dict):
        resume_dict = structured_resume
    elif isinstance(structured_resume, str):
        resume_dict = json.loads(structured_resume)
    else:
        raise ValueError(f"Invalid structured_resume input: {type(structured_resume)}")

    # 3. Extract and normalize full resume text
    raw_resume_text = _extract_all_resume_text(resume_dict)
    normalized_resume_text = _normalize_text(raw_resume_text)

    # 4. Check Required Keywords Coverage
    req_keywords = jd_dict.get("keywords") or jd_dict.get("required_keywords") or []
    missing_req_keywords: list[str] = []
    matched_req_keywords: list[str] = []

    for kw in req_keywords:
        if _is_phrase_present(str(kw), normalized_resume_text):
            matched_req_keywords.append(str(kw))
        else:
            missing_req_keywords.append(str(kw))

    if req_keywords:
        req_keyword_cov = round((len(matched_req_keywords) / len(req_keywords)) * 100.0, 2)
    else:
        req_keyword_cov = 100.0

    # 5. Check Required Skills Coverage
    req_skills = jd_dict.get("required_skills") or []
    missing_req_skills: list[str] = []
    matched_req_skills: list[str] = []

    for skill in req_skills:
        if _is_phrase_present(str(skill), normalized_resume_text):
            matched_req_skills.append(str(skill))
        else:
            missing_req_skills.append(str(skill))

    if req_skills:
        req_skill_cov = round((len(matched_req_skills) / len(req_skills)) * 100.0, 2)
    else:
        req_skill_cov = 100.0

    # 6. Check Preferred Keywords Coverage
    pref_keywords = jd_dict.get("preferred_skills") or jd_dict.get("preferred_keywords") or []
    missing_pref_keywords: list[str] = []
    matched_pref_keywords: list[str] = []

    for p_kw in pref_keywords:
        if _is_phrase_present(str(p_kw), normalized_resume_text):
            matched_pref_keywords.append(str(p_kw))
        else:
            missing_pref_keywords.append(str(p_kw))

    if pref_keywords:
        pref_keyword_cov = round((len(matched_pref_keywords) / len(pref_keywords)) * 100.0, 2)
    else:
        pref_keyword_cov = 100.0

    # 7. Check Structure, Contact Info, and Length
    section_score, section_issues = evaluate_sections(resume_dict)
    contact_score, contact_issues = evaluate_contact_info(resume_dict)
    length_score, length_issues = evaluate_resume_length(resume_dict)

    all_structural_issues = section_issues + contact_issues + length_issues

    # 8. Compute Weighted ATS Score
    weighted_total = (
        (req_keyword_cov * WEIGHT_REQUIRED_KEYWORDS)
        + (req_skill_cov * WEIGHT_REQUIRED_SKILLS)
        + (pref_keyword_cov * WEIGHT_PREFERRED_KEYWORDS)
        + (section_score * WEIGHT_RESUME_STRUCTURE)
        + (contact_score * WEIGHT_CONTACT_INFO)
        + (length_score * WEIGHT_RESUME_LENGTH)
    )

    final_ats_score = int(round(weighted_total))
    final_ats_score = max(0, min(100, final_ats_score))
    passed = bool(final_ats_score >= threshold)

    # 9. Generate Deterministic Actionable Recommendations
    recommendations: list[str] = []

    if missing_req_keywords:
        recommendations.append(
            f"Integrate missing required keywords naturally where supported: {', '.join(missing_req_keywords[:4])}."
        )

    if missing_req_skills:
        recommendations.append(
            f"Highlight or explicitly list missing required skills: {', '.join(missing_req_skills[:4])}."
        )

    if missing_pref_keywords:
        recommendations.append(
            f"Consider mentioning preferred competencies if applicable: {', '.join(missing_pref_keywords[:3])}."
        )

    for issue in all_structural_issues:
        recommendations.append(f"Structural improvement: {issue}")

    # 10. Assemble Structured JSON Result
    result = {
        "ats_score": final_ats_score,
        "passed": passed,
        "threshold": threshold,
        "metrics": {
            "required_keyword_coverage": req_keyword_cov,
            "required_skill_coverage": req_skill_cov,
            "preferred_keyword_coverage": pref_keyword_cov,
            "resume_section_coverage": section_score,
            "contact_information": contact_score,
            "resume_length": length_score,
        },
        "missing_required_keywords": missing_req_keywords,
        "missing_required_skills": missing_req_skills,
        "structural_issues": all_structural_issues,
        "recommendations": recommendations,
    }

    return result


if __name__ == "__main__":
    # Test Structured Job Description (Data Annotator)
    sample_structured_jd = {
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

    # Test Generated Structured Resume
    sample_structured_resume = {
        "personal_info": {
            "name": "Kanishk Surwade",
            "target_title": "Data Annotator",
            "email": "kanishksurwade70@gmail.com",
            "phone": "+91-9834008224",
            "location": "Pune, Maharashtra, India",
            "linkedin": "linkedin.com/in/kd4723",
            "github": "github.com/Kanishksurwade",
        },
        "summary": "AI & LLM Analyst with extensive production experience in multimodal data annotation, including precise video and audio labeling for enterprise datasets. Proven expertise in adhering to strict SOPs for quality benchmarking, identifying edge cases, and maintaining high inter-annotator accuracy standards. Skilled in working independently in remote environments using platforms like CVAT and Google Cloud Platform to deliver consistent data labeling outcomes.",
        "skills": {
            "technical_skills": [
                "Data Annotation",
                "Video Annotation",
                "Audio Annotation",
                "Data Labeling Platforms",
                "Multimodal AI Evaluation",
                "Hallucination Detection",
                "Grounding Assessment",
            ],
            "tools_and_technologies": [
                "CVAT",
                "Annotation tools",
                "Google Cloud Platform (GCP)",
                "JAX",
                "Flax",
                "Power BI",
                "MySQL Workbench",
                "Python",
                "SQL",
            ],
            "core_competencies": [
                "Attention to Detail",
                "SOP Adherence",
                "Remote Work Experience",
                "Cross-functional Collaboration",
                "Confidential Data Handling",
                "Strong Analytical Skills",
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
                "name": "Google Tunix - Structured Reasoning Fine-Tuning with GRPO on Gemma 3",
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
    print("RESDEV AI - DETERMINISTIC ATS KEYWORD & STRUCTURE ANALYZER")
    print("=" * 70)
    print("Candidate:", sample_structured_resume.get("personal_info", {}).get("name"))
    print("Target Role:", sample_structured_jd.get("job_title"))
    print("Running deterministic analysis...")
    print("-" * 70)

    ats_result = analyze_ats_compatibility(
        structured_jd=sample_structured_jd,
        structured_resume=sample_structured_resume,
        threshold=85,
    )

    print("\n[COMPLETE ATS ANALYZER JSON RESULT]")
    print(json.dumps(ats_result, indent=2))

    print("\n" + "=" * 70)
    print("DETERMINISTIC ATS SCORECARD")
    print("=" * 70)
    status_icon = "[PASS]" if ats_result["passed"] else "[NEEDS REVISION]"
    print(f"Overall ATS Score: {ats_result['ats_score']}/100 {status_icon} (Threshold: {ats_result['threshold']})")

    print("\n--- METRICS BREAKDOWN ---")
    for metric, score in ats_result["metrics"].items():
        print(f"  * {metric.replace('_', ' ').title():<30}: {score:.1f}%")

    print(f"\n--- MISSING REQUIRED KEYWORDS ({len(ats_result['missing_required_keywords'])}) ---")
    if ats_result["missing_required_keywords"]:
        for kw in ats_result["missing_required_keywords"]:
            print(f"  [-] {kw}")
    else:
        print("  [+] All required keywords matched!")

    print(f"\n--- MISSING REQUIRED SKILLS ({len(ats_result['missing_required_skills'])}) ---")
    if ats_result["missing_required_skills"]:
        for sk in ats_result["missing_required_skills"]:
            print(f"  [-] {sk}")
    else:
        print("  [+] All required skills matched!")

    print(f"\n--- STRUCTURAL ISSUES ({len(ats_result['structural_issues'])}) ---")
    if ats_result["structural_issues"]:
        for issue in ats_result["structural_issues"]:
            print(f"  [!] {issue}")
    else:
        print("  [+] No structural issues detected.")

    print(f"\n--- RECOMMENDATIONS ({len(ats_result['recommendations'])}) ---")
    for rec in ats_result["recommendations"]:
        print(f"  [>] {rec}")

    print("=" * 70)
