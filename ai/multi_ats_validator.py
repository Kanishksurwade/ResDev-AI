"""
ResDev AI - Multi-ATS Compatibility Validator

Simulates resume evaluation across multiple ATS platforms using deterministic
logic. Reuses core analysis from ai/ats_analyzer.py and applies platform-specific
behavioral profiles from data/ats_profiles.json.

DISCLAIMER:
    These are behavioral simulations based on publicly available information
    about ATS parsing behaviors. This module does NOT claim access to proprietary
    ATS algorithms, and results do NOT represent guaranteed real ATS scores.

No LLM calls. Fully offline. Fully deterministic.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.ats_analyzer import (
    _normalize_text,
    _extract_all_resume_text,
    _is_phrase_present,
    evaluate_contact_info,
    evaluate_sections,
    evaluate_resume_length,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROFILES_PATH = _project_root / "data" / "ats_profiles.json"

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


# ---------------------------------------------------------------------------
# Profile Loader
# ---------------------------------------------------------------------------
def load_ats_profiles(path: str | Path | None = None) -> dict[str, Any]:
    """Load ATS platform profiles from the JSON file."""
    profile_path = Path(path) if path else _PROFILES_PATH
    if not profile_path.exists():
        raise FileNotFoundError(f"ATS profiles not found at {profile_path}")
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("profiles", {})


# ---------------------------------------------------------------------------
# Date Parsing Utilities
# ---------------------------------------------------------------------------
def _parse_date_string(date_str: str) -> tuple[int | None, int | None]:
    """
    Parse a date string into (year, month).
    Supports: 'Month YYYY', 'MM/YYYY', 'YYYY', 'Present'.
    Returns (None, None) for unparseable or 'Present'.
    """
    if not date_str or not isinstance(date_str, str):
        return None, None

    cleaned = date_str.strip().lower()
    if cleaned in ("present", "current", "ongoing", "now"):
        return None, None

    # Try "Month YYYY" — e.g., "Dec 2025"
    parts = cleaned.split()
    if len(parts) == 2:
        month_str, year_str = parts
        month = _MONTH_MAP.get(month_str[:3])
        try:
            year = int(year_str)
            if month and 1900 <= year <= 2100:
                return year, month
        except ValueError:
            pass

    # Try "MM/YYYY" or "MM-YYYY"
    m = re.match(r"^(\d{1,2})[/\-](\d{4})$", cleaned)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return year, month

    # Try bare "YYYY"
    m = re.match(r"^(\d{4})$", cleaned)
    if m:
        year = int(m.group(1))
        if 1900 <= year <= 2100:
            return year, None

    return None, None


def _check_future_dates(resume: dict[str, Any]) -> list[str]:
    """Return list of future-dated entries found in experience/projects."""
    issues: list[str] = []
    now = datetime.now()
    current_year = now.year
    current_month = now.month

    for section_name, section_key in [("Experience", "experience"), ("Projects", "projects")]:
        entries = resume.get(section_key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for date_field in ("start_date", "end_date"):
                raw = entry.get(date_field, "")
                year, month = _parse_date_string(str(raw))
                if year is None:
                    continue
                if year > current_year or (year == current_year and month and month > current_month):
                    label = entry.get("role") or entry.get("name") or "Unknown"
                    issues.append(
                        f"{section_name} '{label}': {date_field} '{raw}' appears future-dated"
                    )
    return issues


def _check_employment_gaps(resume: dict[str, Any], max_gap_months: int | None) -> list[str]:
    """
    Detect gaps between consecutive experience entries exceeding the threshold.
    Returns empty list if max_gap_months is None (profile doesn't flag gaps).
    """
    if max_gap_months is None:
        return []

    experiences = resume.get("experience", [])
    if not isinstance(experiences, list) or len(experiences) < 2:
        return []

    # Parse end/start dates into sortable tuples
    dated_entries: list[tuple[int, int, str]] = []
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        start_raw = str(exp.get("start_date", ""))
        end_raw = str(exp.get("end_date", ""))
        s_year, s_month = _parse_date_string(start_raw)
        e_year, e_month = _parse_date_string(end_raw)
        role = exp.get("role", "Unknown")
        if s_year:
            s_month = s_month or 1
        if e_year:
            e_month = e_month or 12
        if s_year and e_year:
            dated_entries.append((s_year * 12 + s_month, e_year * 12 + e_month, role))

    if len(dated_entries) < 2:
        return []

    # Sort by start date descending (most recent first)
    dated_entries.sort(key=lambda x: x[0], reverse=True)

    gaps: list[str] = []
    for i in range(len(dated_entries) - 1):
        current_start = dated_entries[i][0]
        prev_end = dated_entries[i + 1][1]
        gap_months = current_start - prev_end
        if gap_months > max_gap_months:
            gaps.append(
                f"Employment gap of ~{gap_months} months detected between "
                f"'{dated_entries[i + 1][2]}' and '{dated_entries[i][2]}'"
            )
    return gaps


# ---------------------------------------------------------------------------
# Keyword Stuffing Detection
# ---------------------------------------------------------------------------
def _check_keyword_stuffing(
    resume_text: str,
    keywords: list[str],
    threshold: int | None,
) -> list[str]:
    """
    Detect if any keyword appears more than threshold times in the resume text.
    Returns list of stuffing warnings.
    """
    if threshold is None:
        return []

    issues: list[str] = []
    normalized = _normalize_text(resume_text)
    for kw in keywords:
        norm_kw = _normalize_text(kw)
        if not norm_kw:
            continue
        count = normalized.count(norm_kw)
        if count > threshold:
            issues.append(
                f"Keyword '{kw}' appears {count} times (threshold: {threshold})"
            )
    return issues


# ---------------------------------------------------------------------------
# Job Title Match
# ---------------------------------------------------------------------------
def _check_job_title_match(
    resume: dict[str, Any],
    structured_jd: dict[str, Any],
    requires_exact: bool,
) -> dict[str, Any]:
    """Check if the resume target title aligns with the JD job title."""
    jd_title = str(structured_jd.get("job_title", "")).strip()
    resume_title = str(
        resume.get("personal_info", {}).get("target_title", "")
        or resume.get("target_role", "")
        or resume.get("tailoring_metadata", {}).get("target_role", "")
    ).strip()

    if not jd_title:
        return {"status": "SKIP", "reason": "No JD job title to compare against"}

    norm_jd = _normalize_text(jd_title)
    norm_resume = _normalize_text(resume_title)

    if requires_exact:
        matched = norm_jd == norm_resume
    else:
        # Token overlap: at least 50% of JD title tokens present in resume title
        jd_tokens = set(norm_jd.split())
        resume_tokens = set(norm_resume.split())
        if jd_tokens:
            overlap = len(jd_tokens & resume_tokens) / len(jd_tokens)
            matched = overlap >= 0.5
        else:
            matched = True

    return {
        "status": "PASS" if matched else "FAIL",
        "jd_title": jd_title,
        "resume_title": resume_title,
        "match_mode": "exact" if requires_exact else "token_overlap",
    }


# ---------------------------------------------------------------------------
# Bullet Point Density Check
# ---------------------------------------------------------------------------
def _check_bullet_density(
    resume: dict[str, Any],
    min_bullets: int,
) -> list[str]:
    """Check that each experience entry has at least min_bullets bullet points."""
    issues: list[str] = []
    experiences = resume.get("experience", [])
    if not isinstance(experiences, list):
        return issues
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        bullets = exp.get("bullets", [])
        role = exp.get("role", "Unknown role")
        company = exp.get("company", "Unknown company")
        if len(bullets) < min_bullets:
            issues.append(
                f"'{role}' at '{company}' has {len(bullets)} bullet(s), "
                f"minimum recommended: {min_bullets}"
            )
    return issues


# ---------------------------------------------------------------------------
# Platform-Specific Contact Validation
# ---------------------------------------------------------------------------
def _evaluate_platform_contact(
    resume: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate contact info against platform-specific requirements."""
    contact_cfg = profile.get("contact_requirements", {})
    required_fields = contact_cfg.get("required_fields", ["name", "email"])
    recommended_fields = contact_cfg.get("recommended_fields", [])
    validates_email = contact_cfg.get("validates_email_format", True)
    validates_phone = contact_cfg.get("validates_phone_format", False)

    p_info = resume.get("personal_info", {})
    if not isinstance(p_info, dict):
        p_info = {}

    missing_required: list[str] = []
    missing_recommended: list[str] = []
    validation_issues: list[str] = []

    # Field presence mapping
    field_values = {
        "name": p_info.get("name", ""),
        "email": p_info.get("email", ""),
        "phone": p_info.get("phone", ""),
        "location": p_info.get("location", ""),
        "linkedin": p_info.get("linkedin", ""),
        "github": p_info.get("github", ""),
    }

    for field in required_fields:
        val = str(field_values.get(field, "")).strip()
        if not val:
            missing_required.append(field)

    for field in recommended_fields:
        val = str(field_values.get(field, "")).strip()
        if not val:
            missing_recommended.append(field)

    # Format validation
    email = str(field_values.get("email", "")).strip()
    if validates_email and email and ("@" not in email or "." not in email):
        validation_issues.append("Email format appears invalid")

    phone = str(field_values.get("phone", "")).strip()
    if validates_phone and phone and len(re.sub(r"\D", "", phone)) < 7:
        validation_issues.append("Phone number appears too short")

    passed = len(missing_required) == 0 and len(validation_issues) == 0
    return {
        "status": "PASS" if passed else "FAIL",
        "missing_required_fields": missing_required,
        "missing_recommended_fields": missing_recommended,
        "validation_issues": validation_issues,
    }


# ---------------------------------------------------------------------------
# Platform-Specific Section Validation
# ---------------------------------------------------------------------------
def _evaluate_platform_sections(
    resume: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate section presence against platform-specific requirements."""
    section_cfg = profile.get("section_requirements", {})
    required = section_cfg.get("required", [])
    recommended = section_cfg.get("recommended", [])

    present_sections: list[str] = []
    missing_required: list[str] = []
    missing_recommended: list[str] = []

    section_checks = {
        "summary": lambda r: isinstance(r.get("summary"), str) and len(r["summary"].strip()) > 10,
        "experience": lambda r: isinstance(r.get("experience"), list) and len(r["experience"]) > 0,
        "education": lambda r: isinstance(r.get("education"), list) and len(r["education"]) > 0,
        "skills": lambda r: (
            (isinstance(r.get("skills"), dict) and any(bool(v) for v in r["skills"].values()))
            or (isinstance(r.get("skills"), list) and len(r["skills"]) > 0)
        ),
        "projects": lambda r: isinstance(r.get("projects"), list) and len(r["projects"]) > 0,
        "certifications": lambda r: isinstance(r.get("certifications"), list) and len(r["certifications"]) > 0,
    }

    for section_name, checker in section_checks.items():
        if checker(resume):
            present_sections.append(section_name)

    for req in required:
        if req.lower() not in present_sections:
            missing_required.append(req)

    for rec in recommended:
        if rec.lower() not in present_sections:
            missing_recommended.append(rec)

    passed = len(missing_required) == 0
    return {
        "status": "PASS" if passed else "FAIL",
        "present_sections": present_sections,
        "missing_required_sections": missing_required,
        "missing_recommended_sections": missing_recommended,
    }


# ---------------------------------------------------------------------------
# Core: Evaluate Against One Platform
# ---------------------------------------------------------------------------
def _evaluate_single_platform(
    platform_key: str,
    profile: dict[str, Any],
    structured_jd: dict[str, Any],
    resume: dict[str, Any],
    normalized_resume_text: str,
    raw_resume_text: str,
) -> dict[str, Any]:
    """
    Run all deterministic checks for a single ATS platform profile.
    Returns a structured result dict with PASS/FAIL checks and measurable metrics.
    """
    checks: dict[str, Any] = {}
    critical_failures: list[str] = []
    warnings: list[str] = []
    platform_name = profile.get("platform_name", platform_key)

    kw_cfg = profile.get("keyword_matching", {})
    date_cfg = profile.get("date_parsing", {})
    exp_cfg = profile.get("experience_evaluation", {})
    weight_cfg = profile.get("scoring_weight_overrides", {})

    # --- 1. Keyword Coverage ---
    req_keywords = structured_jd.get("keywords") or structured_jd.get("required_keywords") or []
    matched_kw: list[str] = []
    missing_kw: list[str] = []

    matching_mode = kw_cfg.get("mode", "exact_and_synonym")

    for kw in req_keywords:
        kw_str = str(kw)
        if matching_mode == "exact_only":
            found = _normalize_text(kw_str) in normalized_resume_text
        else:
            found = _is_phrase_present(kw_str, normalized_resume_text)

        if found:
            matched_kw.append(kw_str)
        else:
            missing_kw.append(kw_str)

    kw_total = len(req_keywords) if req_keywords else 1
    kw_coverage = round(len(matched_kw) / kw_total * 100, 1)

    checks["keyword_coverage"] = {
        "status": "PASS" if kw_coverage >= 70 else "FAIL",
        "coverage_percent": kw_coverage,
        "matched": matched_kw,
        "missing": missing_kw,
        "matching_mode": matching_mode,
    }

    if kw_coverage < 50:
        critical_failures.append(f"[{platform_name}] Required keyword coverage critically low: {kw_coverage}%")

    # --- 2. Required Skills Coverage ---
    req_skills = structured_jd.get("required_skills", [])
    matched_sk: list[str] = []
    missing_sk: list[str] = []

    for sk in req_skills:
        sk_str = str(sk)
        if matching_mode == "exact_only":
            found = _normalize_text(sk_str) in normalized_resume_text
        else:
            found = _is_phrase_present(sk_str, normalized_resume_text)

        if found:
            matched_sk.append(sk_str)
        else:
            missing_sk.append(sk_str)

    sk_total = len(req_skills) if req_skills else 1
    sk_coverage = round(len(matched_sk) / sk_total * 100, 1)

    checks["required_skills"] = {
        "status": "PASS" if sk_coverage >= 70 else "FAIL",
        "coverage_percent": sk_coverage,
        "matched": matched_sk,
        "missing": missing_sk,
    }

    if sk_coverage < 50:
        critical_failures.append(f"[{platform_name}] Required skill coverage critically low: {sk_coverage}%")

    # --- 3. Preferred Skills Coverage ---
    pref_skills = structured_jd.get("preferred_skills", [])
    matched_pref: list[str] = []
    missing_pref: list[str] = []

    for ps in pref_skills:
        ps_str = str(ps)
        found = _is_phrase_present(ps_str, normalized_resume_text)
        if found:
            matched_pref.append(ps_str)
        else:
            missing_pref.append(ps_str)

    pref_total = len(pref_skills) if pref_skills else 1
    pref_coverage = round(len(matched_pref) / pref_total * 100, 1)

    checks["preferred_skills"] = {
        "status": "PASS" if pref_coverage >= 50 else "WARN",
        "coverage_percent": pref_coverage,
        "matched": matched_pref,
        "missing": missing_pref,
    }

    # --- 4. Job Title Match ---
    requires_exact_title = kw_cfg.get("requires_exact_job_title_match", False)
    title_result = _check_job_title_match(resume, structured_jd, requires_exact_title)
    checks["job_title_match"] = title_result

    if title_result["status"] == "FAIL" and requires_exact_title:
        critical_failures.append(
            f"[{platform_name}] Job title mismatch: JD='{title_result['jd_title']}' "
            f"vs Resume='{title_result['resume_title']}'"
        )

    # --- 5. Section Coverage ---
    section_result = _evaluate_platform_sections(resume, profile)
    checks["sections"] = section_result

    if section_result["status"] == "FAIL":
        for ms in section_result["missing_required_sections"]:
            critical_failures.append(f"[{platform_name}] Missing required section: {ms}")

    # --- 6. Contact Information ---
    contact_result = _evaluate_platform_contact(resume, profile)
    checks["contact_info"] = contact_result

    if contact_result["status"] == "FAIL":
        for mf in contact_result["missing_required_fields"]:
            critical_failures.append(f"[{platform_name}] Missing required contact field: {mf}")

    # --- 7. Date Parsing ---
    date_issues: list[str] = []
    strictness = date_cfg.get("strictness", "moderate")

    if date_cfg.get("flags_future_dates", False):
        future_issues = _check_future_dates(resume)
        date_issues.extend(future_issues)

    gap_months = date_cfg.get("flags_gaps_over_months")
    gap_issues = _check_employment_gaps(resume, gap_months)
    date_issues.extend(gap_issues)

    checks["date_parsing"] = {
        "status": "PASS" if not date_issues else ("WARN" if strictness == "lenient" else "FAIL"),
        "strictness": strictness,
        "issues": date_issues,
    }

    if strictness == "strict" and date_issues:
        for di in date_issues:
            warnings.append(f"[{platform_name}] Date issue: {di}")

    # --- 8. Formatting Risks ---
    fmt_cfg = profile.get("formatting_risks", {})
    formatting_risks: list[str] = []

    # For structured JSON resumes, most formatting risks don't apply directly.
    # We flag awareness for downstream PDF/DOCX generation.
    if fmt_cfg.get("rejects_tables", False):
        formatting_risks.append("Platform rejects tables in documents")
    if fmt_cfg.get("rejects_columns", False):
        formatting_risks.append("Platform rejects multi-column layouts")
    if fmt_cfg.get("rejects_headers_footers", False):
        formatting_risks.append("Platform ignores headers/footers content")
    if fmt_cfg.get("rejects_images", False):
        formatting_risks.append("Platform ignores embedded images")

    checks["formatting_risks"] = {
        "status": "INFO",
        "accepted_formats": fmt_cfg.get("accepted_formats", []),
        "risks": formatting_risks,
    }

    # --- 9. Keyword Stuffing ---
    stuffing_threshold = kw_cfg.get("keyword_stuffing_threshold")
    all_keywords = req_keywords + [str(s) for s in req_skills]
    stuffing_issues = _check_keyword_stuffing(raw_resume_text, all_keywords, stuffing_threshold)

    checks["keyword_stuffing"] = {
        "status": "PASS" if not stuffing_issues else "WARN",
        "issues": stuffing_issues,
    }

    if stuffing_issues:
        for si in stuffing_issues:
            warnings.append(f"[{platform_name}] Keyword stuffing: {si}")

    # --- 10. Bullet Point Density ---
    min_bullets = exp_cfg.get("min_bullet_points_per_role", 1)
    bullet_issues = _check_bullet_density(resume, min_bullets)

    checks["bullet_density"] = {
        "status": "PASS" if not bullet_issues else "WARN",
        "min_required": min_bullets,
        "issues": bullet_issues,
    }

    # --- 11. Resume Length (reuse existing) ---
    length_score, length_issues = evaluate_resume_length(resume)
    word_count = len(raw_resume_text.split())
    if length_score >= 85:
        length_status = "PASS"
    elif word_count > 0:
        length_status = "WARN"
    else:
        length_status = "FAIL"

    checks["resume_length"] = {
        "status": length_status,
        "score": length_score,
        "issues": length_issues,
    }

    # --- Compute Platform Compatibility Status ---
    fail_count = sum(
        1 for c in checks.values()
        if isinstance(c, dict) and c.get("status") == "FAIL"
    )
    warn_count = sum(
        1 for c in checks.values()
        if isinstance(c, dict) and c.get("status") == "WARN"
    )

    if fail_count > 0:
        overall_status = "FAIL"
    elif warn_count > 0:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    return {
        "platform_name": platform_name,
        "overall_status": overall_status,
        "checks": checks,
        "critical_failures": critical_failures,
        "warnings": warnings,
        "fail_count": fail_count,
        "warn_count": warn_count,
    }


# ---------------------------------------------------------------------------
# Public API: Multi-Platform Validation
# ---------------------------------------------------------------------------
def validate_multi_ats(
    structured_jd: dict[str, Any] | str | Path,
    structured_resume: dict[str, Any] | str | Path,
    profiles_path: str | Path | None = None,
    platforms: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a structured resume against multiple ATS platform simulations.

    Parameters:
        structured_jd: Structured Job Description dict or path to JSON file.
        structured_resume: Structured Resume dict or path to JSON file.
        profiles_path: Optional path to ats_profiles.json (defaults to data/ats_profiles.json).
        platforms: Optional list of platform keys to evaluate. None = all platforms.

    Returns:
        Structured multi-ATS validation report.
    """
    # Resolve inputs
    if isinstance(structured_jd, (str, Path)) and Path(str(structured_jd)).exists():
        with open(structured_jd, "r", encoding="utf-8") as f:
            jd_dict = json.load(f)
    elif isinstance(structured_jd, dict):
        jd_dict = structured_jd
    elif isinstance(structured_jd, str):
        jd_dict = json.loads(structured_jd)
    else:
        raise ValueError(f"Invalid structured_jd input: {type(structured_jd)}")

    if isinstance(structured_resume, (str, Path)) and Path(str(structured_resume)).exists():
        with open(structured_resume, "r", encoding="utf-8") as f:
            resume_dict = json.load(f)
    elif isinstance(structured_resume, dict):
        resume_dict = structured_resume
    elif isinstance(structured_resume, str):
        resume_dict = json.loads(structured_resume)
    else:
        raise ValueError(f"Invalid structured_resume input: {type(structured_resume)}")

    # Load profiles
    all_profiles = load_ats_profiles(profiles_path)

    # Determine which platforms to evaluate
    if platforms:
        target_keys = [k for k in platforms if k in all_profiles]
    else:
        target_keys = list(all_profiles.keys())

    if not target_keys:
        raise ValueError("No valid ATS platform profiles found to evaluate.")

    # Pre-compute shared text extraction (used by all platforms)
    raw_resume_text = _extract_all_resume_text(resume_dict)
    normalized_resume_text = _normalize_text(raw_resume_text)

    # Evaluate each platform
    platform_results: dict[str, Any] = {}
    all_critical_failures: list[str] = []
    all_recommendations: list[str] = []
    pass_count = 0
    fail_count = 0
    warn_count = 0

    for key in target_keys:
        profile = all_profiles[key]
        result = _evaluate_single_platform(
            platform_key=key,
            profile=profile,
            structured_jd=jd_dict,
            resume=resume_dict,
            normalized_resume_text=normalized_resume_text,
            raw_resume_text=raw_resume_text,
        )
        platform_results[key] = result
        all_critical_failures.extend(result.get("critical_failures", []))

        status = result.get("overall_status", "FAIL")
        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1
        else:
            warn_count += 1

    # Build cross-platform recommendations
    # Collect missing keywords/skills across all strict platforms
    missing_kw_across: dict[str, int] = {}
    missing_sk_across: dict[str, int] = {}

    for key, result in platform_results.items():
        checks = result.get("checks", {})
        for mk in checks.get("keyword_coverage", {}).get("missing", []):
            missing_kw_across[mk] = missing_kw_across.get(mk, 0) + 1
        for ms in checks.get("required_skills", {}).get("missing", []):
            missing_sk_across[ms] = missing_sk_across.get(ms, 0) + 1

    # Keywords missing across 2+ platforms are high-priority
    for kw, count in sorted(missing_kw_across.items(), key=lambda x: -x[1]):
        if count >= 2:
            all_recommendations.append(
                f"High priority: Add keyword '{kw}' (missing on {count}/{len(target_keys)} platforms)"
            )
        else:
            all_recommendations.append(
                f"Consider adding keyword '{kw}' (missing on 1 platform)"
            )

    for sk, count in sorted(missing_sk_across.items(), key=lambda x: -x[1]):
        if count >= 2:
            all_recommendations.append(
                f"High priority: Add skill '{sk}' (missing on {count}/{len(target_keys)} platforms)"
            )

    # Determine overall status
    if fail_count > 0:
        overall_status = "FAIL"
    elif warn_count > 0:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    return {
        "disclaimer": (
            "These results are behavioral simulations based on publicly available "
            "ATS documentation. They do NOT represent guaranteed real ATS scores "
            "or access to proprietary algorithms."
        ),
        "platforms": platform_results,
        "summary": {
            "total_platforms": len(target_keys),
            "passed": pass_count,
            "warned": warn_count,
            "failed": fail_count,
        },
        "critical_failures": all_critical_failures,
        "recommendations": all_recommendations,
        "overall_status": overall_status,
    }


# ---------------------------------------------------------------------------
# CLI Test Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Reuse the same test data as ats_analyzer.py for consistency
    sample_jd = {
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

    sample_resume = {
        "personal_info": {
            "name": "Kanishk Surwade",
            "target_title": "Data Annotator",
            "email": "kanishksurwade70@gmail.com",
            "phone": "+91-9834008224",
            "location": "Pune, Maharashtra, India",
            "linkedin": "linkedin.com/in/kd4723",
            "github": "github.com/Kanishksurwade",
        },
        "summary": (
            "AI & LLM Analyst with extensive production experience in multimodal data annotation, "
            "including precise video and audio labeling for enterprise datasets. Proven expertise in "
            "adhering to strict SOPs for quality benchmarking, identifying edge cases, and maintaining "
            "high inter-annotator accuracy standards. Skilled in working independently in remote "
            "environments using platforms like CVAT and Google Cloud Platform."
        ),
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
                    "Executed precise multimodal data annotation for text, audio, video, and image samples.",
                    "Performed comprehensive quality assurance by reviewing annotations and documenting edge cases.",
                    "Collaborated cross-functionally to resolve annotation ambiguities.",
                ],
            },
            {
                "company": "Deloitte",
                "role": "Data Analytics Intern",
                "location": "Virtual Internship",
                "start_date": "Sep 2025",
                "end_date": "Sep 2025",
                "bullets": [
                    "Analyzed operational datasets to identify performance gaps.",
                    "Developed KPI dashboards and calculated metrics using Excel.",
                ],
            },
        ],
        "projects": [
            {
                "name": "Google Tunix - Structured Reasoning Fine-Tuning",
                "technologies": ["Gemma 3 (1B)", "GRPO", "LoRA", "JAX", "Flax", "Google Cloud", "TPU"],
                "start_date": "Dec 2025",
                "end_date": "Jan 2026",
                "bullets": [
                    "Designed reward functions and prompt templates for output consistency.",
                    "Built TPU-based training and evaluation pipeline using JAX and Flax.",
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
        },
    }

    print("=" * 70)
    print("RESDEV AI - MULTI-ATS COMPATIBILITY VALIDATOR")
    print("=" * 70)
    print("Candidate:", sample_resume.get("personal_info", {}).get("name"))
    print("Target Role:", sample_jd.get("job_title"))
    print("Platforms: Workday | Taleo | iCIMS | Greenhouse | Lever | SuccessFactors")
    print("-" * 70)

    result = validate_multi_ats(
        structured_jd=sample_jd,
        structured_resume=sample_resume,
    )

    # Print per-platform summary
    for pkey, presult in result["platforms"].items():
        pname = presult["platform_name"]
        pstatus = presult["overall_status"]
        fails = presult["fail_count"]
        warns = presult["warn_count"]
        icon = "[+]" if pstatus == "PASS" else ("[!]" if pstatus == "WARN" else "[x]")
        print(f"  {icon} {pname:<25} {pstatus:<6}  (Fails: {fails}, Warnings: {warns})")

        # Show keyword and skill coverage
        checks = presult.get("checks", {})
        kw_cov = checks.get("keyword_coverage", {}).get("coverage_percent", "?")
        sk_cov = checks.get("required_skills", {}).get("coverage_percent", "?")
        title_status = checks.get("job_title_match", {}).get("status", "?")
        print(f"    Keywords: {kw_cov}% | Skills: {sk_cov}% | Title Match: {title_status}")

        # Show critical failures for this platform
        for cf in presult.get("critical_failures", []):
            print(f"    [x] {cf}")
        print()

    # Overall summary
    print("-" * 70)
    summary = result["summary"]
    print(f"Overall: {result['overall_status']} "
          f"({summary['passed']} passed, {summary['warned']} warned, {summary['failed']} failed "
          f"out of {summary['total_platforms']})")

    if result["critical_failures"]:
        print(f"\nCritical Failures ({len(result['critical_failures'])}):")
        for cf in result["critical_failures"]:
            print(f"  [x] {cf}")

    if result["recommendations"]:
        print(f"\nCross-Platform Recommendations ({len(result['recommendations'])}):")
        for rec in result["recommendations"]:
            print(f"  [>] {rec}")

    print("\n" + result["disclaimer"])
    print("=" * 70)

    # Also dump full JSON for inspection
    print("\n[FULL JSON RESULT]")
    print(json.dumps(result, indent=2, default=str))
