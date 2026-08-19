"""
Deterministic unit tests for the Multi-ATS Compatibility Validator.

Tests cover:
    - Profile loading
    - Date parsing
    - Future date detection
    - Employment gap detection
    - Keyword stuffing detection
    - Job title matching (exact and token overlap)
    - Bullet density checks
    - Platform-specific contact validation
    - Platform-specific section validation
    - Single-platform evaluation
    - Full multi-platform validation
    - Edge cases (empty resume, missing sections)

No LLM calls. No network. Fully deterministic.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.multi_ats_validator import (
    load_ats_profiles,
    validate_multi_ats,
    _parse_date_string,
    _check_future_dates,
    _check_employment_gaps,
    _check_keyword_stuffing,
    _check_job_title_match,
    _check_bullet_density,
    _evaluate_platform_contact,
    _evaluate_platform_sections,
    _evaluate_single_platform,
)
from ai.ats_analyzer import _normalize_text, _extract_all_resume_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_jd():
    """Minimal structured JD for testing."""
    return {
        "job_title": "Data Annotator",
        "required_skills": [
            "Data annotation",
            "Attention to detail",
            "Video annotation",
        ],
        "preferred_skills": [
            "Remote work experience",
            "Strong analytical skills",
        ],
        "keywords": [
            "Data Annotator",
            "Data annotation",
            "Video annotation",
            "Quality assurance",
        ],
    }


@pytest.fixture
def sample_resume():
    """Minimal structured resume for testing."""
    return {
        "personal_info": {
            "name": "Test Candidate",
            "target_title": "Data Annotator",
            "email": "test@example.com",
            "phone": "+1-555-123-4567",
            "location": "New York, NY",
            "linkedin": "linkedin.com/in/test",
            "github": "github.com/test",
        },
        "summary": (
            "Experienced data annotator with expertise in video and audio annotation, "
            "quality assurance, and attention to detail across multiple labeling platforms."
        ),
        "skills": {
            "technical_skills": [
                "Data Annotation",
                "Video Annotation",
                "Audio Annotation",
                "Quality Assurance",
            ],
            "core_competencies": [
                "Attention to Detail",
                "Strong Analytical Skills",
                "Remote Work Experience",
            ],
        },
        "experience": [
            {
                "company": "AnnotateCo",
                "role": "Senior Data Annotator",
                "location": "Remote",
                "start_date": "Jan 2024",
                "end_date": "Present",
                "bullets": [
                    "Annotated 10,000+ video frames with bounding boxes.",
                    "Conducted quality assurance reviews on peer annotations.",
                    "Documented edge cases and updated labeling guidelines.",
                ],
            },
            {
                "company": "DataCorp",
                "role": "Data Analyst",
                "location": "New York, NY",
                "start_date": "Jun 2022",
                "end_date": "Dec 2023",
                "bullets": [
                    "Analyzed datasets for pattern recognition.",
                    "Created dashboards for operational metrics.",
                ],
            },
        ],
        "projects": [
            {
                "name": "Image Classification Pipeline",
                "technologies": ["Python", "TensorFlow", "OpenCV"],
                "start_date": "Mar 2023",
                "end_date": "Jun 2023",
                "bullets": [
                    "Built automated image classification pipeline.",
                    "Achieved 95% accuracy on test dataset.",
                ],
            },
        ],
        "education": [
            {
                "degree": "B.S. in Computer Science",
                "institution": "State University",
                "location": "New York, NY",
                "start_year": "2018",
                "end_year": "2022",
            }
        ],
        "certifications": [
            {
                "name": "AWS Certified Cloud Practitioner",
                "issuer": "Amazon Web Services",
            }
        ],
        "tailoring_metadata": {
            "target_role": "Data Annotator",
            "primary_keywords_integrated": [
                "Data annotation",
                "Video annotation",
                "Quality assurance",
            ],
        },
    }


@pytest.fixture
def workday_profile():
    """Workday profile extracted for unit testing."""
    profiles = load_ats_profiles()
    return profiles["workday"]


@pytest.fixture
def taleo_profile():
    """Taleo profile extracted for unit testing."""
    profiles = load_ats_profiles()
    return profiles["taleo"]


@pytest.fixture
def greenhouse_profile():
    """Greenhouse profile extracted for unit testing."""
    profiles = load_ats_profiles()
    return profiles["greenhouse"]


# ---------------------------------------------------------------------------
# Test: Profile Loading
# ---------------------------------------------------------------------------
class TestProfileLoading:
    def test_load_all_profiles(self):
        """All 6 profiles should load successfully."""
        profiles = load_ats_profiles()
        assert isinstance(profiles, dict)
        expected_keys = {"workday", "taleo", "icims", "greenhouse", "lever", "successfactors"}
        assert set(profiles.keys()) == expected_keys

    def test_each_profile_has_required_fields(self):
        """Each profile must have core configuration sections."""
        profiles = load_ats_profiles()
        required_sections = [
            "platform_name",
            "keyword_matching",
            "section_requirements",
            "contact_requirements",
            "formatting_risks",
            "date_parsing",
            "experience_evaluation",
            "scoring_weight_overrides",
        ]
        for key, profile in profiles.items():
            for section in required_sections:
                assert section in profile, f"Profile '{key}' missing '{section}'"

    def test_scoring_weights_sum_to_one(self):
        """Each profile's scoring weights should sum to 1.0."""
        profiles = load_ats_profiles()
        for key, profile in profiles.items():
            weights = profile["scoring_weight_overrides"]
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, (
                f"Profile '{key}' weights sum to {total}, expected 1.0"
            )

    def test_invalid_path_raises(self):
        """Loading from a non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_ats_profiles("nonexistent/path.json")


# ---------------------------------------------------------------------------
# Test: Date Parsing
# ---------------------------------------------------------------------------
class TestDateParsing:
    def test_month_year_format(self):
        assert _parse_date_string("Dec 2025") == (2025, 12)
        assert _parse_date_string("January 2023") == (2023, 1)

    def test_mm_yyyy_format(self):
        assert _parse_date_string("01/2024") == (2024, 1)
        assert _parse_date_string("12-2023") == (2023, 12)

    def test_bare_year(self):
        year, month = _parse_date_string("2022")
        assert year == 2022
        assert month is None

    def test_present_keyword(self):
        assert _parse_date_string("Present") == (None, None)
        assert _parse_date_string("current") == (None, None)

    def test_empty_string(self):
        assert _parse_date_string("") == (None, None)

    def test_none_input(self):
        assert _parse_date_string(None) == (None, None)

    def test_garbage_input(self):
        assert _parse_date_string("not_a_date") == (None, None)


# ---------------------------------------------------------------------------
# Test: Future Date Detection
# ---------------------------------------------------------------------------
class TestFutureDateDetection:
    def test_no_future_dates(self):
        """Resume with all past dates should return empty list."""
        resume = {
            "experience": [
                {"role": "Analyst", "start_date": "Jan 2020", "end_date": "Dec 2023"},
            ],
            "projects": [],
        }
        issues = _check_future_dates(resume)
        assert issues == []

    def test_detects_future_end_date(self):
        """Should detect an end_date far in the future."""
        resume = {
            "experience": [
                {"role": "Engineer", "start_date": "Jan 2024", "end_date": "Dec 2099"},
            ],
            "projects": [],
        }
        issues = _check_future_dates(resume)
        assert len(issues) >= 1
        assert "future-dated" in issues[0].lower()

    def test_present_is_not_flagged(self):
        """'Present' as end_date should not be flagged as future."""
        resume = {
            "experience": [
                {"role": "Developer", "start_date": "Jan 2023", "end_date": "Present"},
            ],
            "projects": [],
        }
        issues = _check_future_dates(resume)
        assert issues == []


# ---------------------------------------------------------------------------
# Test: Employment Gap Detection
# ---------------------------------------------------------------------------
class TestEmploymentGaps:
    def test_no_gaps(self):
        """Continuous employment should have no gaps."""
        resume = {
            "experience": [
                {"role": "Role A", "start_date": "Jan 2023", "end_date": "Dec 2023"},
                {"role": "Role B", "start_date": "Jan 2024", "end_date": "Present"},
            ],
        }
        gaps = _check_employment_gaps(resume, max_gap_months=6)
        assert gaps == []

    def test_detects_large_gap(self):
        """Should detect a gap > threshold."""
        resume = {
            "experience": [
                {"role": "Role A", "start_date": "Jan 2020", "end_date": "Jun 2020"},
                {"role": "Role B", "start_date": "Jun 2022", "end_date": "Dec 2023"},
            ],
        }
        gaps = _check_employment_gaps(resume, max_gap_months=6)
        assert len(gaps) >= 1
        assert "gap" in gaps[0].lower()

    def test_none_threshold_skips(self):
        """max_gap_months=None should skip gap detection."""
        resume = {
            "experience": [
                {"role": "Role A", "start_date": "Jan 2018", "end_date": "Jan 2019"},
                {"role": "Role B", "start_date": "Jan 2023", "end_date": "Dec 2023"},
            ],
        }
        gaps = _check_employment_gaps(resume, max_gap_months=None)
        assert gaps == []

    def test_single_entry_no_gap(self):
        """Single experience entry should never report gaps."""
        resume = {
            "experience": [
                {"role": "Only Role", "start_date": "Jan 2020", "end_date": "Dec 2023"},
            ],
        }
        gaps = _check_employment_gaps(resume, max_gap_months=6)
        assert gaps == []


# ---------------------------------------------------------------------------
# Test: Keyword Stuffing
# ---------------------------------------------------------------------------
class TestKeywordStuffing:
    def test_no_stuffing(self):
        text = "data annotation is important for video annotation work"
        issues = _check_keyword_stuffing(text, ["data annotation", "video annotation"], threshold=5)
        assert issues == []

    def test_detects_stuffing(self):
        text = " ".join(["data annotation"] * 10)
        issues = _check_keyword_stuffing(text, ["data annotation"], threshold=5)
        assert len(issues) == 1
        assert "data annotation" in issues[0].lower()

    def test_none_threshold_skips(self):
        text = " ".join(["data annotation"] * 100)
        issues = _check_keyword_stuffing(text, ["data annotation"], threshold=None)
        assert issues == []


# ---------------------------------------------------------------------------
# Test: Job Title Match
# ---------------------------------------------------------------------------
class TestJobTitleMatch:
    def test_exact_match_pass(self):
        resume = {"personal_info": {"target_title": "Data Annotator"}}
        jd = {"job_title": "Data Annotator"}
        result = _check_job_title_match(resume, jd, requires_exact=True)
        assert result["status"] == "PASS"

    def test_exact_match_fail(self):
        resume = {"personal_info": {"target_title": "Senior Data Analyst"}}
        jd = {"job_title": "Data Annotator"}
        result = _check_job_title_match(resume, jd, requires_exact=True)
        assert result["status"] == "FAIL"

    def test_token_overlap_pass(self):
        resume = {"personal_info": {"target_title": "Senior Data Annotator"}}
        jd = {"job_title": "Data Annotator"}
        result = _check_job_title_match(resume, jd, requires_exact=False)
        assert result["status"] == "PASS"

    def test_case_insensitive(self):
        resume = {"personal_info": {"target_title": "data annotator"}}
        jd = {"job_title": "Data Annotator"}
        result = _check_job_title_match(resume, jd, requires_exact=True)
        assert result["status"] == "PASS"

    def test_missing_jd_title_skips(self):
        resume = {"personal_info": {"target_title": "Anything"}}
        jd = {"job_title": ""}
        result = _check_job_title_match(resume, jd, requires_exact=True)
        assert result["status"] == "SKIP"


# ---------------------------------------------------------------------------
# Test: Bullet Density
# ---------------------------------------------------------------------------
class TestBulletDensity:
    def test_sufficient_bullets(self):
        resume = {
            "experience": [
                {"role": "Dev", "company": "Co", "bullets": ["Did X", "Did Y"]},
            ]
        }
        issues = _check_bullet_density(resume, min_bullets=2)
        assert issues == []

    def test_insufficient_bullets(self):
        resume = {
            "experience": [
                {"role": "Dev", "company": "Co", "bullets": ["Did X"]},
            ]
        }
        issues = _check_bullet_density(resume, min_bullets=3)
        assert len(issues) == 1
        assert "Dev" in issues[0]


# ---------------------------------------------------------------------------
# Test: Platform Contact Validation
# ---------------------------------------------------------------------------
class TestPlatformContact:
    def test_workday_full_contact_passes(self, sample_resume, workday_profile):
        result = _evaluate_platform_contact(sample_resume, workday_profile)
        assert result["status"] == "PASS"
        assert result["missing_required_fields"] == []

    def test_missing_email_fails(self, workday_profile):
        resume = {
            "personal_info": {
                "name": "Test",
                "phone": "+1-555-123-4567",
            }
        }
        result = _evaluate_platform_contact(resume, workday_profile)
        assert result["status"] == "FAIL"
        assert "email" in result["missing_required_fields"]

    def test_greenhouse_only_name_email_required(self, greenhouse_profile):
        """Greenhouse only requires name and email."""
        resume = {
            "personal_info": {
                "name": "Test",
                "email": "test@example.com",
            }
        }
        result = _evaluate_platform_contact(resume, greenhouse_profile)
        assert result["status"] == "PASS"

    def test_invalid_email_format(self, workday_profile):
        """Invalid email should fail validation."""
        resume = {
            "personal_info": {
                "name": "Test",
                "email": "not-an-email",
                "phone": "+1-555-123-4567",
            }
        }
        result = _evaluate_platform_contact(resume, workday_profile)
        assert result["status"] == "FAIL"
        assert any("email" in i.lower() for i in result["validation_issues"])


# ---------------------------------------------------------------------------
# Test: Platform Section Validation
# ---------------------------------------------------------------------------
class TestPlatformSections:
    def test_full_resume_passes_workday(self, sample_resume, workday_profile):
        result = _evaluate_platform_sections(sample_resume, workday_profile)
        assert result["status"] == "PASS"

    def test_missing_experience_fails_workday(self, workday_profile):
        """Workday requires experience section."""
        resume = {
            "summary": "A valid summary with enough words to pass.",
            "skills": {"technical": ["Python"]},
            "education": [{"degree": "BS", "institution": "U"}],
        }
        result = _evaluate_platform_sections(resume, workday_profile)
        assert result["status"] == "FAIL"
        assert "experience" in result["missing_required_sections"]

    def test_lever_minimal_passes(self):
        """Lever only requires experience and education."""
        profiles = load_ats_profiles()
        lever = profiles["lever"]
        resume = {
            "experience": [
                {"role": "Dev", "company": "Co", "bullets": ["Did work"]},
            ],
            "education": [{"degree": "BS", "institution": "U"}],
        }
        result = _evaluate_platform_sections(resume, lever)
        assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# Test: Single Platform Evaluation
# ---------------------------------------------------------------------------
class TestSinglePlatformEvaluation:
    def test_workday_full_resume(self, sample_jd, sample_resume, workday_profile):
        raw_text = _extract_all_resume_text(sample_resume)
        norm_text = _normalize_text(raw_text)
        result = _evaluate_single_platform(
            "workday", workday_profile, sample_jd, sample_resume, norm_text, raw_text
        )
        assert "overall_status" in result
        assert result["platform_name"] == "Workday"
        assert "checks" in result
        assert "keyword_coverage" in result["checks"]
        assert "required_skills" in result["checks"]
        assert "job_title_match" in result["checks"]
        assert "sections" in result["checks"]
        assert "contact_info" in result["checks"]
        assert "date_parsing" in result["checks"]

    def test_result_has_measurable_metrics(self, sample_jd, sample_resume, workday_profile):
        """Coverage percentages should be numeric, not invented scores."""
        raw_text = _extract_all_resume_text(sample_resume)
        norm_text = _normalize_text(raw_text)
        result = _evaluate_single_platform(
            "workday", workday_profile, sample_jd, sample_resume, norm_text, raw_text
        )
        kw_cov = result["checks"]["keyword_coverage"]["coverage_percent"]
        sk_cov = result["checks"]["required_skills"]["coverage_percent"]
        assert isinstance(kw_cov, (int, float))
        assert isinstance(sk_cov, (int, float))
        assert 0 <= kw_cov <= 100
        assert 0 <= sk_cov <= 100


# ---------------------------------------------------------------------------
# Test: Full Multi-ATS Validation
# ---------------------------------------------------------------------------
class TestMultiATSValidation:
    def test_validates_all_six_platforms(self, sample_jd, sample_resume):
        result = validate_multi_ats(sample_jd, sample_resume)
        assert "platforms" in result
        assert len(result["platforms"]) == 6
        expected = {"workday", "taleo", "icims", "greenhouse", "lever", "successfactors"}
        assert set(result["platforms"].keys()) == expected

    def test_has_summary_fields(self, sample_jd, sample_resume):
        result = validate_multi_ats(sample_jd, sample_resume)
        assert "summary" in result
        assert "total_platforms" in result["summary"]
        assert result["summary"]["total_platforms"] == 6
        assert "passed" in result["summary"]
        assert "warned" in result["summary"]
        assert "failed" in result["summary"]

    def test_has_overall_status(self, sample_jd, sample_resume):
        result = validate_multi_ats(sample_jd, sample_resume)
        assert result["overall_status"] in ("PASS", "WARN", "FAIL")

    def test_has_disclaimer(self, sample_jd, sample_resume):
        result = validate_multi_ats(sample_jd, sample_resume)
        assert "disclaimer" in result
        assert "simulation" in result["disclaimer"].lower()

    def test_critical_failures_is_list(self, sample_jd, sample_resume):
        result = validate_multi_ats(sample_jd, sample_resume)
        assert isinstance(result["critical_failures"], list)

    def test_recommendations_is_list(self, sample_jd, sample_resume):
        result = validate_multi_ats(sample_jd, sample_resume)
        assert isinstance(result["recommendations"], list)

    def test_selective_platforms(self, sample_jd, sample_resume):
        """Should only evaluate specified platforms."""
        result = validate_multi_ats(
            sample_jd, sample_resume, platforms=["workday", "greenhouse"]
        )
        assert len(result["platforms"]) == 2
        assert "workday" in result["platforms"]
        assert "greenhouse" in result["platforms"]
        assert "taleo" not in result["platforms"]

    def test_result_is_json_serializable(self, sample_jd, sample_resume):
        """Full result must be JSON serializable."""
        result = validate_multi_ats(sample_jd, sample_resume)
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert "platforms" in parsed


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_resume(self, sample_jd):
        """Empty resume should not crash, should produce FAIL results."""
        empty_resume = {}
        result = validate_multi_ats(sample_jd, empty_resume)
        assert result["overall_status"] == "FAIL"
        assert result["summary"]["failed"] > 0

    def test_empty_jd(self, sample_resume):
        """Empty JD should still produce results (all keywords trivially match)."""
        empty_jd = {}
        result = validate_multi_ats(empty_jd, sample_resume)
        assert "platforms" in result

    def test_resume_with_no_experience(self, sample_jd):
        """Resume without experience should fail section checks on strict platforms."""
        resume = {
            "personal_info": {
                "name": "Test",
                "email": "test@test.com",
                "phone": "+1-555-0000",
            },
            "summary": "A summary with enough words to be considered valid by the validator.",
            "skills": {"technical": ["Python", "Data Annotation"]},
            "education": [{"degree": "BS", "institution": "U"}],
        }
        result = validate_multi_ats(sample_jd, resume)
        # At least some platforms should flag missing experience
        failed_platforms = [
            k for k, v in result["platforms"].items()
            if v["overall_status"] == "FAIL"
        ]
        assert len(failed_platforms) > 0

    def test_resume_with_no_skills(self, sample_jd):
        """Resume without skills section should fail on platforms requiring it."""
        resume = {
            "personal_info": {
                "name": "Test",
                "email": "test@test.com",
                "phone": "+1-555-0000",
            },
            "summary": "A summary with enough words to be considered valid here.",
            "experience": [
                {"role": "Dev", "company": "Co", "bullets": ["Did X", "Did Y"]},
            ],
            "education": [{"degree": "BS", "institution": "U"}],
        }
        result = validate_multi_ats(sample_jd, resume)
        # Workday, Taleo, iCIMS, SuccessFactors require skills
        workday_sections = result["platforms"]["workday"]["checks"]["sections"]
        assert "skills" in workday_sections.get("missing_required_sections", [])
