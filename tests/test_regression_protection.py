"""
Regression protection and runtime date validation tests.

Tests verify:
    - Date parsing uses runtime system date (datetime.now()), avoiding hard-coded or stale dates
    - Past employment (e.g. Dec 2025 - Jul 2026 relative to Aug 2026) is recognized as valid
    - Present employment is recognized as ongoing and not flagged
    - Far future employment (e.g. 2099) is correctly flagged
    - Requirement Matrix prevents unsupported JD keywords from entering the resume
    - End-to-end optimizer automatically rolls back and preserves the best candidate
"""

import copy
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.multi_ats_validator import _check_future_dates, _parse_date_string
from ai.edit_plan import build_requirement_matrix, get_unsupported_gaps, get_unmet_valid_gaps
from ai.resume_optimizer import optimize_resume


class TestRuntimeDateValidation:
    def test_past_employment_not_flagged_as_future(self):
        """Employment up to current runtime month/year must not be flagged as future."""
        now = datetime.now()
        # Create a date in the past
        past_year = now.year - 1
        resume = {
            "experience": [
                {
                    "role": "AI Analyst",
                    "company": "Innodata",
                    "start_date": f"Jan {past_year}",
                    "end_date": f"Dec {past_year}",
                }
            ],
            "projects": [],
        }
        issues = _check_future_dates(resume)
        assert len(issues) == 0

    def test_dec_2025_to_jul_2026_valid_for_aug_2026(self):
        """Dec 2025 - Jul 2026 is in the past when runtime is August 2026 or later."""
        now = datetime.now()
        resume = {
            "experience": [
                {
                    "role": "AI & LLM Analyst",
                    "company": "Innodata Inc.",
                    "start_date": "Dec 2025",
                    "end_date": "Jul 2026",
                }
            ],
            "projects": [],
        }
        issues = _check_future_dates(resume)
        # If current year is >= 2026 and current month >= 8, Jul 2026 should not be flagged
        if now.year > 2026 or (now.year == 2026 and now.month >= 7):
            assert len(issues) == 0

    def test_present_employment_not_flagged(self):
        """'Present', 'Current', 'Ongoing' must never be flagged as future."""
        resume = {
            "experience": [
                {
                    "role": "Senior Engineer",
                    "company": "TechCorp",
                    "start_date": "Jan 2024",
                    "end_date": "Present",
                }
            ],
            "projects": [],
        }
        issues = _check_future_dates(resume)
        assert len(issues) == 0

    def test_far_future_date_correctly_flagged(self):
        """Dates in year 2099 must be flagged as future dates."""
        resume = {
            "experience": [
                {
                    "role": "Future Role",
                    "company": "FutureCorp",
                    "start_date": "Jan 2099",
                    "end_date": "Dec 2099",
                }
            ],
            "projects": [],
        }
        issues = _check_future_dates(resume)
        assert len(issues) >= 1
        assert any("future-dated" in i.lower() for i in issues)


class TestRequirementMatrixProtection:
    def test_unsupported_jd_terms_blocked(self):
        """JD terms with zero master resume evidence must have allowed_action='do_not_add'."""
        master = {
            "capabilities": {
                "skills": {
                    "tools_platforms": ["CVAT", "Python", "SQL"],
                    "ai_llm": ["Data annotation"],
                }
            }
        }
        jd = {
            "job_title": "Data Annotator",
            "required_skills": ["Data annotation", "CVAT", "Scale AI", "Labelbox"],
            "preferred_skills": ["Amazon Mechanical Turk"],
            "keywords": ["Quality assurance"],
        }
        current_res = {"skills": {"technical_skills": ["Data annotation"]}}

        matrix = build_requirement_matrix(jd, master, current_res)
        unsupported = get_unsupported_gaps(matrix)

        unsupported_texts = [u["text"] for u in unsupported]
        assert "Scale AI" in unsupported_texts
        assert "Labelbox" in unsupported_texts
        assert "Amazon Mechanical Turk" in unsupported_texts

        for u in unsupported:
            assert u["allowed_action"] == "do_not_add"
            assert u["evidence_status"] == "unsupported"


class TestOptimizerRollbackProtection:
    @patch("ai.resume_optimizer.analyze_job_description")
    @patch("ai.resume_optimizer.match_resume_to_jd")
    @patch("ai.resume_optimizer.generate_tailored_resume")
    @patch("ai.resume_optimizer.apply_edit_plan")
    @patch("ai.resume_optimizer.generate_targeted_edit_plan")
    @patch("ai.resume_optimizer.evaluate_resume")
    def test_optimizer_preserves_best_when_iteration_3_regresses(
        self,
        mock_eval,
        mock_gen_plan,
        mock_apply_plan,
        mock_gen,
        mock_match,
        mock_analyze,
    ):
        """
        Simulates:
            Iteration 1: Base ATS 80, Qwen 80 (Baseline)
            Iteration 2: High ATS 95, Qwen 88 (Best)
            Iteration 3: Regressed ATS 72, Qwen 75 (Worse)
        Asserts:
            Final result is Iteration 2 (ATS 95), not Iteration 3.
        """
        mock_match.return_value = {"overall_match_score": 80}
        mock_analyze.return_value = {"job_title": "Data Annotator"}

        # Candidate resumes for each iteration
        cand_1 = {
            "personal_info": {"name": "Test", "target_title": "Data Annotator", "email": "t@t.com", "phone": "1234567"},
            "summary": "AI Analyst with annotation experience in CVAT.",
            "skills": {"technical_skills": ["Data Annotation"]},
            "experience": [{"company": "Innodata Inc.", "role": "Analyst", "bullets": ["Annotated data."]}],
            "education": [{"degree": "BS", "institution": "College"}],
            "certifications": [{"name": "GenAI"}],
        }
        cand_2 = copy.deepcopy(cand_1)
        cand_2["skills"]["technical_skills"] = ["Data Annotation", "Video Annotation", "Quality Assurance"]

        cand_3_regressed = copy.deepcopy(cand_1)
        cand_3_regressed["skills"]["technical_skills"] = []  # Regressed

        # Iteration 1 uses generate_tailored_resume; Iterations 2 & 3 use apply_edit_plan
        mock_gen.return_value = cand_1
        mock_gen_plan.return_value = {"edits": []}
        mock_apply_plan.side_effect = [
            (cand_2, ["Applied valid edit"], []),
            (cand_3_regressed, ["Applied regression edit"], []),
        ]

        # Evaluation sequence
        mock_eval.side_effect = [
            {"overall_score": 80, "pass_status": False, "weaknesses": []},
            {"overall_score": 88, "pass_status": True, "weaknesses": []},
            {"overall_score": 75, "pass_status": False, "weaknesses": ["Missing skills"]},
        ]

        master = {
            "candidate": {"personal_info": {"name": "Test", "email": "t@t.com", "phone": "1234567"}},
            "capabilities": {
                "skills": {
                    "tools_platforms": ["CVAT", "Python"],
                    "ai_llm": ["Data Annotation", "Video Annotation", "Quality Assurance"],
                },
                "certifications": [{"name": "GenAI"}],
            },
            "experience": [{"company": "Innodata Inc.", "role": "Analyst", "bullets": ["Annotated data."]}],
            "education": [{"degree": "BS", "institution": "College"}],
        }

        jd = {
            "job_title": "Data Annotator",
            "required_skills": ["Data Annotation", "Video Annotation", "Quality Assurance"],
        }

        res = optimize_resume(
            master_resume=master,
            job_description=jd,
            target_score=90,
            max_iterations=3,
        )

        # Final best result MUST be from iteration 2
        assert res["best_iteration"] == 2
        assert res["best_ats_score"] >= 85
        # Decision history must record iteration 3 rejection
        it3_decision = [d for d in res["decision_history"] if d["iteration"] == 3][0]
        assert it3_decision["accepted_as_best"] is False
        assert "REJECTED" in it3_decision["decision_status"]
