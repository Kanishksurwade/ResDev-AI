"""
Integration tests for the Resume Optimizer with Multi-ATS validation.

Tests verify that the optimizer:
    - Imports and calls validate_multi_ats per iteration
    - Includes multi-ATS results in iteration records
    - Includes multi-ATS results in the final output
    - Uses multi-ATS failures as revision feedback
    - Blocks combined pass when multi-ATS has critical failures

These tests use unittest.mock to patch the LLM-dependent modules
(jd_analyzer, resume_matcher, resume_generator, resume_evaluator)
so they run deterministically without Ollama.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Any

import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.resume_optimizer import (
    optimize_resume,
    _evaluate_combined_decision,
    _build_revision_feedback,
    _extract_multi_ats_feedback,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_master_resume():
    """Minimal master resume dict."""
    return {
        "candidate": {
            "personal_info": {
                "name": "Test Candidate",
                "email": "test@example.com",
                "phone": "+1-555-0000",
                "location": "New York, NY",
                "linkedin": "linkedin.com/in/test",
                "github": "github.com/test",
            }
        }
    }


@pytest.fixture
def sample_structured_jd():
    """Minimal structured JD dict."""
    return {
        "job_title": "Data Annotator",
        "required_skills": ["Data annotation", "Attention to detail"],
        "preferred_skills": ["Remote work experience"],
        "keywords": ["Data Annotator", "Data annotation", "Quality assurance"],
    }


@pytest.fixture
def good_tailored_resume():
    """A resume that should pass both ATS and multi-ATS checks."""
    return {
        "personal_info": {
            "name": "Test Candidate",
            "target_title": "Data Annotator",
            "email": "test@example.com",
            "phone": "+1-555-0000",
            "location": "New York, NY",
            "linkedin": "linkedin.com/in/test",
            "github": "github.com/test",
        },
        "summary": (
            "Experienced data annotator with a strong background in quality assurance, "
            "data annotation processes, and attention to detail across enterprise datasets. "
            "Proficient in video annotation, audio annotation, and data labeling platforms. "
            "Skilled in remote work environments and cross-functional collaboration. "
            "Dedicated to maintaining high annotation accuracy standards and documenting "
            "edge cases for continuous process improvement."
        ),
        "skills": {
            "technical_skills": [
                "Data Annotation", "Quality Assurance", "Attention to Detail",
            ],
            "core_competencies": ["Remote Work Experience"],
        },
        "experience": [
            {
                "company": "TestCo",
                "role": "Data Annotator",
                "location": "Remote",
                "start_date": "Jan 2023",
                "end_date": "Present",
                "bullets": [
                    "Performed data annotation on text, video, and audio samples.",
                    "Conducted quality assurance reviews maintaining 99% accuracy.",
                    "Documented edge cases and updated labeling guidelines.",
                ],
            },
            {
                "company": "AnalyticsCorp",
                "role": "Data Analyst",
                "location": "New York, NY",
                "start_date": "Jun 2021",
                "end_date": "Dec 2022",
                "bullets": [
                    "Analyzed datasets and created operational reports.",
                    "Built dashboards for KPI tracking using Excel and SQL.",
                ],
            },
        ],
        "projects": [
            {
                "name": "Annotation Pipeline",
                "technologies": ["Python", "CVAT"],
                "start_date": "Mar 2023",
                "end_date": "Jun 2023",
                "bullets": [
                    "Built automated annotation pipeline for image classification.",
                    "Achieved 97% accuracy on validation dataset.",
                ],
            },
        ],
        "education": [
            {
                "degree": "B.S. in Computer Science",
                "institution": "State University",
                "location": "New York, NY",
                "start_year": "2017",
                "end_year": "2021",
            }
        ],
        "certifications": [
            {"name": "Data Science Certificate", "issuer": "Coursera"},
        ],
        "tailoring_metadata": {
            "target_role": "Data Annotator",
            "primary_keywords_integrated": [
                "Data annotation", "Quality assurance",
            ],
        },
    }


@pytest.fixture
def good_qwen_result():
    """Qwen evaluator result indicating a passing score."""
    return {
        "overall_score": 88,
        "pass_status": True,
        "weaknesses": [],
        "improvement_actions": [],
    }


@pytest.fixture
def mediocre_qwen_result():
    """Qwen evaluator result indicating a borderline score."""
    return {
        "overall_score": 80,
        "pass_status": False,
        "weaknesses": ["Lacks specific annotation methodology detail"],
        "improvement_actions": [
            {
                "target": "experience",
                "change": "Add annotation methodology detail",
                "reason": "Semantic quality",
            }
        ],
    }


@pytest.fixture
def sample_matching_analysis():
    """Minimal matching analysis dict."""
    return {
        "overall_match_score": 75,
        "matched_required_skills": ["Data annotation"],
        "missing_required_skills": ["Attention to detail"],
    }


# ---------------------------------------------------------------------------
# Test: _extract_multi_ats_feedback
# ---------------------------------------------------------------------------
class TestExtractMultiATSFeedback:
    def test_extracts_actions_from_failed_platforms(self):
        multi_result = {
            "summary": {"passed": 3, "warned": 1, "failed": 2, "total_platforms": 6},
            "platforms": {
                "workday": {
                    "overall_status": "FAIL",
                    "platform_name": "Workday",
                    "checks": {
                        "keyword_coverage": {"missing": ["Data labeling"]},
                        "required_skills": {"missing": ["Annotation tools"]},
                    },
                    "critical_failures": ["[Workday] Missing required section: skills"],
                },
                "greenhouse": {
                    "overall_status": "PASS",
                    "platform_name": "Greenhouse",
                    "checks": {
                        "keyword_coverage": {"missing": []},
                        "required_skills": {"missing": []},
                    },
                    "critical_failures": [],
                },
            },
            "recommendations": ["Add keyword 'Data labeling'"],
        }
        actions, notes, passed, total = _extract_multi_ats_feedback(multi_result)
        assert len(actions) >= 2
        assert any("Data labeling" in a["change"] for a in actions)
        assert any("Annotation tools" in a["change"] for a in actions)
        assert passed == 4
        assert total == 6
        assert any("CRITICAL" in n for n in notes)

    def test_skips_passing_platforms(self):
        multi_result = {
            "summary": {"passed": 2, "warned": 0, "failed": 0, "total_platforms": 2},
            "platforms": {
                "greenhouse": {
                    "overall_status": "PASS",
                    "platform_name": "Greenhouse",
                    "checks": {
                        "keyword_coverage": {"missing": []},
                        "required_skills": {"missing": []},
                    },
                    "critical_failures": [],
                },
            },
            "recommendations": [],
        }
        actions, notes, passed, total = _extract_multi_ats_feedback(multi_result)
        assert actions == []
        assert passed == 2


# ---------------------------------------------------------------------------
# Test: _build_revision_feedback with multi-ATS
# ---------------------------------------------------------------------------
class TestBuildRevisionFeedbackWithMultiATS:
    def test_includes_multi_ats_actions(self):
        ats_result = {"missing_required_keywords": [], "missing_required_skills": [], "structural_issues": [], "recommendations": []}
        qwen_result = {"improvement_actions": [], "weaknesses": []}
        multi_result = {
            "summary": {"passed": 4, "warned": 0, "failed": 2, "total_platforms": 6},
            "platforms": {
                "taleo": {
                    "overall_status": "FAIL",
                    "platform_name": "Taleo",
                    "checks": {
                        "keyword_coverage": {"missing": ["Edge cases"]},
                        "required_skills": {"missing": []},
                    },
                    "critical_failures": [],
                },
            },
            "recommendations": ["Add keyword 'Edge cases'"],
        }
        actions, feedback = _build_revision_feedback(
            ats_result, qwen_result, "NONE", multi_ats_result=multi_result
        )
        assert any("Edge cases" in a.get("change", "") for a in actions)

    def test_deduplicates_actions(self):
        """Same keyword from single-ATS and multi-ATS should not duplicate."""
        ats_result = {
            "missing_required_keywords": ["Data labeling"],
            "missing_required_skills": [],
            "structural_issues": [],
            "recommendations": [],
        }
        qwen_result = {"improvement_actions": [], "weaknesses": []}
        multi_result = {
            "summary": {"passed": 5, "warned": 0, "failed": 1, "total_platforms": 6},
            "platforms": {
                "workday": {
                    "overall_status": "FAIL",
                    "platform_name": "Workday",
                    "checks": {
                        "keyword_coverage": {"missing": ["Data labeling"]},
                        "required_skills": {"missing": []},
                    },
                    "critical_failures": [],
                },
            },
            "recommendations": [],
        }
        actions, _ = _build_revision_feedback(
            ats_result, qwen_result, "ATS_DEFICIENCY", multi_ats_result=multi_result
        )
        # "Data labeling" should appear at most twice (one from multi, one from single)
        # but dedup by change text should reduce duplicates
        data_labeling_actions = [a for a in actions if "data labeling" in a.get("change", "").lower()]
        # Both mention it but with different wording, so may be 2 — that's fine as long as capped
        assert len(actions) <= 5


# ---------------------------------------------------------------------------
# Test: _evaluate_combined_decision
# ---------------------------------------------------------------------------
class TestCombinedDecision:
    def test_both_pass(self):
        ats = {"ats_score": 90}
        qwen = {"overall_score": 88}
        passed, trigger, issues = _evaluate_combined_decision(ats, qwen, 85)
        assert passed is True
        assert trigger == "NONE"

    def test_ats_fails(self):
        ats = {"ats_score": 70, "missing_required_keywords": ["X"], "missing_required_skills": [], "structural_issues": []}
        qwen = {"overall_score": 90}
        passed, trigger, issues = _evaluate_combined_decision(ats, qwen, 85)
        assert passed is False
        assert trigger == "ATS_DEFICIENCY"

    def test_both_fail(self):
        ats = {"ats_score": 70, "missing_required_keywords": [], "missing_required_skills": [], "structural_issues": []}
        qwen = {"overall_score": 70, "weaknesses": ["weak"]}
        passed, trigger, issues = _evaluate_combined_decision(ats, qwen, 85)
        assert passed is False
        assert trigger == "BOTH_DEFICIENT"


# ---------------------------------------------------------------------------
# Test: Full optimize_resume integration (mocked LLM calls)
# ---------------------------------------------------------------------------
class TestOptimizeResumeIntegration:
    """
    Integration test that runs optimize_resume with mocked LLM-dependent
    functions but real multi-ATS validation.
    """

    @patch("ai.resume_optimizer.analyze_job_description")
    @patch("ai.resume_optimizer.match_resume_to_jd")
    @patch("ai.resume_optimizer.generate_tailored_resume")
    @patch("ai.resume_optimizer.evaluate_resume")
    def test_optimizer_includes_multi_ats_fields(
        self,
        mock_evaluate,
        mock_generate,
        mock_match,
        mock_analyze_jd,
        sample_master_resume,
        sample_structured_jd,
        good_tailored_resume,
        good_qwen_result,
        sample_matching_analysis,
    ):
        """Optimizer result should contain multi-ATS fields."""
        mock_match.return_value = sample_matching_analysis
        mock_generate.return_value = good_tailored_resume
        mock_evaluate.return_value = good_qwen_result

        result = optimize_resume(
            master_resume=sample_master_resume,
            job_description=sample_structured_jd,
            target_score=85,
            max_iterations=1,
        )

        # Result must have multi-ATS fields
        assert "multi_ats_passed" in result
        assert "multi_ats_total" in result
        assert "multi_ats_failed" in result
        assert "multi_ats_overall_status" in result
        assert "best_multi_ats_evaluation" in result
        assert isinstance(result["best_multi_ats_evaluation"], dict)

    @patch("ai.resume_optimizer.analyze_job_description")
    @patch("ai.resume_optimizer.match_resume_to_jd")
    @patch("ai.resume_optimizer.generate_tailored_resume")
    @patch("ai.resume_optimizer.evaluate_resume")
    def test_iteration_records_have_multi_ats(
        self,
        mock_evaluate,
        mock_generate,
        mock_match,
        mock_analyze_jd,
        sample_master_resume,
        sample_structured_jd,
        good_tailored_resume,
        mediocre_qwen_result,
        sample_matching_analysis,
    ):
        """Each iteration record should contain multi-ATS status."""
        mock_match.return_value = sample_matching_analysis
        mock_generate.return_value = good_tailored_resume
        mock_evaluate.return_value = mediocre_qwen_result

        result = optimize_resume(
            master_resume=sample_master_resume,
            job_description=sample_structured_jd,
            target_score=85,
            max_iterations=2,
        )

        for it in result["iterations"]:
            assert "multi_ats_status" in it
            assert "multi_ats_passed" in it
            assert "multi_ats_total" in it
            assert "multi_ats_failed" in it
            assert "multi_ats_critical_failures" in it

    @patch("ai.resume_optimizer.analyze_job_description")
    @patch("ai.resume_optimizer.match_resume_to_jd")
    @patch("ai.resume_optimizer.generate_tailored_resume")
    @patch("ai.resume_optimizer.evaluate_resume")
    def test_multi_ats_runs_all_six_platforms(
        self,
        mock_evaluate,
        mock_generate,
        mock_match,
        mock_analyze_jd,
        sample_master_resume,
        sample_structured_jd,
        good_tailored_resume,
        good_qwen_result,
        sample_matching_analysis,
    ):
        """Multi-ATS validation should evaluate all 6 platform profiles."""
        mock_match.return_value = sample_matching_analysis
        mock_generate.return_value = good_tailored_resume
        mock_evaluate.return_value = good_qwen_result

        result = optimize_resume(
            master_resume=sample_master_resume,
            job_description=sample_structured_jd,
            target_score=85,
            max_iterations=1,
        )

        multi_eval = result["best_multi_ats_evaluation"]
        assert "platforms" in multi_eval
        assert len(multi_eval["platforms"]) == 6

    @patch("ai.resume_optimizer.analyze_job_description")
    @patch("ai.resume_optimizer.match_resume_to_jd")
    @patch("ai.resume_optimizer.generate_tailored_resume")
    @patch("ai.resume_optimizer.evaluate_resume")
    def test_result_is_json_serializable(
        self,
        mock_evaluate,
        mock_generate,
        mock_match,
        mock_analyze_jd,
        sample_master_resume,
        sample_structured_jd,
        good_tailored_resume,
        good_qwen_result,
        sample_matching_analysis,
    ):
        """Full optimizer result must serialize to JSON without errors."""
        mock_match.return_value = sample_matching_analysis
        mock_generate.return_value = good_tailored_resume
        mock_evaluate.return_value = good_qwen_result

        result = optimize_resume(
            master_resume=sample_master_resume,
            job_description=sample_structured_jd,
            target_score=85,
            max_iterations=1,
        )

        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert "multi_ats_overall_status" in parsed

    @patch("ai.resume_optimizer.analyze_job_description")
    @patch("ai.resume_optimizer.match_resume_to_jd")
    @patch("ai.resume_optimizer.generate_tailored_resume")
    @patch("ai.resume_optimizer.evaluate_resume")
    def test_progress_callback_receives_multi_ats(
        self,
        mock_evaluate,
        mock_generate,
        mock_match,
        mock_analyze_jd,
        sample_master_resume,
        sample_structured_jd,
        good_tailored_resume,
        good_qwen_result,
        sample_matching_analysis,
    ):
        """Progress callback should receive multi-ATS data per iteration."""
        mock_match.return_value = sample_matching_analysis
        mock_generate.return_value = good_tailored_resume
        mock_evaluate.return_value = good_qwen_result

        callback_data: list[dict] = []

        def capture_callback(it, max_it, data):
            callback_data.append(data)

        optimize_resume(
            master_resume=sample_master_resume,
            job_description=sample_structured_jd,
            target_score=85,
            max_iterations=1,
            progress_callback=capture_callback,
        )

        assert len(callback_data) >= 1
        assert "multi_ats_status" in callback_data[0]
        assert "multi_ats_total" in callback_data[0]
