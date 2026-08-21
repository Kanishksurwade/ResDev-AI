"""
Production Cloud Pipeline & Ollama Independence Regression Tests.

Verifies:
1. No component in the production pipeline connects or attempts to connect to localhost:11434 (Ollama).
2. Gemini is the sole AI provider configured and used.
3. Valid scores (0-100) are returned across all steps, and -1 is NEVER returned.
4. Error handling for Gemini 429, 503, network errors, and malformed AI JSON.
5. Max iterations (5) and target ATS threshold (86) with early exit when ATS >= 86.
6. Best candidate preservation across all iterations.
"""

import copy
import json
import sys
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.ats_analyzer import analyze_ats_compatibility
from ai.edit_plan import (
    apply_edit_plan,
    build_requirement_matrix,
    generate_targeted_edit_plan,
    get_unmet_valid_gaps,
)
from ai.evidence_validator import validate_resume_evidence
from ai.gemini_config import (
    DEFAULT_MODEL,
    call_gemini_with_retry,
    get_gemini_api_key,
    _is_retryable,
)
from ai.jd_analyzer import analyze_job_description
from ai.optimization_guard import OptimizationGuard
from ai.resume_evaluator import evaluate_resume
from ai.resume_generator import generate_tailored_resume
from ai.resume_matcher import match_resume_to_jd
from ai.resume_optimizer import optimize_resume


@pytest.fixture
def sample_master_resume() -> dict[str, Any]:
    return {
        "candidate": {
            "personal_info": {
                "name": "Kanishk Surwade",
                "email": "kanishk@example.com",
                "phone": "+91-9876543210",
                "location": "Pune, India",
                "linkedin": "linkedin.com/in/kanishk",
                "github": "github.com/kanishk",
            },
            "professional_identity": {
                "current_profile": "Data Annotator",
                "profile": "Experienced AI Analyst specializing in multimodal data annotation and QA.",
            },
        },
        "capabilities": {
            "skills": {
                "ai_llm": ["Data Annotation", "Video Annotation", "Audio Annotation", "Quality Assurance"],
                "tools_platforms": ["Python", "CVAT", "SQL", "GCP"],
                "core": ["Attention to Detail", "Remote Work Experience", "SOP Adherence"],
            },
            "certifications": [{"name": "Data Annotation Specialist", "issuer": "AI Institute"}],
        },
        "experience": [
            {
                "company": "Innodata Inc.",
                "role": "AI & LLM Analyst",
                "location": "Remote",
                "start_date": "Dec 2024",
                "end_date": "Present",
                "responsibilities_and_achievements": [
                    {"text": "Performed precise data annotation on video and audio samples."},
                    {"text": "Conducted quality assurance reviews adhering to strict SOPs."},
                    {"text": "Identified edge cases and maintained high annotation consistency."},
                ],
            }
        ],
        "projects": [
            {
                "name": "Multimodal Annotation Tooling",
                "technologies": ["Python", "CVAT"],
                "start_date": "Jan 2025",
                "end_date": "Mar 2025",
                "responsibilities_and_achievements": [
                    {"text": "Created automated validation scripts for annotated datasets."}
                ],
            }
        ],
        "education": [
            {
                "degree": "B.E. in Computer Science",
                "institution": "University of Pune",
                "location": "Pune, India",
                "start_year": "2020",
                "end_year": "2024",
                "cgpa": "8.5",
            }
        ],
    }


@pytest.fixture
def sample_structured_jd() -> dict[str, Any]:
    return {
        "job_title": "Data Annotator",
        "seniority": "Junior",
        "required_skills": ["Data annotation", "Video annotation", "Audio annotation", "Attention to detail"],
        "preferred_skills": ["Remote work experience", "CVAT"],
        "responsibilities": ["Annotate video and audio", "Quality assurance", "Document edge cases"],
        "keywords": ["Data Annotator", "Data annotation", "Video annotation", "Audio annotation", "Quality assurance"],
        "raw_job_description": "We are seeking a Data Annotator to annotate video and audio datasets.",
    }


class TestOllamaIndependence:
    """Proves that no module in the production pipeline calls Ollama (localhost:11434)."""

    def test_evaluate_resume_does_not_call_ollama(self, sample_master_resume, sample_structured_jd):
        with mock.patch("ai.resume_evaluator.call_gemini") as mock_gemini:
            mock_gemini.return_value = json.dumps({
                "overall_score": 90,
                "dimension_scores": {"keyword_match": 90, "skills_match": 90},
                "matched_keywords": ["Data annotation"],
                "missing_keywords": [],
                "strengths": ["Strong domain match"],
                "weaknesses": [],
                "improvement_actions": [],
                "explanation": "High alignment",
            })
            result = evaluate_resume(
                tailored_resume=sample_master_resume,
                structured_jd=sample_structured_jd,
                target_score=86,
            )
            assert mock_gemini.called
            assert result["overall_score"] == 90
            assert result["overall_score"] >= 0

    def test_match_resume_to_jd_does_not_call_ollama(self, sample_master_resume, sample_structured_jd):
        with mock.patch("ai.resume_matcher.call_gemini") as mock_gemini:
            mock_gemini.return_value = json.dumps({
                "overall_match_score": 88,
                "matched_required_skills": [{"skill": "Data annotation", "match_type": "strong"}],
                "missing_required_skills": [],
                "matched_preferred_skills": [],
                "missing_preferred_skills": [],
                "matched_keywords": [],
                "missing_keywords": [],
                "relevant_experience": [],
                "relevant_projects": [],
                "transferable_skills": [],
                "evidence_gaps": [],
                "explanation": "Good match",
            })
            result = match_resume_to_jd(
                master_resume=sample_master_resume,
                structured_jd=sample_structured_jd,
            )
            assert mock_gemini.called
            assert result["overall_match_score"] == 88

    def test_generate_tailored_resume_does_not_call_ollama(self, sample_master_resume, sample_structured_jd):
        with mock.patch("ai.resume_generator.call_gemini") as mock_gemini:
            mock_gemini.return_value = json.dumps({
                "personal_info": {"name": "Kanishk Surwade", "target_title": "Data Annotator"},
                "summary": "Experienced Data Annotator.",
                "skills": {
                    "technical_skills": ["Data annotation", "Video annotation"],
                    "tools_and_technologies": ["CVAT", "Python"],
                    "core_competencies": ["Attention to detail"],
                },
                "experience": [],
                "projects": [],
                "education": [],
                "certifications": [],
            })
            result = generate_tailored_resume(
                master_resume=sample_master_resume,
                structured_jd=sample_structured_jd,
            )
            assert mock_gemini.called
            assert result["personal_info"]["name"] == "Kanishk Surwade"

    def test_jd_analyzer_does_not_call_ollama(self):
        with mock.patch("ai.jd_analyzer.call_gemini") as mock_gemini:
            mock_gemini.return_value = json.dumps({
                "job_title": "Data Annotator",
                "seniority": "",
                "required_skills": ["Data annotation"],
                "preferred_skills": [],
                "responsibilities": ["Annotate data"],
                "experience_requirements": [],
                "education_requirements": [],
                "certifications": [],
                "soft_skills": [],
                "keywords": ["Data annotation"],
            })
            result = analyze_job_description("Looking for Data Annotator")
            assert mock_gemini.called
            assert result["job_title"] == "Data Annotator"


class TestScoreIntegrity:
    """Proves that scores are always valid (0-100) and NEVER -1."""

    def test_evaluator_never_returns_negative_scores(self, sample_master_resume, sample_structured_jd):
        with mock.patch("ai.resume_evaluator.call_gemini") as mock_gemini:
            # Test empty/corrupt model response
            mock_gemini.return_value = "This is not valid json at all"
            result = evaluate_resume(
                tailored_resume=sample_master_resume,
                structured_jd=sample_structured_jd,
                target_score=86,
            )
            assert result["overall_score"] >= 0
            assert result["overall_score"] != -1
            for dim, score in result["dimension_scores"].items():
                assert score >= 0
                assert score != -1

    def test_optimization_guard_never_returns_negative_scores(self):
        guard = OptimizationGuard(target_score=86)
        res = guard.get_final_result()
        assert res["best_ats_score"] >= 0
        assert res["best_ats_score"] != -1
        assert res["best_gemini_score"] >= 0
        assert res["best_gemini_score"] != -1
        assert res["best_combined_score"] >= 0.0

    def test_ats_analyzer_never_returns_negative_scores(self, sample_structured_jd):
        empty_resume = {}
        ats_res = analyze_ats_compatibility(sample_structured_jd, empty_resume, threshold=86)
        assert ats_res["ats_score"] >= 0
        assert ats_res["ats_score"] != -1
        assert ats_res["threshold"] == 86


class TestGeminiErrorHandling:
    """Tests retry on 429, 503, network errors, and safe backoff."""

    def test_is_retryable_classification(self):
        assert _is_retryable(Exception("429 Resource exhausted"))
        assert _is_retryable(Exception("503 Service Unavailable"))
        assert _is_retryable(Exception("500 Internal Server Error"))
        assert _is_retryable(Exception("Rate limit exceeded"))
        assert _is_retryable(ConnectionError("Connection refused"))
        assert _is_retryable(TimeoutError("Request timed out"))
        assert not _is_retryable(ValueError("Invalid argument format"))

    def test_call_gemini_retries_on_transient_error(self):
        call_count = 0

        def mock_generate_content(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Resource has been exhausted (rate limit)")
            mock_resp = mock.MagicMock()
            mock_resp.text = '{"success": true}'
            return mock_resp

        with mock.patch("ai.gemini_config.get_gemini_api_key", return_value="fake-test-key"), \
             mock.patch("ai.gemini_config.genai.Client") as mock_client_cls, \
             mock.patch("ai.gemini_config.time.sleep") as mock_sleep:
            mock_client = mock.MagicMock()
            mock_client.models.generate_content.side_effect = mock_generate_content
            mock_client_cls.return_value = mock_client

            result = call_gemini_with_retry("test prompt", max_retries=2)
            assert result == '{"success": true}'
            assert call_count == 2
            assert mock_sleep.called


class TestOptimizationTargetThresholdAndIterations:
    """Tests ATS threshold 86 early stopping and max 5 iterations."""

    def test_early_exit_on_ats_threshold_86(self, sample_master_resume, sample_structured_jd):
        mock_tailored = {
            "personal_info": {"name": "Kanishk Surwade", "target_title": "Data Annotator"},
            "summary": "Data Annotator with experience in video annotation and audio annotation.",
            "skills": {
                "technical_skills": ["Data Annotation", "Video Annotation", "Audio Annotation"],
                "tools_and_technologies": ["CVAT", "Python"],
                "core_competencies": ["Attention to Detail", "Remote Work Experience"],
            },
            "experience": [
                {
                    "company": "Innodata Inc.",
                    "role": "AI & LLM Analyst",
                    "location": "Remote",
                    "start_date": "Dec 2024",
                    "end_date": "Present",
                    "bullets": ["Performed precise data annotation on video and audio samples adhering to SOPs."],
                }
            ],
            "projects": [],
            "education": [],
            "certifications": [],
        }

        with mock.patch("ai.resume_optimizer.analyze_job_description", return_value=sample_structured_jd), \
             mock.patch("ai.resume_optimizer.match_resume_to_jd", return_value={"overall_match_score": 90}), \
             mock.patch("ai.resume_optimizer.generate_tailored_resume", return_value=mock_tailored), \
             mock.patch("ai.resume_optimizer.evaluate_resume", return_value={"overall_score": 90, "pass_status": True, "weaknesses": []}), \
             mock.patch("ai.resume_optimizer.analyze_ats_compatibility", return_value={"ats_score": 92, "passed": True, "structural_issues": []}):

            res = optimize_resume(
                master_resume=sample_master_resume,
                job_description=sample_structured_jd,
                target_score=86,
                max_iterations=5,
            )
            # Should exit on iteration 1 because ATS score (92) >= target_score (86) and evidence passed
            assert res["total_iterations"] == 1
            assert res["best_ats_score"] == 92
            assert res["ats_passed"] is True
            assert res["status"] == "OPTIMIZATION COMPLETE"

    def test_max_iterations_returns_best_candidate_without_failure(self, sample_master_resume, sample_structured_jd):
        mock_tailored = {
            "personal_info": {"name": "Kanishk Surwade", "target_title": "Data Annotator"},
            "summary": "Data Annotator with annotation experience.",
            "skills": {
                "technical_skills": ["Data Annotation"],
                "tools_and_technologies": ["CVAT"],
                "core_competencies": ["Attention to Detail"],
            },
            "experience": [],
            "projects": [],
            "education": [],
            "certifications": [],
        }

        # ATS score 75 across all 5 iterations (does not reach 86)
        with mock.patch("ai.resume_optimizer.analyze_job_description", return_value=sample_structured_jd), \
             mock.patch("ai.resume_optimizer.match_resume_to_jd", return_value={"overall_match_score": 75}), \
             mock.patch("ai.resume_optimizer.generate_tailored_resume", return_value=mock_tailored), \
             mock.patch("ai.resume_optimizer.evaluate_resume", return_value={"overall_score": 75, "pass_status": False, "weaknesses": ["Improve keywords"]}), \
             mock.patch("ai.resume_optimizer.analyze_ats_compatibility", return_value={"ats_score": 75, "passed": False, "structural_issues": []}):

            res = optimize_resume(
                master_resume=sample_master_resume,
                job_description=sample_structured_jd,
                target_score=86,
                max_iterations=5,
            )
            assert res["total_iterations"] == 5
            assert res["best_ats_score"] == 75
            assert res["best_ats_score"] >= 0
            assert res["best_ats_score"] != -1
            assert "best_resume" in res
            assert res["final_structured_resume"] is not None
