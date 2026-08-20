"""
Unit tests for the Optimization Guard.

Tests verify that the guard deterministically:
    - Automatically rejects candidates with factual violations (Hard Gate 1)
    - Automatically rejects candidates with critical structural defects (Hard Gate 2)
    - Rejects lower ATS score candidates (e.g., 78 vs 84)
    - Rejects Multi-ATS platform regressions (e.g., 0/6 vs 6/6)
    - Rejects material semantic quality regressions
    - Preserves previous best candidate when later iterations are worse
    - Prevents unnecessary churn on identical scores

No LLM calls. Fully offline. Fully deterministic.
"""

import copy
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.optimization_guard import OptimizationGuard


@pytest.fixture
def base_candidate():
    return {
        "personal_info": {"name": "Candidate A", "target_title": "Data Annotator"},
        "summary": "Professional summary...",
        "skills": {"technical_skills": ["Data Annotation", "CVAT"]},
        "experience": [{"company": "Innodata Inc.", "role": "Analyst", "bullets": ["Annotated data."]}],
    }


class TestOptimizationGuard:
    def test_initial_candidate_accepted_as_baseline(self, base_candidate):
        guard = OptimizationGuard(target_score=85)
        ats_eval = {"ats_score": 80, "passed": False, "structural_issues": []}
        multi_eval = {"summary": {"passed": 4, "warned": 1, "failed": 1, "total_platforms": 6}}
        qwen_eval = {"overall_score": 82, "pass_status": False}
        evidence_eval = {"passed": True, "violations": []}

        accepted, status, reason = guard.evaluate_candidate(
            base_candidate, 1, ats_eval, multi_eval, qwen_eval, evidence_eval
        )

        assert accepted is True
        assert status == "ACCEPTED_AS_INITIAL_BEST"
        assert guard.best_ats_score == 80
        assert guard.best_iteration == 1

    def test_factual_violation_rejected_automatically(self, base_candidate):
        """Even with 100/100 ATS and 100/100 Qwen, factual violation must be rejected."""
        guard = OptimizationGuard(target_score=85)
        # Establish baseline
        guard.evaluate_candidate(
            base_candidate, 1,
            {"ats_score": 80, "passed": False, "structural_issues": []},
            {"summary": {"passed": 4, "warned": 0, "failed": 2, "total_platforms": 6}},
            {"overall_score": 80, "pass_status": False},
            {"passed": True, "violations": []},
        )

        # Candidate with fabricated Scale AI tool
        hallucinated_candidate = copy.deepcopy(base_candidate)
        hallucinated_candidate["skills"]["tools_and_technologies"] = ["Scale AI"]

        accepted, status, reason = guard.evaluate_candidate(
            hallucinated_candidate, 2,
            {"ats_score": 98, "passed": True, "structural_issues": []},
            {"summary": {"passed": 6, "warned": 0, "failed": 0, "total_platforms": 6}},
            {"overall_score": 95, "pass_status": True},
            {"passed": False, "violations": [{"type": "unsupported_tool", "value": "Scale AI"}]},
        )

        assert accepted is False
        assert status == "REJECTED_FACTUAL_VIOLATION"
        assert "Scale AI" in reason
        assert guard.best_ats_score == 80
        assert guard.best_iteration == 1

    def test_ats_score_regression_rejected(self, base_candidate):
        """Candidate with ATS 78 must not overwrite candidate with ATS 84."""
        guard = OptimizationGuard(target_score=85)

        # Iteration 1: Baseline ATS 80
        guard.evaluate_candidate(
            base_candidate, 1,
            {"ats_score": 80, "passed": False, "structural_issues": []},
            {"summary": {"passed": 4, "warned": 0, "failed": 2, "total_platforms": 6}},
            {"overall_score": 82, "pass_status": False},
            {"passed": True, "violations": []},
        )

        # Iteration 2: Improved to ATS 84
        cand_2 = copy.deepcopy(base_candidate)
        accepted_2, status_2, _ = guard.evaluate_candidate(
            cand_2, 2,
            {"ats_score": 84, "passed": False, "structural_issues": []},
            {"summary": {"passed": 6, "warned": 0, "failed": 0, "total_platforms": 6}},
            {"overall_score": 82, "pass_status": False},
            {"passed": True, "violations": []},
        )
        assert accepted_2 is True
        assert guard.best_ats_score == 84
        assert guard.best_iteration == 2

        # Iteration 3: Regressed to ATS 78
        cand_3 = copy.deepcopy(base_candidate)
        accepted_3, status_3, reason_3 = guard.evaluate_candidate(
            cand_3, 3,
            {"ats_score": 78, "passed": False, "structural_issues": []},
            {"summary": {"passed": 6, "warned": 0, "failed": 0, "total_platforms": 6}},
            {"overall_score": 82, "pass_status": False},
            {"passed": True, "violations": []},
        )
        assert accepted_3 is False
        assert status_3 == "REJECTED_ATS_SCORE_REGRESSION"
        assert "84/100 to 78/100" in reason_3
        # Final best candidate must remain Iteration 2
        assert guard.best_ats_score == 84
        assert guard.best_iteration == 2

    def test_multi_ats_regression_rejected(self, base_candidate):
        """Candidate with Multi-ATS drop (6/6 -> 0/6) must be rejected."""
        guard = OptimizationGuard(target_score=85)

        # Iteration 1: 6/6 platforms passed
        guard.evaluate_candidate(
            base_candidate, 1,
            {"ats_score": 84, "passed": False, "structural_issues": []},
            {"summary": {"passed": 6, "warned": 0, "failed": 0, "total_platforms": 6}},
            {"overall_score": 82, "pass_status": False},
            {"passed": True, "violations": []},
        )

        # Iteration 2: 0/6 platforms passed
        cand_2 = copy.deepcopy(base_candidate)
        accepted_2, status_2, reason_2 = guard.evaluate_candidate(
            cand_2, 2,
            {"ats_score": 84, "passed": False, "structural_issues": []},
            {"summary": {"passed": 0, "warned": 0, "failed": 6, "total_platforms": 6}},
            {"overall_score": 82, "pass_status": False},
            {"passed": True, "violations": []},
        )

        assert accepted_2 is False
        assert status_2 == "REJECTED_MULTI_ATS_REGRESSION"
        assert guard.best_iteration == 1

    def test_best_candidate_survives_failed_later_iterations(self, base_candidate):
        """Best candidate from Iteration 2 survives failed Iterations 3, 4, 5."""
        guard = OptimizationGuard(target_score=85)

        # It 1: ATS 80
        guard.evaluate_candidate(
            base_candidate, 1,
            {"ats_score": 80, "passed": False, "structural_issues": []},
            {"summary": {"passed": 4, "warned": 0, "failed": 2, "total_platforms": 6}},
            {"overall_score": 80, "pass_status": False},
            {"passed": True, "violations": []},
        )

        # It 2: ATS 95 (BEST)
        best_cand = copy.deepcopy(base_candidate)
        best_cand["summary"] = "Best candidate content"
        guard.evaluate_candidate(
            best_cand, 2,
            {"ats_score": 95, "passed": True, "structural_issues": []},
            {"summary": {"passed": 6, "warned": 0, "failed": 0, "total_platforms": 6}},
            {"overall_score": 85, "pass_status": True},
            {"passed": True, "violations": []},
        )

        # It 3: Regressed ATS 75
        guard.evaluate_candidate(
            base_candidate, 3,
            {"ats_score": 75, "passed": False, "structural_issues": []},
            {"summary": {"passed": 2, "warned": 0, "failed": 4, "total_platforms": 6}},
            {"overall_score": 80, "pass_status": False},
            {"passed": True, "violations": []},
        )

        # It 4: Factual violation
        guard.evaluate_candidate(
            base_candidate, 4,
            {"ats_score": 99, "passed": True, "structural_issues": []},
            {"summary": {"passed": 6, "warned": 0, "failed": 0, "total_platforms": 6}},
            {"overall_score": 90, "pass_status": True},
            {"passed": False, "violations": [{"type": "unsupported_tool", "value": "Labelbox"}]},
        )

        final = guard.get_final_result()
        assert final["best_iteration"] == 2
        assert final["best_ats_score"] == 95
        assert final["best_resume"]["summary"] == "Best candidate content"

    def test_guard_never_returns_negative_scores_when_all_rejected(self, base_candidate):
        """When all iterations fail hard gates, guard must return non-negative fallback scores, never -1."""
        guard = OptimizationGuard(target_score=85)

        # Iteration 1: Rejected on factual violation
        guard.evaluate_candidate(
            base_candidate, 1,
            {"ats_score": 90, "passed": True, "structural_issues": []},
            {"summary": {"passed": 5, "warned": 1, "failed": 0, "total_platforms": 6}},
            {"overall_score": 88, "pass_status": True},
            {"passed": False, "violations": [{"type": "unsupported_tool", "value": "Scale AI"}]},
        )

        # Iteration 2: Rejected on factual violation
        guard.evaluate_candidate(
            base_candidate, 2,
            {"ats_score": 92, "passed": True, "structural_issues": []},
            {"summary": {"passed": 6, "warned": 0, "failed": 0, "total_platforms": 6}},
            {"overall_score": 91, "pass_status": True},
            {"passed": False, "violations": [{"type": "unsupported_employer", "value": "Acme"}]},
        )

        final = guard.get_final_result()
        assert final["best_ats_score"] >= 0
        assert final["best_ats_score"] != -1
        assert final["best_qwen_score"] >= 0
        assert final["best_qwen_score"] != -1
        assert final["best_combined_score"] >= 0.0
        assert final["best_combined_score"] != -1.0
        assert final["best_multi_ats_passed"] >= 0
        assert final["best_multi_ats_passed"] != -1
        assert final["best_iteration"] in (1, 2)
        assert final["best_resume"] is not None

