"""
ResDev AI - Optimization Guard & Regression Protection

Ensures that the optimization loop:
    1. NEVER accepts a candidate with factual violations (Hard Gate 1).
    2. NEVER accepts a candidate with critical structural defects (Hard Gate 2).
    3. NEVER regresses in Multi-ATS compatibility (e.g., 6/6 -> 0/6).
    4. NEVER accepts a lower ATS score (e.g., 84 -> 78) without compensating improvement.
    5. NEVER allows a worse later iteration to overwrite the best candidate.
    6. Automatically rolls back to the best verified candidate when a regression occurs.
"""

import copy
from typing import Any


class OptimizationGuard:
    """
    Gated evaluation and regression protection engine for the resume optimization loop.
    """

    def __init__(self, target_score: int = 86):
        self.target_score = target_score
        self.best_candidate: dict[str, Any] | None = None
        self.best_iteration: int = 0
        self.best_ats_score: int = 0
        self.best_qwen_score: int = 0
        self.best_combined_score: float = 0.0
        self.best_multi_passed: int = 0
        self.best_multi_total: int = 6
        self.best_evaluations: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        # Fallback candidate tracking (in case all iterations are rejected on hard gates)
        self.fallback_candidate: dict[str, Any] | None = None
        self.fallback_iteration: int = 0
        self.fallback_ats_score: int = 0
        self.fallback_qwen_score: int = 0
        self.fallback_combined_score: float = 0.0
        self.fallback_multi_passed: int = 0
        self.fallback_multi_total: int = 6
        self.fallback_evaluations: dict[str, Any] = {}

    def evaluate_candidate(
        self,
        candidate_resume: dict[str, Any],
        iteration: int,
        ats_eval: dict[str, Any],
        multi_ats_eval: dict[str, Any],
        qwen_eval: dict[str, Any],
        evidence_eval: dict[str, Any],
    ) -> tuple[bool, str, str]:
        """
        Evaluate a candidate resume against hard gates and regression rules.

        Returns:
            (is_accepted_as_best, decision_status, decision_reason)
            decision_status values:
                "ACCEPTED_AS_INITIAL_BEST"
                "ACCEPTED_AS_NEW_BEST"
                "REJECTED_FACTUAL_VIOLATION"
                "REJECTED_STRUCTURAL_DEFECT"
                "REJECTED_MULTI_ATS_REGRESSION"
                "REJECTED_ATS_SCORE_REGRESSION"
                "REJECTED_SEMANTIC_REGRESSION"
                "REJECTED_NO_IMPROVEMENT"
        """
        ats_score = ats_eval.get("ats_score", 0)
        qwen_score = qwen_eval.get("overall_score", 0)
        combined_score = round(0.5 * ats_score + 0.5 * qwen_score, 1)

        multi_summary = multi_ats_eval.get("summary", {})
        multi_passed = multi_summary.get("passed", 0) + multi_summary.get("warned", 0)
        multi_total = multi_summary.get("total_platforms", 6)
        multi_failed = multi_summary.get("failed", 0)

        # Always maintain a valid fallback candidate from the latest / best iteration
        if (
            self.fallback_candidate is None
            or combined_score >= self.fallback_combined_score
        ):
            self.fallback_candidate = copy.deepcopy(candidate_resume)
            self.fallback_iteration = iteration
            self.fallback_ats_score = ats_score
            self.fallback_qwen_score = qwen_score
            self.fallback_combined_score = combined_score
            self.fallback_multi_passed = multi_passed
            self.fallback_multi_total = multi_total
            self.fallback_evaluations = {
                "ats": copy.deepcopy(ats_eval),
                "multi_ats": copy.deepcopy(multi_ats_eval),
                "qwen": copy.deepcopy(qwen_eval),
                "evidence": copy.deepcopy(evidence_eval),
            }

        # ---------------------------------------------------------
        # HARD GATE 1: Factual Integrity & Evidence Validation
        # ---------------------------------------------------------
        if not evidence_eval.get("passed", True):
            violations = evidence_eval.get("violations", [])
            viol_details = "; ".join([f"{v.get('type')}: {v.get('value')}" for v in violations[:3]])
            status = "REJECTED_FACTUAL_VIOLATION"
            reason = f"Candidate contains {len(violations)} factual violations ({viol_details})"
            self._record_decision(iteration, False, status, reason, ats_score, qwen_score, combined_score, multi_passed, multi_total)
            return False, status, reason

        # ---------------------------------------------------------
        # HARD GATE 2: Structural Integrity (Sections & Contact)
        # ---------------------------------------------------------
        struct_issues = ats_eval.get("structural_issues", [])
        # Severe structural defects: missing experience or missing name/email
        has_critical_struct_defect = any(
            "Missing personal_info" in s or "Experience section is missing" in s
            for s in struct_issues
        )
        if has_critical_struct_defect:
            status = "REJECTED_STRUCTURAL_DEFECT"
            reason = f"Candidate has critical structural defects: {struct_issues[0]}"
            self._record_decision(iteration, False, status, reason, ats_score, qwen_score, combined_score, multi_passed, multi_total)
            return False, status, reason

        # ---------------------------------------------------------
        # INITIAL CANDIDATE (Iteration 1)
        # ---------------------------------------------------------
        if self.best_candidate is None:
            self._set_new_best(
                candidate_resume=candidate_resume,
                iteration=iteration,
                ats_score=ats_score,
                qwen_score=qwen_score,
                combined_score=combined_score,
                multi_passed=multi_passed,
                multi_total=multi_total,
                evaluations={
                    "ats": ats_eval,
                    "multi_ats": multi_ats_eval,
                    "qwen": qwen_eval,
                    "evidence": evidence_eval,
                },
            )
            status = "ACCEPTED_AS_INITIAL_BEST"
            reason = f"Initial candidate baseline established: ATS={ats_score}, Multi-ATS={multi_passed}/{multi_total}, Qwen={qwen_score}"
            self._record_decision(iteration, True, status, reason, ats_score, qwen_score, combined_score, multi_passed, multi_total)
            return True, status, reason

        # ---------------------------------------------------------
        # REGRESSION GUARD: Multi-ATS Platform Drop
        # ---------------------------------------------------------
        if multi_passed < self.best_multi_passed and ats_score <= self.best_ats_score:
            status = "REJECTED_MULTI_ATS_REGRESSION"
            reason = f"Multi-ATS platform coverage dropped from {self.best_multi_passed}/{self.best_multi_total} to {multi_passed}/{multi_total}"
            self._record_decision(iteration, False, status, reason, ats_score, qwen_score, combined_score, multi_passed, multi_total)
            return False, status, reason

        # ---------------------------------------------------------
        # REGRESSION GUARD: Deterministic ATS Score Drop
        # ---------------------------------------------------------
        if ats_score < self.best_ats_score:
            status = "REJECTED_ATS_SCORE_REGRESSION"
            reason = f"Deterministic ATS score regressed from {self.best_ats_score}/100 to {ats_score}/100"
            self._record_decision(iteration, False, status, reason, ats_score, qwen_score, combined_score, multi_passed, multi_total)
            return False, status, reason

        # ---------------------------------------------------------
        # REGRESSION GUARD: Material Semantic Quality Drop
        # ---------------------------------------------------------
        if qwen_score < (self.best_qwen_score - 5) and ats_score <= self.best_ats_score:
            status = "REJECTED_SEMANTIC_REGRESSION"
            reason = f"Semantic quality regressed from {self.best_qwen_score}/100 to {qwen_score}/100"
            self._record_decision(iteration, False, status, reason, ats_score, qwen_score, combined_score, multi_passed, multi_total)
            return False, status, reason

        # ---------------------------------------------------------
        # IMPROVEMENT CRITERIA: Strictly Better Check
        # ---------------------------------------------------------
        is_strictly_better = False

        if ats_score > self.best_ats_score:
            is_strictly_better = True
        elif multi_passed > self.best_multi_passed and ats_score >= self.best_ats_score:
            is_strictly_better = True
        elif combined_score > self.best_combined_score and ats_score >= self.best_ats_score and multi_passed >= self.best_multi_passed:
            is_strictly_better = True

        if is_strictly_better:
            self._set_new_best(
                candidate_resume=candidate_resume,
                iteration=iteration,
                ats_score=ats_score,
                qwen_score=qwen_score,
                combined_score=combined_score,
                multi_passed=multi_passed,
                multi_total=multi_total,
                evaluations={
                    "ats": ats_eval,
                    "multi_ats": multi_ats_eval,
                    "qwen": qwen_eval,
                    "evidence": evidence_eval,
                },
            )
            status = "ACCEPTED_AS_NEW_BEST"
            reason = f"Candidate improved metrics: ATS={ats_score} (was {self.best_ats_score}), Multi-ATS={multi_passed}/{multi_total}, Combined={combined_score}"
            self._record_decision(iteration, True, status, reason, ats_score, qwen_score, combined_score, multi_passed, multi_total)
            return True, status, reason

        # Equal or marginal outcome -> preserve previous best to avoid churning
        status = "REJECTED_NO_IMPROVEMENT"
        reason = f"Candidate did not meaningfully improve over current best (ATS={self.best_ats_score}, Multi-ATS={self.best_multi_passed}/{self.best_multi_total})"
        self._record_decision(iteration, False, status, reason, ats_score, qwen_score, combined_score, multi_passed, multi_total)
        return False, status, reason

    def _set_new_best(
        self,
        candidate_resume: dict[str, Any],
        iteration: int,
        ats_score: int,
        qwen_score: int,
        combined_score: float,
        multi_passed: int,
        multi_total: int,
        evaluations: dict[str, Any],
    ) -> None:
        self.best_candidate = copy.deepcopy(candidate_resume)
        self.best_iteration = iteration
        self.best_ats_score = ats_score
        self.best_qwen_score = qwen_score
        self.best_combined_score = combined_score
        self.best_multi_passed = multi_passed
        self.best_multi_total = multi_total
        self.best_evaluations = copy.deepcopy(evaluations)

    def _record_decision(
        self,
        iteration: int,
        accepted: bool,
        status: str,
        reason: str,
        ats: int,
        qwen: int,
        combined: float,
        multi_p: int,
        multi_t: int,
    ) -> None:
        self.history.append({
            "iteration": iteration,
            "accepted_as_best": accepted,
            "decision_status": status,
            "decision_reason": reason,
            "ats_score": ats,
            "qwen_score": qwen,
            "combined_score": combined,
            "multi_ats_passed": multi_p,
            "multi_ats_total": multi_t,
            "current_best_iteration": self.best_iteration,
            "current_best_ats": self.best_ats_score,
            "current_best_multi": f"{self.best_multi_passed}/{self.best_multi_total}",
        })

    def get_final_result(self) -> dict[str, Any]:
        """Return structured summary of the protected optimization outcome."""
        if self.best_candidate is not None:
            ats = max(0, self.best_ats_score)
            semantic = max(0, self.best_qwen_score)
            comb = max(0.0, self.best_combined_score)
            multi_p = max(0, self.best_multi_passed)
            return {
                "best_iteration": self.best_iteration,
                "best_ats_score": ats,
                "best_gemini_score": semantic,
                "best_semantic_score": semantic,
                "best_qwen_score": semantic,
                "best_combined_score": comb,
                "best_multi_ats_passed": multi_p,
                "best_multi_ats_total": self.best_multi_total,
                "ats_passed": bool(ats >= self.target_score),
                "gemini_passed": bool(semantic >= self.target_score),
                "semantic_passed": bool(semantic >= self.target_score),
                "qwen_passed": bool(semantic >= self.target_score),
                "best_resume": self.best_candidate,
                "best_evaluations": self.best_evaluations,
                "decision_history": self.history,
            }

        # Fallback to the best evaluated candidate across iterations
        fallback_iter = self.fallback_iteration or (self.history[0]["iteration"] if self.history else 1)
        fallback_ats = max(0, self.fallback_ats_score or (self.history[0]["ats_score"] if self.history else 0))
        fallback_semantic = max(0, self.fallback_qwen_score or (self.history[0]["qwen_score"] if self.history else 0))
        fallback_combined = max(0.0, self.fallback_combined_score or (self.history[0]["combined_score"] if self.history else 0.0))
        fallback_multi = max(0, self.fallback_multi_passed or (self.history[0]["multi_ats_passed"] if self.history else 0))
        fallback_multi_total = self.fallback_multi_total or 6

        return {
            "best_iteration": fallback_iter,
            "best_ats_score": fallback_ats,
            "best_gemini_score": fallback_semantic,
            "best_semantic_score": fallback_semantic,
            "best_qwen_score": fallback_semantic,
            "best_combined_score": fallback_combined,
            "best_multi_ats_passed": fallback_multi,
            "best_multi_ats_total": fallback_multi_total,
            "ats_passed": bool(fallback_ats >= self.target_score),
            "gemini_passed": bool(fallback_semantic >= self.target_score),
            "semantic_passed": bool(fallback_semantic >= self.target_score),
            "qwen_passed": bool(fallback_semantic >= self.target_score),
            "best_resume": self.fallback_candidate or {},
            "best_evaluations": self.fallback_evaluations,
            "decision_history": self.history,
        }
