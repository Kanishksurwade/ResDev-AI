"""
ResDev AI - Evidence-Grounded Multi-ATS Resume Optimizer

Orchestrates the resume optimization loop with:
    1. Deterministic Master Resume Evidence Validation (Hard Gate)
    2. Requirement Matrix & Controlled Edit Planning
    3. Deterministic ATS Analysis & Multi-Platform Simulation
    4. Qualitative Semantic Review (Qwen)
    5. Optimization Guard with Automatic Rollback on Regressions

Architecture:
    Base Resume
        ↓
    Evidence Validation (Hard Gate 1)
        ↓
    Requirement Matrix & Valid Gap Extraction
        ↓
    Targeted Edit Plan / Revision
        ↓
    Deterministic ATS + Multi-ATS + Semantic Evaluation
        ↓
    Optimization Guard (Accept as Best OR Rollback)
        ↓
    Repeat for remaining valid gaps
"""

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ai.jd_analyzer import analyze_job_description
    from ai.resume_matcher import match_resume_to_jd
    from ai.resume_generator import generate_tailored_resume
    from ai.resume_evaluator import evaluate_resume
    from ai.ats_analyzer import analyze_ats_compatibility
    from ai.multi_ats_validator import validate_multi_ats
    from ai.evidence_validator import validate_resume_evidence
    from ai.edit_plan import (
        build_requirement_matrix,
        get_unmet_valid_gaps,
        get_unsupported_gaps,
        generate_targeted_edit_plan,
        apply_edit_plan,
    )
    from ai.optimization_guard import OptimizationGuard
except ImportError:
    from jd_analyzer import analyze_job_description
    from resume_matcher import match_resume_to_jd
    from resume_generator import generate_tailored_resume
    from resume_evaluator import evaluate_resume
    from ats_analyzer import analyze_ats_compatibility
    from multi_ats_validator import validate_multi_ats
    from evidence_validator import validate_resume_evidence
    from edit_plan import (
        build_requirement_matrix,
        get_unmet_valid_gaps,
        get_unsupported_gaps,
        generate_targeted_edit_plan,
        apply_edit_plan,
    )
    from optimization_guard import OptimizationGuard

DEFAULT_TARGET_SCORE = 85
DEFAULT_MAX_ITERATIONS = 5


def _evaluate_combined_decision(
    ats_result: dict[str, Any],
    qwen_result: dict[str, Any],
    target_score: int = DEFAULT_TARGET_SCORE,
) -> tuple[bool, str, list[str]]:
    """
    Determine combined pass status, identifying which system triggered revision if needed.

    Returns:
        (combined_passed, revision_trigger, primary_issues)
        revision_trigger values: "NONE", "ATS_DEFICIENCY", "QWEN_SEMANTIC_WEAKNESS", "BOTH_DEFICIENT"
    """
    ats_score = ats_result.get("ats_score", 0)
    qwen_score = qwen_result.get("overall_score", 0)

    ats_passed = ats_score >= target_score
    qwen_passed = qwen_score >= target_score

    combined_passed = ats_passed and qwen_passed

    if combined_passed:
        return True, "NONE", []

    issues: list[str] = []

    if not ats_passed and not qwen_passed:
        trigger = "BOTH_DEFICIENT"
        for kw in ats_result.get("missing_required_keywords", []):
            issues.append(f"Missing required keyword: '{kw}'")
        for sk in ats_result.get("missing_required_skills", []):
            issues.append(f"Missing required skill: '{sk}'")
        for w in qwen_result.get("weaknesses", []):
            issues.append(f"Semantic weakness: {w}")
    elif not ats_passed:
        trigger = "ATS_DEFICIENCY"
        for kw in ats_result.get("missing_required_keywords", []):
            issues.append(f"Missing required keyword: '{kw}'")
        for sk in ats_result.get("missing_required_skills", []):
            issues.append(f"Missing required skill: '{sk}'")
        for struct_issue in ats_result.get("structural_issues", []):
            issues.append(f"Structural gap: {struct_issue}")
    else:
        trigger = "QWEN_SEMANTIC_WEAKNESS"
        for w in qwen_result.get("weaknesses", []):
            issues.append(f"Semantic weakness: {w}")

    return False, trigger, issues


def _extract_multi_ats_feedback(
    multi_ats_result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], int, int]:
    """
    Extract actionable improvement actions and feedback notes from multi-ATS validation.
    """
    actions: list[dict[str, Any]] = []
    notes: list[str] = []
    platforms = multi_ats_result.get("platforms", {})
    summary = multi_ats_result.get("summary", {})
    platforms_passed = summary.get("passed", 0) + summary.get("warned", 0)
    platforms_total = summary.get("total_platforms", len(platforms))

    seen_kw: set[str] = set()
    seen_sk: set[str] = set()

    for _pkey, presult in platforms.items():
        if presult.get("overall_status") == "PASS":
            continue
        checks = presult.get("checks", {})

        for mk in checks.get("keyword_coverage", {}).get("missing", []):
            if mk not in seen_kw:
                seen_kw.add(mk)
                actions.append({
                    "target": "experience",
                    "change": f"Integrate missing keyword '{mk}' where supported by candidate experience",
                    "reason": f"Missing on ATS platform: {presult.get('platform_name', _pkey)}",
                })

        for ms in checks.get("required_skills", {}).get("missing", []):
            if ms not in seen_sk:
                seen_sk.add(ms)
                actions.append({
                    "target": "skills",
                    "change": f"List missing required skill '{ms}' if supported by master resume",
                    "reason": f"Missing on ATS platform: {presult.get('platform_name', _pkey)}",
                })

        for cf in presult.get("critical_failures", []):
            notes.append(f"CRITICAL: {cf}")

    for rec in multi_ats_result.get("recommendations", [])[:2]:
        notes.append(f"Multi-ATS: {rec}")

    return actions, notes, platforms_passed, platforms_total


def _build_revision_feedback(
    ats_result: dict[str, Any],
    qwen_result: dict[str, Any],
    trigger: str,
    multi_ats_result: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Assemble prioritized improvement actions and context notes based on the revision trigger.
    """
    combined_actions: list[dict[str, Any]] = []
    feedback_notes: list[str] = []

    if multi_ats_result:
        multi_actions, multi_notes, _, _ = _extract_multi_ats_feedback(multi_ats_result)
        combined_actions.extend(multi_actions)
        feedback_notes.extend(multi_notes)

    if trigger in ("ATS_DEFICIENCY", "BOTH_DEFICIENT", "MULTI_ATS_FAILURES"):
        for kw in ats_result.get("missing_required_keywords", []):
            combined_actions.append({
                "target": "experience",
                "change": f"Naturally integrate missing required keyword '{kw}' where supported by candidate experience",
                "reason": "Deterministic ATS keyword requirement",
            })
        for sk in ats_result.get("missing_required_skills", []):
            combined_actions.append({
                "target": "skills",
                "change": f"Explicitly list missing required skill '{sk}' if supported by candidate master background",
                "reason": "Deterministic ATS skill requirement",
            })
        for issue in ats_result.get("structural_issues", []):
            combined_actions.append({
                "target": "structure",
                "change": f"Resolve structural issue: {issue}",
                "reason": "Deterministic ATS structure requirement",
            })

    qwen_actions = qwen_result.get("improvement_actions", [])
    if isinstance(qwen_actions, list):
        for act in qwen_actions:
            if isinstance(act, dict):
                combined_actions.append(act)

    seen_changes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for a in combined_actions:
        change_key = a.get("change", "").lower()
        if change_key not in seen_changes:
            seen_changes.add(change_key)
            deduped.append(a)
    combined_actions = deduped[:5]

    qwen_weaknesses = qwen_result.get("weaknesses", [])
    if qwen_weaknesses:
        feedback_notes.append("Semantic Weaknesses to Address:")
        for w in qwen_weaknesses[:2]:
            feedback_notes.append(f"- {w}")

    ats_recs = ats_result.get("recommendations", [])
    if ats_recs and trigger in ("ATS_DEFICIENCY", "BOTH_DEFICIENT", "MULTI_ATS_FAILURES"):
        feedback_notes.append("ATS Guidance:")
        for rec in ats_recs[:2]:
            feedback_notes.append(f"- {rec}")

    current_feedback = "\n".join(feedback_notes) if feedback_notes else None
    return combined_actions, current_feedback


def optimize_resume(
    master_resume: dict[str, Any] | str | Path,
    job_description: dict[str, Any] | str | Path,
    target_score: int = DEFAULT_TARGET_SCORE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Run the evidence-grounded, multi-ATS protected resume optimization loop.

    Parameters:
        master_resume: Master resume data dictionary or path to JSON file.
        job_description: Raw job description text or pre-structured JD dict/path.
        target_score: Minimum evaluation score required to pass (default: 85).
        max_iterations: Maximum revision attempts allowed (default: 5).
        progress_callback: Optional callback(iteration, max_iterations, iteration_data).

    Returns:
        Structured optimization result with regression protection and evidence audit.
    """
    # 1. Load Master Resume
    if isinstance(master_resume, (str, Path)) and Path(str(master_resume)).exists():
        with open(master_resume, "r", encoding="utf-8") as f:
            master_resume_data = json.load(f)
    elif isinstance(master_resume, dict):
        master_resume_data = master_resume
    elif isinstance(master_resume, str):
        master_resume_data = json.loads(master_resume)
    else:
        raise ValueError(f"Invalid master_resume input: {type(master_resume)}")

    # 2. Resolve or Analyze Job Description
    if isinstance(job_description, (str, Path)) and Path(str(job_description)).exists():
        with open(job_description, "r", encoding="utf-8") as f:
            structured_jd = json.load(f)
    elif isinstance(job_description, dict):
        structured_jd = job_description
    elif isinstance(job_description, str):
        trimmed = job_description.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                structured_jd = json.loads(trimmed)
            except json.JSONDecodeError:
                structured_jd = analyze_job_description(job_description)
        else:
            structured_jd = analyze_job_description(job_description)
    else:
        raise ValueError(f"Invalid job_description input: {type(job_description)}")

    # 3. Perform Initial JD <-> Master Resume Matching
    matching_analysis = match_resume_to_jd(
        master_resume=master_resume_data,
        structured_jd=structured_jd,
    )

    # 4. Initialize Optimization Guard
    guard = OptimizationGuard(target_score=target_score)

    active_resume: dict[str, Any] = {}
    current_improvement_actions: list[dict[str, Any]] | None = None
    current_feedback: str | None = None
    iterations_history: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        # A. Candidate Generation / Targeted Edit Plan
        if iteration == 1:
            # Baseline candidate generation
            candidate_resume = generate_tailored_resume(
                master_resume=master_resume_data,
                structured_jd=structured_jd,
                matching_analysis=matching_analysis,
                revision_feedback=None,
                improvement_actions=None,
            )
            applied_edits: list[str] = ["Baseline structured resume generated from master evidence"]
            rejected_edits: list[str] = []
        else:
            # Build Requirement Matrix to identify genuine remaining valid gaps
            req_matrix = build_requirement_matrix(structured_jd, master_resume_data, active_resume)
            unmet_valid_gaps = get_unmet_valid_gaps(req_matrix)

            if unmet_valid_gaps:
                # Targeted edit plan for verified unmet gaps
                edit_plan = generate_targeted_edit_plan(
                    current_resume=active_resume,
                    master_resume=master_resume_data,
                    unmet_valid_gaps=unmet_valid_gaps,
                )
                candidate_resume, applied_edits, rejected_edits = apply_edit_plan(
                    current_resume=active_resume,
                    edit_plan=edit_plan,
                    master_resume=master_resume_data,
                )
            else:
                # If no valid gaps remain, apply targeted semantic revision actions
                candidate_resume = generate_tailored_resume(
                    master_resume=master_resume_data,
                    structured_jd=structured_jd,
                    matching_analysis=matching_analysis,
                    revision_feedback=current_feedback,
                    improvement_actions=current_improvement_actions,
                )
                applied_edits = ["Applied structured semantic revision feedback"]
                rejected_edits = []

        # B. Hard Gate 1: Deterministic Evidence Validation
        evidence_eval = validate_resume_evidence(
            candidate_resume=candidate_resume,
            master_resume=master_resume_data,
        )

        # C. Deterministic ATS Compatibility Analysis
        ats_eval = analyze_ats_compatibility(
            structured_jd=structured_jd,
            structured_resume=candidate_resume,
            threshold=target_score,
        )

        # D. Multi-ATS Platform Simulation Validation
        multi_ats_eval = validate_multi_ats(
            structured_jd=structured_jd,
            structured_resume=candidate_resume,
        )

        # E. Qualitative Semantic Review
        qwen_eval = evaluate_resume(
            tailored_resume=candidate_resume,
            structured_jd=structured_jd,
            target_score=target_score,
        )

        # F. Optimization Guard: Decision & Regression Protection
        is_accepted, decision_status, decision_reason = guard.evaluate_candidate(
            candidate_resume=candidate_resume,
            iteration=iteration,
            ats_eval=ats_eval,
            multi_ats_eval=multi_ats_eval,
            qwen_eval=qwen_eval,
            evidence_eval=evidence_eval,
        )

        # G. Update Active Resume or Rollback
        if is_accepted:
            active_resume = copy.deepcopy(candidate_resume)
        else:
            # Roll back to the best verified candidate
            active_resume = copy.deepcopy(guard.best_candidate) if guard.best_candidate else copy.deepcopy(candidate_resume)

        ats_score = ats_eval.get("ats_score", 0)
        qwen_score = qwen_eval.get("overall_score", 0)
        combined_score = round(0.5 * ats_score + 0.5 * qwen_score, 1)
        multi_summary = multi_ats_eval.get("summary", {})
        multi_p = multi_summary.get("passed", 0) + multi_summary.get("warned", 0)
        multi_t = multi_summary.get("total_platforms", 6)

        iteration_record = {
            "iteration": iteration,
            "ats_score": ats_score,
            "ats_passed": ats_eval.get("passed", False),
            "ats_metrics": ats_eval.get("metrics", {}),
            "qwen_score": qwen_score,
            "qwen_passed": qwen_eval.get("pass_status", False),
            "combined_score": combined_score,
            "multi_ats_status": multi_ats_eval.get("overall_status", "FAIL"),
            "multi_ats_passed": multi_p,
            "multi_ats_total": multi_t,
            "multi_ats_failed": multi_summary.get("failed", 0),
            "multi_ats_critical_failures": multi_ats_eval.get("critical_failures", []),
            "evidence_passed": evidence_eval.get("passed", False),
            "evidence_violations": evidence_eval.get("violations", []),
            "decision_status": decision_status,
            "decision_reason": decision_reason,
            "applied_edits": applied_edits,
            "rejected_edits": rejected_edits,
            "is_best_candidate": is_accepted,
            "current_best_iteration": guard.best_iteration,
            "current_best_ats": guard.best_ats_score,
            "current_best_multi": f"{guard.best_multi_passed}/{guard.best_multi_total}",
        }
        iterations_history.append(iteration_record)

        if progress_callback:
            progress_callback(iteration, max_iterations, iteration_record)

        # H. Formulate Feedback for next iteration
        current_improvement_actions, current_feedback = _build_revision_feedback(
            ats_result=ats_eval,
            qwen_result=qwen_eval,
            trigger="ATS_DEFICIENCY" if not ats_eval.get("passed") else "QWEN_SEMANTIC_WEAKNESS",
            multi_ats_result=multi_ats_eval,
        )

        # Check for Early Exit if all criteria comfortably met
        if (
            guard.best_ats_score >= target_score
            and guard.best_multi_passed >= 5
            and guard.best_qwen_score >= target_score
            and evidence_eval.get("passed", False)
        ):
            break

    # 5. Requirement Matrix Final Audit
    final_best_resume = guard.best_candidate or active_resume
    final_req_matrix = build_requirement_matrix(structured_jd, master_resume_data, final_best_resume)
    unsupported_jd_gaps = get_unsupported_gaps(final_req_matrix)

    guard_result = guard.get_final_result()

    result = {
        "status": "OPTIMIZATION COMPLETE" if guard_result["ats_passed"] else "MAX_ITERATIONS_REACHED",
        "target_score": target_score,
        "max_iterations": max_iterations,
        "total_iterations": len(iterations_history),
        "best_iteration": guard_result["best_iteration"],
        "best_ats_score": guard_result["best_ats_score"],
        "best_qwen_score": guard_result["best_qwen_score"],
        "best_combined_score": guard_result["best_combined_score"],
        "ats_passed": guard_result["ats_passed"],
        "qwen_passed": guard_result["qwen_passed"],
        "multi_ats_passed": guard_result["best_multi_ats_passed"],
        "multi_ats_total": guard_result["best_multi_ats_total"],
        "multi_ats_failed": guard_result["best_multi_ats_total"] - guard_result["best_multi_ats_passed"],
        "multi_ats_overall_status": guard_result.get("best_evaluations", {}).get("multi_ats", {}).get(
            "overall_status", "PASS" if guard_result["best_multi_ats_passed"] == guard_result["best_multi_ats_total"] else "FAIL"
        ),
        "best_resume": final_best_resume,
        "best_ats_evaluation": guard_result.get("best_evaluations", {}).get("ats", {}),
        "best_multi_ats_evaluation": guard_result.get("best_evaluations", {}).get("multi_ats", {}),
        "best_qwen_evaluation": guard_result.get("best_evaluations", {}).get("qwen", {}),
        "best_evidence_evaluation": guard_result.get("best_evaluations", {}).get("evidence", {}),
        "unsupported_jd_requirements": [
            f"Unsupported JD requirement: '{ug['text']}' (No evidence in Master Resume -> Correctly omitted)"
            for ug in unsupported_jd_gaps
        ],
        "remaining_gaps": (
            guard_result.get("best_evaluations", {}).get("qwen", {}).get("weaknesses", [])
            + [f"Unsupported JD requirement: {ug['text']}" for ug in unsupported_jd_gaps]
        ),
        "iterations": iterations_history,
        "decision_history": guard_result["decision_history"],
    }

    return result


if __name__ == "__main__":
    master_resume_file = REPO_ROOT / "data" / "master_resume.json"

    if not master_resume_file.exists():
        raise FileNotFoundError(f"Master resume not found at {master_resume_file}")

    with open(master_resume_file, "r", encoding="utf-8") as f:
        master_resume_content = json.load(f)

    # Sample Data Annotator Structured JD
    target_structured_jd = {
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

    print("=" * 70, flush=True)
    print("RESDEV AI - EVIDENCE-GROUNDED RESUME OPTIMIZATION PIPELINE", flush=True)
    print("=" * 70, flush=True)
    print("Candidate:", master_resume_content.get("candidate", {}).get("personal_info", {}).get("name"), flush=True)
    print("Target Role:", target_structured_jd.get("job_title"), flush=True)
    print(f"Target Threshold: {DEFAULT_TARGET_SCORE}/100 | Max Iterations: 2", flush=True)
    print("Engines: Evidence Validator + Deterministic ATS + Multi-ATS + Qwen", flush=True)
    print("-" * 70, flush=True)

    def on_iteration_progress(it: int, max_it: int, data: dict[str, Any]) -> None:
        ats_tag = "[PASS]" if data["ats_passed"] else "[FAIL]"
        qwen_tag = "[PASS]" if data["qwen_passed"] else "[NEEDS REVISION]"
        ev_tag = "[VERIFIED]" if data["evidence_passed"] else "[VIOLATION]"
        print(f"\n>>> Iteration {it}/{max_it}:", flush=True)
        print(f"    Evidence Integrity : {ev_tag}", flush=True)
        print(f"    Deterministic ATS  : {data['ats_score']}/100 {ats_tag}", flush=True)
        print(f"    Multi-ATS Coverage : {data['multi_ats_passed']}/{data['multi_ats_total']} platforms OK [{data['multi_ats_status']}]", flush=True)
        print(f"    Qwen Semantic Score: {data['qwen_score']}/100 {qwen_tag}", flush=True)
        print(f"    Guard Decision     : {data['decision_status']}", flush=True)
        print(f"    Decision Reason    : {data['decision_reason']}", flush=True)
        if data.get("evidence_violations"):
            print("    [!] FACTUAL VIOLATIONS:", flush=True)
            for v in data["evidence_violations"][:2]:
                print(f"        - {v.get('type')}: {v.get('value')} ({v.get('reason')})", flush=True)

    optimization_result = optimize_resume(
        master_resume=master_resume_content,
        job_description=target_structured_jd,
        target_score=85,
        max_iterations=2,
        progress_callback=on_iteration_progress,
    )

    print("\n" + "=" * 70, flush=True)
    print("FINAL OPTIMIZATION SUMMARY & DUAL SCORECARD", flush=True)
    print("=" * 70, flush=True)
    print(f"Final Status: {optimization_result['status']}", flush=True)
    print(f"Best Candidate Selected from Iteration: #{optimization_result['best_iteration']} of {optimization_result['total_iterations']}", flush=True)
    print(f"Deterministic ATS Score: {optimization_result['best_ats_score']}/100 {'[PASS]' if optimization_result['ats_passed'] else '[FAIL]'}", flush=True)
    print(f"Multi-ATS Platforms    : {optimization_result['multi_ats_passed']}/{optimization_result['multi_ats_total']} OK", flush=True)
    print(f"Qwen Semantic Score    : {optimization_result['best_qwen_score']}/100 {'[PASS]' if optimization_result['qwen_passed'] else '[NEEDS REVISION]'}", flush=True)
    print(f"Combined Composite     : {optimization_result['best_combined_score']}/100 (Threshold: {optimization_result['target_score']})", flush=True)

    print("\nOptimization Guard Decision History (Rollback Protection):", flush=True)
    for d in optimization_result["decision_history"]:
        status_symbol = "[+] ACCEPTED" if d["accepted_as_best"] else "[-] REJECTED"
        print(f"  * Iteration {d['iteration']}: {status_symbol} -> {d['decision_status']}", flush=True)
        print(f"      Details: {d['decision_reason']}", flush=True)

    print("\nBest Resume Details:", flush=True)
    best_res = optimization_result.get("best_resume", {})
    print(f"  Target Title: {best_res.get('personal_info', {}).get('target_title')}", flush=True)
    print(f"  Summary: {best_res.get('summary', '')[:140]}...", flush=True)
    tech_skills = best_res.get("skills", {}).get("technical_skills", [])
    print(f"  Technical Skills ({len(tech_skills)}): {', '.join(tech_skills)}", flush=True)

    if optimization_result.get("unsupported_jd_requirements"):
        print("\nUnsupported JD Requirements (Honest Fact-Check):", flush=True)
        for u in optimization_result["unsupported_jd_requirements"]:
            print(f"  [x] {u}", flush=True)

    print("=" * 70, flush=True)
