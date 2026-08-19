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
except ImportError:
    from jd_analyzer import analyze_job_description
    from resume_matcher import match_resume_to_jd
    from resume_generator import generate_tailored_resume
    from resume_evaluator import evaluate_resume
    from ats_analyzer import analyze_ats_compatibility
    from multi_ats_validator import validate_multi_ats

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

    Returns:
        (actions, feedback_notes, platforms_passed, platforms_total)
    """
    actions: list[dict[str, Any]] = []
    notes: list[str] = []
    platforms = multi_ats_result.get("platforms", {})
    summary = multi_ats_result.get("summary", {})
    platforms_passed = summary.get("passed", 0) + summary.get("warned", 0)
    platforms_total = summary.get("total_platforms", len(platforms))

    # Collect cross-platform missing keywords/skills (deduplicated)
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

    # Append cross-platform recommendations (top 2)
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
    Incorporates multi-ATS validation feedback when available.
    """
    combined_actions: list[dict[str, Any]] = []
    feedback_notes: list[str] = []

    # 1. Prioritize multi-ATS cross-platform failures (most actionable)
    if multi_ats_result:
        multi_actions, multi_notes, _, _ = _extract_multi_ats_feedback(multi_ats_result)
        combined_actions.extend(multi_actions)
        feedback_notes.extend(multi_notes)

    # 2. Prioritize single-engine ATS deficiencies if present
    if trigger in ("ATS_DEFICIENCY", "BOTH_DEFICIENT"):
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

    # 3. Append Qwen semantic actions
    qwen_actions = qwen_result.get("improvement_actions", [])
    if isinstance(qwen_actions, list):
        for act in qwen_actions:
            if isinstance(act, dict):
                combined_actions.append(act)

    # Deduplicate by change text, cap to top 5
    seen_changes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for a in combined_actions:
        change_key = a.get("change", "").lower()
        if change_key not in seen_changes:
            seen_changes.add(change_key)
            deduped.append(a)
    combined_actions = deduped[:5]

    # 4. Build text feedback notes
    qwen_weaknesses = qwen_result.get("weaknesses", [])
    if qwen_weaknesses:
        feedback_notes.append("Semantic Weaknesses to Address:")
        for w in qwen_weaknesses[:2]:
            feedback_notes.append(f"- {w}")

    ats_recs = ats_result.get("recommendations", [])
    if ats_recs and trigger in ("ATS_DEFICIENCY", "BOTH_DEFICIENT"):
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
    Run the end-to-end automated resume generation, deterministic ATS analysis,
    Qwen semantic evaluation, and optimization loop.

    Parameters:
        master_resume: Master resume data dictionary or path to JSON file.
        job_description: Raw job description text or pre-structured JD dict/path.
        target_score: Minimum evaluation score required to pass (default: 85).
        max_iterations: Maximum revision attempts allowed (default: 5).
        progress_callback: Optional callback(iteration, max_iterations, iteration_data).

    Returns:
        Structured optimization result containing history, best scores, and best resume.
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

    # 4. Optimization Loop State
    iterations_history: list[dict[str, Any]] = []
    best_combined_score = -1.0
    best_ats_score = -1
    best_qwen_score = -1
    best_resume: dict[str, Any] = {}
    best_ats_eval: dict[str, Any] = {}
    best_qwen_eval: dict[str, Any] = {}
    best_multi_ats_eval: dict[str, Any] = {}
    current_improvement_actions: list[dict[str, Any]] | None = None
    current_feedback: str | None = None
    previous_combined_score: float | None = None
    loop_status = "MAX_ITERATIONS_REACHED"

    for iteration in range(1, max_iterations + 1):
        # A. Generate Tailored Resume (initial or applying structured improvement actions)
        tailored_resume = generate_tailored_resume(
            master_resume=master_resume_data,
            structured_jd=structured_jd,
            matching_analysis=matching_analysis,
            revision_feedback=current_feedback,
            improvement_actions=current_improvement_actions,
        )

        # B. Run Deterministic ATS Compatibility Analysis (Primary Check)
        ats_result = analyze_ats_compatibility(
            structured_jd=structured_jd,
            structured_resume=tailored_resume,
            threshold=target_score,
        )

        # C. Run Qwen Semantic Quality Evaluation
        qwen_result = evaluate_resume(
            tailored_resume=tailored_resume,
            structured_jd=structured_jd,
            target_score=target_score,
        )

        # D. Run Multi-ATS Platform Validation
        multi_ats_result = validate_multi_ats(
            structured_jd=structured_jd,
            structured_resume=tailored_resume,
        )
        multi_summary = multi_ats_result.get("summary", {})
        multi_passed_count = multi_summary.get("passed", 0) + multi_summary.get("warned", 0)
        multi_total_count = multi_summary.get("total_platforms", 0)
        multi_failed_count = multi_summary.get("failed", 0)
        multi_overall_status = multi_ats_result.get("overall_status", "FAIL")

        ats_score = ats_result.get("ats_score", 0)
        qwen_score = qwen_result.get("overall_score", 0)
        combined_score = round(0.5 * ats_score + 0.5 * qwen_score, 1)

        # E. Combined Decision Logic (includes multi-ATS as gate)
        combined_passed, trigger, primary_issues = _evaluate_combined_decision(
            ats_result=ats_result,
            qwen_result=qwen_result,
            target_score=target_score,
        )

        # Multi-ATS gate: if any platform has critical failures, block pass
        if combined_passed and multi_failed_count > 0:
            combined_passed = False
            trigger = "MULTI_ATS_FAILURES"
            for cf in multi_ats_result.get("critical_failures", [])[:3]:
                primary_issues.append(f"Multi-ATS: {cf}")

        # F. Score progression and outcome tracking
        score_before = previous_combined_score if previous_combined_score is not None else combined_score
        score_after = combined_score
        score_delta = round(score_after - score_before, 1)

        if iteration == 1:
            outcome = "INITIAL_GENERATION"
        elif score_delta > 0:
            outcome = "IMPROVED"
        else:
            outcome = "NO_IMPROVEMENT"

        combined_status = "OPTIMIZATION COMPLETE" if combined_passed else "NEEDS REVISION"

        # Record iteration metadata
        iteration_record = {
            "iteration": iteration,
            "ats_score": ats_score,
            "ats_passed": ats_result.get("passed", False),
            "ats_metrics": ats_result.get("metrics", {}),
            "qwen_score": qwen_score,
            "qwen_passed": qwen_result.get("pass_status", False),
            "combined_score": combined_score,
            "combined_status": combined_status,
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_delta,
            "outcome": outcome,
            "revision_trigger": trigger,
            "primary_issues": primary_issues,
            "ats_missing_keywords": ats_result.get("missing_required_keywords", []),
            "ats_missing_skills": ats_result.get("missing_required_skills", []),
            "qwen_weaknesses": qwen_result.get("weaknesses", []),
            "improvement_actions": qwen_result.get("improvement_actions", []),
            "multi_ats_status": multi_overall_status,
            "multi_ats_passed": multi_passed_count,
            "multi_ats_total": multi_total_count,
            "multi_ats_failed": multi_failed_count,
            "multi_ats_critical_failures": multi_ats_result.get("critical_failures", []),
        }
        iterations_history.append(iteration_record)

        if progress_callback:
            progress_callback(iteration, max_iterations, iteration_record)

        # G. Track Best Resume
        if combined_score > best_combined_score or iteration == 1:
            best_combined_score = combined_score
            best_ats_score = ats_score
            best_qwen_score = qwen_score
            best_resume = tailored_resume
            best_ats_eval = ats_result
            best_qwen_eval = qwen_result
            best_multi_ats_eval = multi_ats_result

        # H. Check Combined Pass Criteria
        if combined_passed:
            loop_status = "OPTIMIZATION COMPLETE"
            break

        # I. Formulate Next Iteration Actions based on Trigger + Multi-ATS
        current_improvement_actions, current_feedback = _build_revision_feedback(
            ats_result=ats_result,
            qwen_result=qwen_result,
            trigger=trigger,
            multi_ats_result=multi_ats_result,
        )

        previous_combined_score = combined_score

    # 5. Assemble Final Optimization Result
    multi_summary_final = best_multi_ats_eval.get("summary", {})
    result = {
        "status": loop_status,
        "target_score": target_score,
        "max_iterations": max_iterations,
        "total_iterations": len(iterations_history),
        "best_ats_score": best_ats_score,
        "best_qwen_score": best_qwen_score,
        "best_combined_score": best_combined_score,
        "ats_passed": bool(best_ats_score >= target_score),
        "qwen_passed": bool(best_qwen_score >= target_score),
        "multi_ats_passed": multi_summary_final.get("passed", 0) + multi_summary_final.get("warned", 0),
        "multi_ats_total": multi_summary_final.get("total_platforms", 0),
        "multi_ats_failed": multi_summary_final.get("failed", 0),
        "multi_ats_overall_status": best_multi_ats_eval.get("overall_status", "UNKNOWN"),
        "best_resume": best_resume,
        "best_ats_evaluation": best_ats_eval,
        "best_qwen_evaluation": best_qwen_eval,
        "best_multi_ats_evaluation": best_multi_ats_eval,
        "remaining_gaps": best_qwen_eval.get("weaknesses", []) + best_ats_eval.get("structural_issues", []),
        "iterations": iterations_history,
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

    print("=" * 70)
    print("RESDEV AI - TRIPLE-ENGINE RESUME OPTIMIZATION PIPELINE")
    print("=" * 70)
    print("Candidate:", master_resume_content.get("candidate", {}).get("personal_info", {}).get("name"))
    print("Target Role:", target_structured_jd.get("job_title"))
    print(f"Target Threshold: {DEFAULT_TARGET_SCORE}/100 | Max Iterations: {DEFAULT_MAX_ITERATIONS}")
    print("Engines: Deterministic ATS + Multi-ATS Validator + Qwen Semantic Evaluator")
    print("-" * 70)

    def on_iteration_progress(it: int, max_it: int, data: dict[str, Any]) -> None:
        ats_tag = "[PASS]" if data["ats_passed"] else "[FAIL]"
        qwen_tag = "[PASS]" if data["qwen_passed"] else "[NEEDS REVISION]"
        multi_tag = data.get("multi_ats_status", "?")
        multi_p = data.get("multi_ats_passed", 0)
        multi_t = data.get("multi_ats_total", 0)
        print(f"\n>>> Iteration {it}/{max_it}:")
        print(f"    ATS Score: {data['ats_score']}/100 {ats_tag}")
        print(f"    Multi-ATS: {multi_p}/{multi_t} platforms OK [{multi_tag}]")
        print(f"    Qwen Score: {data['qwen_score']}/100 {qwen_tag}")
        print(f"    Combined Status: {data['combined_status']} ({data['outcome']})")
        if data["revision_trigger"] != "NONE":
            print(f"    Revision Trigger: {data['revision_trigger']}")
            if data["revision_trigger"] in ("ATS_DEFICIENCY", "BOTH_DEFICIENT", "MULTI_ATS_FAILURES"):
                print("    ATS ISSUE:")
                for issue in data.get("ats_missing_keywords", [])[:2]:
                    print(f"      - Missing required keyword: \"{issue}\"")
                for issue in data.get("ats_missing_skills", [])[:2]:
                    print(f"      - Missing required skill: \"{issue}\"")
            if data.get("multi_ats_critical_failures"):
                print("    MULTI-ATS CRITICAL:")
                for cf in data["multi_ats_critical_failures"][:2]:
                    print(f"      - {cf}")
            if data["revision_trigger"] in ("QWEN_SEMANTIC_WEAKNESS", "BOTH_DEFICIENT"):
                print("    QWEN ISSUE:")
                for w in data.get("qwen_weaknesses", [])[:2]:
                    print(f"      - {w}")

    optimization_result = optimize_resume(
        master_resume=master_resume_content,
        job_description=target_structured_jd,
        target_score=85,
        max_iterations=3,
        progress_callback=on_iteration_progress,
    )

    print("\n" + "=" * 70)
    print("FINAL OPTIMIZATION SUMMARY & DUAL SCORECARD")
    print("=" * 70)
    print(f"Final Status: {optimization_result['status']}")
    print(f"Total Iterations: {optimization_result['total_iterations']}/{optimization_result['max_iterations']}")
    print(f"Deterministic ATS Score: {optimization_result['best_ats_score']}/100 {'[PASS]' if optimization_result['ats_passed'] else '[FAIL]'}")
    print(f"Multi-ATS Platforms: {optimization_result['multi_ats_passed']}/{optimization_result['multi_ats_total']} OK [{optimization_result['multi_ats_overall_status']}]")
    print(f"Qwen Semantic Score: {optimization_result['best_qwen_score']}/100 {'[PASS]' if optimization_result['qwen_passed'] else '[NEEDS REVISION]'}")
    print(f"Combined Composite Score: {optimization_result['best_combined_score']}/100 (Threshold: {optimization_result['target_score']})")

    print("\nIteration History & Score Progression:")
    for it in optimization_result["iterations"]:
        print(f"  * Iteration {it['iteration']}: ATS={it['ats_score']}/100 | Qwen={it['qwen_score']}/100 | Combined={it['combined_score']}/100 -> {it['combined_status']} ({it['revision_trigger']})")

    print("\nBest Resume Details:")
    best_res = optimization_result.get("best_resume", {})
    print(f"  Target Title: {best_res.get('personal_info', {}).get('target_title')}")
    print(f"  Summary: {best_res.get('summary')[:140]}...")
    print(f"  Technical Skills ({len(best_res.get('skills', {}).get('technical_skills', []))}): {', '.join(best_res.get('skills', {}).get('technical_skills', []))}")

    print("\nATS Metrics Breakdown:")
    for m_k, m_v in optimization_result.get("best_ats_evaluation", {}).get("metrics", {}).items():
        print(f"  * {m_k.replace('_', ' ').title():<28}: {m_v:.1f}%")

    if optimization_result.get("remaining_gaps"):
        print("\nRemaining Factual Gaps / Unmet Nuances:")
        for gap in optimization_result["remaining_gaps"]:
            print(f"  [-] {gap}")

    print("=" * 70)
