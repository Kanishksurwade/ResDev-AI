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
except ImportError:
    from jd_analyzer import analyze_job_description
    from resume_matcher import match_resume_to_jd
    from resume_generator import generate_tailored_resume
    from resume_evaluator import evaluate_resume

DEFAULT_TARGET_SCORE = 85
DEFAULT_MAX_ITERATIONS = 5


def optimize_resume(
    master_resume: dict[str, Any] | str | Path,
    job_description: dict[str, Any] | str | Path,
    target_score: int = DEFAULT_TARGET_SCORE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Run the end-to-end automated resume generation, evaluation, and optimization loop.

    Parameters:
        master_resume: Master resume data dictionary or path to JSON file.
        job_description: Raw job description text or pre-structured JD dict/path.
        target_score: Minimum evaluation score required to pass (default: 85).
        max_iterations: Maximum revision attempts allowed (default: 5).
        progress_callback: Optional callback(iteration, max_iterations, iteration_data).

    Returns:
        Structured optimization result containing history, best score, and best resume.
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
        # Determine if string is JSON or raw text
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
    best_score = -1
    best_resume: dict[str, Any] = {}
    best_evaluation: dict[str, Any] = {}
    current_feedback: str | None = None
    loop_status = "MAX_ITERATIONS_REACHED"

    for iteration in range(1, max_iterations + 1):
        # A. Generate Tailored Resume (initial or with accumulated feedback)
        tailored_resume = generate_tailored_resume(
            master_resume=master_resume_data,
            structured_jd=structured_jd,
            matching_analysis=matching_analysis,
            revision_feedback=current_feedback,
        )

        # B. Evaluate Resume Quality
        eval_result = evaluate_resume(
            tailored_resume=tailored_resume,
            structured_jd=structured_jd,
            target_score=target_score,
        )

        current_score = eval_result.get("overall_score", 0)
        is_passed = eval_result.get("pass_status", False)
        instructions = eval_result.get("improvement_instructions", [])
        weaknesses = eval_result.get("weaknesses", [])

        # Record iteration metadata
        iteration_record = {
            "iteration": iteration,
            "overall_score": current_score,
            "pass_status": is_passed,
            "dimension_scores": eval_result.get("dimension_scores", {}),
            "weaknesses": weaknesses,
            "improvement_instructions": instructions,
            "explanation": eval_result.get("explanation", ""),
        }
        iterations_history.append(iteration_record)

        if progress_callback:
            progress_callback(iteration, max_iterations, iteration_record)

        # C. Track Best Resume
        if current_score > best_score or iteration == 1:
            best_score = current_score
            best_resume = tailored_resume
            best_evaluation = eval_result

        # D. Check Pass Criteria
        if is_passed:
            loop_status = "TARGET_REACHED"
            break

        # E. Formulate Actionable Feedback for Next Iteration
        feedback_lines = []
        if weaknesses:
            feedback_lines.append("Identified Weaknesses to Fix:")
            for w in weaknesses:
                feedback_lines.append(f"- {w}")
        if instructions:
            feedback_lines.append("\nRequired Improvements:")
            for inst in instructions:
                feedback_lines.append(f"- {inst}")

        current_feedback = "\n".join(feedback_lines)

    # 5. Assemble Final Optimization Result
    result = {
        "status": loop_status,
        "target_score": target_score,
        "max_iterations": max_iterations,
        "total_iterations": len(iterations_history),
        "best_score": best_score,
        "best_resume": best_resume,
        "best_evaluation": best_evaluation,
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
    print("RESDEV AI - AUTOMATIC RESUME OPTIMIZATION ENGINE")
    print("=" * 70)
    print("Candidate:", master_resume_content.get("candidate", {}).get("personal_info", {}).get("name"))
    print("Target Role:", target_structured_jd.get("job_title"))
    print(f"Target Quality Score: {DEFAULT_TARGET_SCORE}/100 | Max Iterations: {DEFAULT_MAX_ITERATIONS}")
    print("Starting automated optimization pipeline...")
    print("-" * 70)

    def on_iteration_progress(it: int, max_it: int, data: dict[str, Any]) -> None:
        status_label = "[PASS]" if data["pass_status"] else "[NEEDS REVISION]"
        print(f"\n>>> Iteration {it}/{max_it}: Score = {data['overall_score']}/100 {status_label}")
        if data["weaknesses"]:
            print("    Key Weaknesses:", "; ".join(data["weaknesses"][:2]))
        if not data["pass_status"] and data["improvement_instructions"]:
            print("    Applying Feedback:", data["improvement_instructions"][0])

    optimization_result = optimize_resume(
        master_resume=master_resume_content,
        job_description=target_structured_jd,
        target_score=85,
        max_iterations=3,
        progress_callback=on_iteration_progress,
    )

    print("\n" + "=" * 70)
    print("FINAL OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Status: {optimization_result['status']}")
    print(f"Total Iterations: {optimization_result['total_iterations']}/{optimization_result['max_iterations']}")
    print(f"Best Score Achieved: {optimization_result['best_score']}/100 (Target: {optimization_result['target_score']})")

    print("\nIteration History:")
    for it in optimization_result["iterations"]:
        status_tag = "PASS" if it["pass_status"] else "REVISE"
        print(f"  * Iteration {it['iteration']}: Score = {it['overall_score']}/100 [{status_tag}]")

    print("\nBest Resume Details:")
    best_res = optimization_result.get("best_resume", {})
    print(f"  Title: {best_res.get('personal_info', {}).get('target_title')}")
    print(f"  Summary: {best_res.get('summary')[:150]}...")
    print(f"  Technical Skills ({len(best_res.get('skills', {}).get('technical_skills', []))}): {', '.join(best_res.get('skills', {}).get('technical_skills', []))}")

    print("\nBest Evaluation Explanation:")
    print(" ", optimization_result.get("best_evaluation", {}).get("explanation"))
    print("=" * 70)
