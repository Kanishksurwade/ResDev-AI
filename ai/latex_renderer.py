"""
ResDev AI - LaTeX / PDF Resume Renderer (Standard Library Offline-First)

Renders the validated structured resume JSON (generated/final_resume.json)
into a clean, professional, ATS-compliant LaTeX document (.tex) and compiles
it to PDF via pdflatex when a TeX distribution is installed.
"""

import copy
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "templates" / "resume_template.tex"
DEFAULT_TEX_OUTPUT_PATH = REPO_ROOT / "generated" / "final_resume.tex"
DEFAULT_PDF_OUTPUT_PATH = REPO_ROOT / "generated" / "final_resume.pdf"

# Dictionary mapping LaTeX special characters to their escaped representations
LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Regex to match all LaTeX special characters in a single pass
_LATEX_ESCAPE_REGEX = re.compile(r"([\\&%$#_{}~^])")


def escape_latex(text: str) -> str:
    """
    Safely escape LaTeX special characters in a given string using single-pass regex replacement.
    Prevents nested/cascading escaping issues.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    return _LATEX_ESCAPE_REGEX.sub(lambda m: LATEX_ESCAPE_MAP.get(m.group(1), m.group(1)), text)


def sanitize_resume_for_latex(data: Any) -> Any:
    """
    Recursively traverse a resume data structure and escape all strings for LaTeX.
    """
    if isinstance(data, dict):
        return {k: sanitize_resume_for_latex(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_resume_for_latex(item) for item in data]
    elif isinstance(data, str):
        return escape_latex(data)
    else:
        return data


def is_pdflatex_available() -> bool:
    """
    Check whether pdflatex executable is installed and available in the system PATH.
    """
    return shutil.which("pdflatex") is not None


def _render_summary(summary: str) -> str:
    if not summary or not summary.strip():
        return ""
    return f"\\section*{{Professional Summary}}\n{summary.strip()}\n"


def _render_skills(skills: dict[str, Any]) -> str:
    if not isinstance(skills, dict):
        return ""

    technical = skills.get("technical", [])
    tools = skills.get("tools", [])
    soft = skills.get("soft", [])

    items = []
    if technical and isinstance(technical, list):
        items.append(f"\\item \\textbf{{Technical Skills:}} {', '.join(str(s) for s in technical)}")
    if tools and isinstance(tools, list):
        items.append(f"\\item \\textbf{{Tools \\& Technologies:}} {', '.join(str(t) for t in tools)}")
    if soft and isinstance(soft, list):
        items.append(f"\\item \\textbf{{Core Competencies:}} {', '.join(str(s) for s in soft)}")

    if not items:
        return ""

    joined_items = "\n    ".join(items)
    return (
        "\\section*{Skills \\& Competencies}\n"
        "\\begin{itemize}[leftmargin=1.25em, itemsep=2pt]\n"
        f"    {joined_items}\n"
        "\\end{itemize}\n"
    )


def _render_experience(experience: list[dict[str, Any]]) -> str:
    if not isinstance(experience, list) or not experience:
        return ""

    entries = []
    for exp in experience:
        if not isinstance(exp, dict):
            continue

        role = exp.get("role", "")
        company = exp.get("company", "")
        location = exp.get("location", "")
        start_date = exp.get("start_date", "")
        end_date = exp.get("end_date", "")
        bullets = exp.get("bullets", [])

        if not role and not company and not bullets:
            continue

        # Format header line
        header_parts = []
        if role:
            header_parts.append(f"\\textbf{{{role}}}")
            if company:
                header_parts.append(f"--- \\textit{{{company}}}")
        elif company:
            header_parts.append(f"\\textbf{{{company}}}")

        if location:
            header_parts.append(f"| {location}")

        left_side = " ".join(header_parts)

        dates = ""
        if start_date and end_date:
            dates = f"{start_date} -- {end_date}"
        elif start_date:
            dates = start_date
        elif end_date:
            dates = end_date

        if dates:
            line = f"\\noindent\n{left_side} \\hfill {dates}"
        else:
            line = f"\\noindent\n{left_side}"

        if bullets and isinstance(bullets, list):
            bullet_items = "\n    ".join(f"\\item {b}" for b in bullets if b)
            entry_tex = (
                f"{line}\n"
                "\\begin{itemize}\n"
                f"    {bullet_items}\n"
                "\\end{itemize}"
            )
        else:
            entry_tex = f"{line}\n\\vspace{{4pt}}"

        entries.append(entry_tex)

    if not entries:
        return ""

    joined_entries = "\n\n".join(entries)
    return f"\\section*{{Professional Experience}}\n{joined_entries}\n"


def _render_education(education: list[dict[str, Any]]) -> str:
    if not isinstance(education, list) or not education:
        return ""

    entries = []
    for edu in education:
        if not isinstance(edu, dict):
            continue

        degree = edu.get("degree", "")
        field = edu.get("field", "")
        institution = edu.get("institution", "")
        start_date = edu.get("start_date", "")
        end_date = edu.get("end_date", "")

        if not degree and not field and not institution:
            continue

        header_parts = []
        if degree:
            if field:
                header_parts.append(f"\\textbf{{{degree}}} in {field}")
            else:
                header_parts.append(f"\\textbf{{{degree}}}")
        elif field:
            header_parts.append(f"\\textbf{{{field}}}")

        if institution:
            header_parts.append(f"--- \\textit{{{institution}}}")

        left_side = " ".join(header_parts)

        dates = ""
        if start_date and end_date:
            dates = f"{start_date} -- {end_date}"
        elif start_date:
            dates = start_date
        elif end_date:
            dates = end_date

        if dates:
            line = f"\\noindent\n{left_side} \\hfill {dates}\n\\vspace{{2pt}}"
        else:
            line = f"\\noindent\n{left_side}\n\\vspace{{2pt}}"

        entries.append(line)

    if not entries:
        return ""

    joined_entries = "\n".join(entries)
    return f"\\section*{{Education}}\n{joined_entries}\n"


def _render_projects(projects: list[dict[str, Any]]) -> str:
    if not isinstance(projects, list) or not projects:
        return ""

    entries = []
    for proj in projects:
        if not isinstance(proj, dict):
            continue

        name = proj.get("name", "")
        tech = proj.get("technologies", [])
        desc = proj.get("description", "")
        bullets = proj.get("bullets", [])

        if not name and not desc and not bullets:
            continue

        header = f"\\textbf{{{name}}}"
        if tech and isinstance(tech, list):
            header += f" \\textit{{({', '.join(str(t) for t in tech)})}}"

        line = f"\\noindent\n{header}"
        desc_tex = f"\n\\par {desc}" if desc else ""

        if bullets and isinstance(bullets, list):
            bullet_items = "\n    ".join(f"\\item {b}" for b in bullets if b)
            entry_tex = (
                f"{line}{desc_tex}\n"
                "\\begin{itemize}\n"
                f"    {bullet_items}\n"
                "\\end{itemize}"
            )
        else:
            entry_tex = f"{line}{desc_tex}\n\\vspace{{4pt}}"

        entries.append(entry_tex)

    if not entries:
        return ""

    joined_entries = "\n\n".join(entries)
    return f"\\section*{{Projects}}\n{joined_entries}\n"


def _render_certifications(certifications: list[Any]) -> str:
    if not isinstance(certifications, list) or not certifications:
        return ""

    items = []
    for cert in certifications:
        if isinstance(cert, dict):
            c_name = cert.get("name", "")
            c_issuer = cert.get("issuer", "")
            if c_name and c_issuer:
                items.append(f"\\item \\textbf{{{c_name}}} --- {c_issuer}")
            elif c_name:
                items.append(f"\\item \\textbf{{{c_name}}}")
            elif c_issuer:
                items.append(f"\\item {c_issuer}")
        elif isinstance(cert, str) and cert.strip():
            items.append(f"\\item {cert.strip()}")

    if not items:
        return ""

    joined_items = "\n    ".join(items)
    return (
        "\\section*{Certifications}\n"
        "\\begin{itemize}\n"
        f"    {joined_items}\n"
        "\\end{itemize}\n"
    )


def _render_achievements(achievements: list[str]) -> str:
    if not isinstance(achievements, list) or not achievements:
        return ""

    items = [f"\\item {a}" for a in achievements if isinstance(a, str) and a.strip()]
    if not items:
        return ""

    joined_items = "\n    ".join(items)
    return (
        "\\section*{Achievements}\n"
        "\\begin{itemize}\n"
        f"    {joined_items}\n"
        "\\end{itemize}\n"
    )


def _render_additional_sections(additional: list[Any]) -> str:
    if not isinstance(additional, list) or not additional:
        return ""

    sections = []
    for sec in additional:
        if isinstance(sec, dict):
            title = sec.get("title", "Additional Information")
            items = sec.get("items", [])
            content = sec.get("content", "")
            if items and isinstance(items, list):
                item_lines = "\n    ".join(f"\\item {i}" for i in items if i)
                sections.append(
                    f"\\section*{{{title}}}\n"
                    "\\begin{itemize}\n"
                    f"    {item_lines}\n"
                    "\\end{itemize}"
                )
            elif content:
                sections.append(f"\\section*{{{title}}}\n{content}")
        elif isinstance(sec, str) and sec.strip():
            sections.append(f"\\section*{{Additional Information}}\n{sec.strip()}")

    return "\n\n".join(sections) + ("\n" if sections else "")


def render_latex(resume_data: dict[str, Any], template_path: Path | str | None = None) -> str:
    """
    Render a validated resume dictionary into a valid LaTeX document string.
    """
    if not isinstance(resume_data, dict) or not resume_data:
        raise ValueError("Resume data must be a non-empty dictionary.")

    if "personal_info" not in resume_data:
        raise ValueError("Invalid resume data: missing required 'personal_info' section.")

    if template_path is None:
        tpl_path = DEFAULT_TEMPLATE_PATH
    else:
        tpl_path = Path(template_path)

    if not tpl_path.exists():
        raise FileNotFoundError(f"LaTeX template not found at {tpl_path}")

    template_content = tpl_path.read_text(encoding="utf-8")

    # Sanitize all string fields safely for LaTeX
    sanitized = sanitize_resume_for_latex(copy.deepcopy(resume_data))

    p_info = sanitized.get("personal_info", {})
    name = p_info.get("name", "")
    target_role = sanitized.get("target_role", "")

    # Contact line
    contact_parts = []
    for key in ["location", "phone", "email", "linkedin", "github"]:
        val = p_info.get(key, "")
        if val:
            contact_parts.append(val)
    contact_line = " $\\cdot$ ".join(contact_parts)

    target_role_str = f"{{\\large \\textbf{{{target_role}}}}} \\\\ \\vspace{{2pt}}\n" if target_role else ""

    rendered = template_content.replace("__NAME__", name)
    rendered = rendered.replace("__TARGET_ROLE__", target_role_str)
    rendered = rendered.replace("__CONTACT_LINE__", contact_line)
    rendered = rendered.replace("__SUMMARY_SECTION__", _render_summary(sanitized.get("summary", "")))
    rendered = rendered.replace("__SKILLS_SECTION__", _render_skills(sanitized.get("skills", {})))
    rendered = rendered.replace("__EXPERIENCE_SECTION__", _render_experience(sanitized.get("experience", [])))
    rendered = rendered.replace("__EDUCATION_SECTION__", _render_education(sanitized.get("education", [])))
    rendered = rendered.replace("__PROJECTS_SECTION__", _render_projects(sanitized.get("projects", [])))
    rendered = rendered.replace("__CERTIFICATIONS_SECTION__", _render_certifications(sanitized.get("certifications", [])))
    rendered = rendered.replace("__ACHIEVEMENTS_SECTION__", _render_achievements(sanitized.get("achievements", [])))
    rendered = rendered.replace("__ADDITIONAL_SECTIONS__", _render_additional_sections(sanitized.get("additional_sections", [])))

    # Collapse multiple consecutive empty lines
    cleaned = re.sub(r"\n{3,}", "\n\n", rendered)
    return cleaned.strip() + "\n"


def compile_pdf(
    tex_path: Path | str,
    output_pdf_path: Path | str | None = None,
    timeout: int = 30,
) -> tuple[bool, str, Path | None]:
    """
    Compile a .tex file into a .pdf file using pdflatex.

    Returns:
        (success: bool, message: str, pdf_path: Path | None)
    """
    source_tex = Path(tex_path)
    if not source_tex.exists():
        return False, f"Source .tex file not found at {source_tex}", None

    if output_pdf_path is None:
        target_pdf = DEFAULT_PDF_OUTPUT_PATH
    else:
        target_pdf = Path(output_pdf_path)

    target_pdf.parent.mkdir(parents=True, exist_ok=True)

    if not is_pdflatex_available():
        return (
            False,
            "pdflatex is not installed or not found on the system PATH. "
            "A LaTeX distribution (such as MiKTeX, TeX Live, or MacTeX) is required to compile PDFs.",
            None,
        )

    # Use a temporary directory for compilation to keep aux/log files isolated
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_tex = Path(tmp_dir) / source_tex.name
        shutil.copy2(source_tex, tmp_tex)

        try:
            # Run pdflatex non-interactively
            proc = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={tmp_dir}",
                    str(tmp_tex),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )

            generated_tmp_pdf = Path(tmp_dir) / f"{source_tex.stem}.pdf"
            if proc.returncode == 0 and generated_tmp_pdf.exists():
                shutil.copy2(generated_tmp_pdf, target_pdf)
                return True, "PDF compiled successfully.", target_pdf
            else:
                log_file = Path(tmp_dir) / f"{source_tex.stem}.log"
                log_excerpt = ""
                if log_file.exists():
                    log_excerpt = log_file.read_text(encoding="utf-8", errors="ignore")[-1000:]
                error_msg = proc.stderr or proc.stdout or log_excerpt or "pdflatex compilation failed."
                return False, f"Compilation error (exit code {proc.returncode}): {error_msg}", None

        except subprocess.TimeoutExpired:
            return False, f"pdflatex compilation timed out after {timeout} seconds.", None
        except Exception as e:
            return False, f"Unexpected error during PDF compilation: {e}", None


def render_resume_files(
    resume_data: dict[str, Any],
    output_tex_path: Path | str | None = None,
    output_pdf_path: Path | str | None = None,
    compile_pdf_if_available: bool = True,
) -> dict[str, Any]:
    """
    Render LaTeX and optionally compile PDF for a structured resume.

    Returns:
        {
            "status": "SUCCESS" | "TEX_ONLY" | "ERROR",
            "tex_path": str,
            "pdf_path": str | None,
            "pdflatex_available": bool,
            "message": str
        }
    """
    try:
        # 1. Determine Output Paths
        tex_path = Path(output_tex_path) if output_tex_path else DEFAULT_TEX_OUTPUT_PATH
        pdf_path = Path(output_pdf_path) if output_pdf_path else DEFAULT_PDF_OUTPUT_PATH

        tex_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Render LaTeX String
        rendered_tex = render_latex(resume_data)

        # 3. Write .tex File
        tex_path.write_text(rendered_tex, encoding="utf-8")

        # 4. Check pdflatex availability & compile
        has_pdflatex = is_pdflatex_available()

        if compile_pdf_if_available and has_pdflatex:
            success, msg, generated_pdf = compile_pdf(tex_path, pdf_path)
            if success and generated_pdf:
                return {
                    "status": "SUCCESS",
                    "tex_path": str(tex_path),
                    "pdf_path": str(generated_pdf),
                    "pdflatex_available": True,
                    "message": "LaTeX and PDF generated successfully.",
                }
            else:
                return {
                    "status": "TEX_ONLY",
                    "tex_path": str(tex_path),
                    "pdf_path": None,
                    "pdflatex_available": True,
                    "message": f"LaTeX file created, but PDF compilation failed: {msg}",
                }
        else:
            return {
                "status": "TEX_ONLY",
                "tex_path": str(tex_path),
                "pdf_path": None,
                "pdflatex_available": has_pdflatex,
                "message": (
                    "LaTeX file created successfully at "
                    f"{tex_path}. PDF compilation skipped (pdflatex not detected)."
                    if not has_pdflatex
                    else f"LaTeX file created successfully at {tex_path}."
                ),
            }

    except Exception as e:
        return {
            "status": "ERROR",
            "tex_path": str(output_tex_path) if output_tex_path else "",
            "pdf_path": None,
            "pdflatex_available": is_pdflatex_available(),
            "message": f"Rendering error: {e}",
        }
