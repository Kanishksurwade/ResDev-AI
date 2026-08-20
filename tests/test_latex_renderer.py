"""
Tests for LaTeX / PDF Resume Renderer
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ai.latex_renderer import (
    escape_latex,
    sanitize_resume_for_latex,
    is_pdflatex_available,
    render_latex,
    compile_pdf,
    render_resume_files,
)


@pytest.fixture
def sample_structured_resume():
    return {
        "personal_info": {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "+1-555-123-4567",
            "location": "San Francisco, CA",
            "linkedin": "linkedin.com/in/janedoe",
            "github": "github.com/janedoe",
        },
        "target_role": "Senior Data Annotator & Quality Lead",
        "summary": "Experienced AI data analyst specializing in high-accuracy data annotation & quality control (99.5% accuracy).",
        "skills": {
            "technical": ["Data Annotation", "Python", "SQL & ETL"],
            "tools": ["CVAT", "Labelbox", "Google Cloud (GCP)"],
            "soft": ["Attention to Detail", "Quality Assurance"],
        },
        "experience": [
            {
                "company": "InnoData & Co.",
                "role": "Data Annotation Specialist",
                "location": "Remote, USA",
                "start_date": "Jan 2023",
                "end_date": "Present",
                "bullets": [
                    "Annotated >100,000 multimodal items with 99.4% precision & low latency.",
                    "Trained & mentored 10 junior annotators on edge-case guidelines.",
                ],
            }
        ],
        "education": [
            {
                "institution": "University of California, Berkeley",
                "degree": "B.S.",
                "field": "Data Science",
                "start_date": "2018",
                "end_date": "2022",
            }
        ],
        "projects": [
            {
                "name": "Audio Pipeline #1 & Toolset",
                "description": "Custom pipeline for audio & speech labeling.",
                "technologies": ["Python", "Librosa", "PyTorch"],
                "bullets": [
                    "Reduced manual labeling time by 35% across 500+ audio files.",
                ],
            }
        ],
        "certifications": [
            {"name": "Certified Annotation Pro", "issuer": "Data Guild International"}
        ],
        "achievements": [
            "Employee of the Quarter Q3 2024 (Top 1% performer)"
        ],
        "additional_sections": [],
    }


class TestEscapeLatex:
    """Test suite for LaTeX special character escaping."""

    def test_escapes_all_ten_special_characters(self):
        # &, %, $, #, _, {, }, ~, ^, \
        raw_text = r"Symbols: & % $ # _ { } ~ ^ \ and more"
        escaped = escape_latex(raw_text)
        assert r"\&" in escaped
        assert r"\%" in escaped
        assert r"\$" in escaped
        assert r"\#" in escaped
        assert r"\_" in escaped
        assert r"\{" in escaped
        assert r"\}" in escaped
        assert r"\textasciitilde{}" in escaped
        assert r"\textasciicircum{}" in escaped
        assert r"\textbackslash{}" in escaped

    def test_preserves_plain_text(self):
        plain = "Data Annotator with Python and SQL experience."
        assert escape_latex(plain) == plain

    def test_handles_empty_and_non_string_types(self):
        assert escape_latex("") == ""
        assert escape_latex(None) == ""
        assert escape_latex(100) == "100"


class TestSanitizeResume:
    """Test suite for recursive data structure escaping."""

    def test_recursively_sanitizes_nested_dicts_and_lists(self, sample_structured_resume):
        sanitized = sanitize_resume_for_latex(sample_structured_resume)
        assert r"\&" in sanitized["target_role"]
        assert r"\%" in sanitized["summary"]
        assert r"\&" in sanitized["skills"]["technical"][2]
        assert r"\&" in sanitized["experience"][0]["company"]
        assert r"\#" in sanitized["projects"][0]["name"]


class TestRenderLatex:
    """Test suite for LaTeX rendering engine."""

    def test_valid_structured_resume_renders_successfully(self, sample_structured_resume):
        tex = render_latex(sample_structured_resume)
        assert r"\documentclass" in tex
        assert r"\begin{document}" in tex
        assert r"\end{document}" in tex
        assert "Jane Doe" in tex

    def test_all_required_sections_appear_in_latex(self, sample_structured_resume):
        tex = render_latex(sample_structured_resume)
        assert "Senior Data Annotator" in tex
        assert "Professional Summary" in tex
        assert "Skills" in tex
        assert "Professional Experience" in tex
        assert "Education" in tex
        assert "Projects" in tex
        assert "Certifications" in tex
        assert "Achievements" in tex

    def test_special_characters_are_escaped_in_rendered_output(self, sample_structured_resume):
        tex = render_latex(sample_structured_resume)
        # Should contain escaped forms
        assert r"\&" in tex
        assert r"\%" in tex
        assert r"\#" in tex
        # Should not contain unescaped raw special characters (e.g. unescaped & inside text)
        assert "InnoData & Co." not in tex
        assert r"InnoData \& Co." in tex

    def test_missing_optional_sections_do_not_crash(self):
        minimal_resume = {
            "personal_info": {
                "name": "Alex Smith",
                "email": "alex@example.com",
                "phone": "+1-555-0000",
                "location": "New York, NY",
                "linkedin": "",
                "github": "",
            },
            "target_role": "",
            "summary": "Minimal professional summary.",
            "skills": {
                "technical": ["Python"],
                "tools": [],
                "soft": [],
            },
            "experience": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "achievements": [],
            "additional_sections": [],
        }
        tex = render_latex(minimal_resume)
        assert "Alex Smith" in tex
        assert "Minimal professional summary" in tex
        assert r"\begin{document}" in tex

    def test_empty_or_invalid_resume_rejected_safely(self):
        with pytest.raises(ValueError, match="Resume data must be a non-empty dictionary"):
            render_latex({})

        with pytest.raises(ValueError, match="missing required 'personal_info'"):
            render_latex({"summary": "No personal info"})

    def test_no_fabricated_content_introduced(self, sample_structured_resume):
        tex = render_latex(sample_structured_resume)
        # Ensure only content present in the dict appears
        assert "Jane Doe" in tex
        assert "InnoData" in tex
        assert "Unreferenced Company" not in tex
        assert "Fabricated Metric" not in tex


class TestRenderResumeFiles:
    """Test suite for high-level render_resume_files and PDF compilation flow."""

    def test_output_tex_file_created(self, sample_structured_resume, tmp_path):
        tex_path = tmp_path / "test_resume.tex"
        pdf_path = tmp_path / "test_resume.pdf"

        result = render_resume_files(
            resume_data=sample_structured_resume,
            output_tex_path=tex_path,
            output_pdf_path=pdf_path,
            compile_pdf_if_available=False,
        )

        assert result["status"] == "TEX_ONLY"
        assert tex_path.exists()
        assert "Jane Doe" in tex_path.read_text(encoding="utf-8")

    def test_missing_pdflatex_handled_gracefully(self, sample_structured_resume, tmp_path):
        tex_path = tmp_path / "test_resume.tex"
        pdf_path = tmp_path / "test_resume.pdf"

        with patch("ai.latex_renderer.is_pdflatex_available", return_value=False):
            result = render_resume_files(
                resume_data=sample_structured_resume,
                output_tex_path=tex_path,
                output_pdf_path=pdf_path,
                compile_pdf_if_available=True,
            )

            assert result["status"] == "TEX_ONLY"
            assert result["pdflatex_available"] is False
            assert tex_path.exists()
            assert "pdflatex not detected" in result["message"]

    def test_compile_pdf_with_mocked_pdflatex_success(self, sample_structured_resume, tmp_path):
        tex_path = tmp_path / "test_resume.tex"
        pdf_path = tmp_path / "test_resume.pdf"

        # Generate .tex first
        tex_path.write_text(render_latex(sample_structured_resume), encoding="utf-8")

        def fake_subprocess_run(cmd, *args, **kwargs):
            # Create a fake output PDF file in the temp output directory
            out_dir = None
            for arg in cmd:
                if str(arg).startswith("-output-directory="):
                    out_dir = Path(str(arg).split("=", 1)[1])
            if out_dir:
                fake_pdf = out_dir / f"{tex_path.stem}.pdf"
                fake_pdf.write_bytes(b"%PDF-1.4 Mock PDF Content")
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "Output written on test_resume.pdf"
            mock_proc.stderr = ""
            return mock_proc

        with patch("ai.latex_renderer.is_pdflatex_available", return_value=True), \
             patch("subprocess.run", side_effect=fake_subprocess_run):

            success, msg, out_pdf = compile_pdf(tex_path, pdf_path)
            assert success is True
            assert out_pdf == pdf_path
            assert pdf_path.exists()
            assert pdf_path.read_bytes().startswith(b"%PDF-1.4")
