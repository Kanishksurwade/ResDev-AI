"""
Unit tests for the Evidence Validator.

Tests verify that the validator deterministically:
    - Rejects unsupported tools and platforms (e.g., Scale AI, Labelbox, MTurk)
    - Rejects unsupported employers and companies
    - Rejects unsupported quantitative metrics ($5M, 1,000,000, 99.9%)
    - Rejects unsupported degrees and institutions
    - Rejects unsupported certifications
    - Accepts grounded semantic rewrites and rephrasings of authentic facts
    - Safely handles edge cases (empty resumes, empty evidence)

No LLM calls. Fully offline. Fully deterministic.
"""

import json
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ai.evidence_validator import (
    validate_resume_evidence,
    extract_master_evidence,
    _is_tool_supported,
    _is_employer_supported,
    _is_degree_supported,
    _is_institution_supported,
    _is_cert_supported,
    _extract_numbers_and_metrics,
)


@pytest.fixture
def master_resume():
    """Master resume fixture with authentic candidate background."""
    return {
        "candidate": {
            "personal_info": {
                "name": "Kanishk Surwade",
                "location": "Pune, Maharashtra, India",
                "email": "kanishksurwade70@gmail.com",
                "phone": "+91-9834008224",
                "linkedin": "linkedin.com/in/kd4723",
                "github": "github.com/Kanishksurwade",
            },
            "professional_identity": {
                "current_profile": "AI & LLM Analyst",
                "profile": "AI & LLM Analyst with production experience in multimodal AI evaluation and data annotation.",
            },
        },
        "capabilities": {
            "skills": {
                "ai_llm": [
                    "LLM Evaluation",
                    "Prompt Engineering",
                    "Prompt Analysis",
                    "Hallucination Detection",
                    "Grounding Assessment",
                    "Side-by-Side Evaluation",
                    "Multimodal AI Annotation (Text, Audio, Video, Image)",
                ],
                "tools_platforms": [
                    "Google Cloud Platform (GCP)",
                    "Power BI",
                    "MySQL Workbench",
                    "Excel",
                    "Python",
                    "SQL",
                    "CVAT",
                    "JAX",
                    "Flax",
                ],
                "core": [
                    "Attention to Detail",
                    "Quality Benchmarking",
                    "SOP Adherence",
                    "Remote Work Experience",
                ],
            },
            "certifications": [
                {
                    "name": "Career Essentials in Generative AI",
                    "issuer": "Microsoft & LinkedIn Learning",
                }
            ],
        },
        "experience": [
            {
                "company": "Innodata Inc.",
                "role": "AI & LLM Analyst",
                "location": "Noida, India (Remote)",
                "start_date": "Dec 2025",
                "end_date": "Jul 2026",
                "responsibilities_and_achievements": [
                    {
                        "text": "Conducted multimodal data annotation and hallucination detection on production datasets.",
                        "skills_used": ["LLM Evaluation", "Hallucination Detection"],
                    }
                ],
            },
            {
                "company": "Deloitte",
                "role": "Data Analytics Intern",
                "location": "Virtual Internship",
                "start_date": "Sep 2025",
                "end_date": "Sep 2025",
                "responsibilities_and_achievements": [
                    {
                        "text": "Analyzed operational datasets and developed KPI dashboards.",
                        "skills_used": ["Excel", "SQL"],
                    }
                ],
            },
        ],
        "projects": [
            {
                "name": "Google Tunix - Structured Reasoning Fine-Tuning with GRPO on Gemma 3",
                "technologies": ["Gemma 3 (1B)", "GRPO", "LoRA", "JAX", "Flax", "Google Cloud", "TPU"],
                "start_date": "Dec 2025",
                "end_date": "Jan 2026",
                "bullets": ["Designed reward functions and prompt templates for output consistency."],
            },
            {
                "name": "Northwind Sales Analysis & Dashboard",
                "technologies": ["SQL", "Excel", "Power BI", "DAX", "ETL"],
                "start_date": "Oct 2025",
                "end_date": "Oct 2025",
                "bullets": ["Cleaned and analyzed 5,000+ sales records using SQL and Excel."],
            },
        ],
        "education": [
            {
                "degree": "B.Tech in Automation and Robotics",
                "institution": "JSPM Rajarshi Shahu College of Engineering",
                "location": "Pune, India",
                "start_year": "2021",
                "end_year": "2025",
                "cgpa": "7.75",
            }
        ],
    }


class TestEvidenceValidator:
    def test_extract_master_evidence(self, master_resume):
        """Master evidence extraction should populate all verifiable entity sets."""
        evidence = extract_master_evidence(master_resume)
        assert "innodata inc." in evidence["employers"]
        assert "deloitte" in evidence["employers"]
        assert "cvat" in evidence["tools_and_technologies"]
        assert "python" in evidence["tools_and_technologies"]
        assert "career essentials in generative ai" in evidence["certifications"]
        assert "5,000+" in evidence["metrics"]
        assert "7.75" in evidence["metrics"]

    def test_grounded_resume_passes(self, master_resume):
        """A properly grounded resume with authentic facts should pass with 0 violations."""
        valid_resume = {
            "personal_info": {
                "name": "Kanishk Surwade",
                "target_title": "Data Annotator",
                "email": "kanishksurwade70@gmail.com",
            },
            "summary": "AI & LLM Analyst experienced in multimodal data annotation and prompt engineering using CVAT.",
            "skills": {
                "technical_skills": ["Data Annotation", "Video Annotation", "Prompt Engineering"],
                "tools_and_technologies": ["CVAT", "Google Cloud Platform", "Python", "SQL"],
            },
            "experience": [
                {
                    "company": "Innodata Inc.",
                    "role": "AI & LLM Analyst",
                    "bullets": ["Executed multimodal data annotation and quality benchmarking using CVAT."],
                }
            ],
            "education": [
                {
                    "degree": "B.Tech in Automation and Robotics",
                    "institution": "JSPM Rajarshi Shahu College of Engineering",
                }
            ],
            "certifications": [
                {
                    "name": "Career Essentials in Generative AI",
                    "issuer": "Microsoft & LinkedIn Learning",
                }
            ],
        }
        res = validate_resume_evidence(valid_resume, master_resume)
        assert res["passed"] is True
        assert len(res["violations"]) == 0

    def test_unsupported_tool_rejected(self, master_resume):
        """Resume containing hallucinated tools like Scale AI or Labelbox must be rejected."""
        resume = {
            "skills": {
                "tools_and_technologies": ["CVAT", "Scale AI", "Labelbox"],
            },
        }
        res = validate_resume_evidence(resume, master_resume)
        assert res["passed"] is False
        tool_violations = [v for v in res["violations"] if v["type"] == "unsupported_tool"]
        assert len(tool_violations) >= 2
        viol_values = [v["value"] for v in tool_violations]
        assert "Scale AI" in viol_values
        assert "Labelbox" in viol_values

    def test_unsupported_employer_rejected(self, master_resume):
        """Resume with fabricated employers must be rejected."""
        resume = {
            "experience": [
                {"company": "Google LLC", "role": "Lead Annotator", "bullets": ["Annotated data."]},
            ],
        }
        res = validate_resume_evidence(resume, master_resume)
        assert res["passed"] is False
        assert any(v["type"] == "unsupported_employer" and "Google LLC" in v["value"] for v in res["violations"])

    def test_unsupported_metric_rejected(self, master_resume):
        """Resume with fabricated metrics ($5M, 1,000,000 samples) must be rejected."""
        resume = {
            "experience": [
                {
                    "company": "Innodata Inc.",
                    "bullets": ["Managed $5M budget and processed 1,000,000 samples with 99.9% accuracy."],
                }
            ],
        }
        res = validate_resume_evidence(resume, master_resume)
        assert res["passed"] is False
        metric_violations = [v for v in res["violations"] if v["type"] == "unsupported_metric"]
        assert len(metric_violations) >= 2

    def test_unsupported_degree_and_institution_rejected(self, master_resume):
        """Resume with fabricated academic credentials must be rejected."""
        resume = {
            "education": [
                {"degree": "Ph.D. in Artificial Intelligence", "institution": "Stanford University"},
            ],
        }
        res = validate_resume_evidence(resume, master_resume)
        assert res["passed"] is False
        assert any(v["type"] == "unsupported_institution" for v in res["violations"])
        assert any(v["type"] == "unsupported_degree" for v in res["violations"])

    def test_unsupported_certification_rejected(self, master_resume):
        """Resume with fabricated certifications must be rejected."""
        resume = {
            "certifications": [
                {"name": "AWS Certified Solutions Architect Professional", "issuer": "Amazon Web Services"},
            ],
        }
        res = validate_resume_evidence(resume, master_resume)
        assert res["passed"] is False
        assert any(v["type"] == "unsupported_certification" for v in res["violations"])

    def test_semantic_rewrite_accepted(self, master_resume):
        """Semantic rephrasing of authentic tasks should be accepted."""
        resume = {
            "experience": [
                {
                    "company": "Innodata Inc.",
                    "role": "AI & LLM Analyst",
                    "bullets": [
                        "Performed multimodal data annotation including audio, video, and image labeling to uphold dataset benchmarks."
                    ],
                }
            ],
        }
        res = validate_resume_evidence(resume, master_resume)
        assert res["passed"] is True

    def test_empty_resume_handling(self, master_resume):
        """Empty resume should not crash the validator."""
        res = validate_resume_evidence({}, master_resume)
        assert res["passed"] is True
        assert res["violations"] == []

    def test_uploaded_text_resume_evidence_validation(self):
        """Uploaded plain text resume with _raw_text must validate grounded facts."""
        raw_text = (
            "Kanishk Surwade\n"
            "Innodata Inc. - AI & LLM Analyst (Dec 2025 - Jul 2026)\n"
            "- Evaluated multimodal LLM responses on 1,000+ prompt pairs using CVAT, Python, and Google Cloud Platform.\n"
            "Education: JSPM Rajarshi Shahu College of Engineering - B.Tech in Automation and Robotics\n"
            "Certifications: Career Essentials in Generative AI - Microsoft\n"
        )
        uploaded_master = {
            "candidate": {
                "personal_info": {"name": "Kanishk Surwade"},
                "professional_identity": {"profile": raw_text},
            },
            "_raw_text": raw_text,
        }
        candidate = {
            "personal_info": {"name": "Kanishk Surwade", "target_title": "Data Annotator"},
            "skills": {
                "technical_skills": ["Multimodal AI Annotation", "Data Annotation"],
                "tools_and_technologies": ["CVAT", "Google Cloud Platform (GCP)", "Python"],
            },
            "experience": [
                {
                    "company": "Innodata Inc.",
                    "role": "AI & LLM Analyst",
                    "bullets": ["Evaluated multimodal LLM responses on 1,000+ prompt pairs using CVAT."],
                }
            ],
            "education": [{"degree": "B.Tech in Automation and Robotics", "institution": "JSPM Rajarshi Shahu College of Engineering"}],
            "certifications": [{"name": "Career Essentials in Generative AI", "issuer": "Microsoft"}],
        }
        res = validate_resume_evidence(candidate, uploaded_master)
        assert res["passed"] is True
        assert res["violations"] == []

    def test_uploaded_text_resume_rejects_hallucinated_facts(self):
        """Uploaded plain text resume must still strictly reject completely fabricated facts."""
        raw_text = "Jane Doe\nAcme Corp - Developer\n- Built web tools in Python.\n"
        uploaded_master = {
            "candidate": {"personal_info": {"name": "Jane Doe"}, "professional_identity": {"profile": raw_text}},
            "_raw_text": raw_text,
        }
        hallucinated_candidate = {
            "personal_info": {"name": "Jane Doe", "target_title": "Data Annotator"},
            "skills": {"tools_and_technologies": ["Scale AI", "Labelbox"]},
            "experience": [{"company": "Google LLC", "bullets": ["Managed $10M budget on 5,000,000 samples."]}],
            "education": [{"degree": "Ph.D. in Neuroscience", "institution": "Harvard University"}],
        }
        res = validate_resume_evidence(hallucinated_candidate, uploaded_master)
        assert res["passed"] is False
        assert res["violation_count"] >= 4
