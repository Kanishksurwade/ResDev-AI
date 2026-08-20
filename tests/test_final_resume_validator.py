"""
Tests for Final Structured Resume Output & Schema Validator
"""

import json
import pytest
from pathlib import Path
from ai.final_resume_validator import (
    build_final_resume,
    validate_final_resume,
    save_final_resume,
    load_final_resume,
    normalize_resume_data,
    REQUIRED_TOP_LEVEL_FIELDS,
)


@pytest.fixture
def valid_final_resume():
    return {
        "personal_info": {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "+1-555-123-4567",
            "location": "San Francisco, CA",
            "linkedin": "linkedin.com/in/janedoe",
            "github": "github.com/janedoe",
        },
        "target_role": "Senior Data Annotator",
        "summary": "Experienced AI data analyst specializing in high-accuracy data annotation and quality control.",
        "skills": {
            "technical": ["Data Annotation", "Python", "SQL"],
            "tools": ["CVAT", "Labelbox", "GCP"],
            "soft": ["Attention to Detail", "Communication"],
        },
        "experience": [
            {
                "company": "Tech Corp",
                "role": "Data Annotation Specialist",
                "location": "San Francisco, CA",
                "start_date": "Jan 2023",
                "end_date": "Present",
                "bullets": [
                    "Annotated over 100,000 multimodal data points with 99.4% precision.",
                    "Trained and mentored 10 junior annotators on domain labeling guidelines.",
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
                "name": "Audio Labeling Automation",
                "description": "Custom pipeline for pre-labeling audio files.",
                "technologies": ["Python", "Librosa"],
                "bullets": [
                    "Reduced manual labeling time by 35%.",
                ],
            }
        ],
        "certifications": [
            {"name": "Certified Annotation Professional", "issuer": "Data Guild"}
        ],
        "achievements": [
            "Employee of the Quarter Q3 2024"
        ],
        "additional_sections": [],
    }


class TestFinalResumeValidator:
    """Test suite for validate_final_resume and schema conformance."""

    def test_valid_resume_passes_validation(self, valid_final_resume):
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is True
        assert len(res["errors"]) == 0
        assert res["resume"]["personal_info"]["name"] == "Jane Doe"

    def test_missing_required_top_level_field(self, valid_final_resume):
        del valid_final_resume["summary"]
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is False
        assert any("Missing required top-level field: 'summary'" in err for err in res["errors"])

    def test_missing_nested_personal_info_field(self, valid_final_resume):
        del valid_final_resume["personal_info"]["email"]
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is False
        assert any("Missing required field in personal_info: 'email'" in err for err in res["errors"])

    def test_empty_candidate_name_rejected(self, valid_final_resume):
        valid_final_resume["personal_info"]["name"] = "   "
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is False
        assert any("Candidate name in personal_info must not be empty" in err for err in res["errors"])

    def test_wrong_data_type_top_level(self, valid_final_resume):
        valid_final_resume["summary"] = 12345
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is False
        assert any("Field 'summary' must be a string" in err for err in res["errors"])

    def test_wrong_data_type_nested_skills(self, valid_final_resume):
        valid_final_resume["skills"]["technical"] = "Not A List"
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is False
        assert any("must be a list of strings" in err for err in res["errors"])

    def test_empty_optional_sections_remain_valid(self, valid_final_resume):
        valid_final_resume["projects"] = []
        valid_final_resume["certifications"] = []
        valid_final_resume["achievements"] = []
        valid_final_resume["additional_sections"] = []
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is True
        assert len(res["errors"]) == 0

    def test_duplicate_bullet_normalization(self, valid_final_resume):
        valid_final_resume["experience"][0]["bullets"] = [
            "Annotated multimodal data.",
            "Annotated multimodal data.",
            "Trained junior annotators.",
        ]
        res = validate_final_resume(valid_final_resume, auto_normalize=True)
        assert res["valid"] is True
        bullets = res["resume"]["experience"][0]["bullets"]
        assert len(bullets) == 2
        assert bullets == ["Annotated multimodal data.", "Trained junior annotators."]

    def test_empty_string_list_cleaning(self, valid_final_resume):
        valid_final_resume["skills"]["technical"] = ["Python", "", "   ", "SQL"]
        res = validate_final_resume(valid_final_resume, auto_normalize=True)
        assert res["valid"] is True
        assert res["resume"]["skills"]["technical"] == ["Python", "SQL"]

    def test_whitespace_normalization_preserves_punctuation(self, valid_final_resume):
        valid_final_resume["summary"] = "  Experienced   data   annotator (99.4% accuracy).  "
        res = validate_final_resume(valid_final_resume, auto_normalize=True)
        assert res["valid"] is True
        assert res["resume"]["summary"] == "Experienced data annotator (99.4% accuracy)."

    def test_date_preservation_exact(self, valid_final_resume):
        valid_final_resume["experience"][0]["start_date"] = "Dec 2025"
        valid_final_resume["experience"][0]["end_date"] = "Jul 2026"
        res = validate_final_resume(valid_final_resume, auto_normalize=True)
        assert res["valid"] is True
        assert res["resume"]["experience"][0]["start_date"] == "Dec 2025"
        assert res["resume"]["experience"][0]["end_date"] == "Jul 2026"

    def test_malformed_experience_entry_missing_keys(self, valid_final_resume):
        valid_final_resume["experience"] = [{"company": "Tech Corp"}]
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is False
        assert any("is missing required field" in err for err in res["errors"])

    def test_malformed_experience_entry_non_dict(self, valid_final_resume):
        valid_final_resume["experience"] = ["String Instead of Dict"]
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is False
        assert any("must be a dictionary object" in err for err in res["errors"])

    def test_malformed_education_entry(self, valid_final_resume):
        valid_final_resume["education"] = [{"institution": "UC Berkeley"}]
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is False
        assert any("missing required field: 'degree'" in err for err in res["errors"])

    def test_malformed_project_entry(self, valid_final_resume):
        valid_final_resume["projects"] = [{"name": "Project A", "bullets": "not a list"}]
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is False
        assert any("Project entry #1 is missing required field" in err or "'bullets' must be a list" in err for err in res["errors"])

    def test_json_serialization(self, valid_final_resume):
        res = validate_final_resume(valid_final_resume)
        assert res["valid"] is True
        serialized = json.dumps(res["resume"])
        deserialized = json.loads(serialized)
        assert deserialized == res["resume"]

    def test_save_and_load_round_trip(self, valid_final_resume, tmp_path):
        out_file = tmp_path / "test_final_resume.json"
        save_path = save_final_resume(valid_final_resume, out_file)
        assert save_path.exists()
        loaded = load_final_resume(save_path)
        assert loaded["personal_info"]["name"] == valid_final_resume["personal_info"]["name"]
        assert loaded["target_role"] == valid_final_resume["target_role"]


class TestBuildFinalResume:
    """Test converting optimizer candidates into Final Structured Resume format."""

    def test_build_from_optimizer_candidate(self):
        candidate = {
            "personal_info": {
                "name": "Kanishk Surwade",
                "target_title": "Data Annotator",
                "email": "kanishk@example.com",
                "phone": "+91-9834008224",
                "location": "Pune, India",
                "linkedin": "linkedin.com/in/kd4723",
                "github": "github.com/Kanishksurwade",
            },
            "summary": "Experienced Data Annotator skilled in video and audio annotation.",
            "skills": {
                "technical_skills": ["Data Annotation", "Video Annotation"],
                "tools_and_technologies": ["CVAT", "Python"],
                "core_competencies": ["Attention to Detail", "Remote Work"],
            },
            "experience": [
                {
                    "company": "Innodata Inc.",
                    "role": "AI & LLM Analyst",
                    "location": "Remote",
                    "start_date": "Dec 2025",
                    "end_date": "Jul 2026",
                    "bullets": [
                        "Conducted LLM evaluation and data annotation.",
                        "Conducted LLM evaluation and data annotation.", # duplicate to test deduplication
                    ],
                }
            ],
            "projects": [
                {
                    "name": "Audio Pipeline",
                    "technologies": ["Python", "Librosa"],
                    "bullets": ["Processed audio files."],
                }
            ],
            "education": [
                {
                    "institution": "University",
                    "degree": "B.E. Computer Engineering",
                    "start_year": "2020",
                    "end_year": "2024",
                }
            ],
            "certifications": [
                {"name": "Generative AI", "issuer": "Microsoft"}
            ],
        }

        structured = build_final_resume(candidate, target_role="Data Annotator")
        val = validate_final_resume(structured)

        assert val["valid"] is True
        assert structured["target_role"] == "Data Annotator"
        assert structured["skills"]["technical"] == ["Data Annotation", "Video Annotation"]
        assert structured["skills"]["tools"] == ["CVAT", "Python"]
        assert structured["skills"]["soft"] == ["Attention to Detail", "Remote Work"]
        assert len(structured["experience"][0]["bullets"]) == 1 # duplicate removed
        assert structured["education"][0]["start_date"] == "2020"
        assert structured["education"][0]["end_date"] == "2024"
