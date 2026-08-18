# ResDev AI — Project Roadmap

This roadmap outlines the phased development plan for **ResDev AI**, an offline-first, one-click AI Resume Automation System.

---

## Phase 0: Project Scaffolding & Foundation ✅
- [x] Create modular directory architecture (`app/`, `backend/`, `ai/`, `prompts/`, `templates/`, `data/`, `generated/`, `database/`, `utils/`, `tests/`, `docs/`).
- [x] Establish foundational documentation (`PROJECT_INSTRUCTIONS.md`, `CURRENT_CONTEXT.md`, `ROADMAP.md`, `README.md`).
- [x] Configure repository definitions (`requirements.txt`, `.gitignore`).

---

## Phase 1: Data Modeling & Storage Layer ⏳
- [ ] Define canonical JSON Resume schema (`data/resume_schema.json`).
- [ ] Implement SQLite schema and migrations in `database/schema.sql`.
  - Tables: `master_resumes`, `job_descriptions`, `tailored_resumes`, `generation_history`, `evaluation_logs`.
- [ ] Build SQLite data access layer / repository in `database/db_manager.py`.
- [ ] Create initial validation utilities for JSON schemas and inputs in `utils/validators.py`.
- [ ] Add sample master resume datasets in `data/sample_master_resume.json`.

---

## Phase 2: AI Engine & Local LLM Integration
- [ ] Implement local Ollama API communication client in `ai/ollama_client.py`.
- [ ] Configure `qwen3.5:4b` model parameters and prompt handlers.
- [ ] Develop and refine prompt templates in `prompts/`:
  - `jd_analysis.prompt`: Extract skills, keywords, responsibilities, culture fit.
  - `resume_tailor.prompt`: Align achievements to JD requirements using action-oriented framing.
  - `resume_evaluator.prompt`: Score relevance, ATS keyword match, and impact metrics.
- [ ] Implement the automated self-correction loop in `ai/evaluator.py`:
  - Score against quality threshold.
  - Re-prompt with specific refinement instructions if score is below threshold.

---

## Phase 3: LaTeX & PDF Generation Pipeline
- [ ] Design and test base LaTeX resume templates in `templates/` (modern, classic, technical layouts).
- [ ] Implement LaTeX template renderer using Jinja2 in `backend/latex_renderer.py` (with LaTeX special character escaping).
- [ ] Implement local PDF compiler wrapper in `utils/pdf_compiler.py` (`pdflatex` / `xelatex`).
- [ ] Add error parsing and safe fallback handling for LaTeX compilation failures.
- [ ] Implement file artifact manager in `backend/file_manager.py` to organize outputs in `generated/`.

---

## Phase 4: Pipeline Orchestration & Streamlit UI
- [ ] Build the core orchestrator in `backend/pipeline.py` executing the complete 10-step workflow.
- [ ] Build the Streamlit user interface in `app/`:
  - **Dashboard / One-Click Generator**: Input job title, paste JD, select template, and trigger one-click generation.
  - **Master Resume Editor**: View and update the candidate master profile.
  - **Live Preview & Downloads**: View compiled PDF, tailored JSON, LaTeX source, and evaluation metrics.
  - **History Viewer**: Browse past generations, compare scores, and retrieve previous resumes.

---

## Phase 5: Cloud Integrations & Production Hardening
- [ ] Add Google Drive API integration for automatic cloud backup of generated PDFs.
- [ ] Add Google Sheets API integration for automated job application tracking.
- [ ] Implement end-to-end automated tests in `tests/` covering:
  - Schema validation.
  - Pipeline orchestration.
  - LaTeX compilation stability.
  - SQLite transactions.
- [ ] Optimize offline inference speed and memory footprint.
