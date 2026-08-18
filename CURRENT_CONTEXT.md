# ResDev AI — Current Context & State Tracker

## Current State
- **Phase**: Phase 0 — Initial Foundation & Scaffolding
- **Status**: Scaffolding complete; project structure initialized.
- **Application Logic**: Not yet started (by design).
- **Environment**: Offline-first configuration ready for subsequent implementation phases.

---

## Active Configuration & Decisions

- **System Purpose**: Offline-first, one-click AI Resume Automation System.
- **AI Backend**: Ollama running `qwen3.5:4b`.
- **UI Framework**: Streamlit.
- **Database**: SQLite (local embedded storage).
- **Document Engine**: LaTeX (`pdflatex` / `xelatex`).
- **Dependencies**: Minimal (`streamlit`, `requests`, `jinja2`, `pytest`).

---

## Directory Setup Checklist

- [x] `app/` (Streamlit UI)
- [x] `backend/` (Pipeline orchestration)
- [x] `ai/` (Ollama integration & evaluation loops)
- [x] `prompts/` (Prompt templates)
- [x] `templates/` (LaTeX templates)
- [x] `data/` (Master resume data & schemas)
- [x] `generated/` (Outputs: JSON, TeX, PDF)
- [x] `database/` (SQLite schema & DB managers)
- [x] `utils/` (Helper utilities & compilation)
- [x] `tests/` (Test suites)
- [x] `docs/` (Documentation)

---

## Immediate Next Steps (Phase 1)
1. Define the canonical JSON resume schema in `data/schema.json`.
2. Implement the SQLite schema and database manager in `database/`.
3. Create sample master resume data for validation and testing.
