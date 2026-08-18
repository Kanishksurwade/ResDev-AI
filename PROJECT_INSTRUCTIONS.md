# ResDev AI — Project Instructions & System Architecture

## 1. System Overview
**ResDev AI** is an offline-first, one-click AI Resume Automation System. It transforms a user's master resume and target job description into an optimized, highly tailored, professional resume through an automated AI generation, evaluation, and LaTeX-to-PDF compilation pipeline.

---

## 2. Inputs & Expected Outputs

### User Inputs:
1. **Job Title**: The target role title.
2. **Job Description (JD)**: Raw text or file containing the target job posting.
3. **Master Resume**: Structured JSON or comprehensive Markdown document containing the user's complete career history, skills, education, projects, and achievements.
4. **Resume Template**: Target LaTeX template defining layout, styling, and structural constraints.

### Generated Outputs:
1. **Tailored Resume JSON**: Validated structured data adhering to a canonical resume schema.
2. **LaTeX Source (`.tex`)**: Rendered LaTeX code dynamically populated from the tailored JSON.
3. **Compiled PDF (`.pdf`)**: Formatted document ready for job application submission.
4. **Evaluation & Quality Report**: Detailed score and breakdown against the job description requirements.
5. **Historical Record**: Persisted snapshot in SQLite for auditing, tracking, and future retrieval.

---

## 3. End-to-End Workflow (10 Steps)

1. **Analyze the Job Description**: Extract key requirements, mandatory technical skills, soft skills, domain keywords, and responsibilities.
2. **Analyze the Master Resume**: Parse candidate experiences, metrics, competencies, and achievements.
3. **Generate a Tailored Resume**: Align candidate experience directly to the target JD while preserving factual truthfulness and tone.
4. **Evaluate Resume Quality**: Score the tailored resume against clarity, ATS keyword alignment, impact metrics (e.g., Google X-Y-Z formula), and relevance.
5. **Regenerate if Quality Threshold Not Met**: Trigger an automated revision loop with feedback if the evaluation score falls below the target threshold.
6. **Generate Structured JSON**: Standardize the finalized content into a strict, validated JSON resume schema.
7. **Generate LaTeX**: Inject the structured JSON data into the selected LaTeX template via templating.
8. **Generate PDF**: Compile the LaTeX source into a PDF using local TeX engines (`pdflatex` or `xelatex`).
9. **Save Generated Files**: Store artifacts (`.json`, `.tex`, `.pdf`, metadata) into structured directories (`generated/`).
10. **Maintain Resume History**: Record run metadata, JD details, scores, and file references in a local SQLite database.

---

## 4. Technology Decisions & Constraints

> [!IMPORTANT]
> The technology stack is strictly fixed. Do not introduce alternative frameworks, cloud LLMs, or unauthorized dependencies.

| Category | Technology | Usage |
|---|---|---|
| **Core Language** | Python 3.10+ | Core application runtime and logic |
| **User Interface** | Streamlit | Clean, reactive one-click dashboard and history browser |
| **Local AI Engine** | Ollama | Local LLM host and inference engine |
| **Model** | Qwen3.5 4B (`qwen3.5:4b`) | Core reasoning, extraction, tailoring, and evaluation |
| **Data Format** | JSON | Structured interchange between AI prompts, backend, and templates |
| **Database** | SQLite | Local relational storage for runs, resumes, jobs, and audit history |
| **Typesetting & PDF** | LaTeX (`pdflatex` / `xelatex`) | Professional document rendering and compilation |
| **Version Control** | Git & GitHub | Source tracking and codebase management |
| **Future Integrations** | Google Drive API, Google Sheets API | Automated cloud backup and application tracking sync |

---

## 5. Architectural Structure & Module Boundaries

```
ResDev-AI/
├── app/          # Streamlit UI layers, views, components, and state management
├── backend/      # Pipeline orchestrator, workflow engine, file handlers
├── ai/           # Ollama client, model manager, structured output parsers, evaluator
├── prompts/      # Text & jinja prompt templates (JD analysis, tailoring, scoring)
├── templates/    # LaTeX resume templates (.tex) and asset files
├── data/         # Schemas, sample inputs, and master resume storage
├── generated/    # Output storage for generated JSON, LaTeX, and compiled PDFs
├── database/     # SQLite database models, schema setup, and query repositories
├── utils/        # General utilities (LaTeX compilers, file helpers, validators)
├── tests/        # Unit, integration, and prompt evaluation tests
└── docs/         # System architecture, schemas, and usage documentation
```

---

## 6. Engineering & Operational Rules

1. **Offline-First**: All LLM processing runs strictly on local hardware via Ollama. No external AI API calls.
2. **Deterministic Output**: Always enforce strict JSON validation on LLM responses. Use robust retry mechanisms on malformed outputs.
3. **Factual Integrity**: Tailoring must highlight and rephrase existing master resume experiences without hallucinating qualifications or employment records.
4. **Separation of Concerns**:
   - `app/` strictly handles rendering and user interactions; it never executes raw prompts or database queries directly.
   - `backend/` orchestrates workflows between `ai/`, `database/`, and `utils/`.
   - `ai/` encapsulates all Ollama communication and prompt logic.
   - `database/` isolates all SQLite operations.
5. **Idempotent & Safe Compilation**: LaTeX compilation runs in sandboxed or isolated temporary directories before copying clean artifacts to `generated/`.
