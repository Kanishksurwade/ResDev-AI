# ResDev AI — Engineering Black Book

> From offline-first prototype to public AI resume automation system.

**Author:** Kanishk Surwade  
**Public demo:** https://resdev-ai.streamlit.app  
**Repository:** https://github.com/KanishkSurwade/ResDev-AI

## Executive Summary

ResDev AI automates resume customization from a Job Title, Job Description, Master Resume, and Resume Template. It analyzes the job, matches candidate evidence, generates a targeted resume, evaluates quality, iteratively optimizes it, and produces JSON, LaTeX and PDF outputs.

The central engineering principle is **evidence-grounded generation**: the master resume remains the source of truth so the system does not invent unsupported experience.

## Evolution

```text
Offline-first prototype
    ↓
Master Resume + validation
    ↓
JD Analysis
    ↓
JD ↔ Resume Matching
    ↓
Resume Generation
    ↓
Quality Evaluation
    ↓
Automatic Optimization
    ↓
Deterministic ATS
    ↓
Multi-ATS Validation
    ↓
Evidence-Grounded Optimization
    ↓
Structured Validation
    ↓
Streamlit + PDF
    ↓
Ollama → Gemini cloud migration
    ↓
Secure secrets + cloud dependencies
    ↓
Production tests + PDF hardening
    ↓
Public ResDev AI
```

## Major Engineering Problems Solved

### Ollama localhost failure

A hosted Streamlit container cannot reach the developer's laptop through `localhost:11434`. The solution was to separate local and cloud AI execution and migrate the public path to Gemini.

### Cloud LaTeX/PDF failure

The cloud environment initially generated LaTeX but lacked the tooling required to compile the PDF. The deployment was hardened with the required system package configuration and PDF error handling.

### Streamlit runtime / stopping behavior

Interrupted local Streamlit processes sometimes remained in a stopping state. The issue was handled as a process/runtime problem and verified through clean restarts and repeatable test runs.

### Multi-ATS regression

A regression test exposed a fixture that produced 0/6 expected Multi-ATS passes. The validator/test behavior was corrected without weakening the intended quality requirement.

### ATS bottlenecks

Some iterations had strong semantic scores but weak ATS scores. This motivated deterministic ATS analysis, revision planning, optimization guards and regression tests instead of relying on a single LLM score.

## Final Architecture

```text
                    ResDev AI
                        │
                  Streamlit UI
                        │
        ┌───────────────┴────────────────┐
        ↓                                ↓
   JD Analysis                     Resume Parser
        │                                │
        └───────────────┬────────────────┘
                        ↓
                Resume Matching
                        ↓
          Evidence-Grounded Generation
                        ↓
               Quality Evaluation
                        ↓
             Deterministic ATS
                        ↓
              Multi-ATS Validator
                        ↓
             Optimization Guard
                        ↓
              Structured Resume
                 /      |                     JSON   LaTeX    PDF
```

## Final User Workflow

1. Enter Job Title.
2. Paste Job Description.
3. Upload Master Resume.
4. Select Resume Template.
5. Click **Generate Resume**.
6. Automated analysis and optimization runs.
7. Download PDF, LaTeX and JSON.

## Verified Final State

- **Automated tests:** 136 passed; 1 external-library deprecation warning.
- **Best verified run:** ATS 94/100.
- **Semantic:** 92/100.
- **Combined:** 93/100.
- **Multi-ATS:** 6/6 passed.
- **Outputs:** PDF, LaTeX and JSON.
- **Public demo:** https://resdev-ai.streamlit.app

> These ATS numbers are internal evaluation metrics, not official scores from commercial ATS vendors.

## Demo Evidence

The uploaded 111.9-second demo records the actual product workflow: landing page, job input, master resume upload, template selection, optimization pipeline, iteration log, results and downloadable artifacts.

The demo captures an earlier development state. That is useful evidence because later engineering iterations improved the system to the verified 94/92/93 + 6/6 state.

## Git History as an Engineering Record

The repository history records the progression through:

- Project foundation
- Master resume data and validation
- JD analysis
- JD/master-resume matching
- Resume generation
- Resume quality evaluation
- Automatic optimization
- Deterministic ATS
- Multi-ATS validation
- Evidence-grounded optimization
- Structured validation
- Streamlit frontend and PDF rendering
- Ollama → Gemini migration
- Secure cloud configuration
- Cloud import fixes
- Resume upload/parsing
- Production optimization
- Production pipeline finalization
- Cloud pipeline tests
- Streamlit Cloud PDF hardening

## Lessons Learned

The main lesson is that an AI prompt is not a production system by itself. Production reliability required deterministic checks, structured validation, regression tests, error handling, secure configuration, environment-aware deployment, PDF tooling, and repeated real-world verification.

## Future Roadmap

- More resume templates
- Resume history
- Optional Google Drive / Sheets integrations
- Additional evaluation methods
- Stronger privacy controls
- More deployment options

## Final Reflection

ResDev AI was built as a learning-by-building project. The final result is both a usable resume automation system and a record of practical software engineering: architecture, AI integration, deterministic validation, testing, debugging, cloud deployment, and production hardening.
