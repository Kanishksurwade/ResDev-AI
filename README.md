# ResDev AI

> **Offline-first, one-click AI Resume Automation System**

ResDev AI is an automated resume customization platform that runs completely offline on your local machine. It analyzes job descriptions and master candidate profiles, generates highly tailored resumes with iterative self-evaluation loops, and compiles professional LaTeX-formatted PDFs with full history tracking.

---

## 🚀 Key Features (Planned)

- **Offline-First Intelligence**: Powered locally by [Ollama](https://ollama.com/) and `qwen3.5:4b`.
- **One-Click Tailoring**: Provide a job title and description, and receive a tailored resume in seconds.
- **Automated Quality Evaluation**: Built-in self-scoring and regeneration loop ensuring high-impact bullet points and keyword alignment.
- **LaTeX Typesetting**: Crisp, ATS-friendly, professional PDF generation.
- **History & Tracking**: Local SQLite database logging past generations, scores, and job postings.
- **Cloud Sync (Future)**: Optional integration with Google Drive and Google Sheets for application tracking.

---

## 🛠️ Technology Stack

- **Application & UI**: Python 3.10+, [Streamlit](https://streamlit.io/)
- **Local AI Engine**: [Ollama](https://ollama.com/) (`qwen3.5:4b`)
- **Data & Storage**: JSON, SQLite
- **Typesetting & PDF Engine**: LaTeX (`pdflatex` / `xelatex`)
- **Version Control**: Git, GitHub
- **Integrations**: Google Drive API, Google Sheets API (planned)

---

## 📂 Directory Layout

```
ResDev-AI/
├── app/          # Streamlit UI application and pages
├── backend/      # Core pipeline orchestrator & workflow logic
├── ai/           # Ollama client, model manager, evaluation loops
├── prompts/      # Prompt templates (analysis, tailoring, scoring)
├── templates/    # LaTeX resume templates (.tex)
├── data/         # Master resume files and JSON schemas
├── generated/    # Generated JSON, LaTeX, and PDF files
├── database/     # SQLite schema definitions and repositories
├── utils/        # LaTeX compiler helpers, validators, file utilities
├── tests/        # Test suites
└── docs/         # Documentation and architectural guides
```

---

## 📋 Prerequisites

Before running ResDev AI, ensure you have the following installed:

1. **Python 3.10+**
2. **Ollama**: Installed and running locally.
   ```bash
   ollama pull qwen3.5:4b
   ```
3. **LaTeX Distribution**:
   - Windows: [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)
   - Linux: `sudo apt install texlive-latex-extra`
   - macOS: [MacTeX](https://www.tug.org/mactex/)

---

## 📄 Documentation

- [Project Instructions & System Architecture](file:///d:/ResDev-AI/PROJECT_INSTRUCTIONS.md)
- [Current Context & State Tracker](file:///d:/ResDev-AI/CURRENT_CONTEXT.md)
- [Development Roadmap](file:///d:/ResDev-AI/ROADMAP.md)
