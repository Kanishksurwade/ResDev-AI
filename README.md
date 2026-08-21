# ResDev AI 🚀

> **Evidence-grounded AI resume automation — tailor a real master resume to a specific job, evaluate it, optimize it, and export a professional PDF.**

🌐 **Live Demo:** https://resdev-ai.streamlit.app  
💻 **GitHub:** https://github.com/KanishkSurwade/ResDev-AI

---

## What is ResDev AI?

ResDev AI automates the repetitive work of tailoring a resume for a job application.

You provide only:

1. **Job Title**
2. **Job Description**
3. **Master Resume**
4. **Resume Template**

Then ResDev AI:

```text
Job Description
       +
Master Resume
       ↓
JD Analysis
       ↓
Resume Matching
       ↓
Evidence-Grounded Generation
       ↓
Quality Evaluation
       ↓
ATS Analysis
       ↓
Multi-ATS Validation
       ↓
Optimization / Regeneration
       ↓
Structured JSON
       ↓
LaTeX
       ↓
PDF
```

The core rule is simple:

> **The Master Resume is the source of truth. The system should optimize the presentation of real experience, not invent experience.**

---

## ✨ Features

- Job description analysis
- Master resume upload and parsing
- Evidence-grounded resume generation
- Keyword and skills matching
- Experience relevance evaluation
- Grammar and readability checks
- ATS structure analysis
- Deterministic ATS scoring
- Multi-ATS compatibility validation
- Automatic optimization loop
- Iteration history
- Resume length and action-verb checks
- Unsupported-requirement audit
- JSON output
- LaTeX output
- PDF output
- Streamlit web interface
- Local/offline-first architecture
- Cloud-compatible Gemini deployment
- Automated regression tests

---

## 🧠 Why Evidence-Grounded?

A resume generator should not turn a job description into imaginary experience.

### ❌ Not this

```text
Job Description
      ↓
AI guesses qualifications
      ↓
Fake resume
```

### ✅ ResDev AI

```text
Verified Master Resume
      ↓
Relevant candidate evidence
      ↓
Job-specific presentation
      ↓
ATS-aware optimization
```

The system is designed to avoid inventing companies, responsibilities, certifications, achievements, statistics, skills, or project results that are not supported by the source resume.

---

## 🔄 Optimization Engine

ResDev AI does not have to accept the first generated resume.

```text
Generate
   ↓
Evaluate
   ↓
Improve?
 ┌─┴─┐
No  Yes
↓     ↓
Keep  Revision Plan
       ↓
    Regenerate
       ↓
     Evaluate
       ↓
      Keep
```

The optimization process considers ATS score, semantic relevance and Multi-ATS compatibility while protecting the evidence boundary.

---

## 📊 Evaluation

The internal evaluator considers:

| Area | Purpose |
|---|---|
| Keyword Match | Relevant job keywords |
| Skills Match | Required/preferred skills |
| Experience Relevance | Job-to-experience alignment |
| Grammar | Writing quality |
| Formatting | Resume structure |
| ATS Structure | Machine-readable organization |
| Readability | Clarity and scanability |
| Resume Length | Avoids excessive length |
| Action Verbs | Stronger bullet wording |
| Missing Keywords | Identifies useful gaps |
| Multi-ATS | Compatibility across internal ATS profiles |

### Final verified development result

**ATS:** 94/100  
**Semantic:** 92/100  
**Combined:** 93/100  
**Multi-ATS:** 6/6 passed

> These are ResDev AI's internal evaluation metrics. They are **not official scores from commercial ATS vendors** and do not guarantee interviews.

---

## 🏗️ Architecture

```text
                         ResDev AI
                             │
                       Streamlit UI
                             │
              ┌──────────────┴──────────────┐
              ↓                             ↓
        JD Analyzer                   Resume Parser
              │                             │
              └──────────────┬──────────────┘
                             ↓
                     Resume Matching
                             ↓
                Evidence-Grounded Generator
                             ↓
                     Quality Evaluator
                             ↓
                  Deterministic ATS Analyzer
                             ↓
                    Multi-ATS Validator
                             ↓
                    Optimization Guard
                             ↓
                     Structured Resume
                       /      |      \
                     JSON    LaTeX    PDF
```

### Local mode

```text
Streamlit
    ↓
Local Python pipeline
    ↓
Ollama + Qwen
    ↓
Local outputs
```

### Public cloud mode

```text
Streamlit Cloud
    ↓
Python pipeline
    ↓
Gemini
    ↓
LaTeX/PDF
    ↓
Downloads
```

The core application pipeline remains shared; only the AI runtime changes according to the environment.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and AI pipeline |
| Streamlit | Web interface |
| Ollama | Local AI runtime |
| Qwen | Local language model |
| Gemini | Public cloud AI runtime |
| JSON | Structured resume data |
| SQLite | Local storage/history components |
| LaTeX | Resume rendering |
| pytest | Automated testing |
| Git | Version control |
| GitHub | Source code and portfolio |

---

## 📁 Repository Structure

```text
ResDev-AI/
├── ai/                 # Analysis, generation, evaluation, optimization
├── app/                # Streamlit application
├── backend/            # Backend/pipeline components
├── data/               # Structured data
├── database/           # Database components
├── docs/               # Engineering documentation
├── generated/          # Generated artifacts
├── prompts/            # AI prompts
├── templates/          # Resume templates
├── tests/               # Automated tests
├── utils/              # Utility functions
├── .gitignore
├── PROJECT_INSTRUCTIONS.md
├── CURRENT_CONTEXT.md
├── ROADMAP.md
├── requirements.txt
└── packages.txt
```

---

## 💻 Run Locally

### Requirements

- Python
- Ollama
- Configured Qwen model
- MiKTeX or TeX Live
- Git

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Pull the configured local model:

```bash
ollama pull qwen3.5:4b
```

Run the application:

```bash
python -m streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## 🧪 Testing

Run the complete test suite:

```bash
python -m pytest -q
```

Final verified local result during production hardening:

```text
136 passed
1 warning
```

The warning came from an external Google GenAI dependency and did not fail the test suite.

---

## 🌐 Public Demo

Try ResDev AI:

**https://resdev-ai.streamlit.app**

The public version demonstrates the cloud-compatible workflow.

> Do not upload confidential or sensitive resumes to a public service unless you are comfortable with the service's current privacy and data-handling practices.

---

## 📚 Engineering Black Book

The complete development journey is documented separately:

**[ResDev AI Engineering Black Book](docs/RESDEV_AI_BLACK_BOOK.md)**

It covers:

- Original problem and vision
- Offline-first architecture
- JD analysis
- Resume matching
- Generation
- Evaluation
- ATS architecture
- Multi-ATS validation
- Optimization
- Local-to-cloud migration
- Ollama localhost failure
- Streamlit deployment problems
- LaTeX/PDF cloud failure
- Package/import problems
- Regression failures
- Debugging and fixes
- Testing
- Production verification
- Final architecture
- Engineering lessons
- Future roadmap

A print-ready PDF version is included in the project documentation package.

---

## 🎥 Product Demo

The project includes a recorded end-to-end demonstration showing:

```text
Landing Page
     ↓
Job Information
     ↓
Master Resume Upload
     ↓
Template Selection
     ↓
Generate Resume
     ↓
Optimization Pipeline
     ↓
Iteration Log
     ↓
ATS / Semantic / Multi-ATS Results
     ↓
PDF / LaTeX / JSON
```

---

## 🔐 Security & Privacy

Never commit:

- Personal master resumes
- API keys
- Passwords
- Access tokens
- Private credentials
- Sensitive personal information

The local architecture is designed to keep the core workflow local when Ollama is used.

Cloud deployment requires cloud AI credentials, which should be stored using deployment secrets/environment variables rather than source code.

---

## ⚠️ Limitations

ResDev AI is an automated resume assistant, not a hiring or interview guarantee.

ATS scores are internal estimates. Real applicant-tracking systems differ in their parsing and ranking behavior.

Resume quality also depends on the quality and completeness of the master resume.

---

## 🧭 Project Evolution

The Git history records the project growing from a foundation into a production-oriented system:

```text
Project Foundation
      ↓
Master Resume
      ↓
JD Analysis
      ↓
Resume Matching
      ↓
Resume Generation
      ↓
Quality Evaluation
      ↓
Automatic Optimization
      ↓
Deterministic ATS
      ↓
Multi-ATS
      ↓
Evidence-Grounded Optimization
      ↓
Structured Validation
      ↓
Streamlit + PDF
      ↓
Ollama → Gemini
      ↓
Cloud Hardening
      ↓
Production Tests
      ↓
Public Demo
```

---

## 🎯 Vision

The long-term goal is a simple one-click resume workflow:

```text
Job Title
   +
Job Description
   +
Master Resume
   +
Template
      ↓
   ResDev AI
      ↓
 Resume Ready
```

---

## 👨‍💻 Author

### Kanishk Surwade

Built as a hands-on project to learn and demonstrate:

**Software Development · AI Engineering · Automation · Python · LLM Integration · Prompt Engineering · ATS Optimization · Streamlit · LaTeX · Testing · Git · GitHub · Cloud Deployment**

---

## ⭐ Support

If you find ResDev AI useful or interesting:

- ⭐ Star the repository
- Try the public demo
- Explore the source code
- Open an issue
- Share feedback

---

## 📄 License

License details can be added before a formal open-source release.

---

**ResDev AI — Built by Kanishk Surwade**
