[# ResDev AI

> **AI-powered resume automation system for creating job-specific,
> ATS-friendly resumes**

ResDev AI is a project I built to automate one of the most repetitive
parts of job applications: customizing the same master resume for
different jobs.

Instead of manually reading a job description, finding relevant
experience, rewriting bullet points, checking keywords, and formatting a
new PDF, ResDev AI is designed to bring these steps into one workflow.

**Built by Kanishk Surwade.**

------------------------------------------------------------------------

## 🎯 Why I Built This

Applying for different jobs often means changing the resume again and
again.

The same process is repeated:

1.  Read the job description.
2.  Find important skills and keywords.
3.  Read the master resume.
4.  Find relevant experience and projects.
5.  Rewrite the resume.
6.  Check ATS-style requirements.
7.  Fix formatting.
8.  Generate a PDF.

I wanted to build a system that could automate this process while
keeping the candidate's real experience as the source of truth.

This project is also my hands-on way of learning software development
and AI engineering by building a real application.

------------------------------------------------------------------------

## 🚀 What ResDev AI Does

The basic idea is:

``` text
Job Description + Master Resume
              ↓
        Job Analysis
              ↓
      Resume Analysis
              ↓
    Resume Generation
              ↓
      Quality Evaluation
              ↓
        Optimization
              ↓
       Structured JSON
              ↓
           LaTeX
              ↓
            PDF
```

The system is designed to create a targeted resume without simply
copying the job description or inventing experience.

------------------------------------------------------------------------

## 🧩 Main Features

### Job Description Analysis

The system analyzes the job description to identify relevant skills,
keywords, responsibilities, tools, technologies, experience
requirements, and other job-specific requirements.

### Master Resume as Source of Truth

The master resume is stored in structured form and contains candidate
information, capabilities, professional experience, projects, education,
achievements, and generation rules.

This helps keep generated resumes grounded in the candidate's actual
background.

### AI Resume Generation

The system selects relevant information from the master resume and
adapts it toward the target job.

It aims to highlight relevant experience, match useful skills, improve
bullet points, use stronger action verbs, remove unnecessary
information, keep the resume ATS-friendly, and avoid unsupported claims.

### Resume Evaluation

The generated resume is evaluated using several criteria:

  Area                   Purpose
  ---------------------- ---------------------------------------
  Keyword Match          Checks relevant job keywords
  Skills Match           Checks alignment with required skills
  Experience Relevance   Measures relevance of experience
  Grammar                Checks writing quality
  Formatting             Checks resume structure
  ATS Structure          Checks machine-readable structure
  Readability            Checks clarity
  Resume Length          Helps avoid unnecessary content
  Action Verbs           Checks strength of bullet wording
  Missing Keywords       Identifies important gaps

The project also contains multi-ATS validation logic.

------------------------------------------------------------------------

## 🔄 Optimization Loop

The project does not have to accept the first generated resume as final.

The workflow can evaluate a generated resume and use the results to
improve it:

``` text
Generate
   ↓
Evaluate
   ↓
Passed?
 ┌─┴─┐
Yes  No
 ↓    ↓
PDF  Improve
      ↓
   Regenerate
      ↓
    Evaluate
```

This is intended to make generation more reliable than a single AI
prompt.

------------------------------------------------------------------------

## 📊 Example Evaluation

During development, the system produced results such as:

``` text
ATS Score:        85/100
Semantic Score:   82/100
Combined Score:   83.5/100
```

These are **internal evaluation metrics**, not official scores from a
commercial ATS provider.

The application can also report multi-ATS validation results and
unsupported job-description requirements.

------------------------------------------------------------------------

## 🛡️ Evidence-Grounded Generation

One of the most important design principles is avoiding invented
experience.

A resume generator should not create fake companies, responsibilities,
skills, certifications, achievements, project results, or statistics.

The intended flow is:

``` text
Verified Master Resume
        ↓
Relevant Information
        ↓
Targeted Resume
```

not:

``` text
Job Description
        ↓
AI guesses experience
        ↓
Fake resume
```

------------------------------------------------------------------------

## 🖥️ User Interface

The application uses Streamlit.

The user provides:

-   Job Title
-   Job Description
-   Master Resume
-   Resume Template

Then the application starts the resume-generation pipeline and displays
the optimization results.

------------------------------------------------------------------------

## 📄 Output

The system can generate:

``` text
JSON
 ↓
LaTeX
 ↓
PDF
```

LaTeX is used to create the final professional resume PDF.

------------------------------------------------------------------------

## 🏗️ Architecture

``` text
                 ResDev AI
                     │
          ┌──────────┴──────────┐
          │                     │
     Streamlit UI           Backend
          │                     │
          └──────────┬──────────┘
                     │
              Resume Pipeline
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   JD Analysis   Resume Data   Generation
        │            │            │
        └────────────┼────────────┘
                     ↓
              Quality Evaluation
                     ↓
                 Optimization
                     ↓
              Structured JSON
                     ↓
                   LaTeX
                     ↓
                    PDF
```

------------------------------------------------------------------------

## 📁 Project Structure

``` text
ResDev-AI/
│
├── ai/                 # AI analysis, generation and optimization
├── app/                # Streamlit application
├── backend/            # Backend and pipeline components
├── data/               # Structured resume/project data
├── database/           # Database components
├── docs/               # Documentation
├── generated/          # Generated output
├── prompts/            # AI prompts
├── templates/          # Resume templates
├── tests/              # Automated tests
├── utils/              # Utility functions
│
├── .gitignore
├── CURRENT_CONTEXT.md
├── PROJECT_INSTRUCTIONS.md
├── README.md
├── ROADMAP.md
└── requirements.txt
```

------------------------------------------------------------------------

## 🛠️ Technology Stack

  Technology   Used For
  ------------ ----------------------------------
  Python       Main application and AI pipeline
  Streamlit    Web interface
  Ollama       Local AI runtime
  Qwen         Local language model
  JSON         Structured resume data
  SQLite       Local storage/history
  LaTeX        Resume and PDF generation
  Git          Version control
  GitHub       Source code and portfolio
  pytest       Automated testing

The local version follows a simple, modular and offline-first approach.

------------------------------------------------------------------------

## 💻 Run Locally

### Requirements

Install Python, Ollama, the Qwen model used by the project, a LaTeX
distribution such as MiKTeX or TeX Live, and Git.

Check Python:

``` bash
python --version
```

Install dependencies:

``` bash
python -m pip install -r requirements.txt
```

Install the configured Ollama model:

``` bash
ollama pull qwen3.5:4b
```

Make sure Ollama is running.

A LaTeX distribution such as MiKTeX or TeX Live is also required for PDF
generation.

### Start the application

``` bash
python -m streamlit run app/streamlit_app.py
```

Then open:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## 🧪 Testing

Run:

``` bash
pytest
```

For detailed output:

``` bash
pytest -v
```

------------------------------------------------------------------------

## 🔐 Privacy

The local version is designed to be offline-first.

When running locally, the application can run on the user's computer,
Ollama can run the AI model locally, resume files can remain on the
local machine, and generated files can remain local.

Never commit personal master resumes, passwords, API keys, access
tokens, private credentials, or sensitive personal information to
GitHub.

------------------------------------------------------------------------

## 🌐 Public Demo

A public web version is being prepared as the project showcase.

The intended experience is:

``` text
Open Website
     ↓
Enter Job Title
     ↓
Paste Job Description
     ↓
Upload Master Resume
     ↓
Choose Template
     ↓
Generate Resume
     ↓
Download PDF
```

The local application uses Ollama on the developer's computer. A public
deployment therefore needs a cloud-compatible AI setup instead of
depending on that local Ollama process.

**Public Demo: Coming soon**

------------------------------------------------------------------------

## 📌 Project Status

### Completed

-   [x] Project foundation
-   [x] Modular architecture
-   [x] Master resume structure
-   [x] Job description analysis
-   [x] Resume matching
-   [x] Resume generation
-   [x] Resume evaluation
-   [x] Resume optimization
-   [x] ATS analysis
-   [x] Multi-ATS validation
-   [x] Structured JSON generation
-   [x] LaTeX generation
-   [x] PDF generation
-   [x] Streamlit interface
-   [x] Automated testing
-   [x] Git/GitHub repository

### In Progress

-   [ ] Public working demo
-   [ ] Cloud-compatible AI workflow
-   [ ] Final portfolio screenshots
-   [ ] Final documentation

### Future Ideas

-   [ ] More resume templates
-   [ ] Optional Google Drive integration
-   [ ] Optional Google Sheets tracking
-   [ ] More evaluation methods
-   [ ] Additional output formats

------------------------------------------------------------------------

## ⚠️ Current Limitations

ResDev AI is an automated assistant, not a guarantee of interview
results.

The ATS scores are estimates produced by the project's own evaluation
logic. They are not official scores from every ATS platform.

Different companies use different recruiting systems and screening
processes.

The quality of the final resume also depends on the quality and
completeness of the master resume.

> A high internal score does not guarantee an interview or job offer.

------------------------------------------------------------------------

## 📚 What I Learned

This project was built as a practical learning project.

While building it, I worked with Python development, modular
architecture, JSON data structures, prompt engineering, local AI models,
resume generation, evaluation pipelines, ATS-style validation, LaTeX,
PDF rendering, Streamlit, automated testing, Git, GitHub, and debugging
a real application.

The goal was to learn by building instead of separating learning from
development.

------------------------------------------------------------------------

## 🎯 Project Vision

The final vision is a simple one-click workflow:

``` text
Job Title
    +
Job Description
    +
Master Resume
    ↓
  ResDev AI
    ↓
Analyze
    ↓
Generate
    ↓
Evaluate
    ↓
Improve if required
    ↓
Generate LaTeX
    ↓
Generate PDF
    ↓
Resume Ready
```

The aim is to make resume customization faster while keeping the content
grounded in the candidate's real experience.

------------------------------------------------------------------------

## 👨‍💻 Author

### Kanishk Surwade

**Creator and developer of ResDev AI**

This project was built to learn and demonstrate practical skills in
Software Development, AI Engineering, Automation, Python, Prompt
Engineering, Resume and ATS Optimization, Streamlit, LaTeX, Testing,
Git, and GitHub.

------------------------------------------------------------------------

## ⭐ Feedback

If you find the project interesting, you can:

-   ⭐ Star the repository
-   Explore the source code
-   Try the public demo when available
-   Open an issue with suggestions
-   Share feedback

------------------------------------------------------------------------

## 📜 License

License details will be added before the final public release.

------------------------------------------------------------------------

**ResDev AI --- Built by Kanishk Surwade**
](https://resdev-ai.streamlit.app/)
