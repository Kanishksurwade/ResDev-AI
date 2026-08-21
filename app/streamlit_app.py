"""
ResDev AI — Streamlit Frontend
Wired to the existing resume optimization pipeline (ai/resume_optimizer.py).
"""

import json
import random
import sys
import textwrap
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ai.* imports resolve correctly
# whether Streamlit is launched from the repo root or the app/ subdirectory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai.resume_optimizer import optimize_resume
from ai.latex_renderer import render_resume_files
from ai.final_resume_validator import save_final_resume
from utils.resume_parser import parse_uploaded_resume, ResumeParseError, SUPPORTED_EXTENSIONS

_GENERATED_DIR = _REPO_ROOT / "generated"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResDev AI — Evidence-Grounded Resume Automation",
    page_icon="🕹️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SAFE HTML RENDER HELPER
# ------------------------------------------------------------
# textwrap.dedent + strip() guarantees no line ever starts with
# 4+ leading spaces, which is what makes Streamlit's markdown
# parser mistake HTML for a code block.
# ============================================================

def md(html_string: str) -> None:
    st.markdown(textwrap.dedent(html_string).strip(), unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS — CYBERPUNK ARCADE THEME
# ============================================================

md(
    """
    <style>

    :root {
        --bg: #0a0a14;
        --panel: #12121f;
        --ink: #f2f2f7;
        --muted: #8e8ea8;
        --cyan: #00f6ff;
        --pink: #ff2e9a;
        --yellow: #ffd60a;
        --green: #39ff6a;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(255,46,154,0.10), transparent 40%),
            radial-gradient(circle at 85% 10%, rgba(0,246,255,0.10), transparent 40%),
            var(--bg);
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 30px;
        padding-bottom: 60px;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent; }

    * {
        font-family: "Courier New", ui-monospace, "SFMono-Regular", monospace;
    }

    /* CRT scanline overlay */
    .scanlines {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 9999;
        background: repeating-linear-gradient(
            to bottom,
            rgba(0,0,0,0) 0px,
            rgba(0,0,0,0) 2px,
            rgba(0,0,0,0.10) 3px
        );
        mix-blend-mode: overlay;
    }

    /* ================= MARQUEE / HERO ================= */

    .arcade-frame {
        border: 3px solid var(--cyan);
        clip-path: polygon(
            0 10px, 10px 10px, 10px 0, calc(100% - 10px) 0,
            calc(100% - 10px) 10px, 100% 10px,
            100% calc(100% - 10px), calc(100% - 10px) calc(100% - 10px),
            calc(100% - 10px) 100%, 10px 100%,
            10px calc(100% - 10px), 0 calc(100% - 10px)
        );
        background: var(--panel);
        padding: 26px 28px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 0 24px rgba(0,246,255,0.15);
    }

    .blink {
        animation: blink 1.1s steps(2, start) infinite;
        color: var(--yellow);
        font-size: 12px;
        letter-spacing: 3px;
        margin-bottom: 10px;
    }

    @keyframes blink { to { visibility: hidden; } }

    .logo {
        font-size: 46px;
        font-weight: 900;
        letter-spacing: 4px;
        margin-bottom: 10px;
    }
    .logo span:nth-child(1) { color: var(--pink); }
    .logo span:nth-child(2) { color: var(--yellow); }
    .logo span:nth-child(3) { color: var(--cyan); }
    .logo span:nth-child(4) { color: var(--green); }
    .logo span:nth-child(5) { color: var(--pink); }
    .logo span:nth-child(6) { color: var(--yellow); }
    .logo span {
        text-shadow: 0 0 8px currentColor, 0 0 18px currentColor;
    }

    .tagline {
        color: var(--muted);
        font-size: 13px;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ================= LEVEL PANELS ================= */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: var(--panel) !important;
        border: 3px solid var(--pink) !important;
        clip-path: polygon(
            0 8px, 8px 8px, 8px 0, calc(100% - 8px) 0,
            calc(100% - 8px) 8px, 100% 8px,
            100% calc(100% - 8px), calc(100% - 8px) calc(100% - 8px),
            calc(100% - 8px) 100%, 8px 100%,
            8px calc(100% - 8px), 0 calc(100% - 8px)
        ) !important;
        padding: 20px 20px 6px 20px !important;
        margin-bottom: 18px;
        box-shadow: 0 0 16px rgba(255,46,154,0.12);
    }

    .level-tag {
        display: inline-block;
        background: var(--yellow);
        color: #1a1400;
        font-weight: 900;
        font-size: 11px;
        padding: 3px 8px;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .level-title {
        color: var(--ink);
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .level-sub {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 14px;
    }

    /* ================= INPUTS ================= */

    label { color: var(--cyan) !important; font-size: 12px !important; font-weight: 700 !important; letter-spacing: 0.5px; }

    input, textarea {
        background: #0a0a14 !important;
        color: var(--ink) !important;
        border: 2px solid #2a2a45 !important;
        border-radius: 2px !important;
    }
    input:focus, textarea:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 2px rgba(0,246,255,0.25) !important;
    }
    div[data-baseweb="select"] > div {
        background: #0a0a14 !important;
        border: 2px solid #2a2a45 !important;
        border-radius: 2px !important;
    }

    /* ================= FILE UPLOADER ================= */

    section[data-testid="stFileUploader"] {
        background: #0a0a14 !important;
        border: 2px dashed #2a2a45 !important;
        border-radius: 2px !important;
        padding: 8px !important;
    }
    section[data-testid="stFileUploader"]:hover {
        border-color: var(--cyan) !important;
    }

    /* ================= GENERATE BUTTON ================= */

    .stButton > button {
        width: 100%;
        height: 56px;
        border: 3px solid var(--yellow) !important;
        border-radius: 2px;
        background: linear-gradient(135deg, var(--pink), #b3006b);
        color: #fff;
        font-size: 15px;
        font-weight: 900;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 0 18px rgba(255,46,154,0.4);
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--cyan), #0077a8);
        box-shadow: 0 0 22px rgba(0,246,255,0.5);
        transform: translateY(-2px);
        border-color: var(--cyan) !important;
    }

    /* ================= QUEST TRACKER ================= */

    .quest-heart { font-size: 20px; margin-right: 4px; }
    .quest-label { color: var(--muted); font-size: 11px; letter-spacing: 1px; margin-bottom: 6px; }

    /* ================= TIP TICKER ================= */

    .tip-box {
        color: var(--green);
        font-size: 12.5px;
        line-height: 1.6;
        border-left: 3px solid var(--green);
        padding-left: 10px;
        margin: 8px 0 14px 0;
    }

    /* ================= FOOTER ================= */

    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 10.5px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 36px;
        padding-top: 12px;
        border-top: 1px solid #1e1e30;
    }
    .footer a {
        color: var(--cyan);
        text-decoration: none;
    }

    </style>
    <div class="scanlines"></div>
    """
)


# ============================================================
# HERO / MARQUEE
# ============================================================

md(
    """
    <div class="arcade-frame">
        <div class="blink">★ PRESS START TO BUILD YOUR RESUME ★</div>
        <div class="logo">
            <span>R</span><span>E</span><span>S</span><span>D</span><span>E</span><span>V</span> AI
        </div>
        <div class="tagline">
            Upload your resume. Paste a job description. ResDev AI tailors
            a fully ATS-optimised resume — grounded only in <em>your</em>
            real experience. No fluff, no invented stats.
        </div>
    </div>
    """
)


# ============================================================
# TIP QUOTES
# ============================================================

TIPS = [
    "TIP: Mirror the exact keywords from the job title in your first bullet.",
    "TIP: Numbers beat adjectives. 'Reduced errors 18%' > 'Very accurate'.",
    "TIP: One resume per job. Generic resumes lose to ATS filters.",
    "TIP: Recruiters skim for 6-8 seconds. Lead with impact, not duties.",
    "TIP: Keep bullet lines under 2 rows -- dense paragraphs get skipped.",
    "TIP: Match your resume's job titles to the market's common titles.",
    "TIP: Every claim on your resume should be provable in an interview.",
]

if "tip_idx" not in st.session_state:
    st.session_state.tip_idx = random.randrange(len(TIPS))


# ============================================================
# LAYOUT — 3 COLUMNS
# ============================================================

col_left, col_center, col_right = st.columns([1, 2.3, 1], gap="medium")


# ─────────────────────────────────────────────────────────────
# CENTER: main form
# ─────────────────────────────────────────────────────────────
with col_center:

    # ── LEVEL 01 — Job Information ────────────────────────────
    with st.container(border=True):
        md(
            """
            <div class="level-tag">LEVEL 01</div>
            <div class="level-title">Job Information</div>
            <div class="level-sub">Tell ResDev AI which role you're applying for.</div>
            """
        )

        job_title = st.text_input(
            "Job title",
            placeholder="e.g. Data Annotator",
            key="job_title_input",
        )

        job_description = st.text_area(
            "Job description",
            placeholder="Paste the complete job description here...",
            height=180,
            key="job_desc_input",
        )

    # ── LEVEL 02 — Master Resume Upload ───────────────────────
    with st.container(border=True):
        md(
            """
            <div class="level-tag">LEVEL 02</div>
            <div class="level-title">Upload Your Master Resume</div>
            <div class="level-sub">Your resume is the only source of truth — nothing is invented.</div>
            """
        )

        uploaded_file = st.file_uploader(
            "Master Resume",
            type=["pdf", "docx", "txt", "json"],
            help=(
                "Supported: PDF, DOCX, TXT, JSON.  \n"
                "Legacy .doc files are not supported — please save as DOCX first."
            ),
            key="resume_uploader",
            label_visibility="collapsed",
        )

        md(
            '<div class="level-sub" style="margin-top:6px;">'
            "Supported: <strong>PDF · DOCX · TXT · JSON</strong>"
            " &nbsp;|&nbsp; Legacy .doc not supported"
            "</div>"
        )

        # Parse on upload — cache per session so we don't re-parse on every rerun
        master_resume_data = None
        resume_ready = False

        if uploaded_file is not None:
            cache_key = f"parsed_resume_{uploaded_file.name}_{uploaded_file.size}"
            if cache_key not in st.session_state:
                try:
                    parsed = parse_uploaded_resume(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                    )
                    st.session_state[cache_key] = parsed
                except ResumeParseError as _parse_err:
                    st.error(f"⚠️ Resume upload failed:  \n{_parse_err}")
                    st.session_state[cache_key] = None
                except Exception as _unexpected:
                    st.error(f"⚠️ Unexpected error reading resume: {_unexpected}")
                    st.session_state[cache_key] = None

            _cached = st.session_state.get(cache_key)
            if _cached is not None:
                master_resume_data = _cached
                resume_ready = True
                candidate_name = (
                    _cached.get("candidate", {})
                    .get("personal_info", {})
                    .get("name", uploaded_file.name)
                )
                st.success(
                    f"✅ **{uploaded_file.name}** loaded successfully  \n"
                    f"Candidate: **{candidate_name}**"
                )
        else:
            st.info(
                "⬆️ Upload your master resume to continue.  \n"
                "Supported formats: **PDF, DOCX, TXT, JSON**"
            )

    # ── LEVEL 03 — Resume Template ────────────────────────────
    with st.container(border=True):
        md(
            """
            <div class="level-tag">LEVEL 03</div>
            <div class="level-title">Resume Template</div>
            <div class="level-sub">Choose an ATS-friendly layout for your generated resume.</div>
            """
        )

        resume_template = st.selectbox(
            "Resume template",
            ["Professional", "Modern", "Minimal", "Executive"],
            key="template_select",
        )

        if resume_template != "Professional":
            st.caption(
                f"'{resume_template}' is a placeholder — only "
                f"'Professional' is wired to the generation pipeline."
            )

    # ── GENERATE BUTTON ───────────────────────────────────────
    if st.button("▶ Generate Resume", key="generate_btn"):

        if not job_title.strip():
            st.error("⚠️ Please enter a Job Title.")

        elif not job_description.strip():
            st.error("⚠️ Please enter the Job Description.")

        elif not resume_ready or master_resume_data is None:
            st.error(
                "⚠️ **No resume uploaded.**  \n"
                "Please upload your master resume in LEVEL 02 before generating."
            )

        else:
            # ── Run the optimization pipeline ──────────────────
            st.markdown("---")
            st.markdown("### ⚙️ Optimization Pipeline")

            _progress_bar = st.progress(0, text="Initialising pipeline…")
            _status_area = st.empty()
            _iter_log = st.expander("📋 Iteration Log", expanded=False)

            def _on_progress(iteration: int, max_it: int, data: dict) -> None:
                pct = int((iteration / max_it) * 80)
                ats_tag = "✅ PASS" if data.get("ats_passed") else "❌ FAIL"
                semantic_tag = "✅ PASS" if data.get("semantic_passed", data.get("qwen_passed")) else "⚠️ NEEDS REVISION"
                ev_tag = "🔒 VERIFIED" if data.get("evidence_passed") else "🚨 VIOLATION"
                sem_score = data.get("semantic_score", data.get("gemini_score", data.get("qwen_score", 0)))
                _progress_bar.progress(
                    pct,
                    text=f"Iteration {iteration}/{max_it} — ATS {data['ats_score']}/100 {ats_tag}",
                )
                _status_area.info(
                    f"**Iteration {iteration}/{max_it}**  \n"
                    f"Evidence: {ev_tag} · "
                    f"ATS: {data['ats_score']}/100 {ats_tag} · "
                    f"Semantic: {sem_score}/100 {semantic_tag} · "
                    f"Guard: `{data['decision_status']}`"
                )
                with _iter_log:
                    st.markdown(
                        f"**Iter {iteration}** — ATS `{data['ats_score']}` | "
                        f"Semantic `{sem_score}` | "
                        f"Multi-ATS `{data['multi_ats_passed']}/{data['multi_ats_total']}` | "
                        f"Decision: `{data['decision_status']}` — {data['decision_reason']}"
                    )
                    if data.get("evidence_violations"):
                        for _v in data["evidence_violations"][:2]:
                            st.warning(
                                f"Evidence violation: {_v.get('type')} → "
                                f"`{_v.get('value')}` ({_v.get('reason')})"
                            )

            try:
                optimization_result = optimize_resume(
                    master_resume=master_resume_data,
                    job_description=job_description.strip(),
                    target_score=86,
                    max_iterations=5,
                    progress_callback=_on_progress,
                )
            except Exception as _e:
                _progress_bar.empty()
                _status_area.empty()
                _err_msg = str(_e)
                # Classify the error for a clean user-facing message
                _is_api_error = any(s in _err_msg.lower() for s in [
                    "429", "503", "quota", "rate limit", "api key", "gemini", "timeout",
                    "network", "connection", "socket", "transport",
                ])
                if _is_api_error:
                    st.error(
                        "**AI provider temporarily unavailable.**  \n"
                        "The Gemini API returned an error. This is usually a rate-limit or "
                        "temporary outage. Please wait 30–60 seconds and try again.  \n"
                        f"*Details:* `{_err_msg[:200]}`"
                    )
                else:
                    st.error(
                        "**Resume optimization failed.**  \n"
                        "An unexpected error occurred during generation. "
                        "Please check your Job Description and Master Resume, then try again.  \n"
                        f"*Details:* `{_err_msg[:300]}`"
                    )
                st.stop()

            # ── Save final_resume.json ──────────────────────────
            _progress_bar.progress(85, text="Saving final_resume.json…")
            _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            _json_out_path = _GENERATED_DIR / "final_resume.json"
            final_structured_resume = optimization_result["final_structured_resume"]

            try:
                save_final_resume(final_structured_resume, _json_out_path)
            except Exception as _e:
                _progress_bar.empty()
                st.error(
                    "**Failed to save resume JSON.**  \n"
                    "The optimization completed but the output file could not be written. "
                    "Please check disk space or file permissions and retry.  \n"
                    f"*Details:* `{str(_e)[:200]}`"
                )
                st.stop()

            # ── Render LaTeX + compile PDF ──────────────────────
            _progress_bar.progress(90, text="Rendering LaTeX & compiling PDF…")
            _tex_out_path = _GENERATED_DIR / "final_resume.tex"
            _pdf_out_path = _GENERATED_DIR / "final_resume.pdf"

            try:
                render_result = render_resume_files(
                    resume_data=final_structured_resume,
                    output_tex_path=_tex_out_path,
                    output_pdf_path=_pdf_out_path,
                    compile_pdf_if_available=True,
                )
            except Exception as _e:
                _progress_bar.empty()
                _err_str = str(_e)
                render_result = {
                    "status": "ERROR",
                    "tex_path": str(_tex_out_path) if _tex_out_path.is_file() else "",
                    "pdf_path": None,
                    "pdflatex_available": False,
                    "message": f"Rendering error: {_err_str}",
                }

            _progress_bar.progress(100, text="Done!")
            _status_area.empty()

            # ── Results summary ─────────────────────────────────
            opt_status = optimization_result.get("status", "UNKNOWN")
            best_ats = optimization_result.get("best_ats_score", 0)
            best_semantic = optimization_result.get("best_semantic_score", optimization_result.get("best_gemini_score", optimization_result.get("best_qwen_score", 0)))
            best_combined = optimization_result.get("best_combined_score", 0)
            best_iter = optimization_result.get("best_iteration", "-")
            total_iters = optimization_result.get("total_iterations", "-")
            multi_passed = optimization_result.get("multi_ats_passed", 0)
            multi_total = optimization_result.get("multi_ats_total", 0)

            render_status = render_result.get("status", "ERROR")
            render_msg = render_result.get("message", "")

            st.markdown("---")
            st.markdown("### 🏁 Results")

            if render_status == "SUCCESS":
                st.success(
                    f"**Resume generated successfully!**  \n"
                    f"Status: `{opt_status}` · Best iteration: #{best_iter} of {total_iters}  \n"
                    f"ATS Score: **{best_ats}/100** · Semantic: **{best_semantic}/100** · "
                    f"Combined: **{best_combined}/100**  \n"
                    f"Multi-ATS platforms: **{multi_passed}/{multi_total}** passed"
                )
            elif render_status == "TEX_ONLY":
                st.warning(
                    f"**Resume optimised — LaTeX source created, but PDF compilation was skipped or unavailable.**  \n"
                    f"{render_msg}  \n"
                    f"ATS: {best_ats}/100 · Semantic: {best_semantic}/100 · "
                    f"Multi-ATS: {multi_passed}/{multi_total}"
                )
            else:
                st.error(f"**Rendering error:** {render_msg}")

            # Schema validation warnings
            val = optimization_result.get("final_resume_validation", {})
            if val.get("errors"):
                st.warning(
                    f"Schema validation issues ({len(val['errors'])}):\n"
                    + "\n".join(f"- {e}" for e in val["errors"][:5])
                )

            # ── Download buttons ────────────────────────────────
            st.markdown("#### 📥 Downloads")
            _dl_cols = st.columns(3)

            # 1. PDF Download
            with _dl_cols[0]:
                _raw_pdf = render_result.get("pdf_path")
                _pdf_path = Path(_raw_pdf) if _raw_pdf else None
                if _pdf_path and _pdf_path.is_file():
                    try:
                        with open(_pdf_path, "rb") as _pdf_file:
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=_pdf_file.read(),
                                file_name="final_resume.pdf",
                                mime="application/pdf",
                                key="download_pdf",
                            )
                    except Exception as _pdf_read_err:
                        st.caption(f"Could not read PDF: {_pdf_read_err}")
                else:
                    st.caption("📄 PDF not available (pdflatex not found or compilation failed).")

            # 2. LaTeX Source Download (.tex)
            with _dl_cols[1]:
                if _tex_out_path.is_file():
                    try:
                        with open(_tex_out_path, "r", encoding="utf-8") as _tf:
                            st.download_button(
                                label="⬇️ Download LaTeX (.tex)",
                                data=_tf.read(),
                                file_name="final_resume.tex",
                                mime="text/plain",
                                key="download_tex",
                            )
                    except Exception as _tex_read_err:
                        st.caption(f"LaTeX source unreadable: {_tex_read_err}")
                else:
                    st.caption("LaTeX source unavailable.")

            # 3. JSON Resume Download (.json)
            with _dl_cols[2]:
                if _json_out_path.is_file():
                    try:
                        with open(_json_out_path, "r", encoding="utf-8") as _jf:
                            st.download_button(
                                label="⬇️ Download JSON",
                                data=_jf.read(),
                                file_name="final_resume.json",
                                mime="application/json",
                                key="download_json",
                            )
                    except Exception as _json_read_err:
                        st.caption(f"JSON resume unreadable: {_json_read_err}")
                else:
                    st.caption("JSON resume unavailable.")

            # Honest audit
            unsupported = optimization_result.get("unsupported_jd_requirements", [])
            if unsupported:
                with st.expander("ℹ️ Unsupported JD Requirements (honest audit)", expanded=False):
                    for _u in unsupported:
                        st.markdown(f"- {_u}")

    # ── BRANDING FOOTER ────────────────────────────────────────
    md(
        """
        <div class="footer">
            ResDev AI &nbsp;&middot;&nbsp; Evidence-Grounded Resume Automation<br>
            Built by <span style="color:#00f6ff;">Kanishk Surwade</span>
        </div>
        """
    )


# ─────────────────────────────────────────────────────────────
# LEFT: quest progress + tip ticker
# ─────────────────────────────────────────────────────────────
with col_left:

    with st.container(border=True):
        _resume_uploaded = resume_ready if "resume_ready" in dir() else False
        steps_done = sum([
            bool(job_title.strip()) if job_title else False,
            bool(job_description.strip()) if job_description else False,
            bool(uploaded_file is not None),
        ])
        hearts = "".join(
            "❤️" if i < steps_done else "🖤" for i in range(3)
        )
        md(
            f"""
            <div class="level-tag">QUEST LOG</div>
            <div class="quest-label">PROGRESS: {steps_done}/3</div>
            <div class="quest-heart">{hearts}</div>
            <div style="height:10px"></div>
            <div class="quest-label">DAILY TIP</div>
            <div class="tip-box">{TIPS[st.session_state.tip_idx]}</div>
            """
        )
        if st.button("🔄 New tip"):
            st.session_state.tip_idx = random.randrange(len(TIPS))
            st.rerun()


# ─────────────────────────────────────────────────────────────
# RIGHT: Neon Runner game (improved)
# ─────────────────────────────────────────────────────────────
with col_right:

    with st.container(border=True):
        md(
            """
            <div class="level-tag">BONUS ROUND</div>
            <div class="level-title" style="font-size:14px;">Neon Runner</div>
            <div class="level-sub">Space / click to jump. Dodge the blocks.</div>
            """
        )

        components.html(
            """
            <canvas id="game" width="230" height="280"
                style="background:#0a0a14;border:2px solid #00f6ff;image-rendering:pixelated;display:block;margin:0 auto;">
            </canvas>
            <div id="scoreEl" style="color:#ffd60a;font-family:monospace;font-size:12px;
                text-align:center;margin-top:8px;">SCORE: 0 &nbsp;|&nbsp; HI: 0</div>
            <div id="msgEl" style="color:#00f6ff;font-family:monospace;font-size:10px;
                text-align:center;height:14px;margin-top:2px;"></div>
            <script>
            (function(){
                const canvas   = document.getElementById('game');
                const ctx      = canvas.getContext('2d');
                const scoreEl  = document.getElementById('scoreEl');
                const msgEl    = document.getElementById('msgEl');
                const W = canvas.width, H = canvas.height;
                const GROUND   = H - 30;
                const GRAVITY  = 0.72;
                const JUMP_V   = -13;
                const BASE_SPD = 4;
                const SPD_INC  = 0.0015;   // speed increase per frame
                const OBS_GAP_MIN = 55;    // minimum gap between obstacles (frames)
                const OBS_GAP_MAX = 110;
                const OBS_COLORS = ['#ffd60a','#ff2e9a','#39ff6a'];

                let player, obstacles, frame, score, hiScore, gameOver, speed, nextObs, trail;

                hiScore = 0;

                function reset() {
                    player    = { x: 28, y: GROUND - 20, w: 18, h: 18, vy: 0, grounded: true };
                    obstacles = [];
                    trail     = [];
                    frame     = 0;
                    score     = 0;
                    gameOver  = false;
                    speed     = BASE_SPD;
                    nextObs   = randomGap();
                    msgEl.textContent = '';
                    requestAnimationFrame(loop);
                }

                function randomGap() {
                    return OBS_GAP_MIN + Math.floor(Math.random() * (OBS_GAP_MAX - OBS_GAP_MIN));
                }

                function randomObstacle() {
                    // Vary height between 12 and 30, width between 10 and 20
                    const h = 12 + Math.floor(Math.random() * 19);
                    const w = 10 + Math.floor(Math.random() * 11);
                    // Occasionally a double obstacle (two stacked)
                    const isDouble = Math.random() < 0.18;
                    const color = OBS_COLORS[Math.floor(Math.random() * OBS_COLORS.length)];
                    const obs = [{ x: W, y: GROUND - h, w, h, color }];
                    if (isDouble) {
                        const h2 = 10 + Math.floor(Math.random() * 10);
                        obs.push({ x: W + w + 4, y: GROUND - h2, w: w - 2, h: h2, color });
                    }
                    return obs;
                }

                function jump() {
                    if (gameOver) { reset(); return; }
                    if (player.grounded) {
                        player.vy = JUMP_V;
                        player.grounded = false;
                    }
                }

                document.addEventListener('keydown', function(e) {
                    if (e.code === 'Space') { e.preventDefault(); jump(); }
                });
                canvas.addEventListener('click', jump);

                function drawGrid() {
                    ctx.strokeStyle = 'rgba(0,246,255,0.05)';
                    ctx.lineWidth = 1;
                    for (let gx = (frame * speed * 0.3) % 18; gx < W; gx += 18) {
                        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
                    }
                }

                function drawGround() {
                    // Scrolling ground line
                    const goff = (frame * speed * 0.5) % 18;
                    ctx.strokeStyle = '#ff2e9a';
                    ctx.lineWidth = 2;
                    ctx.beginPath(); ctx.moveTo(0, GROUND); ctx.lineTo(W, GROUND); ctx.stroke();
                    ctx.strokeStyle = 'rgba(255,46,154,0.25)';
                    ctx.lineWidth = 1;
                    for (let tx = -goff; tx < W; tx += 18) {
                        ctx.beginPath(); ctx.moveTo(tx, GROUND); ctx.lineTo(tx + 9, GROUND + 6); ctx.stroke();
                    }
                }

                function drawPlayer() {
                    // Ghost trail
                    trail.push({ x: player.x, y: player.y, alpha: 0.35 });
                    if (trail.length > 6) trail.shift();
                    trail.forEach(function(t, i) {
                        ctx.globalAlpha = t.alpha * (i / trail.length);
                        ctx.fillStyle = '#00f6ff';
                        ctx.fillRect(t.x, t.y, player.w, player.h);
                    });
                    ctx.globalAlpha = 1;

                    ctx.fillStyle = '#00f6ff';
                    ctx.shadowColor = '#00f6ff';
                    ctx.shadowBlur = 12;
                    ctx.fillRect(player.x, player.y, player.w, player.h);
                    ctx.shadowBlur = 0;
                }

                function drawObstacles() {
                    obstacles.forEach(function(o) {
                        ctx.fillStyle = o.color;
                        ctx.shadowColor = o.color;
                        ctx.shadowBlur = 8;
                        ctx.fillRect(o.x, o.y, o.w, o.h);
                        ctx.shadowBlur = 0;
                    });
                }

                function collides(a, b) {
                    const pad = 3; // small forgiveness
                    return (
                        a.x + pad < b.x + b.w &&
                        a.x + a.w - pad > b.x &&
                        a.y + pad < b.y + b.h &&
                        a.y + a.h - pad > b.y
                    );
                }

                function getDifficultyLabel() {
                    if (speed < 5.5) return 'EASY';
                    if (speed < 7)   return 'MED';
                    if (speed < 9)   return 'HARD';
                    return 'INSANE';
                }

                function loop() {
                    if (gameOver) return;

                    frame++;
                    speed = BASE_SPD + frame * SPD_INC;
                    score = Math.floor(frame / 6);

                    // clear
                    ctx.fillStyle = '#0a0a14';
                    ctx.fillRect(0, 0, W, H);

                    drawGrid();
                    drawGround();

                    // spawn obstacles
                    if (frame >= nextObs) {
                        randomObstacle().forEach(function(o) { obstacles.push(o); });
                        nextObs = frame + randomGap();
                    }

                    // move obstacles
                    obstacles.forEach(function(o) { o.x -= speed; });
                    obstacles = obstacles.filter(function(o) { return o.x + o.w > 0; });

                    // physics
                    player.vy += GRAVITY;
                    player.y  += player.vy;
                    if (player.y >= GROUND - player.h) {
                        player.y = GROUND - player.h;
                        player.vy = 0;
                        player.grounded = true;
                    }

                    drawObstacles();
                    drawPlayer();

                    // HUD
                    ctx.fillStyle = '#ffd60a';
                    ctx.font = '9px monospace';
                    ctx.textAlign = 'left';
                    ctx.fillText(getDifficultyLabel(), 4, 12);
                    ctx.textAlign = 'right';
                    ctx.fillText('SPD x' + speed.toFixed(1), W - 4, 12);

                    // collision
                    for (let i = 0; i < obstacles.length; i++) {
                        if (collides(player, obstacles[i])) {
                            gameOver = true;
                            if (score > hiScore) hiScore = score;
                            break;
                        }
                    }

                    scoreEl.textContent = 'SCORE: ' + score + '  |  HI: ' + hiScore;

                    if (gameOver) {
                        ctx.fillStyle = 'rgba(0,0,0,0.65)';
                        ctx.fillRect(0, 0, W, H);
                        ctx.fillStyle = '#ff2e9a';
                        ctx.font = 'bold 15px monospace';
                        ctx.textAlign = 'center';
                        ctx.shadowColor = '#ff2e9a';
                        ctx.shadowBlur = 12;
                        ctx.fillText('GAME OVER', W/2, H/2 - 12);
                        ctx.shadowBlur = 0;
                        ctx.font = '11px monospace';
                        ctx.fillStyle = '#ffd60a';
                        ctx.fillText('SCORE: ' + score, W/2, H/2 + 6);
                        ctx.font = '9px monospace';
                        ctx.fillStyle = '#00f6ff';
                        ctx.fillText('click or SPACE to retry', W/2, H/2 + 22);
                        msgEl.textContent = score > 80 ? '★ GREAT RUN! ★' : (score > 40 ? 'Nice try!' : 'Keep going!');
                        return;
                    }

                    requestAnimationFrame(loop);
                }

                reset();
            })();
            </script>
            """,
            height=345,
        )