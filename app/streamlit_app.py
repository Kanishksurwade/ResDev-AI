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

_MASTER_RESUME_PATH = _REPO_ROOT / "data" / "master_resume.json"
_GENERATED_DIR = _REPO_ROOT / "generated"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResDev AI",
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
            Paste a job description + your master resume. ResDev AI reads
            both and produces a tailored, ATS-friendly resume — grounded
            in what's actually on your resume. No fluff, no fake stats.
        </div>
    </div>
    """
)


# ============================================================
# TIP QUOTES (fills the side space, purely cosmetic / no backend)
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
# LAYOUT -- 3 COLUMNS SO THE SIDE SPACE ISN'T EMPTY
# ============================================================

col_left, col_center, col_right = st.columns([1, 2.3, 1], gap="medium")


# ---------- CENTER: the actual form ----------
with col_center:

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

    with st.container(border=True):
        md(
            """
            <div class="level-tag">LEVEL 02</div>
            <div class="level-title">Master Resume</div>
            <div class="level-sub">Your saved master resume will be used automatically.</div>
            """
        )

        # Show whether the canonical master resume is present — no uploader needed yet.
        if _MASTER_RESUME_PATH.exists():
            st.info(
                "✅ **Master Resume loaded:** `data/master_resume.json`  \n"
                "All resume generation uses this verified source of truth."
            )
        else:
            st.error(
                f"❌ Master resume not found at `{_MASTER_RESUME_PATH}`.  \n"
                "Please ensure `data/master_resume.json` exists in the project root."
            )

        resume_template = st.selectbox(
            "Resume template",
            ["Professional", "Modern", "Minimal", "Executive"],
            key="template_select",
        )

        if resume_template != "Professional":
            st.caption(
                f"'{resume_template}' is a placeholder for now -- only "
                f"'Professional' is wired up to the generation pipeline."
            )

    # -------------------------------------------------------
    # GENERATE BUTTON — wired to the optimization pipeline
    # -------------------------------------------------------
    if st.button("▶ Generate Resume", key="generate_btn"):

        if not job_title.strip():
            st.error("Please enter a Job Title.")

        elif not job_description.strip():
            st.error("Please enter the Job Description.")

        elif not _MASTER_RESUME_PATH.exists():
            st.error(
                "Cannot generate: `data/master_resume.json` is missing.  \n"
                f"Expected at: `{_MASTER_RESUME_PATH}`"
            )

        else:
            # --------------------------------------------------
            # Load master resume
            # --------------------------------------------------
            try:
                with open(_MASTER_RESUME_PATH, "r", encoding="utf-8") as _f:
                    master_resume_data = json.load(_f)
            except Exception as _e:
                st.error(f"Failed to load master resume JSON: {_e}")
                st.exception(_e)
                st.stop()

            # --------------------------------------------------
            # Run the optimization pipeline
            # --------------------------------------------------
            st.markdown("---")
            st.markdown("### ⚙️ Optimization Pipeline")

            _progress_bar = st.progress(0, text="Initialising pipeline…")
            _status_area = st.empty()
            _iter_log = st.expander("📋 Iteration Log", expanded=False)

            def _on_progress(iteration: int, max_it: int, data: dict) -> None:
                pct = int((iteration / max_it) * 80)  # reserve 20% for rendering
                ats_tag = "✅ PASS" if data["ats_passed"] else "❌ FAIL"
                qwen_tag = "✅ PASS" if data["qwen_passed"] else "⚠️ NEEDS REVISION"
                ev_tag = "🔒 VERIFIED" if data["evidence_passed"] else "🚨 VIOLATION"
                _progress_bar.progress(
                    pct,
                    text=f"Iteration {iteration}/{max_it} — ATS {data['ats_score']}/100 {ats_tag}",
                )
                _status_area.info(
                    f"**Iteration {iteration}/{max_it}**  \n"
                    f"Evidence: {ev_tag} · "
                    f"ATS: {data['ats_score']}/100 {ats_tag} · "
                    f"Semantic: {data['qwen_score']}/100 {qwen_tag} · "
                    f"Guard: `{data['decision_status']}`"
                )
                with _iter_log:
                    st.markdown(
                        f"**Iter {iteration}** — ATS `{data['ats_score']}` | "
                        f"Qwen `{data['qwen_score']}` | "
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
                    target_score=85,
                    max_iterations=5,
                    progress_callback=_on_progress,
                )
            except Exception as _e:
                _progress_bar.empty()
                _status_area.empty()
                st.error("**Optimization pipeline failed.**")
                st.exception(_e)
                st.stop()

            # --------------------------------------------------
            # Save final_resume.json
            # --------------------------------------------------
            _progress_bar.progress(85, text="Saving final_resume.json…")
            _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            _json_out_path = _GENERATED_DIR / "final_resume.json"
            final_structured_resume = optimization_result["final_structured_resume"]

            try:
                save_final_resume(final_structured_resume, _json_out_path)
            except Exception as _e:
                _progress_bar.empty()
                st.error(f"Failed to save final_resume.json: {_e}")
                st.exception(_e)
                st.stop()

            # --------------------------------------------------
            # Render LaTeX + compile PDF
            # --------------------------------------------------
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
                st.error(f"LaTeX rendering failed: {_e}")
                st.exception(_e)
                st.stop()

            _progress_bar.progress(100, text="Done!")
            _status_area.empty()

            # --------------------------------------------------
            # Results summary
            # --------------------------------------------------
            opt_status = optimization_result.get("status", "UNKNOWN")
            best_ats = optimization_result.get("best_ats_score", 0)
            best_qwen = optimization_result.get("best_qwen_score", 0)
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
                    f"ATS Score: **{best_ats}/100** · Semantic: **{best_qwen}/100** · "
                    f"Combined: **{best_combined}/100**  \n"
                    f"Multi-ATS platforms: **{multi_passed}/{multi_total}** passed"
                )
            elif render_status == "TEX_ONLY":
                st.warning(
                    f"**Resume optimised — LaTeX created, but PDF not compiled.**  \n"
                    f"{render_msg}  \n"
                    f"ATS: {best_ats}/100 · Semantic: {best_qwen}/100 · "
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

            # --------------------------------------------------
            # Download buttons
            # --------------------------------------------------
            st.markdown("#### 📥 Downloads")
            _dl_cols = st.columns(2)

            with _dl_cols[0]:
                _pdf_path = Path(render_result.get("pdf_path") or "")
                if _pdf_path.exists():
                    with open(_pdf_path, "rb") as _pdf_file:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=_pdf_file.read(),
                            file_name="final_resume.pdf",
                            mime="application/pdf",
                            key="download_pdf",
                        )
                else:
                    st.caption("PDF not available (pdflatex not found or compilation failed).")

            with _dl_cols[1]:
                if _json_out_path.exists():
                    with open(_json_out_path, "r", encoding="utf-8") as _jf:
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=_jf.read(),
                            file_name="final_resume.json",
                            mime="application/json",
                            key="download_json",
                        )

            # Honest audit of JD requirements that couldn't be met
            unsupported = optimization_result.get("unsupported_jd_requirements", [])
            if unsupported:
                with st.expander("ℹ️ Unsupported JD Requirements (honest audit)", expanded=False):
                    for _u in unsupported:
                        st.markdown(f"- {_u}")

    md('<div class="footer">ResDev AI &middot; Evidence-grounded resume automation</div>')


# ---------- LEFT: quest progress + tip ticker ----------
with col_left:

    with st.container(border=True):
        steps_done = sum([
            bool(job_title.strip()) if job_title else False,
            bool(job_description.strip()) if job_description else False,
            _MASTER_RESUME_PATH.exists(),  # master resume is always "ready" if file exists
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


# ---------- RIGHT: playable mini-game ----------
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
            <div id="score" style="color:#ffd60a;font-family:monospace;font-size:12px;
                text-align:center;margin-top:8px;">SCORE: 0</div>
            <script>
            const canvas = document.getElementById('game');
            const ctx = canvas.getContext('2d');
            const scoreEl = document.getElementById('score');
            const groundY = 250;
            let player, obstacles, frame, score, gameOver;

            function reset(){
                player = {x:24, y:groundY-18, w:18, h:18, vy:0};
                obstacles = [];
                frame = 0;
                score = 0;
                gameOver = false;
                requestAnimationFrame(loop);
            }

            function jump(){
                if(gameOver){ reset(); return; }
                if(player.y >= groundY-18-1){ player.vy = -12.5; }
            }

            document.addEventListener('keydown', function(e){
                if(e.code === 'Space'){ e.preventDefault(); jump(); }
            });
            canvas.addEventListener('click', jump);

            function loop(){
                if(gameOver) return;
                frame++;
                ctx.fillStyle = '#0a0a14';
                ctx.fillRect(0,0,canvas.width,canvas.height);

                ctx.strokeStyle = 'rgba(0,246,255,0.08)';
                for(let gx=0; gx<canvas.width; gx+=18){
                    ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,canvas.height); ctx.stroke();
                }

                ctx.fillStyle = '#ff2e9a';
                ctx.fillRect(0, groundY, canvas.width, 4);

                player.vy += 0.75;
                player.y += player.vy;
                if(player.y > groundY-18){ player.y = groundY-18; player.vy = 0; }

                ctx.fillStyle = '#00f6ff';
                ctx.shadowColor = '#00f6ff';
                ctx.shadowBlur = 10;
                ctx.fillRect(player.x, player.y, player.w, player.h);
                ctx.shadowBlur = 0;

                if(frame % 65 === 0){
                    obstacles.push({x:canvas.width, y:groundY-18, w:14, h:18});
                }
                ctx.fillStyle = '#ffd60a';
                obstacles.forEach(function(o){ o.x -= 4; ctx.fillRect(o.x, o.y, o.w, o.h); });
                obstacles = obstacles.filter(function(o){ return o.x + o.w > 0; });

                obstacles.forEach(function(o){
                    if(player.x < o.x+o.w && player.x+player.w > o.x &&
                       player.y < o.y+o.h && player.y+player.h > o.y){
                        gameOver = true;
                    }
                });

                if(!gameOver){ score++; }
                scoreEl.textContent = 'SCORE: ' + score + (gameOver ? '  ·  GAME OVER' : '');

                if(gameOver){
                    ctx.fillStyle = 'rgba(0,0,0,0.6)';
                    ctx.fillRect(0,0,canvas.width,canvas.height);
                    ctx.fillStyle = '#ff2e9a';
                    ctx.font = '14px monospace';
                    ctx.textAlign = 'center';
                    ctx.fillText('GAME OVER', canvas.width/2, canvas.height/2);
                    ctx.font = '10px monospace';
                    ctx.fillStyle = '#00f6ff';
                    ctx.fillText('click to retry', canvas.width/2, canvas.height/2+16);
                    return;
                }
                requestAnimationFrame(loop);
            }
            reset();
            </script>
            """,
            height=330,
        )