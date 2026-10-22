"""
ClassInsight AI — Class Quality Assessment Tool
Humaniti AI Recruitment Challenge
"""
import os, gradio as gr, traceback
from utils.video_processor import VideoProcessor
from utils.analyzer import ClassAnalyzer
from utils.report_generator import ReportGenerator


import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def _rating_emoji(rating: str) -> str:
    return {
        "Excellent": "🟢  Excellent",
        "Good":      "🔵  Good",
        "Average":   "🟡  Average",
        "Poor":      "🔴  Poor",
    }.get(rating, rating)


def analyze_class(video_file, progress=gr.Progress(track_tqdm=True)):
    if video_file is None:
        return (gr.update(visible=False),) * 3 + ("⚠️ Please upload a recording.", gr.update(visible=False))

    # Resolve file path from string or Gradio File object
    if hasattr(video_file, "name"):
        video_path = video_file.name
    elif isinstance(video_file, dict) and "path" in video_file:
        video_path = video_file["path"]
    else:
        video_path = str(video_file)

    if not os.path.exists(video_path):
        return (gr.update(visible=False),) * 3 + (f"⚠️ Uploaded file not found: `{video_path}`", gr.update(visible=False))

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        msg = (
            "❌ **OPENROUTER_API_KEY not set.**\n\n"
            "Create a `.env` file in the project root:\n```\nOPENROUTER_API_KEY=sk-or-...\n```"
        )
        return (gr.update(visible=False),) * 3 + (msg, gr.update(visible=False))

    import time
    pipeline_start = time.time()
    print(f"[PIPELINE] Starting analysis of: {video_path}", flush=True)
    try:
        processor = VideoProcessor()
        analyzer  = ClassAnalyzer(api_key=api_key)
        reporter  = ReportGenerator()

        progress(0.10, desc="🎬 Extracting audio and frames…")
        t0 = time.time()
        extraction = processor.extract(video_path)
        print(f"[PIPELINE] Step 1 DONE — Video extracted in {time.time()-t0:.1f}s | duration={extraction['duration_seconds']:.0f}s frames={len(extraction['frame_paths'])}", flush=True)

        progress(0.30, desc="🎙️ Transcribing audio…")
        t0 = time.time()
        transcript_data = analyzer.transcribe(extraction["audio_path"])
        print(f"[PIPELINE] Step 2 DONE — Transcription in {time.time()-t0:.1f}s | words={transcript_data['word_count']} wpm={transcript_data['words_per_minute']}", flush=True)

        progress(0.55, desc="🔍 Analysing transcript signals…")
        t0 = time.time()
        transcript_analysis = analyzer.analyze_transcript(transcript_data)
        print(f"[PIPELINE] Step 3 DONE — Transcript analysis in {time.time()-t0:.1f}s", flush=True)

        progress(0.75, desc="👁️ Analysing visual signals…")
        t0 = time.time()
        visual_analysis = analyzer.analyze_frames(extraction["frame_paths"])
        print(f"[PIPELINE] Step 4 DONE — Visual analysis in {time.time()-t0:.1f}s", flush=True)

        progress(0.92, desc="🧠 Synthesising final rating…")
        t0 = time.time()
        final_result = analyzer.synthesize(
            transcript_data, transcript_analysis, visual_analysis,
            extraction["duration_seconds"]
        )
        print(f"[PIPELINE] Step 5 DONE — Synthesis in {time.time()-t0:.1f}s | Rating={final_result.get('rating')} Score={final_result.get('score')}", flush=True)

        progress(1.00, desc="✅ Building report…")
        report_md = reporter.build(final_result, extraction)
        processor.cleanup(extraction)
        print(f"[PIPELINE] COMPLETE — Total time: {time.time()-pipeline_start:.1f}s", flush=True)

        return (
            gr.update(value=_rating_emoji(final_result["rating"]), visible=True),
            gr.update(value=f"{final_result['score']} / 100", visible=True),
            gr.update(value=final_result["summary"], visible=True),
            report_md,
            gr.update(visible=True),
        )

    except Exception as exc:
        print(f"[PIPELINE] ERROR after {time.time()-pipeline_start:.1f}s: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        err = f"❌ **Analysis failed:** {exc}\n\n```\n{traceback.format_exc()}\n```"
        return (gr.update(visible=False),) * 3 + (err, gr.update(visible=False))


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
body, .gradio-container {
    background: linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%) !important;
    min-height: 100vh;
}
.gr-button-primary {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 1.05rem !important;
    height: 54px !important;
    box-shadow: 0 4px 24px rgba(124,58,237,.35) !important;
    transition: all .3s ease !important;
}
.gr-button-primary:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(124,58,237,.55) !important; }
.gr-box, .gr-form { background: rgba(255,255,255,.05) !important; border: 1px solid rgba(255,255,255,.1) !important; border-radius: 16px !important; backdrop-filter: blur(12px) !important; }
label span { color:#94a3b8 !important; font-size:.82rem !important; text-transform:uppercase; letter-spacing:.5px; }
textarea, input[type=text] { background: rgba(255,255,255,.06) !important; color:#e2e8f0 !important; border: 1px solid rgba(255,255,255,.12) !important; border-radius: 10px !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-thumb { background:rgba(167,139,250,.4); border-radius:3px; }
"""


def build_ui():
    with gr.Blocks(title="ClassInsight AI - Class Quality Assessor") as demo:

        gr.HTML("""
        <div style="text-align:center;padding:36px 0 12px;border-radius:16px;margin-bottom:8px;">
          <h1 style="background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     font-size:2.8rem;font-weight:800;margin:0;font-family:Inter,sans-serif;">
            🎓 ClassInsight AI
          </h1>
          <p style="color:#94a3b8;font-size:1rem;margin:10px 0 0;font-family:Inter,sans-serif;">
            AI-powered class quality assessment &nbsp;·&nbsp; Poor &nbsp;·&nbsp; Average &nbsp;·&nbsp; Good &nbsp;·&nbsp; Excellent
          </p>
        </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown("### 📤 Upload Class Recording")
                video_input = gr.File(
                    label="Drop or click to upload class recording (MP4 · MOV · AVI · WebM · MKV)",
                    file_types=["video", ".mp4", ".mov", ".avi", ".webm", ".mkv"],
                    file_count="single",
                    height=200,
                )
                gr.Markdown(
                    "<small style='color:#475569;'>🔒 Processed locally — audio sent only to OpenRouter Whisper endpoint</small>"
                )
                analyse_btn = gr.Button("🔍  Analyse Class Quality", variant="primary", size="lg")

            with gr.Column(scale=4):
                gr.Markdown("### 📊 Assessment Summary")
                rating_output  = gr.Textbox(label="Overall Rating",        interactive=False, visible=False)
                score_output   = gr.Textbox(label="Quality Score (0–100)", interactive=False, visible=False)
                summary_output = gr.Textbox(label="Executive Summary",     lines=7, interactive=False, visible=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Detailed Analysis Report")
                report_output = gr.Markdown(
                    value=(
                        "*Upload a class recording and click **Analyse Class Quality** to see the full assessment.*\n\n"
                        "> Examines transcripts, pacing, teacher-student interaction, visual engagement, "
                        "content structure, and pedagogy to produce a specific, evidence-based rating."
                    )
                )

        with gr.Row():
            download_btn = gr.DownloadButton("⬇️  Download Report (Markdown)", visible=False)

        with gr.Accordion("ℹ️  How ClassInsight AI Works", open=False):
            gr.Markdown("""
**Five-stage pipeline — all free models via OpenRouter:**

| Stage | What Happens | Model |
|---|---|---|
| 🎬 Video Processing | Extract audio + sample frames every 30s | FFmpeg (local) |
| 🎙️ Transcription | Full speech-to-text with timestamps | `openai/whisper-large-v3` |
| 🔍 Transcript Analysis | Pacing, questions, interaction, vocabulary | `meta-llama/llama-4-maverick:free` |
| 👁️ Visual Analysis | Teacher presence, slides, student screens | `google/gemini-2.0-flash-exp:free` |
| 🧠 Synthesis | Combine all signals → final rating | `meta-llama/llama-4-maverick:free` |

**Key signals evaluated:**
- Teacher–student interaction ratio  ·  Question frequency  ·  Speech pacing (WPM)
- Content structure (intro → body → summary)  ·  Filler-word density
- Teacher on-screen visibility  ·  Slide / screen-share quality  ·  Student engagement
            """)

        analyse_btn.click(
            fn=analyze_class,
            inputs=[video_input],
            outputs=[rating_output, score_output, summary_output, report_output, download_btn],
            show_progress=True,
        )

    return demo


def get_available_port(start_port: int = 7860) -> int:
    import socket
    for port in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start_port


if __name__ == "__main__":
    # Refresh PATH on Windows so freshly installed CLI tools (e.g. ffmpeg from winget) are visible immediately
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"System\CurrentControlSet\Control\Session Manager\Environment") as k:
                sys_path = winreg.QueryValueEx(k, "Path")[0]
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
                usr_path = winreg.QueryValueEx(k, "Path")[0]
            os.environ["PATH"] = sys_path + os.pathsep + usr_path + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    port = int(os.environ.get("PORT", get_available_port(7860)))
    share = os.environ.get("SHARE", "false").lower() in ("true", "1")
    print(f"Launching ClassInsight AI on port {port}...")
    build_ui().launch(server_name="0.0.0.0", server_port=port, share=share, show_error=True, css=CSS)
