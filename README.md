---
title: ClassInsight AI
emoji: 🎓
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
short_description: AI-powered class quality assessment for online education
---

# 🎓 ClassInsight AI

AI-powered class quality assessment for online education recordings.
Built for the **Humaniti AI Recruitment Challenge**.

## What It Does

Accepts an online class video recording and classifies its quality as:
**Poor · Average · Good · Excellent**

Provides specific, evidence-based observations — never generic statements.

## How It Works

| Stage | What Happens | Model |
|---|---|---|
| 🎬 Video Processing | Extract audio + sample frames every 30s | FFmpeg (local) |
| 🎙️ Transcription | Full speech-to-text | `openai/whisper-large-v3` via OpenRouter |
| 🔍 Transcript Analysis | Pacing, interaction, vocabulary | `meta-llama/llama-4-maverick:free` |
| 👁️ Visual Analysis | Teacher presence, slides, students | `google/gemini-2.0-flash-exp:free` |
| 🧠 Synthesis | Final rating + observations | `meta-llama/llama-4-maverick:free` |

All AI via **OpenRouter free tier** — no paid API needed.

## Setup

### Prerequisites

- Python 3.10+
- FFmpeg installed and in PATH
  - Windows: `winget install ffmpeg` or download from https://ffmpeg.org
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### Install

```bash
git clone <your-repo-url>
cd classinsight-ai
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
# Free key: https://openrouter.ai/keys
```

### Run

```bash
python app.py
```

Visit `http://localhost:7860` — a public shareable link will also appear in the terminal.

## Project Structure

```
classinsight-ai/
├── app.py                    # Gradio UI + analysis pipeline
├── utils/
│   ├── video_processor.py    # FFmpeg: extract audio + frames
│   ├── analyzer.py           # OpenRouter: transcribe, analyse, synthesise
│   └── report_generator.py   # Markdown report builder
├── requirements.txt
├── .env.example
└── README.md
```

## Key Quality Signals Evaluated

- **Teacher–student interaction ratio** — monologue vs. dialogue
- **Question frequency & wait time** — Socratic teaching markers
- **Speech pacing (WPM)** — appropriate for age group
- **Content structure** — intro → body → summary
- **Filler-word density** — "um", "uh", etc. per minute
- **Teacher on-screen visibility** — engagement presence
- **Slide / screen-share quality** — content clarity
- **Student engagement indicators** — cameras, responses

## Privacy Note

- Video files are processed entirely on your local machine
- Only the extracted audio is sent to OpenRouter (Whisper endpoint)
- Sampled JPEG frames are sent to OpenRouter (Vision endpoint)
- No raw video is ever uploaded to any server
- Do not deploy publicly with identifiable recordings of children

## Limitations

- Analysis quality depends on audio clarity; noisy recordings reduce accuracy
- Visual analysis from 30-second frame samples may miss brief interactions
- Free OpenRouter models may have rate limits for long recordings
- Whisper transcription may struggle with heavy accents or multiple speakers simultaneously

## Built With

- [Gradio](https://gradio.app) — UI framework
- [OpenRouter](https://openrouter.ai) — AI model gateway (free tier)
- FFmpeg — video processing
