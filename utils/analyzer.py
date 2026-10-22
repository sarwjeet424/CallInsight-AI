"""
ClassAnalyzer — Core AI analysis engine.
All LLM requests go through OpenRouter (free-tier models).
Transcription runs locally via faster-whisper (no API cost).
"""
import os, base64, json, math, re, time, subprocess, tempfile
from typing import Optional
import requests

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Free models verified active on OpenRouter (Oct 2026)
MODEL_TEXT   = "nvidia/nemotron-3.5-lightning:free"
MODEL_VISION = "nvidia/nemotron-3.5-super-vl:free"

FALLBACK_TEXT_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

FALLBACK_VISION_MODELS = [
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
    "dots-studio/dots-3-note-preview:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
]


def _log(stage: str, msg: str):
    """Unified timestamped logger for the analysis pipeline."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{stage}] {msg}", flush=True)


class ClassAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://classinsight.ai",
            "X-Title":       "ClassInsight AI",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 1. TRANSCRIPTION
    # ──────────────────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribes audio using local faster-whisper.
        Uses a subprocess-based audio reader to avoid the av/Python3.12
        'metadata_errors' incompatibility.
        Falls back to duration-only metrics if anything fails.
        """
        _log("TRANSCRIBE", f"Starting transcription of: {audio_path}")
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        _log("TRANSCRIBE", f"Audio file size: {file_size_mb:.1f} MB")

        # ── 1. Local faster-whisper ───────────────────────────────────────────
        try:
            import numpy as np
            from faster_whisper import WhisperModel

            _log("TRANSCRIBE", "Loading WhisperModel (base.en, int8, CPU)...")
            model = WhisperModel("base.en", device="cpu", compute_type="int8")
            _log("TRANSCRIBE", "Model loaded. Reading audio via ffmpeg subprocess...")

            # Use ffmpeg to decode audio to raw PCM float32 — avoids all av/av2 issues
            audio_np = self._audio_to_numpy(audio_path)

            if audio_np is not None and len(audio_np) > 0:
                _log("TRANSCRIBE", f"Audio decoded: {len(audio_np)} samples ({len(audio_np)/16000:.1f}s). Starting transcription...")

                # faster-whisper accepts numpy arrays directly
                segments, info = model.transcribe(
                    audio_np,
                    beam_size=1,
                    language="en",
                    vad_filter=True,       # skip silent segments — much faster
                    vad_parameters={"min_silence_duration_ms": 500},
                )

                _log("TRANSCRIBE", "Collecting transcript segments...")
                text_parts = []
                for seg in segments:
                    text_parts.append(seg.text.strip())

                text = " ".join(text_parts).strip()
                _log("TRANSCRIBE", f"Transcription complete. Words: {len(text.split())}, Duration: {getattr(info, 'duration', 0):.0f}s")

                if text:
                    words    = text.split()
                    duration = getattr(info, "duration", 0) or 0
                    wpm      = int(len(words) / (duration / 60)) if duration > 0 and len(words) > 0 else 130
                    return {
                        "text":             text,
                        "word_count":       len(words),
                        "words_per_minute": wpm,
                        "duration_seconds": duration,
                    }
                else:
                    _log("TRANSCRIBE", "WARNING: Transcription returned empty text (possibly silent or music-only audio)")
            else:
                _log("TRANSCRIBE", "WARNING: Audio decoding returned empty buffer")

        except ImportError as e:
            _log("TRANSCRIBE", f"faster-whisper not available: {e}")
        except Exception as e:
            import traceback
            _log("TRANSCRIBE", f"ERROR in local Whisper: {e}")
            _log("TRANSCRIBE", traceback.format_exc())

        # ── 2. Fallback: duration from ffprobe ────────────────────────────────
        _log("TRANSCRIBE", "Falling back to duration-only metrics")
        try:
            from utils.video_processor import VideoProcessor
            duration = VideoProcessor()._get_duration(audio_path)
            _log("TRANSCRIBE", f"Fallback duration: {duration:.0f}s")
        except Exception as e:
            _log("TRANSCRIBE", f"Duration fallback also failed: {e}")
            duration = 0.0

        return {
            "text":             "[Audio successfully extracted — transcript not available. Analysis based on visual signals only.]",
            "word_count":       max(10, int(duration * 2)),
            "words_per_minute": 130,
            "duration_seconds": duration,
        }

    def _audio_to_numpy(self, audio_path: str):
        """
        Decode audio file to a 16kHz mono float32 numpy array using
        ffmpeg subprocess — no av/soundfile dependency, works on all platforms.
        """
        import numpy as np
        try:
            from utils.video_processor import VideoProcessor
            ffmpeg_bin = VideoProcessor._find_bin("ffmpeg")
        except Exception:
            ffmpeg_bin = "ffmpeg"

        cmd = [
            ffmpeg_bin, "-y",
            "-i", audio_path,
            "-ac", "1",           # mono
            "-ar", "16000",       # 16kHz sample rate (Whisper requirement)
            "-f", "f32le",        # raw float32 little-endian PCM
            "-",                  # pipe to stdout
        ]
        _log("AUDIO_DECODE", f"Running: {' '.join(cmd[:5])} ... [truncated]")
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        elapsed = time.time() - t0
        _log("AUDIO_DECODE", f"ffmpeg decode finished in {elapsed:.1f}s, output size: {len(result.stdout)} bytes")

        if result.returncode != 0:
            _log("AUDIO_DECODE", f"ffmpeg decode failed: {result.stderr[-500:]}")
            return None

        audio = np.frombuffer(result.stdout, dtype=np.float32)
        return audio

    # ──────────────────────────────────────────────────────────────────────────
    # 2. TRANSCRIPT ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────

    TRANSCRIPT_SYSTEM = (
        "You are an expert educational quality analyst specialising in online K-12 classes.\n"
        "Your job is to analyse a class transcript and extract objective quality signals.\n"
        "Return ONLY valid JSON with no markdown fences or extra text."
    )

    # NOTE: We do NOT use .format() on TRANSCRIPT_PROMPT to avoid Python mistaking
    # JSON field names like {"pacing_wpm": ...} as format variables.
    # Instead, we append the transcript text manually in analyze_transcript().
    TRANSCRIPT_PROMPT_HEADER = (
        'Analyse this class transcript and return a JSON object with EXACTLY these fields:\n\n'
        '{\n'
        '  "pacing_wpm": <integer, estimated words-per-minute if not provided>,\n'
        '  "pacing_assessment": "<Too Fast | Appropriate | Too Slow>",\n'
        '  "question_count": <integer, number of questions asked by teacher>,\n'
        '  "student_response_indicators": <integer, count of phrases suggesting student responses>,\n'
        '  "filler_word_count": <integer, total \'um\',\'uh\',\'like\',\'you know\',\'basically\',\'literally\'>,\n'
        '  "filler_density_per_min": <float, filler words per minute>,\n'
        '  "has_introduction": <true|false, does class start with a clear intro/agenda?>,\n'
        '  "has_summary_or_closure": <true|false, does class end with summary or clear ending?>,\n'
        '  "teacher_enthusiasm_score": <1-10, energy and positivity level>,\n'
        '  "content_clarity_score": <1-10, how clearly concepts are explained>,\n'
        '  "interaction_type": "<Monologue | Mostly Teacher | Balanced | Student-Led>",\n'
        '  "notable_strengths": [<list of 2-4 specific strengths>],\n'
        '  "notable_weaknesses": [<list of 2-4 specific weaknesses>],\n'
        '  "key_topics_covered": [<list of main topics>],\n'
        '  "age_appropriateness": "<Too Complex | Appropriate | Too Simple>",\n'
        '  "summary": "<2-3 sentence objective summary of what was taught and how>"\n'
        '}\n\n'
        'Transcript (metadata and content follow):\n'
    )

    def analyze_transcript(self, transcript_data: dict) -> dict:
        transcript_text = transcript_data.get("text", "")
        wpm = transcript_data.get("words_per_minute", 0)
        wc  = transcript_data.get("word_count", 0)

        _log("TRANSCRIPT_ANALYSIS", f"Starting. Word count: {wc}, WPM: {wpm}")

        is_fallback = transcript_text.startswith("[")
        if not transcript_text:
            _log("TRANSCRIPT_ANALYSIS", "No transcript text, returning empty analysis")
            return self._empty_transcript_analysis()

        # Build prompt without .format() — concatenate manually
        meta_header = f"[word_count={wc}, wpm={wpm}]\n\n"
        prompt = self.TRANSCRIPT_PROMPT_HEADER + meta_header + transcript_text[:8000]

        if is_fallback:
            _log("TRANSCRIPT_ANALYSIS", "Transcript is a fallback placeholder — returning empty analysis")
            return self._empty_transcript_analysis()

        _log("TRANSCRIPT_ANALYSIS", f"Sending {len(prompt)} chars to LLM...")
        t0 = time.time()
        try:
            raw = self._chat(MODEL_TEXT, self.TRANSCRIPT_SYSTEM, prompt)
            _log("TRANSCRIPT_ANALYSIS", f"LLM response received in {time.time()-t0:.1f}s")
            result = self._parse_json(raw, self._empty_transcript_analysis())
            _log("TRANSCRIPT_ANALYSIS", "JSON parsed successfully")
            return result
        except Exception as e:
            _log("TRANSCRIPT_ANALYSIS", f"ERROR: {e}")
            return self._empty_transcript_analysis()

    def _empty_transcript_analysis(self) -> dict:
        return {
            "pacing_wpm": 0, "pacing_assessment": "Unknown",
            "question_count": 0, "student_response_indicators": 0,
            "filler_word_count": 0, "filler_density_per_min": 0,
            "has_introduction": False, "has_summary_or_closure": False,
            "teacher_enthusiasm_score": 5, "content_clarity_score": 5,
            "interaction_type": "Unknown",
            "notable_strengths": [], "notable_weaknesses": [],
            "key_topics_covered": [], "age_appropriateness": "Unknown",
            "summary": "Transcript not available.",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 3. VISUAL ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────

    VISUAL_SYSTEM = (
        "You are an expert at assessing the visual quality of online class recordings.\n"
        "You will receive one or more sampled frames from a class.\n"
        "Return ONLY valid JSON with no markdown fences or extra text."
    )

    # NOTE: No .format() used — JSON braces don't conflict with template substitution
    VISUAL_PROMPT = (
        'Analyse these sampled frames from an online class and return a JSON object with EXACTLY these fields:\n\n'
        '{\n'
        '  "teacher_visible": <true|false, is the teacher face/body visible in most frames?>,\n'
        '  "teacher_presence_score": <1-10, quality of teacher on-screen presence>,\n'
        '  "screen_share_detected": <true|false>,\n'
        '  "screen_share_quality": "<None | Low | Medium | High>",\n'
        '  "student_cameras_visible": <true|false, are student video feeds visible?>,\n'
        '  "estimated_student_count": <integer or null>,\n'
        '  "environment_professionalism": <1-10, background, lighting, framing>,\n'
        '  "visual_engagement_score": <1-10, overall visual engagement>,\n'
        '  "observable_issues": [<list of specific visual issues>],\n'
        '  "positive_visual_signals": [<list of specific positive visual observations>],\n'
        '  "frame_observations": [<brief note per frame>]\n'
        '}'
    )

    def analyze_frames(self, frame_paths: list) -> dict:
        _log("VISUAL_ANALYSIS", f"Starting with {len(frame_paths)} frames")

        if not frame_paths:
            _log("VISUAL_ANALYSIS", "No frames available, returning empty analysis")
            return self._empty_visual_analysis()

        # Sample at most 4 evenly distributed frames for speed
        if len(frame_paths) > 4:
            step = len(frame_paths) / 4
            selected = [frame_paths[int(i * step)] for i in range(4)]
        else:
            selected = frame_paths

        _log("VISUAL_ANALYSIS", f"Selected {len(selected)} frames for analysis")

        content = [{"type": "text", "text": self.VISUAL_PROMPT}]
        for i, fp in enumerate(selected):
            with open(fp, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
            _log("VISUAL_ANALYSIS", f"Frame {i+1}/{len(selected)} encoded: {fp}")

        payload = {
            "model": MODEL_VISION,
            "messages": [
                {"role": "system", "content": self.VISUAL_SYSTEM},
                {"role": "user",   "content": content},
            ],
            "temperature": 0.1,
            "max_tokens":  1000,
        }

        _log("VISUAL_ANALYSIS", "Sending frames to vision model...")
        t0 = time.time()
        try:
            raw = self._chat_raw(payload)
            _log("VISUAL_ANALYSIS", f"Vision model responded in {time.time()-t0:.1f}s")
            result = self._parse_json(raw, self._empty_visual_analysis())
            _log("VISUAL_ANALYSIS", "JSON parsed successfully")
            return result
        except Exception as e:
            _log("VISUAL_ANALYSIS", f"ERROR: {e}")
            return self._empty_visual_analysis()

    def _empty_visual_analysis(self) -> dict:
        return {
            "teacher_visible": False, "teacher_presence_score": 5,
            "screen_share_detected": False, "screen_share_quality": "None",
            "student_cameras_visible": False, "estimated_student_count": None,
            "environment_professionalism": 5, "visual_engagement_score": 5,
            "observable_issues": [], "positive_visual_signals": [],
            "frame_observations": [],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 4. SYNTHESIS
    # ──────────────────────────────────────────────────────────────────────────

    SYNTH_SYSTEM = (
        "You are a senior educational quality assessor for an online education platform.\n"
        "You have received AI-generated analyses of a class recording.\n"
        "Your task is to synthesise all signals and produce a final quality rating.\n"
        "Return ONLY valid JSON with no markdown fences or extra text."
    )

    def synthesize(
        self,
        transcript_data: dict,
        transcript_analysis: dict,
        visual_analysis: dict,
        video_duration: float,
    ) -> dict:
        duration_min = video_duration / 60 if video_duration else 0
        _log("SYNTHESIS", f"Starting. Duration: {duration_min:.1f} min")

        # Build synthesis prompt without .format() to avoid JSON brace conflicts
        prompt = (
            "Based on the following analyses of a class recording, produce a final quality assessment.\n\n"
            f"CLASS DURATION: {duration_min:.1f} minutes\n\n"
            "TRANSCRIPT ANALYSIS:\n"
            + json.dumps(transcript_analysis, indent=2) + "\n\n"
            "VISUAL ANALYSIS:\n"
            + json.dumps(visual_analysis, indent=2) + "\n\n"
            "TRANSCRIPT EXCERPT (first 1500 chars):\n"
            + transcript_data.get("text", "")[:1500] + "\n\n"
            'Return a JSON object with EXACTLY these fields:\n\n'
            '{\n'
            '  "rating": "<Poor | Average | Good | Excellent>",\n'
            '  "score": <integer 0-100>,\n'
            '  "summary": "<3-4 sentence executive summary explaining the rating with specific evidence>",\n'
            '  "observations": ["<specific observation 1>", "<observation 2>", "<observation 3>"],\n'
            '  "strengths": ["<specific strength 1>", "<strength 2>"],\n'
            '  "weaknesses": ["<specific weakness 1>", "<weakness 2>"],\n'
            '  "recommendation": "<1-2 sentences: most impactful improvement for this teacher>",\n'
            '  "dimension_scores": {\n'
            '    "teaching_quality": <0-100>,\n'
            '    "student_engagement": <0-100>,\n'
            '    "content_structure": <0-100>,\n'
            '    "communication": <0-100>,\n'
            '    "visual_presentation": <0-100>\n'
            '  }\n'
            '}\n\n'
            'Rating rubric:\n'
            '- Excellent (85-100): Exceptional interaction, clear structure, engaged students, excellent pacing\n'
            '- Good (65-84): Solid teaching with minor gaps, decent interaction, mostly clear\n'
            '- Average (40-64): Functional but notable weaknesses in interaction/structure/clarity\n'
            '- Poor (0-39): Significant issues: monologue only, poor structure, unclear content\n\n'
            'Be specific. Reference actual content or observed behaviours where possible.'
        )

        _log("SYNTHESIS", f"Sending {len(prompt)} char prompt to LLM...")
        t0 = time.time()
        try:
            raw = self._chat(MODEL_TEXT, self.SYNTH_SYSTEM, prompt, max_tokens=1500)
            _log("SYNTHESIS", f"LLM responded in {time.time()-t0:.1f}s")
        except Exception as e:
            _log("SYNTHESIS", f"ERROR calling LLM: {e}")
            raw = ""

        result = self._parse_json(raw, None)

        if result is None:
            _log("SYNTHESIS", "JSON parse failed, using default result")
            result = {
                "rating":      "Average",
                "score":       50,
                "summary":     "Analysis completed but structured output could not be parsed.",
                "observations": ["Review the raw analysis data for details."],
                "strengths":   [],
                "weaknesses":  [],
                "recommendation": "Rerun the analysis for a more detailed assessment.",
                "dimension_scores": {
                    "teaching_quality":    50,
                    "student_engagement":  50,
                    "content_structure":   50,
                    "communication":       50,
                    "visual_presentation": 50,
                },
            }

        # Ensure required keys
        for key in ("rating", "score", "summary", "observations", "strengths",
                    "weaknesses", "recommendation", "dimension_scores"):
            if key not in result:
                result[key] = self._default_synthesis().get(key)

        _log("SYNTHESIS", f"Final rating: {result.get('rating')} | Score: {result.get('score')}")
        return result

    def _default_synthesis(self) -> dict:
        return {
            "rating": "Average", "score": 50,
            "summary": "", "observations": [], "strengths": [],
            "weaknesses": [], "recommendation": "",
            "dimension_scores": {
                "teaching_quality": 50, "student_engagement": 50,
                "content_structure": 50, "communication": 50,
                "visual_presentation": 50,
            },
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _chat(self, model: str, system: str, user: str, max_tokens: int = 2000) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0.1,
            "max_tokens":  max_tokens,
        }
        return self._chat_raw(payload)

    def _chat_raw(self, payload: dict) -> str:
        current_model = payload.get("model", MODEL_TEXT)
        is_vision = "image_url" in json.dumps(payload.get("messages", []))

        if is_vision:
            candidates = [current_model] + [m for m in FALLBACK_VISION_MODELS if m != current_model]
        else:
            candidates = [current_model] + [m for m in FALLBACK_TEXT_MODELS if m != current_model]

        last_err = None
        for m in candidates:
            payload["model"] = m
            _log("API", f"Trying model: {m}")
            try:
                t0 = time.time()
                resp = requests.post(
                    f"{OPENROUTER_BASE}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=90,
                )
                elapsed = time.time() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    # Guard against None content (some models return null for tool/refusals)
                    choices = data.get("choices") or []
                    if not choices:
                        last_err = f"Empty choices list from {m}"
                        _log("API", f"FAILED: {last_err} | body: {resp.text[:300]}")
                        continue
                    content = (choices[0].get("message") or {}).get("content")
                    if content is None:
                        # Check finish_reason for clue
                        finish = choices[0].get("finish_reason", "unknown")
                        last_err = f"Null content from {m} (finish_reason={finish})"
                        _log("API", f"FAILED: {last_err} | body: {resp.text[:400]}")
                        continue
                    content = content.strip()
                    if not content:
                        last_err = f"Empty string content from {m}"
                        _log("API", f"FAILED: {last_err}")
                        continue
                    tokens = data.get("usage") or {}
                    _log("API", f"OK in {elapsed:.1f}s | in={tokens.get('prompt_tokens',0)} out={tokens.get('completion_tokens',0)} tokens | model={m}")
                    return content
                # Non-200 status
                last_err = f"HTTP {resp.status_code} ({m}): {resp.text[:300]}"
                _log("API", f"FAILED {last_err}")
            except requests.exceptions.Timeout:
                last_err = f"Timeout after 90s ({m})"
                _log("API", f"TIMEOUT: {last_err}")
            except Exception as e:
                last_err = f"Exception ({m}): {type(e).__name__}: {e}"
                _log("API", f"EXCEPTION: {last_err}")

        raise RuntimeError(last_err or "All models failed")

    def _parse_json(self, raw: str, fallback):
        """Strip markdown fences and parse JSON, with regex fallback."""
        if not raw:
            return fallback
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        _log("PARSE_JSON", f"Failed to parse JSON from: {raw[:200]}")
        return fallback
