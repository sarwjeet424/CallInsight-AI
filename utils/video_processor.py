"""
VideoProcessor — Extract audio track and sample frames from a class recording.
Uses system ffmpeg binary directly via subprocess.
"""
import os, subprocess, tempfile, math, glob, shutil, time
from pathlib import Path


def _log(stage: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{stage}] {msg}", flush=True)


class VideoProcessor:
    # Extract at most 5 evenly spaced frames — enough for visual quality signals
    MAX_FRAMES = 5

    # Cap audio for transcription: analyse first N minutes only.
    # A 45-min class has plenty of teaching signal in the first 20 min.
    # Set to 0 or None to transcribe the full recording.
    AUDIO_ANALYSIS_LIMIT_SEC = 20 * 60   # 20 minutes

    def extract(self, video_path: str) -> dict:
        """
        Extract audio (MP3, capped at AUDIO_ANALYSIS_LIMIT_SEC) and
        sample JPEG frames from the video.
        Returns dict: audio_path, frame_paths, duration_seconds, tmp_dir.
        """
        _log("EXTRACT", f"Starting extraction of: {video_path}")
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        _log("EXTRACT", f"Input file size: {file_size_mb:.1f} MB")

        tmp_dir    = tempfile.mkdtemp(prefix="classinsight_")
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        _log("EXTRACT", f"Temp dir: {tmp_dir}")

        # ── Get duration ──────────────────────────────────────────────────────
        _log("EXTRACT", "Running ffprobe to get video duration...")
        t0 = time.time()
        duration = self._get_duration(video_path)
        _log("EXTRACT", f"Duration: {duration:.0f}s ({duration/60:.1f} min) — detected in {time.time()-t0:.1f}s")

        # ── Extract audio (capped for speed) ─────────────────────────────────
        audio_duration = duration
        audio_args = ["-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k"]

        limit = self.AUDIO_ANALYSIS_LIMIT_SEC
        if limit and duration > limit:
            audio_duration = limit
            audio_args += ["-t", str(limit)]
            _log("EXTRACT", f"Audio capped to first {limit//60} min for faster transcription (full video is {duration/60:.0f} min)")
        else:
            _log("EXTRACT", "Extracting full audio track")

        audio_args.append(audio_path)
        _log("EXTRACT", "Running ffmpeg to extract audio...")
        t0 = time.time()
        self._run_ffmpeg(audio_args)
        audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        _log("EXTRACT", f"Audio extracted in {time.time()-t0:.1f}s → {audio_size_mb:.1f} MB")

        # ── Sample frames evenly across FULL video ────────────────────────────
        # Spread frames over the whole recording to capture different class phases
        interval = max(60, math.ceil(duration / self.MAX_FRAMES))
        _log("EXTRACT", f"Sampling {self.MAX_FRAMES} frames every ~{interval}s...")
        t0 = time.time()
        self._run_ffmpeg([
            "-y", "-i", video_path,
            "-vf", f"fps=1/{interval},scale=640:-1",
            "-q:v", "5",
            os.path.join(frames_dir, "frame_%04d.jpg"),
        ])
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        _log("EXTRACT", f"Frames extracted in {time.time()-t0:.1f}s → {len(frame_paths)} frames")

        return {
            "audio_path":        audio_path,
            "audio_duration_sec": audio_duration,
            "frame_paths":        frame_paths,
            "duration_seconds":   duration,
            "tmp_dir":            tmp_dir,
        }

    def cleanup(self, extraction: dict):
        """Remove temporary files."""
        tmp_dir = extraction.get("tmp_dir")
        if tmp_dir and os.path.exists(tmp_dir):
            _log("CLEANUP", f"Removing temp dir: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_bin(name: str) -> str:
        """Locate ffmpeg/ffprobe: PATH → WinGet → common install dirs."""
        found = shutil.which(name)
        if found:
            return found

        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            winget_path = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
            if os.path.exists(winget_path):
                matches = glob.glob(os.path.join(winget_path, "**", f"{name}.exe"), recursive=True)
                if matches:
                    bin_dir = os.path.dirname(matches[0])
                    if bin_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    return matches[0]

        for candidate in [r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin"]:
            exe = os.path.join(candidate, f"{name}.exe")
            if os.path.exists(exe):
                return exe

        return name  # last resort: hope it's on PATH

    def _get_duration(self, video_path: str) -> float:
        bin_path = self._find_bin("ffprobe")
        try:
            result = subprocess.run(
                [bin_path, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=30,
            )
            return float(result.stdout.strip())
        except Exception as e:
            _log("EXTRACT", f"WARNING: ffprobe duration failed: {e}")
            return 0.0

    def _run_ffmpeg(self, args: list):
        bin_path = self._find_bin("ffmpeg")
        cmd = [bin_path] + args
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-2000:]
            _log("EXTRACT", f"ffmpeg stderr (last 2000 chars):\n{stderr}")
            raise RuntimeError(f"ffmpeg failed (exit {result.returncode}). See logs above.")
