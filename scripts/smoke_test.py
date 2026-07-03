"""Manual end-to-end smoke test for the web app, with the two
external-model stages (UVR separation, DeepFilterNet denoise) stubbed
out, since this sandbox's network policy blocks the GitHub release
downloads those need (works fine on Railway's unrestricted build
network). Everything else -- ffmpeg extraction, ffmpeg mastering, the
FastAPI routes, the in-memory job manager/progress tracking, and
static file serving -- runs for real.

Usage: python scripts/smoke_test.py /path/to/test_input.mp4
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import pipeline  # noqa: E402


def fake_separate_vocals(wav_path: Path, out_dir: Path, progress_cb=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb("isolating voices", 0.5)
    # Stand-in for the UVR model: in real runs this actually separates
    # speech from background; here we just pass the extracted audio
    # through so the rest of the pipeline (which is real) has input.
    out_path = out_dir / "vocals.wav"
    shutil.copy(wav_path, out_path)
    if progress_cb:
        progress_cb("voices isolated", 1.0)
    return out_path


def fake_denoise_speech(vocals_wav: Path, work_dir: Path, progress_cb=None):
    if progress_cb:
        progress_cb("removing background noise", 0.5)
    mono = pipeline._resample_mono(vocals_wav, work_dir / "vocals_48k_mono.wav", pipeline.DENOISE_SR)
    if progress_cb:
        progress_cb("noise removed", 1.0)
    return mono


pipeline.separate_vocals = fake_separate_vocals
pipeline.denoise_speech = fake_denoise_speech

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def main():
    src = Path(sys.argv[1])
    assert src.exists(), f"missing {src}"

    resp = client.get("/api/health")
    assert resp.status_code == 200, resp.text
    print("health ok:", resp.json())

    resp = client.get("/")
    assert resp.status_code == 200 and b"Voice Isolation Studio" in resp.content
    print("static index ok, length", len(resp.content))

    with open(src, "rb") as f:
        resp = client.post("/api/jobs", files={"file": ("test_input.mp4", f, "video/mp4")})
    assert resp.status_code == 202, resp.text
    job = resp.json()
    print("job created:", job)
    job_id = job["id"]

    deadline = time.time() + 60
    last_status = None
    while time.time() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        job = resp.json()
        if job["status"] != last_status:
            print("status:", job)
            last_status = job["status"]
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.3)

    assert job["status"] == "done", f"job did not finish: {job}"

    for name in ("original.wav", "voice_clean.wav", "voice_clean.mp3"):
        resp = client.get(f"/api/jobs/{job_id}/download/{name}")
        assert resp.status_code == 200, (name, resp.status_code, resp.text)
        assert len(resp.content) > 1000, (name, "suspiciously small", len(resp.content))
        print(f"download {name}: {len(resp.content)} bytes, content-type={resp.headers['content-type']}")

    # 404 behavior for unknown job
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
