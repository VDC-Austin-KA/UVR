"""End-to-end smoke test for the whole web app.

Because the pipeline is now pure ffmpeg (no downloaded models), this runs
the *real* pipeline start to finish -- extraction, denoise, amplify --
against a synthetic noisy clip, with nothing stubbed out.

Usage:
    python scripts/smoke_test.py [/path/to/optional/input.mp4]

With no argument it synthesizes a noisy "faint speech" test clip via
ffmpeg and runs that.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def _make_test_clip(dest: Path) -> None:
    # A quiet 300Hz tone (stand-in for faint speech) buried in pink noise.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=5,volume=0.06",
            "-f", "lavfi", "-i", "anoisesrc=d=5:c=pink:a=0.25",
            "-filter_complex", "[0][1]amix=inputs=2:normalize=0",
            "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
        assert src.exists(), f"missing {src}"
        filename = src.name
    else:
        tmp = Path(tempfile.mkdtemp()) / "noisy_test.wav"
        _make_test_clip(tmp)
        src = tmp
        filename = "noisy_test.wav"

    resp = client.get("/api/health")
    assert resp.status_code == 200, resp.text
    print("health ok:", resp.json())

    resp = client.get("/")
    assert resp.status_code == 200 and b"Voice Isolation Studio" in resp.content
    print("static index ok, length", len(resp.content))

    with open(src, "rb") as f:
        resp = client.post("/api/jobs", files={"file": (filename, f, "application/octet-stream")})
    assert resp.status_code == 202, resp.text
    job = resp.json()
    print("job created:", job)
    job_id = job["id"]

    deadline = time.time() + 120
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

    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
