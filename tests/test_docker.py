"""Guards on the container setup: cheap checks that the security-relevant lines
don't silently disappear in a later edit. (Building the image is `make docker-build`.)"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
COMPOSE = (ROOT / "docker-compose.yml").read_text()
IGNORE = (ROOT / ".dockerignore").read_text().split()


def test_secrets_and_data_never_enter_the_build_context():
    for entry in [".env", "storage/", "data/", ".venv/", ".git/"]:
        assert entry in IGNORE


def test_base_images_pinned_by_digest_and_deps_by_hash():
    froms = re.findall(r"^FROM (\S+)", DOCKERFILE, re.M)
    assert froms and all("@sha256:" in f for f in froms)
    assert "--require-hashes" in DOCKERFILE
    assert "sha256:" in (ROOT / "requirements.txt").read_text()


def test_runs_unprivileged_and_read_only():
    assert re.search(r"^USER (?!root)\w+", DOCKERFILE, re.M)
    for line in ["read_only: true", "cap_drop: [ALL]", "no-new-privileges:true", "init: true"]:
        assert line in COMPOSE


def test_only_loopback_ports_are_published():
    for port in re.findall(r'-\s*"([^"]+:\d+)"', COMPOSE):
        assert port.startswith("127.0.0.1:"), port


def test_space_bundle_contains_only_what_should_be_published(tmp_path):
    import subprocess

    subprocess.run([str(ROOT / "deploy" / "build_space.sh")], check=True, capture_output=True)
    out = ROOT / "build" / "hf-space"
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"Dockerfile", "README.md", "requirements.txt", "jarvis_rag/api.py", "eval/qa_set.jsonl",
            "learn/sample_docs/product_specs.md"} <= files
    assert not any(f.startswith((".env", "storage/", "data/", ".git", ".space-secrets")) for f in files)
    assert "sdk: docker" in (out / "README.md").read_text() and "app_port: 7860" in (out / "README.md").read_text()
    space_df = (out / "Dockerfile").read_text()
    assert "--min-hit5" in space_df and "KB_READ_ONLY=true" in space_df and "--require-hashes" in space_df
