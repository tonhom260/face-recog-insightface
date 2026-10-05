#!/usr/bin/env python3
"""Fetch pinned publisher assets, verify SHA256, then atomically install them.

Only standard verified HTTPS and Git/Git LFS routes are used. No ZIP contents
are extracted; the MediaPipe task bundle stays intact for the browser runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ZOO_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MEDIAPIPE_COMMIT = "c2518ec444c3a3a99689e5d31eddadc240c83a0c"
MODELS = (
    {"name": "YuNet", "group": "backend", "filename": "face_detection_yunet_2023mar.onnx",
     "repo": "https://github.com/opencv/opencv_zoo.git", "commit": ZOO_COMMIT,
     "path": "models/face_detection_yunet/face_detection_yunet_2023mar.onnx", "lfs": True,
     "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", "size": 232589},
    {"name": "SFace", "group": "backend", "filename": "face_recognition_sface_2021dec.onnx",
     "repo": "https://github.com/opencv/opencv_zoo.git", "commit": ZOO_COMMIT,
     "path": "models/face_recognition_sface/face_recognition_sface_2021dec.onnx", "lfs": True,
     "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", "size": 38696353},
    {"name": "MediaPipe Face Landmarker float16 v1", "group": "frontend", "filename": "face_landmarker.task",
     "repo": "https://github.com/google-ai-edge/mediapipe-samples.git", "commit": MEDIAPIPE_COMMIT,
     "path": "examples/face_landmarker/ios/FaceLandmarker/face_landmarker.task", "lfs": False,
     "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff", "size": 3758596},
)


def verify(path: Path, model: dict) -> bool:
    if not path.is_file() or path.stat().st_size != model["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == model["sha256"]


def atomic_install(source: Path, destination: Path, model: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Public model assets must remain readable by the unprivileged container,
    # even when the download shell has a restrictive umask.
    destination.parent.chmod(0o755)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as out:
            temporary = Path(out.name)
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, out)
            out.flush()
            os.fsync(out.fileno())
        if not verify(temporary, model):
            raise ValueError(f"Integrity verification failed for {model['name']}; existing asset preserved")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(*args: str, cwd: Path, env=None) -> None:
    if args and args[0] == "git":
        args = ("git", "-c", "http.sslVerify=true", *args[1:])
    subprocess.run(args, cwd=cwd, env=env, check=True, timeout=300)


def fetch_git(model: dict, workspace: Path) -> Path:
    checkout = workspace / ("opencv-zoo" if model["lfs"] else "mediapipe-samples")
    if not checkout.exists():
        checkout.mkdir()
        run("git", "init", "--quiet", cwd=checkout)
        run("git", "remote", "add", "origin", model["repo"], cwd=checkout)
        run("git", "fetch", "--depth", "1", "origin", model["commit"], cwd=checkout)
    source = workspace / model["filename"]
    # Read only the pinned blob; this avoids checking out unrelated model assets.
    with source.open("wb") as out:
        subprocess.run(["git", "show", f"{model['commit']}:{model['path']}"], cwd=checkout,
                       stdout=out, check=True, timeout=300)
    if model["lfs"]:
        pointer = source.read_text(encoding="ascii")
        if f"oid sha256:{model['sha256']}\n" not in pointer or f"size {model['size']}\n" not in pointer:
            raise ValueError(f"Publisher LFS metadata disagrees with manifest for {model['name']}")
        run("git", "lfs", "fetch", "--include=" + model["path"], "--exclude=", "origin", model["commit"], cwd=checkout)
        digest = model["sha256"]
        source = checkout / ".git" / "lfs" / "objects" / digest[:2] / digest[2:4] / digest
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "backend", "frontend"), default="all")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("ATTENDANCE_MODEL_DIR", str(ROOT / ".local/models"))))
    parser.add_argument("--frontend-dir", type=Path, default=ROOT / "frontend/public/models")
    parser.add_argument("--verify-only", action="store_true", help="No network or mutations; fail if assets are missing/corrupt")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="attendance-models-") as directory:
        for model in MODELS:
            if args.only != "all" and args.only != model["group"]:
                continue
            destination = (args.model_dir if model["group"] == "backend" else args.frontend_dir) / model["filename"]
            if not args.verify_only:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.parent.chmod(0o755)
            if verify(destination, model):
                if not args.verify_only:
                    destination.chmod(0o644)
                print(f"Verified {model['name']}: {destination}")
                continue
            if args.verify_only:
                raise SystemExit(f"Missing or corrupt {model['name']}: {destination}; run scripts/prepare_models.py")
            atomic_install(fetch_git(model, Path(directory)), destination, model)
            print(f"Installed verified {model['name']}: {destination}")
        if not args.verify_only:
            for group, destination in (("backend", args.model_dir), ("frontend", args.frontend_dir)):
                if args.only in ("all", group):
                    shutil.copytree(ROOT / "scripts/model_licenses", destination / "licenses", dirs_exist_ok=True)
                    (destination / "licenses").chmod(0o755)
                    for notice in (destination / "licenses").iterdir():
                        notice.chmod(0o644)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        raise SystemExit(f"Model preparation failed: {error}") from error
