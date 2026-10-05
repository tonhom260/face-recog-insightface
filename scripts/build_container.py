#!/usr/bin/env python3
"""Build through the configured HTTPS proxy and CA trust; never disable TLS."""
import argparse
import os
from pathlib import Path
import socket
import ssl
import subprocess
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tag", default="attendance-app:local")
    parser.add_argument("--min-face-luma", default=os.getenv("VITE_MIN_FACE_LUMA"),
                        help="Frontend build threshold; Docker default 45. Rebuild to apply changes.")
    args = parser.parse_args()
    if args.target == "cuda":
        runtimes = subprocess.check_output(["docker", "info", "--format", "{{json .Runtimes}}"], text=True)
        if '"nvidia"' not in runtimes:
            raise SystemExit("CUDA build requested but Docker NVIDIA runtime is unavailable; configure NVIDIA Container Toolkit first")
    command = ["docker", "build", "--target", args.target, "--tag", args.tag]
    if args.min_face_luma is not None:
        command += ["--build-arg", f"VITE_MIN_FACE_LUMA={args.min_face_luma}"]
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if proxy:
        host = urllib.parse.urlparse(proxy).hostname
        if host:
            # Propagate normal workspace DNS resolution for the existing proxy,
            # preserving the proxy's routing and policy enforcement.
            command += ["--add-host", f"{host}:{socket.gethostbyname(host)}"]
        for variable in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
            if variable in os.environ:
                command += ["--build-arg", variable]
        trust = os.getenv("SSL_CERT_FILE") or ssl.get_default_verify_paths().cafile
        if trust and Path(trust).is_file():
            # The trust bundle is mounted only during installation; it is never
            # copied into the resulting image or committed to the repository.
            command += ["--secret", f"id=cloud_ca,src={trust}"]
    command.append(str(ROOT))
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
