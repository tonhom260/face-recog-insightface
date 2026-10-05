#!/usr/bin/env python3
"""Run model initialization/inference, optionally detect and embed a fixture."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attendance.config import Settings
from attendance.face import FaceService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    service = FaceService(Settings.from_env())
    service.load()
    if not service.ready:
        raise SystemExit("YuNet/SFace initialization failed; check model integrity and dependencies")
    result = {"model_version": service.model_version, "detector_device": "cpu",
              "sface_device": service.device, "sface_providers": service.providers}
    if args.image:
        embedding = service.extract(args.image.read_bytes())
        result["embedding_dimensions"] = int(embedding.size)
        result["self_similarity"] = service.similarity(embedding, embedding)
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"Runtime validation failed: {error}") from error
