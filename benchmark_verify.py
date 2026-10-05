"""Benchmark SFace inference against an ignored, model-tagged demo enrollment."""
import argparse
from pathlib import Path
import time
import numpy as np
from attendance.config import Settings
from attendance.face import FaceService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="face2.jpg")
    parser.add_argument("--enrollment", default=".local/demo/sface_enrollment.npz")
    args = parser.parse_args()
    service = FaceService(Settings.from_env())
    service.load()
    with np.load(args.enrollment, allow_pickle=False) as enrollment:
        if str(enrollment["model_version"]) != service.model_version:
            parser.error("Enrollment model differs; rerun register_face.py")
        target = enrollment["embedding"]
    started = time.perf_counter()
    source = service.extract(Path(args.image).read_bytes())
    extraction = time.perf_counter() - started
    started = time.perf_counter()
    score = service.similarity(target, source)
    comparison = time.perf_counter() - started
    print(f"Model: {service.model_version}; YuNet CPU; SFace {service.device}")
    print(f"Cosine similarity: {score:.6f}")
    print(f"Extraction: {extraction:.6f}s; comparison: {comparison:.9f}s")


if __name__ == "__main__":
    main()
