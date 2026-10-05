"""Compare two photographs with the same SFace service used by the API."""
import argparse
from pathlib import Path
from attendance.config import Settings
from attendance.face import FaceService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", nargs="?", default="face1.jpg")
    parser.add_argument("second", nargs="?", default="face2.jpg")
    args = parser.parse_args()
    settings = Settings.from_env()
    service = FaceService(settings)
    service.load()
    first = service.extract(Path(args.first).read_bytes())
    second = service.extract(Path(args.second).read_bytes())
    score = service.similarity(first, second)
    print(f"Model: {service.model_version}; YuNet CPU; SFace {service.device}")
    print(f"Cosine similarity: {score:.6f}; configured threshold: {settings.face_threshold}")
    print("Match" if score >= settings.face_threshold else "Different people")


if __name__ == "__main__":
    main()
