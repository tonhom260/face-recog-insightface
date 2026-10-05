"""Generate an ignored SFace demo enrollment; never reuse InsightFace embeddings."""
import argparse
from pathlib import Path
import numpy as np
from attendance.config import Settings
from attendance.face import FaceService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="face1.jpg")
    parser.add_argument("--output", default=".local/demo/sface_enrollment.npz")
    args = parser.parse_args()
    service = FaceService(Settings.from_env())
    service.load()
    embedding = service.extract(Path(args.image).read_bytes())
    output = Path(args.output)
    if output.resolve() == Path("my_face_constant.npy").resolve() or output.suffix != ".npz":
        parser.error("Use a .npz output; the old InsightFace enrollment must be preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, embedding=embedding, model_version=service.model_version)
    print(f"Saved {service.model_version} enrollment ({embedding.size} dimensions) to {output}")


if __name__ == "__main__":
    main()
