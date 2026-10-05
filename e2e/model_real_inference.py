"""Real CPU inference and preprocessing parity using a distinct-person fixture.

Run in backend environment from repository root:
  PYTHONPATH=. python e2e/model_real_inference.py --different-person /tmp/attendance-astronaut.jpg
OpenCV's published SFace feature() is used as a separate preprocessing reference.
This checks fixture behavior, not population accuracy or physical GPU readiness.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from attendance.config import Settings
from attendance.face import FaceService, SFACE_FILE

ROOT = Path(__file__).resolve().parents[1]


def run(different_person, model_dir):
    service = FaceService(Settings(model_dir=str(model_dir), inference_device='cpu'))
    service.load()
    assert service.ready and service.device == 'cpu'
    inputs = [ROOT / 'face1.jpg', ROOT / 'face2.jpg', different_person]
    embeddings = [service.extract(p.read_bytes()) for p in inputs]
    genuine = service.similarity(embeddings[0], embeddings[1])
    negative = service.similarity(embeddings[0], embeddings[2])
    assert genuine >= service.settings.face_threshold, (genuine, 'Genuine fixture rejected')
    assert negative < service.settings.face_threshold, (negative, 'Different-person fixture accepted')
    reference = cv2.FaceRecognizerSF.create(str(model_dir / SFACE_FILE), '')
    parity = []
    for path, embedding in zip(inputs, embeddings):
        image = service.decode(path.read_bytes())
        height, width = image.shape[:2]
        if max(height, width) > 1600:
            scale = 1600 / max(height, width)
            image = cv2.resize(image, (round(width * scale), round(height * scale)))
        service.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = service.detector.detect(image)
        aligned = reference.alignCrop(image, faces[0].reshape(1, -1))
        reference_embedding = service.normalize(reference.feature(aligned))
        cosine = service.similarity(embedding, reference_embedding)
        assert cosine > 0.9999, (path.name, cosine)
        parity.append({'fixture': path.name, 'opencv_reference_cosine': cosine})
    explicit_cuda = 'not tested: provider available; requires real hardware validation'
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        try:
            FaceService(Settings(model_dir=str(model_dir), inference_device='cuda')).load()
        except RuntimeError as error:
            assert 'CUDA' in str(error)
            explicit_cuda = 'correctly rejected: CUDA execution provider unavailable'
        else:
            raise AssertionError('Explicit CUDA silently fell back to CPU')
    return {'model_version': service.model_version, 'device': service.device,
            'providers': service.providers, 'threshold': service.settings.face_threshold,
            'genuine_cosine': genuine, 'different_person_cosine': negative,
            'reference_parity': parity, 'explicit_cuda': explicit_cuda,
            'limitations': 'Three fixture images only; no accuracy parity, physical GPU, or real-phone liveness claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--different-person', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=ROOT / '.local/models')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = run(args.different_person, args.model_dir)
    output = json.dumps(result, indent=2)
    print(output)
    if args.report:
        args.report.write_text(output + '\n')
