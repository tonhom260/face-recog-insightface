"""YuNet CPU detection and SFace ONNX Runtime inference.

Preprocessing follows OpenCV modules/objdetect/src/face_recognize.cpp:
FaceRecognizerSF.alignCrop, then blobFromImage(scale=1, 112x112, zero mean,
swapRB=True, crop=False). No guessed normalization or custom landmark warp.
"""
import io
import logging
from pathlib import Path
import threading

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import APIError

logger = logging.getLogger(__name__)
MODEL_VERSION = "opencv-sface-2021dec-v1"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"


def create_sface_session(path, requested, runtime=ort):
    if requested not in {"auto", "cpu", "cuda", "coreml"}:
        raise ValueError("Inference device must be auto, cpu, cuda, or coreml")
    available = runtime.get_available_providers()
    provider_names = {"cuda": "CUDAExecutionProvider", "coreml": "CoreMLExecutionProvider", "cpu": "CPUExecutionProvider"}
    if requested != "auto" and provider_names[requested] not in available:
        raise RuntimeError(f"{requested.upper()} requested but ONNX Runtime {provider_names[requested]} is unavailable")
    candidates = [device for device in ("cuda", "coreml", "cpu") if provider_names[device] in available] if requested == "auto" else [requested]
    for device in candidates:
        try:
            options = runtime.SessionOptions()
            options.intra_op_num_threads = 2
            options.log_severity_level = 3
            if device == "cuda":
                # Explicit CUDA must not silently execute the graph on CPU.
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            provider = provider_names[device]
            # CoreML selects supported Apple compute units. Unsupported graph
            # partitions may run on CPU; this is not a promise of all-GPU execution.
            providers = [(provider, {"MLComputeUnits": "ALL"}), "CPUExecutionProvider"] if device == "coreml" else [provider]
            session = runtime.InferenceSession(str(path), sess_options=options, providers=providers)
            if provider not in session.get_providers():
                raise RuntimeError(f"Requested {provider} was not initialized")
            session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 112, 112), dtype=np.float32)})
            return session, device
        except Exception as exc:
            if device != "cpu" and requested == "auto":
                logger.warning("%s initialization failed; auto tries next available device: %s", device.upper(), type(exc).__name__)
                continue
            raise RuntimeError(f"SFace {device.upper()} initialization/inference failed") from exc
    raise RuntimeError("No working SFace execution provider")


class FaceService:
    model_version = MODEL_VERSION

    def __init__(self, settings):
        self.settings = settings
        self.ready = False
        self.device = None
        self.providers = []
        self.lock = threading.Lock()

    def load(self):
        # Validate explicit CUDA even if model assets are missing.
        if self.settings.inference_device in {"cuda", "coreml"}:
            provider = {"cuda": "CUDAExecutionProvider", "coreml": "CoreMLExecutionProvider"}[self.settings.inference_device]
            if provider not in ort.get_available_providers():
                raise RuntimeError(f"{self.settings.inference_device.upper()} requested but ONNX Runtime {provider} is unavailable")
        directory = Path(self.settings.model_dir)
        detector_path, recognizer_path = directory / YUNET_FILE, directory / SFACE_FILE
        if not detector_path.is_file() or not recognizer_path.is_file():
            if self.settings.inference_device in {"cuda", "coreml"}:
                raise RuntimeError(f"{self.settings.inference_device.upper()} requested but YuNet/SFace model assets are missing")
            logger.warning("YuNet/SFace assets unavailable; face endpoints return MODELS_UNAVAILABLE")
            return
        try:
            self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.9, 0.3, 5000)
            self.aligner = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
            self.session, self.device = create_sface_session(recognizer_path, self.settings.inference_device)
            self.input_name = self.session.get_inputs()[0].name
            self.providers = self.session.get_providers()
            self.ready = True
        except Exception:
            if self.settings.inference_device in {"cuda", "coreml"}:
                raise
            logger.exception("Face model initialization failed")

    def decode(self, data):
        if not data or len(data) > self.settings.max_image_bytes:
            raise APIError(422, "INVALID_IMAGE", "Image is empty or exceeds the upload limit")
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Unsupported image format")
                if source.width * source.height > 25_000_000 or min(source.size) < 32:
                    raise ValueError("Unsupported image dimensions")
                source.load()
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise APIError(422, "INVALID_IMAGE", "Provide a valid JPEG, PNG, or WebP photograph") from exc

    def extract(self, data):
        if not self.ready:
            raise APIError(503, "MODELS_UNAVAILABLE", "Face verification is temporarily unavailable")
        image = self.decode(data)
        # Limit detector cost while retaining sufficient pixels for mobile photos.
        height, width = image.shape[:2]
        if max(height, width) > 1600:
            scale = 1600 / max(height, width)
            image = cv2.resize(image, (round(width * scale), round(height * scale)))
        with self.lock:
            self.detector.setInputSize((image.shape[1], image.shape[0]))
            _, faces = self.detector.detect(image)
            if faces is None or len(faces) == 0:
                raise APIError(422, "FACE_NOT_FOUND", "No usable face found")
            if len(faces) != 1:
                raise APIError(422, "MULTIPLE_FACES", "Provide a photograph with exactly one face")
            face = faces[0]
            if not np.isfinite(face).all() or min(face[2:4]) < 40:
                raise APIError(422, "FACE_NOT_FOUND", "Face is too small or unusable")
            aligned = self.aligner.alignCrop(image, face.reshape(1, -1))
            blob = cv2.dnn.blobFromImage(aligned, 1, (112, 112), (0, 0, 0), True, False)
            output = self.session.run(None, {self.input_name: blob})[0]
        return self.normalize(output)

    @staticmethod
    def normalize(embedding):
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if embedding.size != 128 or not np.isfinite(embedding).all() or norm <= 0:
            raise APIError(422, "FACE_NOT_FOUND", "Face features are unusable")
        return embedding / norm

    @staticmethod
    def similarity(first, second):
        return float(np.clip(np.dot(FaceService.normalize(first), FaceService.normalize(second)), -1, 1))
