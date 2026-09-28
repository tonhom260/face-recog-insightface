import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis

# โหลดโมเดลครั้งเดียว (ตอน import)
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


def extract_embedding_from_bytes(file_bytes: bytes):
    """
    แปลงรูปจาก bytes -> embedding (512 vector)
    """
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    faces = face_app.get(img)
    if len(faces) == 0:
        return None

    return faces[0].embedding


def extract_embedding_from_path(filepath: str):
    """
    แปลงรูปจาก path -> embedding
    """
    img = cv2.imread(filepath)
    if img is None:
        return None

    faces = face_app.get(img)
    if len(faces) == 0:
        return None

    return faces[0].embedding


def cosine_similarity(v1, v2):
    """
    คำนวณ cosine similarity ระหว่าง embedding 2 ตัว
    """
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def verify_face(
    source_image_bytes: bytes = None,
    source_image_path: str = None,
    target_embedding_path: str = None,
    target_embedding_vector: np.ndarray = None,
    threshold: float = 0.4
):
    """
    เทียบภาพใหม่ กับ embedding เดิม
    return: {similarity, is_match, extraction_time, math_time}

    • source_image_bytes -> bytes ของรูปใหม่
    • source_image_path  -> path รูปใหม่ (ใช้แทน bytes ได้)
    • target_embedding_path -> ไฟล์ .npy ของ embedding เดิม
    • target_embedding_vector -> vector ของ embedding เดิม (แทน path ได้)
    """

    # ---- Load target embedding ----
    if target_embedding_vector is None:
        if target_embedding_path is None:
            raise ValueError(
                "ต้องระบุ target_embedding_path หรือ target_embedding_vector")
        target_embedding_vector = np.load(target_embedding_path)

    # ---- Extract embedding from source image ----
    t1 = time.time()

    if source_image_bytes:
        source_embedding = extract_embedding_from_bytes(source_image_bytes)
    elif source_image_path:
        source_embedding = extract_embedding_from_path(source_image_path)
    else:
        raise ValueError("ต้องส่ง source_image_bytes หรือ source_image_path")

    extraction_time = time.time() - t1

    # ไม่มีหน้าในรูปใหม่
    if source_embedding is None:
        return {
            "success": False,
            "reason": "no_face_detected",
            "similarity": None
        }

    # ---- Compute similarity ----
    t2 = time.time()
    similarity = cosine_similarity(source_embedding, target_embedding_vector)
    math_time = time.time() - t2

    # ---- Final result ----
    return {
        "success": True,
        "similarity": similarity,
        "is_match": similarity > threshold,
        "extraction_time": extraction_time,
        "math_time": math_time
    }
