import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

# โหลดโมเดลไว้ครั้งเดียว (ประหยัดเวลา)
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


def extract_embedding_from_bytes(file_bytes: bytes, save_path: str = None):
    """
    รับไฟล์รูปแบบ bytes -> คืนค่า embedding (512 vector)
    ถ้า save_path ไม่เป็น None จะบันทึก .npy ลงไฟล์ด้วย
    """

    # bytes -> numpy array
    np_arr = np.frombuffer(file_bytes, np.uint8)

    # decode -> cv2 image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # detect face
    faces = face_app.get(img)
    if len(faces) == 0:
        return None

    embedding = faces[0].embedding

    # ถ้ามี save_path -> บันทึกไฟล์ .npy
    if save_path:
        # ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        np.save(save_path, embedding)

    return embedding


def extract_embedding_from_path(filepath: str, save_path: str = None):
    """
    อ่านไฟล์จาก path -> คืนค่า embedding
    """

    img = cv2.imread(filepath)
    if img is None:
        return None

    faces = face_app.get(img)
    if len(faces) == 0:
        return None

    embedding = faces[0].embedding

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, embedding)

    return embedding
