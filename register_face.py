import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. โหลดโมเดล
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. อ่านรูปต้นฉบับ
filename = "face1.jpg"  # <--- เปลี่ยนเป็นชื่อไฟล์รูปหลักของคุณ
img = cv2.imread(filename)

if img is None:
    print("❌ หาไฟล์รูปไม่เจอ")
    exit()

faces = app.get(img)

if len(faces) > 0:
    # 3. ดึงค่า Embedding (ชุดตัวเลข 512 ตัว)
    embedding = faces[0].embedding

    # 4. บันทึกเป็นไฟล์ Binary ของ Numpy (.npy)
    # ไฟล์นี้จะเล็กมาก และโหลดเร็วมาก
    np.save("my_face_constant.npy", embedding)
    print(f"✅ บันทึกค่า Embedding ของ {filename} เรียบร้อยแล้ว!")
    print(f"   Shape: {embedding.shape}")
else:
    print("❌ ไม่เจอหน้าในรูป")
