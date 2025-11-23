import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. ตั้งค่าโมเดล (ใช้ CPU เพราะรันบน Docker M1)
print("Loading model...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ฟังก์ชันสำหรับดึง "ค่าตัวเลขระบุตัวตน" (Embedding) จากรูป


# cache_data =


def get_embedding(img_path):
    # อ่านรูปภาพ
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return None

    # ให้ InsightFace หาตำแหน่งหน้าและคำนวณค่าต่างๆ
    faces = app.get(img)

    if len(faces) == 0:
        print(f"No face detected in {img_path}")
        return None

    # เอาหน้าแรกที่เจอมาใช้ (faces[0]) และดึงค่า embedding
    return faces[0].embedding


# 2. เริ่มเปรียบเทียบ
file1 = "face1.jpg"  # <-- เปลี่ยนชื่อไฟล์ตรงนี้
file2 = "face2.jpg"  # <-- เปลี่ยนชื่อไฟล์ตรงนี้

print(f"Comparing {file1} vs {file2} ...")

emb1 = get_embedding(file1)

main = emb1

emb2 = get_embedding(file2)

if emb1 is not None and emb2 is not None:
    # 3. คำนวณความเหมือน (Cosine Similarity)
    # สูตร: เอาเวกเตอร์มาคูณกัน (Dot Product)
    similarity = np.dot(emb1, emb2) / \
        (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    print("------------------------------------------------")
    print(f"ความเหมือน (Similarity Score): {similarity:.4f}")
    print("------------------------------------------------")

    # เกณฑ์ตัดสิน (Threshold) ปกติถ้าเกิน 0.4 หรือ 0.5 ถือว่าเป็นคนเดียวกัน
    if similarity > 0.4:
        print("✅ ผลลัพธ์: เป็นคนเดียวกัน (Same Person)")
    else:
        print("❌ ผลลัพธ์: คนละคน (Different People)")
