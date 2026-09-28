import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis

# --- SETUP ---
# โหลดโมเดลเตรียมไว้ก่อน (ปกติ Server จะทำขั้นตอนนี้แค่ครั้งเดียวตอนเปิดเครื่อง)
print("Loading Model...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# โหลดค่า Constant ที่เราเซฟไว้ (Database จำลอง)
target_embedding = np.load("my_face_constant.npy")
print("Database loaded.")
print("------------------------------------------------")

# รูปที่จะเอามาเทส
# test_image_file = "face2.jpg"  # <--- รูปใหม่ที่จะมาสแกน
# test_image_file = "bhum.jpg"  # <--- รูปใหม่ที่จะมาสแกน
# test_image_file = "bhop.jpg"  # <--- รูปใหม่ที่จะมาสแกน
test_image_file = "bhop2.jpg"  # <--- รูปใหม่ที่จะมาสแกน

# --- เริ่มจับเวลา ---
total_start = time.time()

# 1. ขั้นตอนอ่านรูปและแปลงเป็นตัวเลข (Feature Extraction)
# *ส่วนนี้จะกินเวลามากที่สุด เพราะต้องใช้ AI คำนวณ*
t1 = time.time()
img = cv2.imread(test_image_file)
faces = app.get(img)
extraction_time = time.time() - t1

if len(faces) == 0:
    print("ไม่เจอหน้าในรูปใหม่")
    exit()

source_embedding = faces[0].embedding

# 2. ขั้นตอนเปรียบเทียบตัวเลข (Vector Math)
# *ส่วนนี้คือสิ่งที่คุณอยากรู้ ว่าเร็วแค่ไหน*
t2 = time.time()

# คำนวณ Cosine Similarity
similarity = np.dot(source_embedding, target_embedding) / \
    (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))

math_time = time.time() - t2

# --- สรุปผล ---
print(f"ผลลัพธ์ความเหมือน: {similarity:.4f}")
if similarity > 0.4:
    print("✅ คนเดียวกัน")
else:
    print("❌ คนละคน")

print("------------------------------------------------")
print(
    f"⏱️  เวลาที่ใช้แปลงรูปใหม่ (AI Inference): {extraction_time:.5f} วินาที")
print(f"🚀 เวลาที่ใช้เทียบตัวเลข (Math/Matching): {math_time:.10f} วินาที")
print("------------------------------------------------")
