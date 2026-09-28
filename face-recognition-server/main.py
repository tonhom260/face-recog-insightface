from fastapi import FastAPI, File, UploadFile, Form
import platform
from register import extract_embedding_from_bytes
from verify import verify_face

app = FastAPI()


@app.get("/test")
async def read_test():
    # ส่งกลับเป็น JSON (FastAPI แปลง Dict เป็น JSON ให้เอง)
    return {
        "status": "ok",
        "message": "FastAPI is running correctly!",
        "system_info": {
            "platform": platform.system(),
            "processor": platform.processor(),  # เช็คได้ว่าเป็น aarch64 (M1) หรือไม่
            "node": platform.node()
        }
    }


@app.post("/send_img")
async def read_test(
    userId: str = Form(None),
    upload: UploadFile = File(...)
):
    print("Name:", userId)

    print("Filename:", upload.filename)

    # อ่านไฟล์ (ถ้าต้องการ)
    file_data = await upload.read()
    # ตั้งชื่อไฟล์สำหรับบันทึก
    # สำหรับทดสอบ จะ cache ไว้แค่ ไฟล์เดียว
    filename = f"./embeddings/my_face_constant.npy"

    embedding = extract_embedding_from_bytes(
        file_bytes=file_data,
        save_path=filename  # บันทึกไฟล์
    )

    if embedding is None:
        return {"status": "no_face_detected"}

    return {
        "status": "ok",
        "message": "File received",
        "file_info": {
            "filename": upload.filename,
            "content_type": upload.content_type,
            "file_size": len(file_data)
        },
        "system_info": {
            "platform": platform.system(),
            "processor": platform.processor(),
            "node": platform.node()
        }
    }


@app.post("/verify")
async def verify(upload: UploadFile = File(...)):

    print("Filename:", upload.filename)
    file_data = await upload.read()

    result = verify_face(
        source_image_bytes=file_data,
        target_embedding_path="embeddings/my_face_constant.npy",
        threshold=0.4
    )

    return result
