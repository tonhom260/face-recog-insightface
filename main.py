from fastapi import FastAPI, File, UploadFile, Form
import platform

app = FastAPI()


@app.get("/test")
async def read_test(name: str = None, age: int = None):
    print(name, age)
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
    name: str = Form(None),
    age: int = Form(None),
    upload: UploadFile = File(...)
):
    print("Name:", name)
    print("Age:", age)
    print("Filename:", upload.filename)

    # อ่านไฟล์ (ถ้าต้องการ)
    file_data = await upload.read()

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
