FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# เพิ่ม build-essential และ python3-dev สำหรับ compile InsightFace
# และ libgl Libraries สำหรับ OpenCV
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    python3-dev \ 
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp/

# 3. กำหนดโฟลเดอร์ทำงาน (สำคัญมาก ไม่งั้น uvicorn จะหา main.py ไม่เจอ)
# ใน Dev Container ปกติ VS Code จะ Mount งานไว้ที่ /workspaces/ชื่อโปรเจกต์
# แต่ถ้าจะรันเอง ให้ใช้ /app
WORKDIR /workspaces/img_recognition

# 4. เปิด Port ให้รู้ว่า Container นี้ใช้พอร์ตอะไร
EXPOSE 8000

# 5. สั่งรัน FastAPI (CMD ต้องเขียนเป็น Array ของ String)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```
#    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

### ⚠️ ข้อควรระวัง (สำคัญมากสำหรับ Dev Container)

1.  **VS Code จะเมินบรรทัดนี้:**
    ถ้าคุณใช้ไฟล์นี้กับ **Dev Container** (เปิดผ่าน VS Code) ตัว VS Code จะ **Overrride (ทับ)** คำสั่ง `CMD` นี้ด้วยคำสั่งเริ่มต้นของมันเอง (เพื่อให้ Terminal ทำงานได้และไม่ปิดตัวลง)
    * *ผลลัพธ์:* เปิดมา Server จะไม่รันเอง คุณต้องพิมพ์ `uvicorn ...` เอง หรือใส่ใน `postCreateCommand` ใน `devcontainer.json`

2.  **ถ้าใช้ `docker run` (รันเอง):**
    คำสั่ง `CMD` นี้จะทำงานทันที! แต่... มันจะ Error ถ้ามันหาไฟล์ `main.py` ไม่เจอ
    * *วิธีแก้:* ต้องแน่ใจว่าตอนรัน คุณได้ Mount Volume เข้าไปที่ `WORKDIR` แล้ว:
        ```bash
        docker run -p 8000:8000 -v $(pwd):/workspaces/img_recognition my-image-name