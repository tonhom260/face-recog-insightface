# Mobile attendance

ระบบลงเวลาเข้า–ออกสำหรับพนักงาน ใช้กล้องหน้าบนมือถือและ UI ภาษาไทย รองรับ tablet และ desktop ทำงานบน branch `codex/mobile-attendance`.

พนักงานเข้าสู่ระบบ → MediaPipe ตรวจท่าทางบนอุปกรณ์ → ถ่ายภาพเมื่อกลับมามองตรง → backend ใช้ YuNet และ SFace เทียบใบหน้า **1:1 กับบัญชีที่เข้าสู่ระบบ** → บันทึกเวลาเข้า/ออกจากนาฬิกา server และแสดงเวลาไทย

## Stack และการทำงาน

| ส่วน | เครื่องมือ |
| --- | --- |
| UI | React, TypeScript, Vite; mobile first |
| ตรวจท่าทาง | MediaPipe Face Landmarker ใน browser; โมเดลและ WASM เสิร์ฟจากแอป |
| ตรวจ/จัดแนวหน้า | OpenCV YuNet บน CPU |
| เปรียบเทียบใบหน้า | SFace ผ่าน ONNX Runtime; โหลดครั้งเดียวเมื่อเริ่ม backend |
| API และบัญชี | FastAPI, Argon2, session cookie แบบ HttpOnly |
| ข้อมูล | PostgreSQL สำหรับ deployment; SQLite สำหรับ development/test |
| Deployment | Docker Compose; CPU เป็นค่าเริ่มต้น มี NVIDIA CUDA override |

สุ่ม **3 ท่าจาก 6 แบบโดยไม่ซ้ำในครั้งเดียว**: หันซ้าย, หันขวา, ก้ม, เงย, ยิ้ม และหลับตาสั้น ๆ แล้วลืมตา ต้องกลับมามองตรงระหว่างท่า ถ้าใบหน้าหายหรือมีหลายคนจะเริ่มตรวจใหม่ ลำดับสุ่มสร้างบนมือถือ ไม่มีการขอรหัส challenge จาก backend ก่อนเปิดกล้อง.

React ตรวจความสว่างระหว่างเปิดกล้องด้วย หากแสงน้อยเกินไปจะแจ้งให้เปิดไฟหรือย้ายไปที่สว่าง พร้อมหยุดตรวจท่าทางและถ่ายภาพ กล้องยังเปิดเพื่อตรวจแสง เมื่อแสงเพียงพอต่อเนื่องประมาณ 500 ms ปุ่ม **เริ่มตรวจต่อ** จะกดได้ ต้องให้ผู้ใช้กดก่อนจึงเริ่ม challenge ใหม่ตั้งแต่ขั้นที่ 1 ใช้กับการลงทะเบียนด้วย ปรับเกณฑ์ผ่าน `VITE_MIN_FACE_LUMA` ตอน build ได้ ค่าเริ่มต้น `45` จากช่วงความสว่าง 0–255.

ภาพที่ส่งให้ backend เป็นภาพถ่ายหลังทำครบ ระบบไม่ส่งวิดีโอและไม่ใช้โมเดล liveness ที่ backend. UUID ที่แนบกับการลงเวลาใช้ป้องกันรายการซ้ำเมื่อต้อง retry เท่านั้น. Backend ใช้บัญชีจาก session, คำนวณคะแนนเอง และบังคับลำดับเข้า–ออกใน transaction.

ผู้ดูแลสร้างบัญชีแรกผ่านหน้า setup จากนั้นเพิ่ม/ปิดใช้งานพนักงานและ reset ใบหน้าได้ พนักงานลงทะเบียนใบหน้าครั้งแรกเอง การเปลี่ยนใบหน้าที่ลงทะเบียนแล้วต้องให้ผู้ดูแลจัดการ ไม่มีบัญชีหรือรหัสผ่านเริ่มต้น.

## เริ่มใช้งานในเครื่อง

ใช้ Python 3.11/3.12, Node 22+, npm, Git และ Git LFS:

```bash
bash scripts/setup_dev.sh
source .venv/bin/activate
ATTENDANCE_COOKIE_SECURE=false ATTENDANCE_INFERENCE_DEVICE=cpu \
  python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

สคริปต์ติดตั้ง dependency, ดาวน์โหลดโมเดลที่ pin แหล่งและ SHA256, ตรวจ inference จริง และ build หน้าเว็บ Backend เสิร์ฟหน้าเว็บที่ build แล้ว เปิดแอปและสร้างผู้ดูแลคนแรกผ่านหน้า setup. กล้องบนมือถือที่เข้าผ่าน LAN หรืออินเทอร์เน็ตต้องใช้ HTTPS.

ระหว่างพัฒนา UI เปิดอีก terminal แล้วใช้ `npm --prefix frontend run dev`. ดูขั้นตอน cloud environment, PostgreSQL, HTTPS และ Docker ใน [docs/development.md](docs/development.md).

## เลือกอุปกรณ์ผ่าน env

| `ATTENDANCE_INFERENCE_DEVICE` | การทำงานของ SFace |
| --- | --- |
| `cpu` | ใช้ CPU; รองรับ Linux และ Mac |
| `auto` | ลอง accelerator ที่ runtime รองรับ แล้วใช้ CPU หากเริ่มทำงานไม่ได้ |
| `cuda` | ต้องมี NVIDIA GPU และ CUDA runtime ที่เข้ากัน; ใช้ไม่ได้แล้วแจ้ง error |
| `coreml` | ใช้ Core ML เมื่อรัน Python บน macOS โดยตรงและ provider พร้อม; Apple GPU/Neural Engine ตามส่วนของโมเดลที่รองรับ |

YuNet และการจัดแนวหน้ายังใช้ CPU. Docker Linux บน Mac ใช้ Apple GPU ผ่าน Core ML ไม่ได้ ให้รัน backend บน macOS โดยตรง หรือใช้ Docker แบบ CPU. `/api/health` รายงาน device/provider ที่เริ่มทำงานจริง เครื่อง cloud นี้ตรวจ inference แบบ CPU แล้ว; NVIDIA และ Apple hardware ต้องตรวจบนเครื่องปลายทาง.

ตั้งค่า `ATTENDANCE_FACE_THRESHOLD` เพื่อปรับเกณฑ์ cosine ค่าเริ่มต้น `0.363`. SFace และ InsightFace มี embedding/คะแนนคนละแบบ ห้ามใช้ enrollment เดิมร่วมกัน พนักงานต้องลงทะเบียนใบหน้าใหม่.

## ตรวจสอบและเอกสาร

```bash
python scripts/prepare_models.py --verify-only
pytest -q
python -m unittest discover -s scripts -p 'test_*.py' -v
npm --prefix frontend test
npm --prefix frontend run build
```

- [API contract](docs/api-contract.md)
- [แผนและขอบเขตงาน](docs/implementation-plan.md)
- [แหล่งโมเดลและ license](docs/model-licenses.md)
- [ผลตรวจจริงและขอบเขตการทดสอบ](docs/qa-report.md)
- [การ sync โค้ดและ deploy](docs/git-workflow.md)
- [Mobile wireframe 390×844](docs/wireframes/preview.png)

ท่าทางฝั่ง client เหมาะกับระดับการตรวจที่ตกลงสำหรับ attendance และยังต้องลองกับพนักงาน/มือถือจริง คะแนนจาก fixture ไม่ใช่ผลความแม่นยำทั้งประชากรหรือหลักฐานว่าเทียบเท่า InsightFace. รูปตัวอย่างและ `my_face_constant.npy` เดิมถูกเก็บไว้ โมเดลที่ดาวน์โหลด ฐานข้อมูล และ enrollment ใหม่ไม่รวมใน Git.
