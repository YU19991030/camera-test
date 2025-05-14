from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from paddleocr import PaddleOCR
from faster_whisper import WhisperModel
from PIL import Image
import numpy as np
import cv2
import base64
import io
import tempfile

app = FastAPI()

# CORS（讓 Streamlit 前端能跨域連線）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 PaddleOCR & Whisper 模型
ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')
whisper_model = WhisperModel("base", device="cpu", compute_type="float32")

# ----------- OCR 路由 -----------
class OCRRequest(BaseModel):
    image: str  # base64 格式圖片

@app.post("/ocr")
def ocr_endpoint(req: OCRRequest):
    try:
        base64_str = req.image.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        result = ocr_model.ocr(img, cls=True)
        text = "\n".join([line[1][0] for box in result for line in box])
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

# ----------- Whisper 語音辨識路由 -----------
@app.post("/whisper")
async def whisper_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        segments, _ = whisper_model.transcribe(
            tmp_path, language="zh", beam_size=5, vad_filter=True
        )
        text = " ".join([seg.text.strip() for seg in segments])
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}
