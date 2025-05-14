import streamlit as st
import cv2
import av
import numpy as np
import requests
import base64
import tempfile
import sounddevice as sd
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 🔧 請修改成你的後端 FastAPI API 網址
API_BASE = "https://fe67-125-228-143-171.ngrok-free.app"

st.set_page_config(page_title="📷 即時 OCR + 🎙 語音辨識", layout="centered")
st.title("📷 即時 PaddleOCR + 🎙 Whisper 語音辨識")

# ---------------------------
# 📸 相機畫面即時擷取辨識
# ---------------------------
st.header("📸 鏡頭畫面即時辨識")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        self.latest_frame = image.copy()
        return image

ctx = webrtc_streamer(
    key="ocr-cam",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("📸 擷取並辨識畫面") and ctx.video_processor:
    frame = ctx.video_processor.latest_frame
    if frame is not None:
        _, buffer = cv2.imencode(".png", frame)
        base64_img = base64.b64encode(buffer).decode("utf-8")
        payload = {"image": f"data:image/png;base64,{base64_img}"}
        try:
            res = requests.post(f"{API_BASE}/ocr", json=payload)
            text = res.json().get("text", "")
            st.text_area("📄 OCR 辨識結果", value=text, height=200)
        except Exception as e:
            st.error(f"❌ OCR API 錯誤：{e}")
    else:
        st.warning("⚠️ 尚未擷取到畫面")

# ---------------------------
# 🎤 錄音語音辨識（Whisper）
# ---------------------------
st.header("🎤 即時語音輸入辨識")

duration = st.slider("⏱ 錄音時間（秒）", 2, 10, 4)

if st.button("🎙 開始錄音"):
    with st.spinner("🎙 錄音中，請開始說話..."):
        fs = 16000
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, fs)
            tmp.seek(0)
            audio_bytes = tmp.read()

        st.audio(audio_bytes, format="audio/wav")

        with st.spinner("⏳ 辨識語音中..."):
            try:
                res = requests.post(f"{API_BASE}/whisper", files={"file": ("audio.wav", audio_bytes, "audio/wav")})
                result = res.json().get("text", "")
                st.text_area("📝 語音辨識結果", value=result, height=200)
            except Exception as e:
                st.error(f"❌ 語音 API 錯誤：{e}")
 