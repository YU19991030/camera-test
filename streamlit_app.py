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
API_BASE = " https://0f58-125-228-143-171.ngrok-free.app"

st.set_page_config(page_title="📷 即時 OCR + 🎙 語音辨識", layout="centered")
st.title("📷 即時 PaddleOCR + 🎙 Whisper 語音辨識")

# ---------------------------
# 📸 相機拍照並送出辨識
# ---------------------------
st.header("📸 名片拍照辨識")

img_file = st.camera_input("請拍攝一張名片")

if img_file:
    st.image(img_file, caption="名片預覽", use_column_width=True)

    img_bytes = img_file.getvalue()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    payload = {"image": f"data:image/jpeg;base64,{base64_img}"}

    with st.spinner("🧠 PaddleOCR 辨識中..."):
        try:
            res = requests.post(f"{API_BASE}/ocr", json=payload)
            res.raise_for_status()
            text = res.json().get("text", "")
            st.text_area("📄 OCR 辨識結果", value=text, height=200)
        except Exception as e:
            st.error(f"❌ OCR API 錯誤：{e}")

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
 