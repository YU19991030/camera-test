from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from paddleocr import PaddleOCR
from pydub import AudioSegment
from io import BytesIO
import base64, io, os, numpy as np, cv2
from PIL import Image
import tempfile
from faster_whisper import WhisperModel
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# PaddleOCR 初始化（繁體中文 + 自動轉向）
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# Whisper 模型初始化
device = "cpu"
model = WhisperModel("base", device=device, compute_type="float32")
print(f"🚀 faster-whisper 使用設備：{device}")

last_sentences = []

# 📄 圖片 OCR 辨識 API
@app.route("/ocr", methods=["POST"])
def ocr_api():
    try:
        data = request.get_json()
        base64_str = data["image"].split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        result = ocr.ocr(img, cls=True)
        text = "\n".join([line[1][0] for box in result for line in box])

        print("📄 辨識結果：", text)
        return jsonify({"text": text})
    except Exception as e:
        print("❌ 辨識錯誤：", e)
        return jsonify({"text": "", "error": str(e)})

# 🔊 Whisper 語音辨識 WebSocket 接口
@socketio.on("start_recording")
def handle_start():
    print("🎬 開始錄音")

@socketio.on("stop_recording")
def handle_stop():
    print("🛑 停止錄音")

@socketio.on("audio")
def handle_audio(data):
    try:
        audio_bytes = BytesIO(data["audio"])
        audio = AudioSegment.from_file(audio_bytes)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_filename = tmp.name
            audio.export(wav_filename, format="wav")

        print(f"🎧 收到音訊：{wav_filename}")

        segments, _ = model.transcribe(
            wav_filename,
            language="zh",
            beam_size=5,
            vad_filter=True,
        )

        new_sentences = []
        for seg in segments:
            sentence = seg.text.strip()
            new_sentences.append(sentence)
            last_sentences.append(sentence)

        final_text = new_sentences[-1] if new_sentences else "(無辨識到語音)"

        if not os.path.exists("transcriptions"):
            os.makedirs("transcriptions")
        with open(os.path.join("transcriptions", "transcription.txt"), "w", encoding="utf-8") as f:
            f.write(final_text)

        emit("transcription", {
            "text": final_text,
            "download_url": "/download/transcription.txt"
        })

        os.remove(wav_filename)

    except Exception as e:
        print("❌ 語音辨識錯誤：", e)
        emit("transcription", {"text": f"(錯誤：{str(e)})"})

@socketio.on("clear_transcription")
def clear_transcription():
    global last_sentences
    last_sentences = []
    emit("transcription", "")

# 📄 提供下載辨識結果
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(directory="transcriptions", path=filename, as_attachment=True)

# ✅ 啟動伺服器
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000)
