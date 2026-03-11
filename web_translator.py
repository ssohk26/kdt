import streamlit as st
import torch
import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import os
import time
import io
import speech_recognition as sr
from gtts import gTTS
from deep_translator import GoogleTranslator
import pygame
import base64

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="한국어 실시간 번역기",
    page_icon="🎙️",
    layout="centered"
)

# ──────────────────────────────────────────────
# 스타일
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
    background-color: #0f0f1a;
    color: #eaeaea;
}

.main { background-color: #0f0f1a; }

.stApp { background-color: #0f0f1a; }

h1 {
    font-size: 2rem !important;
    color: #eaeaea !important;
    border-bottom: 2px solid #e94560;
    padding-bottom: 10px;
}

.subtitle {
    color: #7a7a9a;
    font-size: 0.8rem;
    font-family: 'Courier New', monospace;
    margin-top: -10px;
    margin-bottom: 20px;
}

.card {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.result-box {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 16px 20px;
    min-height: 100px;
    margin-top: 6px;
    font-family: 'Consolas', monospace;
    font-size: 1rem;
    color: #eaeaea;
    white-space: pre-wrap;
}

.label {
    font-family: 'Courier New', monospace;
    font-size: 0.75rem;
    color: #7a7a9a;
    font-weight: bold;
    margin-bottom: 4px;
}

.status-ok   { color: #4caf50; font-weight: bold; }
.status-err  { color: #e94560; font-weight: bold; }
.status-info { color: #7a9ae9; font-weight: bold; }

div[data-testid="stButton"] button {
    background-color: #e94560 !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
}
div[data-testid="stButton"] button:hover {
    background-color: #c73652 !important;
}

.stRadio label { color: #eaeaea !important; }
.stSlider label { color: #7a7a9a !important; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# VAD 모델 (캐싱)
# ──────────────────────────────────────────────
@st.cache_resource
def load_vad():
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        verbose=False
    )
    return model, utils

@st.cache_resource
def init_pygame():
    pygame.mixer.init()

# ──────────────────────────────────────────────
# 언어 설정
# ──────────────────────────────────────────────
LANG_OPTIONS = {
    "🇺🇸 영어":    {"trans": "en", "tts": "en"},
    "🇫🇷 프랑스어": {"trans": "fr", "tts": "fr"},
    "🇯🇵 일본어":  {"trans": "ja", "tts": "ja"},
}

SAMPLING_RATE = 16000
BLOCK_SIZE    = 512

# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.markdown("<h1>🎙️ 한국어 실시간 번역기</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">VAD · STT · TTS — 말하면 바로 번역!</p>', unsafe_allow_html=True)

# 언어 선택
st.markdown('<div class="card">', unsafe_allow_html=True)
lang_choice = st.radio(
    "번역 언어 선택",
    list(LANG_OPTIONS.keys()),
    horizontal=True,
    label_visibility="visible"
)
st.markdown('</div>', unsafe_allow_html=True)

# 녹음 시간
st.markdown('<div class="card">', unsafe_allow_html=True)
record_sec = st.slider("녹음 시간 (초)", min_value=5, max_value=30, value=10, step=1)
st.markdown('</div>', unsafe_allow_html=True)

# 녹음 버튼
start = st.button("⏺  녹음 시작")

# 상태창
status_box = st.empty()

# 결과 패널
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="label">🇰🇷 인식된 한국어</div>', unsafe_allow_html=True)
    kr_box = st.empty()
    kr_box.markdown('<div class="result-box"> </div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="label">🔤 번역된 텍스트</div>', unsafe_allow_html=True)
    out_box = st.empty()
    out_box.markdown('<div class="result-box"> </div>', unsafe_allow_html=True)

# 오디오 재생 영역
audio_area = st.empty()

# ──────────────────────────────────────────────
# 핵심 함수들
# ──────────────────────────────────────────────
def do_record(model, utils, seconds):
    _, _, _, VADIterator, _ = utils
    vad_iterator  = VADIterator(model)
    speech_buffer = []
    is_speaking   = False
    start_time    = time.time()

    with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, blocksize=BLOCK_SIZE) as stream:
        while True:
            chunk, _ = stream.read(BLOCK_SIZE)
            chunk    = chunk.flatten()
            tensor   = torch.from_numpy(chunk)
            sd_dict  = vad_iterator(tensor)

            if sd_dict:
                if "start" in sd_dict:
                    is_speaking = True
                    status_box.markdown('<p class="status-info">🎤 음성 감지됨!</p>', unsafe_allow_html=True)
                if "end" in sd_dict:
                    is_speaking = False

            if is_speaking:
                speech_buffer.extend(chunk)

            remaining = seconds - (time.time() - start_time)
            if not is_speaking:
                status_box.markdown(
                    f'<p class="status-info">🔴 녹음 중... 남은 시간: {max(0,int(remaining))}s</p>',
                    unsafe_allow_html=True
                )

            if time.time() - start_time > seconds:
                break

    audio = np.array(speech_buffer if speech_buffer else [0.0])
    tmp   = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, SAMPLING_RATE)
    return tmp.name


def do_stt(wav_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data, language='ko-KR')
    except sr.UnknownValueError:
        return ""


def do_translate(text, target):
    return GoogleTranslator(source='ko', target=target).translate(text)


def make_tts_b64(text, lang):
    """gTTS MP3 → base64 (브라우저 자동재생용)"""
    tts    = gTTS(text=text, lang=lang)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return base64.b64encode(mp3_fp.read()).decode()


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
if start:
    lang_info = LANG_OPTIONS[lang_choice]

    try:
        # 1. VAD 모델 로드
        status_box.markdown('<p class="status-info">⏳ 모델 로딩 중...</p>', unsafe_allow_html=True)
        model, utils = load_vad()

        # 2. 녹음
        status_box.markdown('<p class="status-info">🔴 녹음 중... 말씀해 주세요!</p>', unsafe_allow_html=True)
        wav_path = do_record(model, utils, record_sec)

        # 3. STT
        status_box.markdown('<p class="status-info">🔍 음성 인식 중...</p>', unsafe_allow_html=True)
        text = do_stt(wav_path)
        os.remove(wav_path)

        if not text:
            status_box.markdown('<p class="status-err">⚠️ 음성을 인식하지 못했습니다. 다시 시도해 주세요.</p>', unsafe_allow_html=True)
            st.stop()

        kr_box.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)

        # 4. 번역
        status_box.markdown('<p class="status-info">🌐 번역 중...</p>', unsafe_allow_html=True)
        result = do_translate(text, lang_info["trans"])

        if not result:
            status_box.markdown('<p class="status-err">⚠️ 번역 결과가 없습니다.</p>', unsafe_allow_html=True)
            st.stop()

        out_box.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)

        # 5. TTS → 브라우저 자동재생
        status_box.markdown('<p class="status-info">🔊 음성 생성 중...</p>', unsafe_allow_html=True)
        b64 = make_tts_b64(result, lang_info["tts"])

        audio_area.markdown(f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)

        status_box.markdown('<p class="status-ok">✅ 완료! 다시 녹음하려면 버튼을 누르세요.</p>', unsafe_allow_html=True)

    except Exception as e:
        status_box.markdown(f'<p class="status-err">❌ 오류: {e}</p>', unsafe_allow_html=True)
