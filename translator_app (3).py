import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import torch
import sounddevice as sd
import numpy as np
import soundfile as sf
import time
import io
import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
from deep_translator import GoogleTranslator
import pygame

# ──────────────────────────────────────────────
# pygame 초기화
# ──────────────────────────────────────────────
pygame.mixer.init()

# ──────────────────────────────────────────────
# VAD 모델 로드
# ──────────────────────────────────────────────
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    verbose=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

SAMPLING_RATE = 16000
BLOCK_SIZE    = 512
WAV_PATH      = "vad_recorded.wav"

# 언어 설정
LANG_OPTIONS = {
    "🇺🇸 영어":    {"trans": "en", "tts": "en", "flag": "🇺🇸"},
    "🇫🇷 프랑스어": {"trans": "fr", "tts": "fr", "flag": "🇫🇷"},
    "🇯🇵 일본어":  {"trans": "ja", "tts": "ja", "flag": "🇯🇵"},
}


# ══════════════════════════════════════════════
#  메인 앱
# ══════════════════════════════════════════════
class TranslatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎙️ 한국어 실시간 번역기")
        self.root.geometry("740x660")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f0f1a")

        self.is_recording   = False
        self.record_seconds = tk.IntVar(value=10)
        self.target_lang    = tk.StringVar(value=list(LANG_OPTIONS.keys())[0])

        self._build_ui()

    # ─────────────────────────────────────────
    #  UI 구성
    # ─────────────────────────────────────────
    def _build_ui(self):
        BG         = "#0f0f1a"
        CARD       = "#1a1a2e"
        ACCENT     = "#e94560"
        ACCENT2    = "#0f3460"
        TEXT       = "#eaeaea"
        MUTED      = "#7a7a9a"
        FONT_TITLE = ("Georgia", 22, "bold")
        FONT_LABEL = ("Courier New", 10, "bold")
        FONT_TEXT  = ("Consolas", 11)
        FONT_BTN   = ("Georgia", 13, "bold")

        # ── 타이틀 ──────────────────────────────
        title_frame = tk.Frame(self.root, bg=BG)
        title_frame.pack(fill="x", pady=(24, 4), padx=30)

        tk.Label(title_frame, text="한국어 실시간 번역기",
                 font=FONT_TITLE, bg=BG, fg=TEXT).pack(side="left")
        tk.Label(title_frame, text="VAD · STT · TTS",
                 font=("Courier New", 9), bg=BG, fg=MUTED).pack(side="right", anchor="s", pady=6)

        sep = tk.Frame(self.root, bg=ACCENT, height=2)
        sep.pack(fill="x", padx=30, pady=(0, 16))

        # ── 언어 선택 ───────────────────────────
        lang_frame = tk.Frame(self.root, bg=CARD)
        lang_frame.pack(fill="x", padx=30, pady=(0, 10))
        lang_frame.configure(height=60)
        lang_frame.pack_propagate(False)

        tk.Label(lang_frame, text="번역 언어", font=FONT_LABEL,
                 bg=CARD, fg=MUTED).pack(side="left", padx=16, pady=16)

        for lang_name in LANG_OPTIONS.keys():
            rb = tk.Radiobutton(
                lang_frame, text=lang_name,
                variable=self.target_lang, value=lang_name,
                font=("Courier New", 11, "bold"), bg=CARD, fg=TEXT,
                selectcolor=ACCENT2, activebackground=CARD,
                activeforeground=TEXT, cursor="hand2",
                command=self._on_lang_change
            )
            rb.pack(side="left", padx=14)

        # ── 녹음 시간 슬라이더 ───────────────────
        ctrl_frame = tk.Frame(self.root, bg=CARD)
        ctrl_frame.pack(fill="x", padx=30, pady=(0, 12))
        ctrl_frame.configure(height=58)
        ctrl_frame.pack_propagate(False)

        tk.Label(ctrl_frame, text="녹음 시간(초)", font=FONT_LABEL,
                 bg=CARD, fg=MUTED).pack(side="left", padx=16)

        self.sec_label = tk.Label(ctrl_frame, text="10s",
                                  font=("Courier New", 12, "bold"),
                                  bg=CARD, fg=ACCENT, width=4)
        self.sec_label.pack(side="right", padx=16)

        slider = ttk.Scale(ctrl_frame, from_=5, to=30,
                           variable=self.record_seconds,
                           orient="horizontal",
                           command=self._on_slider)
        slider.pack(side="left", fill="x", expand=True, padx=8)

        # ── 녹음 버튼 ───────────────────────────
        self.rec_btn = tk.Button(
            self.root, text="⏺  녹음 시작",
            font=FONT_BTN, bg=ACCENT, fg="white",
            activebackground="#c73652", activeforeground="white",
            bd=0, padx=20, pady=12, cursor="hand2",
            command=self._toggle_record
        )
        self.rec_btn.pack(pady=(0, 12))

        # ── 상태 표시 ───────────────────────────
        self.status_var = tk.StringVar(value="대기 중... 녹음 시작 버튼을 누르세요!")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              font=("Courier New", 10), bg=ACCENT2,
                              fg=TEXT, anchor="w", padx=12, pady=6)
        status_bar.pack(fill="x", padx=30, pady=(0, 10))

        # ── 텍스트 패널 ─────────────────────────
        panels = tk.Frame(self.root, bg=BG)
        panels.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)

        def make_panel(parent, label, col_idx):
            col = tk.Frame(parent, bg=BG)
            col.grid(row=0, column=col_idx, sticky="nsew",
                     padx=(0, 8 if col_idx == 0 else 0))

            tk.Label(col, text=label, font=FONT_LABEL,
                     bg=BG, fg=MUTED, anchor="w").pack(fill="x", pady=(0, 4))

            box = scrolledtext.ScrolledText(
                col, font=FONT_TEXT, bg=CARD, fg=TEXT,
                insertbackground=TEXT, relief="flat",
                wrap="word", bd=0, padx=10, pady=10,
                state="disabled", height=10
            )
            box.pack(fill="both", expand=True)
            return box

        self.kr_box  = make_panel(panels, "🇰🇷  인식된 한국어", 0)
        self.out_box = make_panel(panels, "🔤  번역된 텍스트",  1)

        self._translated_text = ""

    # ─────────────────────────────────────────
    #  콜백
    # ─────────────────────────────────────────
    def _on_lang_change(self):
        lang = self.target_lang.get()
        flag = LANG_OPTIONS[lang]["flag"]
        self._write_box(self.out_box, "")
        self._set_status(f"{flag}  번역 언어 변경됨 → {lang}")

    def _on_slider(self, val):
        self.sec_label.config(text=f"{int(float(val))}s")

    # ─────────────────────────────────────────
    #  녹음 토글
    # ─────────────────────────────────────────
    def _toggle_record(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.rec_btn.config(text="⏳  녹음 중...", state="disabled")
        self._clear_boxes()
        threading.Thread(target=self._record_pipeline, daemon=True).start()

    # ─────────────────────────────────────────
    #  전체 파이프라인
    # ─────────────────────────────────────────
    def _record_pipeline(self):
        try:
            lang_key  = self.target_lang.get()
            lang_info = LANG_OPTIONS[lang_key]
            flag      = lang_info["flag"]

            # 1. 녹음 + VAD
            self._set_status("🔴  녹음 중... (말씀해 주세요)")
            self._do_record()

            # 2. STT
            self._set_status("🔍  음성 인식 중...")
            text = self._do_stt()
            if not text:
                self._set_status("⚠️  음성을 인식하지 못했습니다. 다시 시도해 주세요.")
                return
            self._write_box(self.kr_box, text)

            # 3. 번역
            self._set_status(f"🌐  {flag} 번역 중...")
            result = GoogleTranslator(
                source='ko', target=lang_info["trans"]
            ).translate(text)
            if not result:
                self._set_status("⚠️  번역 결과가 없습니다.")
                return
            self._translated_text = result
            self._write_box(self.out_box, result)

            # 4. TTS 생성 → 자동 재생 (pygame, ffmpeg 불필요!)
            self._set_status(f"🔊  {flag} 음성 생성 중...")
            self._play_tts_pygame(result, lang_info["tts"], flag)

            self._set_status("✅  완료! 다시 녹음하려면 버튼을 누르세요.")

        except Exception as e:
            self._set_status(f"❌  오류: {e}")
        finally:
            self.is_recording = False
            self.root.after(0, lambda: self.rec_btn.config(
                text="⏺  녹음 시작", state="normal"))

    # ─────────────────────────────────────────
    #  녹음 (VAD)
    # ─────────────────────────────────────────
    def _do_record(self):
        vad_iterator  = VADIterator(model)
        speech_buffer = []
        is_speaking   = False
        seconds       = self.record_seconds.get()
        start_time    = time.time()

        with sd.InputStream(samplerate=SAMPLING_RATE,
                            channels=1,
                            blocksize=BLOCK_SIZE) as stream:
            while True:
                chunk, _ = stream.read(BLOCK_SIZE)
                chunk    = chunk.flatten()
                tensor   = torch.from_numpy(chunk)
                sd_dict  = vad_iterator(tensor)

                if sd_dict:
                    if "start" in sd_dict:
                        is_speaking = True
                        self._set_status("🎤  음성 감지됨!")
                    if "end" in sd_dict:
                        is_speaking = False

                if is_speaking:
                    speech_buffer.extend(chunk)

                remaining = seconds - (time.time() - start_time)
                if not is_speaking:
                    self._set_status(f"🔴  녹음 중... 남은 시간: {max(0, int(remaining))}s")

                if time.time() - start_time > seconds:
                    break

        audio = np.array(speech_buffer if speech_buffer else [0.0])
        sf.write(WAV_PATH, audio, SAMPLING_RATE)

    # ─────────────────────────────────────────
    #  STT
    # ─────────────────────────────────────────
    def _do_stt(self) -> str:
        recognizer = sr.Recognizer()
        with sr.AudioFile(WAV_PATH) as source:
            audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language='ko-KR')
        except sr.UnknownValueError:
            return ""

    # ─────────────────────────────────────────
    #  TTS 재생 (pygame 사용 - ffmpeg 불필요!)
    # ─────────────────────────────────────────
    def _play_tts_pygame(self, text: str, lang: str, flag: str):
        # gTTS로 MP3 생성 → 임시파일에 저장
        tts = gTTS(text=text, lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        tmp.close()

        # pygame으로 재생 (ffmpeg 필요 없음!)
        self._set_status(f"🔊  {flag} 번역 음성 재생 중...")
        pygame.mixer.music.load(tmp.name)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # 임시파일 삭제
        pygame.mixer.music.unload()
        os.remove(tmp.name)

    # ─────────────────────────────────────────
    #  헬퍼
    # ─────────────────────────────────────────
    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _write_box(self, box: scrolledtext.ScrolledText, text: str):
        def _w():
            box.config(state="normal")
            box.delete("1.0", "end")
            box.insert("end", text)
            box.config(state="disabled")
        self.root.after(0, _w)

    def _clear_boxes(self):
        self._write_box(self.kr_box,  "")
        self._write_box(self.out_box, "")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = TranslatorApp(root)
    root.mainloop()
