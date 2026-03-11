
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
from pydub import AudioSegment
from deep_translator import GoogleTranslator

# 번역
result = GoogleTranslator(source='ko', target='en').translate(text)

print('원본 텍스트:', text)
print('번역된 텍스트:', result)

# TTS 출력
if result:
    tts = gTTS(text=result, lang='en')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    audio_segment = AudioSegment.from_mp3(mp3_fp)
    wav_fp = io.BytesIO()
    audio_segment.export(wav_fp, format="wav")
    wav_fp.seek(0)

    data, samplerate = sf.read(wav_fp)
    print("스피커로 출력 중...")
    sd.play(data, samplerate)
    sd.wait()
    print("출력 완료")
else:
    print("번역 결과가 비어있습니다.")