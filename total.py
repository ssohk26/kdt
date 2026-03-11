import torch
import sounddevice as sd
import numpy as np
import soundfile as sf
import time

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model ='silero_vad'
)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

vad_iterator = VADIterator(model)

SAMPLING_RATE = 16000
BLOCK_SIZE = 512
RECORD_SECONDS = 10

speech_buffer = []
is_speaking = False

print('녹음 시작')

start_time = time.time()

with sd.InputStream(
    samplerate = SAMPLING_RATE,
    channels = 1,
    blocksize=BLOCK_SIZE
) as stream:
    
    while True:

        audio_chunk, _ = stream.read(BLOCK_SIZE)
        audio_chunk = audio_chunk.flatten()

        audio_tensor = torch.from_numpy(audio_chunk)

        speech_dict = vad_iterator(audio_tensor)

        if speech_dict:

            if"start" in speech_dict:
                is_speaking = True
                print(" speech start")

            if "end" in speech_dict:
                is_speaking = False
                print("  speech end")

        if is_speaking:
            speech_buffer.extend(audio_chunk)

        if time.time() - start_time > RECORD_SECONDS:
            break
print("녹음 종료")

speech_audio = np.array(speech_buffer)

sf.write(
    "vad_recorded.wav",
    speech_audio,
    SAMPLING_RATE
)

print("파일 저장 종료")

import speech_recognition as sr
from IPython.display import display, Javascript
from base64 import b64decode
from io import BytesIO
from pydub import AudioSegment
from translate import Translator

# 음성 인식 객체 생성
recognizer = sr.Recognizer()

# 오디오 파일 로드
audio_file = 'vad_recorded.wav'

with sr.AudioFile(audio_file) as source:
    print("음성을 인식 중입니다....")
    audio_data = recognizer.record(source)
  
text = recognizer.recognize_google(audio_data, language='ko-KR')
print("변환된 텍스트: ", text)

translator = Translator(to_lang='en', from_lang='ko')

# 번역 실행
result = translator.translate(text) # 한국어를 영어로 번역)


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