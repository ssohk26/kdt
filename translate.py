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

translator = Translator(to_lang='ja', from_lang='ko')

# 번역 실행
result = translator.translate(text) # 한국어를 영어로 번역)

print('원본텍스트 :', text)
print('번역된 텍스트 :', result)