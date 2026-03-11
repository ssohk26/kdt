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