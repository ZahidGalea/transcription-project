import datetime
import os

import sounddevice as sd
import whisper
from numpy import ndarray
from pyannote.core import SlidingWindowFeature
from scipy.io.wavfile import write

# Configura el dispositivo de grabación
SAMPLE_RATE = 44100  # Frecuencia de muestreo


def get_query_device(nombre_dispositivo):
	"""
	Obtén el ID del dispositivo de entrada virtual
	:param nombre_dispositivo: 
	:return: int
	"""

	dispositivos = sd.query_devices()
	for i in range(len(dispositivos)):
		if dispositivos[i]['name'] == nombre_dispositivo:
			return i


def mezcla_estero_record(duration=10, record_timestamp=datetime.datetime.now()):
	record_name = f'recording-{record_timestamp.strftime("%Y%m%d%H%M%S")}.wav'
	# Nombre del dispositivo de entrada virtual
	nombre_dispositivo = 'Mezcla Estereo Manual (Realtek(R) Audio)'
	dispositivo_id = get_query_device(nombre_dispositivo)

	recording = sd.rec(frames=int(duration * SAMPLE_RATE),
					   device=dispositivo_id,
					   channels=2,
					   samplerate=SAMPLE_RATE)
	sd.wait()
	write(filename=record_name, rate=SAMPLE_RATE, data=recording)
	return record_name


def transcript(audio_file=None, language='es', prompt=None):
	model = whisper.load_model("base", download_root=os.getcwd())
	audio = whisper.load_audio(audio_file)
	audio = whisper.pad_or_trim(audio)

	_transcription = model.transcribe(audio=audio, language=language, prompt=prompt)
	return _transcription


def diarization(_record_file: str) -> SlidingWindowFeature | ndarray:
	from pyannote.audio import Model
	model = Model.from_pretrained("../dependencies/pytorch_segmentation_model.bin")
	from pyannote.audio import Inference
	inference = Inference(model, step=2.5)
	output = inference(_record_file)
	return output


if __name__ == '__main__':
	raise Exception('Not implemented Yet.')
	from dotenv import load_dotenv

	load_dotenv()
	record_file = mezcla_estero_record(duration=30)
	diarization = diarization(record_file)
	for segment in diarization:
		transcription = transcript(segment[1])
		print(transcription)
