import sounddevice as sd
from pydub import AudioSegment
import numpy as np
import whisper

class SpeechCapture:
    def __init__(self, model_size='tiny'):
        self.model_size = model_size
        self.model = whisper.load_model(model_size)

    def record_audio(self, duration=5, fs=16000):
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        print("Recording finished")
        return recording.flatten()  # Ensure the audio is a 1D array


    def save_audio_to_mp3(self, audio, filename, fs=16000):
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        audio_segment = AudioSegment(
            data=audio_bytes,
            sample_width=2,
            frame_rate=fs,
            channels=1
        )
        audio_segment.export(filename, format='mp3')
        print(f"Audio saved to {filename}")

    def transcribe(self, audio, save_audio=False, audio_filename="recorded_audio.mp3", 
                   save_text=False, text_filename="transcription.txt"):
        if save_audio:
            self.save_audio_to_mp3(audio, audio_filename)
        
        audio = audio.astype(np.float32)  # Ensure audio data type is float32
        if audio.ndim != 1:
            raise ValueError("Audio must be a 1D array")
        
        result = self.model.transcribe(audio_filename if save_audio else audio)
        transcription = result['text']
        print("Transcription:", transcription)

        if save_text:
            with open(text_filename, "w") as text_file:
                text_file.write(transcription)
            print(f"Transcription saved to {text_filename}")

        return transcription
