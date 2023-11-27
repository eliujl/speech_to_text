from speech_capture import SpeechCapture

def test_speech_capture():
    # Initialize the transcriber
    transcriber = SpeechCapture(model_size='tiny')

    # Test audio recording
    audio_data = transcriber.record_audio(duration=5)

    # Test transcription without saving
    print("\n--- Test transcription without saving ---")
    transcription = transcriber.transcribe(audio_data)
    print(transcription)

    # Test saving audio to MP3
    print("\n--- Test saving audio to MP3 ---")
    transcriber.save_audio_to_mp3(audio_data, "test_audio.mp3")

    # Test transcription with saving audio and text
    print("\n--- Test transcription with saving audio and text ---")
    transcription = transcriber.transcribe(
        audio_data,
        save_audio=True,
        audio_filename="test_audio.mp3",
        save_text=True,
        text_filename="test_transcription.txt"
    )

if __name__ == "__main__":
    test_speech_capture()
