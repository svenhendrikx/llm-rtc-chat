import sounddevice as sd
import pygame
import sys
import numpy as np
from scipy.io.wavfile import write
import whisper
from pathlib import Path
from openai import OpenAI


# Initialize Pygame
pygame.init()


# Record audio
def record_audio(duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float64')
    sd.wait()
    print("Recording stopped")
    return recording, fs

# Save the recording to a file
def save_recording(recording, fs, filename='output.wav'):
    write(filename, fs, recording)
    print(f"File saved as {filename}")

# Transcribe audio using OpenAI's Whisper
def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

# Example of recording for 5 seconds and transcribing it
def stt():
    recording, fs = record_audio(duration=5)
    save_recording(recording, fs)
    text = transcribe_audio('output.wav')
    return text



# Function to play the MP3 file
def play_mp3(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for the music to finish playing
        pygame.time.Clock().tick(16)


def tts(input_text):
    client = OpenAI()

    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
      model="tts-1",
      voice="echo",
      input=input_text,
    )

    response.stream_to_file(speech_file_path)
    play_mp3(speech_file_path)


# tts('Mijn naam is Nooitgedacht 1 en ik ben een auto gebouwd door Marcel Hendrikx')
