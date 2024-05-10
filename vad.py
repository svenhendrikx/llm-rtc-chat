import pyaudio
import webrtcvad
import whisper
from pydub import AudioSegment
# from faster_whisper import WhisperModel
import numpy as np


vad = webrtcvad.Vad()
model_size = "tiny"
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

def byte_to_float(audio_data, sample_width):
    if sample_width == 2:  # Assuming 16-bit audio
        data_int = np.frombuffer(audio_data, dtype=np.int16)  # Convert byte data to int16
        data_float = data_int.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
        return data_float
    else:
        raise ValueError("This function only supports 16-bit audio.")


def main():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)

    vad = webrtcvad.Vad()
    vad.set_mode(0)  # Mode can be 0 (normal), 1, or 2 (most aggressive).

    # Record as long as there's speech
    print("Start speaking")
    max_compliance = 30
    compliance = 0
    frames = []
    while True:
        # The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz. A frame must be either 10, 20, or 30 ms in duration:
        sample_rate = 16000
        frame = stream.read(480)
        
        import ipdb; ipdb.set_trace()
        is_speech = vad.is_speech(frame, sample_rate)

        if not is_speech:
            print('no')
            compliance += 1

            if compliance >= max_compliance:
                print("No speech detected, stopping recording")
                break
        else:
            print('yes')
            compliance = 0
            frames.append(frame)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    # Convert the list of frames to a byte array
    audio_data = b''.join(frames)


    # # Run on GPU with FP16
    # transcription = model.transcribe(byte_to_float(audio_data=audio_data, sample_width=2),
    #                                  beam_size=5,
    #                                  )
    # print(transcription)



if __name__ == "__main__":
    main()

