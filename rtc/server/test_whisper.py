from faster_whisper import WhisperModel

model_size = "distil-small.en"

# Run on GPU with FP16
device = 'cpu'
compute_type = 'int8' if device == 'cpu' else 'float16'

model = WhisperModel(model_size,
                     device=device,
                     compute_type=compute_type,
                     )
transcription_gen = model.transcribe("demo_instruct.wav")[0]
print('surw')

for x in transcription_gen:
    print(x)
