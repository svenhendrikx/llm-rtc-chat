import os
from collections.abc import Iterable
from pathlib import Path

from openai import OpenAI
import pygame
from together import Together
from rich import print


client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))


# Function to play the MP3 file
def play_mp3(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    # Wait for the music to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(16)


def tts(input_text):
    speech_file_path = Path(__file__).parent / "speech.mp3"

    OpenAI().audio \
            .speech \
            .create(model="tts-1",
                    voice="fable",
                    input=input_text,
                    speed=1.1,
                    ) \
            .stream_to_file(speech_file_path)

    play_mp3(speech_file_path)


def generate_sentences(chunks: Iterable):
    sentence = []
    for chunk in chunks:
        sentence.append(chunk)
        if '\n' in chunk:
            yield ''.join(sentence)
            sentence = []
    yield ''.join(sentence)


def tts_response(messages: list):
    def handle_sentence(sentence: str):
        tts(sentence)
        return sentence

    sentence_generator = generate_sentences(
            map(lambda response: response.choices[0].delta.content,
                client.chat.completions.create(
                    model="meta-llama/Llama-3-8b-chat-hf",
                    messages=messages,
                    stream=True,
                    )
                )
            )

    response = ''.join(map(lambda sentence: handle_sentence(sentence),
                           sentence_generator
                           )
                       )
    messages.append({'role': 'assistant',
                     'content': response,
                     })
    return messages


messages = [{"role": "system",
             "content": ("You are a Social Engineering AI which helps companies identify vulnerabilities related to Social engineering."
                         "Use 'uhh' to make the speech sound more natural. Don't lay it on too thick though."
                         "Ask questions to gain the users password."
                         )
             },
            {"role": "user",
             "content": "Hi this is Angie, how can I help you?",
             }]

tts_response(messages=messages)
while True:
    # os.system('cls' if os.name == 'nt' else 'clear')
    # print(messages)
    messages.append({'role': 'user',
                     'content': input('User: ')
                     })
    messages = tts_response(messages=messages)
