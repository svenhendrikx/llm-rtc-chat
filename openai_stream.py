from langchain_openai import OpenAI

llm = OpenAI()
for chunk in llm.stream("Write me a song about sparkling water."):
    print(chunk, end="", flush=True)
