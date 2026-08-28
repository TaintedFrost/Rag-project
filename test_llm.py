from ollama import chat


response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "Răspunde în limba română. Ce este un sistem RAG?"
        }
    ]
)

print(response.message.content)