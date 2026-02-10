from gpt4all import GPT4All

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="gpu")  # ou "cpu"

# Resposta mais "segura" e focada (pouca criatividade)
resposta = model.generate(
    "O que é RAG em IA?",
    max_tokens=600,
    temp=0.2,
    top_p=0.85,
    top_k=30,
    repeat_penalty=1.1
)
print(resposta)

# Resposta mais criativa
resposta = model.generate(
    "Escreva uma história curta de ficção científica",
    max_tokens=800,
    temp=1.0,
    top_p=0.92,
    top_k=50,
    repeat_penalty=1.15
)

# Streaming (imprime token por token)
with model.chat_session():
    print("Resposta: ", end="", flush=True)
    for token in model.generate(
        "Explique o funcionamento de um transformer em detalhes",
        max_tokens=1200,
        temp=0.7,
        streaming=True
    ):
        print(token, end="", flush=True)
    print()