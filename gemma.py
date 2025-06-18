import lmstudio as lms
import time as time

modelo = lms.llm("google/gemma-2-9b")

prompt = input("Prompt:")

inicio = time.perf_counter()

resposta = modelo.respond(prompt)

fim = time.perf_counter()

tempoResposta = fim - inicio

print(f"Modelo gemma-2-9b\nPrompt: {prompt}\nResposta do modelo: {resposta}\nTempo de Resposta: {tempoResposta:.2f} segundos")
