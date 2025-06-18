import lmstudio as lms
import time as time

modelo = lms.llm("meta-llama-3.1-8b-instruct")

prompt = input("Prompt: ")

inicio = time.perf_counter()

resposta = modelo.respond(prompt)

fim = time.perf_counter()

tempoResposta = fim - inicio

print(f"Modelo meta-llama-3.1-8b-instruct \nPrompt: {prompt}\nResposta do modelo: {resposta}\nTempo de Resposta: {tempoResposta:.2f} segundos")