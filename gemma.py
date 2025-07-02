import lmstudio as lms
import time as time
import datetime as date
import nltk.downloader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  #type: ignore
from nltk.translate.meteor_score import meteor_score #type: ignore
from rouge_score import rouge_scorer  #type: ignore
import nltk
from planilha import atualizar_planilha

if not nltk.download('wordnet', quiet=True):
    nltk.download('wordnet')

if not nltk.download('omw-1.4', quiet=True):
    nltk.download('omw-1.4')


modelo = lms.llm("google/gemma-2-9b")

dados = []

with open("prompts.txt", "r", encoding= "utf-8") as file:
   prompts = [linha.strip() for linha in file.readlines()]

with open("gabarito.txt", "r", encoding= "utf-8") as file:
   gabarito = [linha.strip() for linha in file.readlines()]

rouge_score = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for i,prompt in enumerate(prompts):

inicio = time.perf_counter()

resposta = modelo.respond(prompt)

fim = time.perf_counter()

tempoResposta = fim - inicio

print(f"Modelo gemma-2-9b\nPrompt: {prompt}\nResposta do modelo: {resposta}\nTempo de Resposta: {tempoResposta:.2f} segundos")
