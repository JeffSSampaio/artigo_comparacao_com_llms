import csv
import os

def atualizar_planilha( nomeArquivo, dadosAtualizados):

 if not os.path.isfile(nomeArquivo):
    with open(nomeArquivo, mode='w', newline='', encoding='utf-8') as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(['Momento da Execução', 'Modelo' ,'Prompt', 'Resposta', 'Tempo de Resposta', 'BLEU', 'METEOR', 'ROUGE1', 'ROUGE2', 'ROUGEL'])

        for linha in dadosAtualizados:
           escritor.writerow(linha)
    print(f"Arquivo {nomeArquivo} criado com sucesso.")
 else:
      with open(nomeArquivo, mode='a', newline='', encoding='utf-8') as arquivo:
        escritor = csv.writer(arquivo)
        for linha in dadosAtualizados:
           escritor.writerow(linha)
        print(f"No Arquivo {nomeArquivo} dados foram atualizados.")