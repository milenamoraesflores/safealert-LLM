import pandas as pd
import os

def carregar_dados_json(caminho='data/deslizamento_brasil.json'):
    """
    Lê um arquivo JSON contendo os dados de deslizamento e retorna um DataFrame.
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    
    try:
        df = pd.read_json(caminho, orient='records')
        return df
    except ValueError as e:
        raise ValueError(f"Erro ao ler JSON: {e}")
