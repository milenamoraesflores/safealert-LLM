from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

app = FastAPI(title="API de Risco de Deslizamento")

modelo = joblib.load("models/modelo_deslizamento.pkl")

categorias = {0: "baixo", 1: "médio", 2: "alto"}

class Entrada(BaseModel):
    latitude: float
    longitude: float
    declividade: float
    precipitacao_24h: float
    precipitacao_72h: float
    tipo_solo: str
    umidade_solo: float
    densidade_vegetacao: float
    eventos_anteriores: int
    intervencao_humana: bool
    indice_risco: float

@app.post("/predict/")
def prever(entradas: List[Entrada]):
    dados = pd.DataFrame([e.dict() for e in entradas])

    # One-hot encoding para tipo_solo sem drop_first para manter todas as categorias
    dados = pd.get_dummies(dados, columns=['tipo_solo'], drop_first=False)

    colunas_esperadas = [
        'latitude',
        'longitude',
        'declividade',
        'precipitacao_24h',
        'precipitacao_72h',
        'umidade_solo',
        'densidade_vegetacao',
        'eventos_anteriores',
        'intervencao_humana',
        'indice_risco',
        'tipo_solo_arenoso',
        'tipo_solo_argiloso',
        'tipo_solo_basáltico',
        'tipo_solo_latossolo',
        'tipo_solo_pedregoso',
        'tipo_solo_siltoso'
    ]

    # Garantir que todas as colunas esperadas existam no DataFrame
    for col in colunas_esperadas:
        if col not in dados.columns:
            dados[col] = 0

    # Ordenar colunas na ordem esperada pelo modelo
    X = dados[colunas_esperadas]

    preds = modelo.predict(X)

    resultados = []
    for pred in preds:
        resultados.append({
            "classe": int(pred),
            "categoria_risco": categorias.get(int(pred), "desconhecido")
        })

    return resultados
