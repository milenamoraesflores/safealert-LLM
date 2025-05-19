# API de Risco de Deslizamento

Este projeto contém uma API REST desenvolvida com FastAPI para previsão de risco de deslizamento com base em dados ambientais.

## Requisitos
- Python 3.11
- pip para instalar pacotes Python

## Setup do ambiente

### Clone o repositório
```bash
git clone <url-do-repositorio>
cd <nome-da-pasta>
```

### Crie um ambiente virtual com Python 3.11
```bash
py -3.11 -m venv venv
```

### Ative o ambiente virtual
No Windows:
```bash
venv\Scripts\activate
```

No Linux/Mac:
```bash
source venv/bin/activate
```

### Instale as dependências
```bash
pip install -r requirements.txt
```

## Como rodar a API
Com o ambiente ativado, execute:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
- `--host 0.0.0.0` permite acesso externo (se desejar)
- `--port 8080` define a porta para 8080
- `--reload` ativa o auto-reload para facilitar o desenvolvimento

A API estará disponível em: http://localhost:8080

## Testando a API
A API possui um endpoint `/predict/` que recebe uma lista de entradas para previsão.

Exemplo de payload JSON para o POST:
```json
[
  {
    "latitude": -23.55052,
    "longitude": -46.633308,
    "declividade": 15.2,
    "precipitacao_24h": 12.0,
    "precipitacao_72h": 35.4,
    "tipo_solo": "latossolo",
    "umidade_solo": 30.5,
    "densidade_vegetacao": 0.7,
    "eventos_anteriores": 1,
    "intervencao_humana": true,
    "indice_risco": 0.45
  }
]
```

## Dependências principais
- fastapi
- uvicorn
- pandas
- scikit-learn
- joblib
- pydantic

## Observações
- Certifique-se de que o arquivo `modelo_deslizamento.pkl` está no caminho correto `models/`.
- Ajuste o nome do arquivo `main.py` caso seu arquivo principal tenha outro nome.