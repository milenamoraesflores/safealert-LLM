import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import carregar_dados_json
from src.feature_engineering import preparar_features
from src.model import treinar_modelo, avaliar_modelo, salvar_modelo

# Caminho para o JSON de entrada
caminho_dados = 'data/deslizamento_brasil.json'

# 1. Carregar os dados
df = carregar_dados_json(caminho_dados)

# 2. Preparar as features e o rótulo
X, y = preparar_features(df)

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinar o modelo
modelo = treinar_modelo(X_train, y_train)

# 5. Avaliar o modelo
relatorio = avaliar_modelo(modelo, X_test, y_test)
print('📊 Relatório de Classificação:\n', relatorio)

# 6. Salvar o modelo
salvar_modelo(modelo)
print('✅ Modelo salvo em models/modelo_deslizamento.pkl')
