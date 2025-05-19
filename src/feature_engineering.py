import pandas as pd

def preparar_features(df):
    # Mapeia categorias conhecidas
    categoria_map = {'baixo': 0, 'médio': 1, 'alto': 2}
    df['categoria_risco_num'] = df['categoria_risco'].map(categoria_map)

    # Remove linhas com rótulos ausentes
    df = df.dropna(subset=['categoria_risco_num'])

    # Codifica tipo de solo
    df = pd.get_dummies(df, columns=['tipo_solo'], drop_first=True)

    # Separa preditores e rótulo
    X = df.drop(columns=['categoria_risco', 'categoria_risco_num', 'id_estacao', 'municipio', 'estado'])
    y = df['categoria_risco_num']

    return X, y
