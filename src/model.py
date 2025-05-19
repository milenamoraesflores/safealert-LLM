from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def treinar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    return classification_report(y_test, y_pred)

def salvar_modelo(modelo, caminho='models/modelo_deslizamento.pkl'):
    joblib.dump(modelo, caminho)

def carregar_modelo(caminho='models/modelo_deslizamento.pkl'):
    return joblib.load(caminho)
