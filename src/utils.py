import matplotlib.pyplot as plt

def plotar_importancia_features(modelo, feature_names):
    importances = modelo.feature_importances_
    indices = importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('Importância das Features')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.show()
