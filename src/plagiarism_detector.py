# plagiarism_detector.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from joblib import dump

def cargar_datos(path, etiqueta):
    """
    Función para cargar datos desde archivos de texto y etiquetarlos.
    Args:
    path (str): La ruta al directorio que contiene los archivos.
    etiqueta (int): La etiqueta para los archivos en este directorio.
    
    Returns:
    list of tuples: Lista de tuplas, cada tupla contiene el texto del archivo y su etiqueta.
    """
    archivos = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                    text = file.read().strip()
            except UnicodeDecodeError:
                # Intentar con una codificación diferente si UTF-8 falla
                with open(os.path.join(path, filename), 'r', encoding='ISO-8859-1') as file:
                    text = file.read().strip()
            archivos.append((text, etiqueta))
    return archivos

def main():
    # Rutas a los directorios de datos
    path_genuinos = 'data/original/'
    path_plagiados = 'data/'

    # Cargar y etiquetar los datos
    datos_genuinos = cargar_datos(path_genuinos, 0)  # 0 para genuinos
    datos_plagiados = cargar_datos(path_plagiados, 1)  # 1 para plagiados

    # Crear un DataFrame con todos los datos
    df = pd.DataFrame(datos_genuinos + datos_plagiados, columns=['texto', 'etiqueta'])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(df['texto'], df['etiqueta'], test_size=0.2, random_state=42)

    # Configurar y aplicar el vectorizador TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Inicializar y entrenar el modelo de regresión logística
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Realizar predicciones y evaluar el modelo
    predictions = model.predict(X_test_tfidf)
    print("Informe de clasificación:\n", classification_report(y_test, predictions))
    
    # Usar validación cruzada para evaluar el modelo
    scores = cross_val_score(LogisticRegression(), X_train_tfidf, y_train, cv=5)
    print("Precisión promedio con validación cruzada:", scores.mean())

    # Guardar el modelo y el vectorizador
    dump(model, 'plagiarism_model.joblib')
    dump(vectorizer, 'tfidf_vectorizer.joblib')

if __name__ == "__main__":
    main()
