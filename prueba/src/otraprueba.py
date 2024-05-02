import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk

# Descargar recursos de NLTK si no están descargados
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Función para preprocesar texto
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    return ' '.join(lemmatized_tokens)

# Cargar datos del archivo Excel
def load_excel_data(excel_path):
    df = pd.read_excel(excel_path)
    return df['original_text'], df['copy_text'], df['copy_type']

# Entrenar y evaluar el modelo Naive Bayes
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Vectorizar los textos
    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    # Entrenar el clasificador Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train_vect, y_train)
    
    # Evaluar el clasificador
    accuracy = clf.score(X_test_vect, y_test)
    return clf, vectorizer, accuracy

# Función principal
def main():
    # Rutas de los directorios y del archivo Excel
    originals_path = "data/original"
    copies_path = "data/copias"
    excel_path = "data/resultado.xlsx"
    
    # Cargar archivos originales y de plagio
    originals = []
    copies = []
    for filename in os.listdir(originals_path):
        with open(os.path.join(originals_path, filename), 'r', encoding='latin-1') as file:
            text = file.read()
            originals.append((filename, text))
    for filename in os.listdir(copies_path):
        with open(os.path.join(copies_path, filename), 'r', encoding='latin-1') as file:
            text = file.read()
            copies.append((filename, text))
    
    # Procesar textos originales y de plagio
    preprocessed_originals = [(name, preprocess_text(text)) for name, text in originals]
    preprocessed_copies = [(name, preprocess_text(text)) for name, text in copies]
    
    # Cargar datos del archivo Excel
    original_texts, copy_texts, copy_types = load_excel_data(excel_path)
    
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(original_texts + copy_texts, copy_types, test_size=0.2, random_state=42)
    
    # Entrenar y evaluar el modelo de detección de tipo de plagio
    clf, vectorizer, accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print("Accuracy of the Naive Bayes classifier:", accuracy)
    
    # Comparar cada archivo de plagio con cada archivo original
    for copy_name, copy_text in preprocessed_copies:
        for original_name, original_text in preprocessed_originals:
            # Detectar el tipo de plagio
            copy_type = clf.predict(vectorizer.transform([original_text, copy_text]))
            
            # Calcular si es plagio
            is_copy = True if any(copy_type) else False
            
            # Agregar resultados al DataFrame
            df_results = pd.DataFrame({
                'original_name': original_name,
                'original_text': original_text,
                'copy_name': copy_name,
                'copy_text': copy_text,
                'is_copy': is_copy,
                'copy_type': copy_type
            })
            
            # Imprimir resultados en la consola
            # print(f"Archivo copia: {copy_name}")
            # print(f"Archivo original: {original_name}")
            # print(f"¿Es plagio?: {'Si' if is_copy else 'No'}")
            # print(f"Tipo de plagio: {copy_type}")
            # print("------------------------------------------")
    
    # Guardar resultados en un archivo de texto
    df_results.to_csv('plagiarism_results.txt', sep='\t', index=False)
    
    # Guardar resultados en un archivo Excel
    df_results.to_excel('plagiarism_results.xlsx', index=False)

if __name__ == "__main__":
    main()
