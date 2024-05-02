# import csv
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# import scipy.sparse
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# import os

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# def preprocess_text(text):
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
#     stemmer = PorterStemmer()
#     lemmatizer = WordNetLemmatizer()
#     stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
#     return ' '.join(lemmatized_tokens)

# def calculate_similarity(text1, text2):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
#     similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
#     return similarity_score

# def load_originals(originals_path):
#     originals = []
#     for filename in os.listdir(originals_path):
#         with open(os.path.join(originals_path, filename), 'r', encoding='latin-1') as file:
#             text = file.read()
#             originals.append((filename, text))  # Guarda el nombre del archivo junto con el texto
#     return originals

# def load_copies(copies_path):
#     copies = []
#     for filename in os.listdir(copies_path):
#         with open(os.path.join(copies_path, filename), 'r', encoding='latin-1') as file:
#             text = file.read()
#             copies.append((filename, text))  # Guarda el nombre del archivo junto con el texto
#     return copies

# def extract_features(texts):
#     tfidf_vectorizer = TfidfVectorizer()
#     features = tfidf_vectorizer.fit_transform(texts)
#     return features

# def train_model(features_originals, features_copies, labels):
#     features_combined = scipy.sparse.vstack([features_originals, features_copies])
#     num_originals = features_originals.shape[0]
#     num_copies = features_copies.shape[0]
#     labels_combined = [0] * num_originals + [1] * num_copies
#     model = SVC(kernel='linear')
#     model.fit(features_combined, labels_combined)
#     return model

# def detect_plagiarism(model, features_copies):
#     predictions = model.predict(features_copies)
#     return predictions



# def detect_paraphrasing(original_text, copy_text):
#     # Aquí implementa tu lógica para detectar parafraseo
#     return False

# # def detect_plagiarism_type(original_text, copy_text):
# #     similarity_score = calculate_similarity(original_text, copy_text)
# #     plagiarism_percentage = round(similarity_score * 100, 2)

# #     if plagiarism_percentage > 0.10 :
# #         if detect_paraphrasing(original_text, copy_text):
# #             return "Parafraseo", plagiarism_percentage
# #         elif detect_time_change(original_text, copy_text):
# #             return "Cambio de tiempo", plagiarism_percentage
# #         elif detect_voice_change(original_text, copy_text):
# #             return "Cambio de voz", plagiarism_percentage
# #         elif detect_inserted_phrases(original_text, copy_text):
# #             return "Insertar o reemplazar frases", plagiarism_percentage
# #         elif detect_disordered_phrases(original_text, copy_text):
# #             return "Desordenar frases", plagiarism_percentage
# #     else:
# #         return "Ninguno", plagiarism_percentage

# def detect_plagiarism_type(original_text, copy_text):
#     similarity_score = calculate_similarity(original_text, copy_text)
#     plagiarism_percentage = round(similarity_score * 100, 2)

#     if plagiarism_percentage >= 90:
#         return "Parafraseo", plagiarism_percentage
#     elif detect_disordered_phrases(original_text, copy_text):
#         return "Desordenar frases", plagiarism_percentage
#     elif detect_time_change(original_text, copy_text):
#         return "Cambio de tiempo", plagiarism_percentage
#     elif detect_voice_change(original_text, copy_text):
#         return "Cambio de voz", plagiarism_percentage
#     elif detect_inserted_phrases(original_text, copy_text):
#         return "Insertar o reemplazar frases", plagiarism_percentage
#     else:
#         return "Ninguno", plagiarism_percentage  # Se devuelve un tipo de plagio "Ninguno" con un porcentaje de 0.0


# def detect_disordered_phrases(original_text, copy_text):
#     original_phrases = original_text.split('.')
#     copy_phrases = copy_text.split('.')
#     if len(original_phrases) != len(copy_phrases):
#         return True  # Si la cantidad de frases es diferente, hay desorden en las frases
#     return False

# def detect_time_change(original_text, copy_text):
#     past_words = {'was', 'were', 'had', 'did'}
#     present_words = {'is', 'are', 'has', 'does'}

#     original_words = set(word_tokenize(original_text.lower()))
#     copy_words = set(word_tokenize(copy_text.lower()))

#     if any(word in past_words for word in original_words) and any(word in present_words for word in copy_words):
#         return True  # Si el tiempo cambió, se considera cambio de tiempo
#     return False

# def detect_voice_change(original_text, copy_text):
#     personal_pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they'}

#     original_tokens = word_tokenize(original_text.lower())
#     copy_tokens = word_tokenize(copy_text.lower())

#     original_count = sum(token in personal_pronouns for token in original_tokens)
#     copy_count = sum(token in personal_pronouns for token in copy_tokens)

#     if original_count != copy_count:
#         return True  # Si el uso de pronombres personales es diferente, se considera cambio de voz
#     return False


# def detect_inserted_phrases(original_text, copy_text):
#     original_tokens = set(word_tokenize(original_text.lower()))
#     copy_tokens = set(word_tokenize(copy_text.lower()))

#     # Si hay tokens en el texto copiado que no están en el texto original,
#     # entonces al menos una frase ha sido insertada o reemplazada
#     if copy_tokens.difference(original_tokens):
#         return True
#     return False

# def main():
#     originals_path = "data/original"
#     copies_path = "data/copias"
#     originals = load_originals(originals_path)
#     copies = load_copies(copies_path)

#     all_documents = originals + copies  # Combinar originales y copias para facilitar la comparación

#     preprocessed_documents = [preprocess_text(text) for _, text in all_documents]
#     features = extract_features(preprocessed_documents)

#     num_originals = len(originals)
#     features_originals = features[:num_originals]
#     features_copies = features[num_originals:]

#     model = train_model(features_originals, features_copies, None)  # No necesitamos labels para entrenamiento
#     predictions = detect_plagiarism(model, features_copies)

#     plagiarism_results = []

#     for i, copy_doc in enumerate(copies):
#         copy_name, copy_text = copy_doc
#         plagiarism_results_copy = []  # Almacena los resultados de comparación de esta copia con todos los originales
#         for j, (original_name, original_text) in enumerate(originals):
#             similarity_score = calculate_similarity(original_text, copy_text)
#             plagiarism_type, plagiarism_percentage = detect_plagiarism_type(original_text, copy_text)
#             plagiarism_found = predictions[i] == 1  # Comprobamos si la copia fue detectada como plagio por el modelo
#             plagiarism_results_copy.append((original_name, plagiarism_found, plagiarism_type, plagiarism_percentage))
#         plagiarism_results.append((copy_name, plagiarism_results_copy))

#     # Guardar los resultados en un archivo de texto
#     with open('results.txt', 'w') as f:
#         writer = csv.writer(f, delimiter='\t')
#         for result in plagiarism_results:
#             copy_name, plagiarism_results_copy = result
#             for original_name, plagiarism_found, plagiarism_type, plagiarism_percentage in plagiarism_results_copy:
#                 writer.writerow([copy_name, original_name, plagiarism_found, plagiarism_type, plagiarism_percentage])

#     # Crear un DataFrame de pandas para los resultados y guardarlo como un archivo Excel
#     results_df = pd.DataFrame(columns=['original_name', 'original_text', 'copy_name', 'copy_text', 'is_copy', 'copy_type', 'percentage'])
#     for result in plagiarism_results:
#         copy_name, plagiarism_results_copy = result
#         for original_name, plagiarism_found, plagiarism_type, plagiarism_percentage in plagiarism_results_copy:
#             results_df = results_df.append(pd.DataFrame({'original_name': [original_name], 
#                                             'original_text': next((text for name, text in originals if name == original_name), None),
#                                             'copy_name': copy_name,
#                                             'copy_text': next((text for name, text in copies if name == copy_name), None),
#                                             'is_copy': 'Si' if plagiarism_found else 'No',
#                                             'copy_type': plagiarism_type,
#                                             'percentage': plagiarism_percentage}, ignore_index=True))

#     results_df.to_excel('results.xlsx', index=False)

# if __name__ == "__main__":
#     main()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    return ' '.join(lemmatized_tokens)

def calculate_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score

def load_originals(originals_path):
    originals = []
    for filename in os.listdir(originals_path):
        with open(os.path.join(originals_path, filename), 'r', encoding='latin-1') as file:
            text = file.read()
            originals.append((filename, text))  # Guarda el nombre del archivo junto con el texto
    return originals

def load_copies(copies_path):
    copies = []
    for filename in os.listdir(copies_path):
        with open(os.path.join(copies_path, filename), 'r', encoding='latin-1') as file:
            text = file.read()
            copies.append((filename, text))  # Guarda el nombre del archivo junto con el texto
    return copies

def extract_features(texts):
    tfidf_vectorizer = TfidfVectorizer()
    features = tfidf_vectorizer.fit_transform(texts)
    return features

def train_model(features_originals, features_copies, labels):
    features_combined = scipy.sparse.vstack([features_originals, features_copies])
    num_originals = features_originals.shape[0]
    num_copies = features_copies.shape[0]
    labels_combined = [0] * num_originals + [1] * num_copies
    model = SVC(kernel='linear')
    model.fit(features_combined, labels_combined)
    return model

def detect_plagiarism(model, features_copies):
    predictions = model.predict(features_copies)
    return predictions

def detect_plagiarism_type(original_text, copy_text):
    similarity_score = calculate_similarity(original_text, copy_text)
    plagiarism_percentage = round(similarity_score * 100, 2)

    if plagiarism_percentage >= 90:
        return "Parafraseo", plagiarism_percentage
    elif plagiarism_percentage >= 60:
        if detect_disordered_phrases(original_text, copy_text):
            return "Desordenar frases", plagiarism_percentage
        elif detect_time_change(original_text, copy_text):
            return "Cambio de tiempo", plagiarism_percentage
        elif detect_voice_change(original_text, copy_text):
            return "Cambio de voz", plagiarism_percentage
        elif detect_inserted_phrases(original_text, copy_text):
            return "Insertar o reemplazar frases", plagiarism_percentage
    else:
        return "Ninguno", plagiarism_percentage



def detect_disordered_phrases(original_text, copy_text):
    original_phrases = original_text.split('.')
    copy_phrases = copy_text.split('.')
    if len(original_phrases) != len(copy_phrases):
        return True  # Si la cantidad de frases es diferente, hay desorden en las frases
    return False

def detect_time_change(original_text, copy_text):
    past_words = {'was', 'were', 'had', 'did'}
    present_words = {'is', 'are', 'has', 'does'}

    original_words = set(word_tokenize(original_text.lower()))
    copy_words = set(word_tokenize(copy_text.lower()))

    if any(word in past_words for word in original_words) and any(word in present_words for word in copy_words):
        return True  # Si el tiempo cambió, se considera cambio de tiempo
    return False

def detect_voice_change(original_text, copy_text):
    personal_pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they'}

    original_tokens = word_tokenize(original_text.lower())
    copy_tokens = word_tokenize(copy_text.lower())

    original_count = sum(token in personal_pronouns for token in original_tokens)
    copy_count = sum(token in personal_pronouns for token in copy_tokens)

    if original_count != copy_count:
        return True  # Si el uso de pronombres personales es diferente, se considera cambio de voz
    return False


def detect_inserted_phrases(original_text, copy_text):
    original_tokens = set(word_tokenize(original_text.lower()))
    copy_tokens = set(word_tokenize(copy_text.lower()))

    # Si hay tokens en el texto copiado que no están en el texto original,
    # entonces al menos una frase ha sido insertada o reemplazada
    if copy_tokens.difference(original_tokens):
        return True
    return False

def detect_paraphrasing(original_text, copy_text):
    # Preprocesamiento de texto
    preprocessed_original = preprocess_text(original_text)
    preprocessed_copy = preprocess_text(copy_text)

    # Extraer características
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_original, preprocessed_copy])

    # Calcular similitud de coseno entre los vectores TF-IDF
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # Si la similitud de coseno es alta, consideramos que hay parafraseo
    return similarity_score > 0.9  # Esto es un umbral arbitrario, puedes ajustarlo según sea necesario


# def main():
#     originals_path = "data/original"
#     copies_path = "data/copias"
#     originals = load_originals(originals_path)
#     copies = load_copies(copies_path)

#     all_documents = originals + copies  # Combinar originales y copias para facilitar la comparación

#     preprocessed_documents = [preprocess_text(text) for _, text in all_documents]
#     features = extract_features(preprocessed_documents)

#     num_originals = len(originals)
#     features_originals = features[:num_originals]
#     features_copies = features[num_originals:]

#     model = train_model(features_originals, features_copies, None)  # No necesitamos labels para entrenamiento
#     predictions = detect_plagiarism(model, features_copies)

#     plagiarism_results = []
#     df = pd.DataFrame(columns=['original_name', 'original_text', 'copy_name', 'copy_text', 'is_copy', 'copy_type', 'percentage'])

#     # Dentro del bucle
#     for i, copy_doc in enumerate(copies):
#         copy_name, copy_text = copy_doc
#         plagiarism_results_copy = []  # Almacena los resultados de comparación de esta copia con todos los originales
#         for j, (original_name, original_text) in enumerate(originals):
#             similarity_score = calculate_similarity(original_text, copy_text)
#             plagiarism_type, plagiarism_percentage = detect_plagiarism_type(original_text, copy_text)
#             plagiarism_found = predictions[i] == 1  # Comprobamos si la copia fue detectada como plagio por el modelo
#             df = pd.concat([df, pd.DataFrame({'original_name': [original_name],
#                                             'original_text': [original_text],
#                                             'copy_name': [copy_name],
#                                             'copy_text': [copy_text],
#                                             'is_copy': [plagiarism_found],
#                                             'copy_type': [plagiarism_type],
#                                             'percentage': [plagiarism_percentage]})])

#     for result in plagiarism_results:
#         contador = 0
#         copy_name, plagiarism_results_copy = result
        
#         for original_name, plagiarism_found, plagiarism_type, plagiarism_percentage in plagiarism_results_copy:
#             print(f'Archivo copia: {copy_name}\n')
#             print(f'Archivo original: {original_name}\n')
#             print(f"¿Es plagio?: {'Si' if plagiarism_found else 'No'}")
#             print(f"Tipo de plagio: {plagiarism_type}")
#             print(f"Porcentaje de plagio: {plagiarism_percentage}%")
#             print("------------------------------------------")

#     # Guardar resultados en un archivo de texto
#     with open('plagiarism_results.txt', 'w') as file:
#         for result in plagiarism_results:
#             copy_name, plagiarism_results_copy = result
#             for original_name, plagiarism_found, plagiarism_type, plagiarism_percentage in plagiarism_results_copy:
#                 file.write(f'Archivo copia: {copy_name}\n')
#                 file.write(f'Archivo original: {original_name}\n')
#                 file.write(f"¿Es plagio?: {'Si' if plagiarism_found else 'No'}\n")
#                 file.write(f"Tipo de plagio: {plagiarism_type}\n")
#                 file.write(f"Porcentaje de plagio: {plagiarism_percentage}%\n")
#                 file.write("------------------------------------------\n")

#     # Guardar resultados en un archivo Excel
#     df.to_excel('plagiarism_results.xlsx', index=False)

# if __name__ == "__main__":
#     main()
def main():
    originals_path = "data/original"
    copies_path = "data/copias"
    originals = load_originals(originals_path)
    copies = load_copies(copies_path)

    all_documents = originals + copies  # Combinar originales y copias para facilitar la comparación

    preprocessed_documents = [preprocess_text(text) for _, text in all_documents]
    features = extract_features(preprocessed_documents)

    num_originals = len(originals)
    features_originals = features[:num_originals]
    features_copies = features[num_originals:]

    model = train_model(features_originals, features_copies, None)  # No necesitamos labels para entrenamiento
    predictions = detect_plagiarism(model, features_copies)

    plagiarism_results = []
    df = pd.DataFrame(columns=['original_name', 'original_text', 'copy_name', 'copy_text', 'is_copy', 'copy_type', 'percentage'])

    # Dentro del bucle
    for i, copy_doc in enumerate(copies):
        copy_name, copy_text = copy_doc
        plagiarism_results_copy = []  # Almacena los resultados de comparación de esta copia con todos los originales
        for j, (original_name, original_text) in enumerate(originals):
            similarity_score = calculate_similarity(original_text, copy_text)
            plagiarism_type, plagiarism_percentage = detect_plagiarism_type(original_text, copy_text)
            plagiarism_found = predictions[i] == 1  # Comprobamos si la copia fue detectada como plagio por el modelo
            plagiarism_results_copy.append((original_name, plagiarism_found, plagiarism_type, plagiarism_percentage))
            df = pd.concat([df, pd.DataFrame({'original_name': [original_name],
                                            'original_text': [original_text],
                                            'copy_name': [copy_name],
                                            'copy_text': [copy_text],
                                            'is_copy': [plagiarism_found],
                                            'copy_type': [plagiarism_type],
                                            'percentage': [plagiarism_percentage]})])
        plagiarism_results.append((copy_name, plagiarism_results_copy))

    for result in plagiarism_results:
        copy_name, plagiarism_results_copy = result
        for original_name, plagiarism_found, plagiarism_type, plagiarism_percentage in plagiarism_results_copy:
            print(f'Archivo copia: {copy_name}\n')
            print(f'Archivo original: {original_name}\n')
            print(f"¿Es plagio?: {'Si' if plagiarism_found else 'No'}")
            print(f"Tipo de plagio: {plagiarism_type}")
            print(f"Porcentaje de plagio: {plagiarism_percentage}%")
            print("------------------------------------------")

    # Guardar resultados en un archivo de texto
    with open('plagiarism_results.txt', 'w') as file:
        for result in plagiarism_results:
            copy_name, plagiarism_results_copy = result
            for original_name, plagiarism_found, plagiarism_type, plagiarism_percentage in plagiarism_results_copy:
                file.write(f'Archivo copia: {copy_name}\n')
                file.write(f'Archivo original: {original_name}\n')
                file.write(f"¿Es plagio?: {'Si' if plagiarism_found else 'No'}\n")
                file.write(f"Tipo de plagio: {plagiarism_type}\n")
                file.write(f"Porcentaje de plagio: {plagiarism_percentage}%\n")
                file.write("------------------------------------------\n")

    # Guardar resultados en un archivo Excel
    df.to_excel('plagiarism_results.xlsx', index=False)

if __name__ == "__main__":
    main()
