import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import os
import numpy as np

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
            originals.append((filename, text))
    return originals

def load_copies(copies_path):
    copies = []
    for filename in os.listdir(copies_path):
        with open(os.path.join(copies_path, filename), 'r', encoding='latin-1') as file:
            text = file.read()
            copies.append((filename, text))
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

def detect_paraphrasing(original_text, copy_text):
    preprocessed_original = preprocess_text(original_text)
    preprocessed_copy = preprocess_text(copy_text)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_original, preprocessed_copy])

    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    paraphrasing_percentage = similarity_score * 100
    paraphrasing_percentage = min(paraphrasing_percentage, 100)  # Ensure not to exceed 100%

    if paraphrasing_percentage >= 50:
        return "Parafraseo", paraphrasing_percentage

    return None, 0

def detect_disordered_phrases(original_text, copy_text):
    original_sentences = nltk.sent_tokenize(original_text)
    copy_sentences = nltk.sent_tokenize(copy_text)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(original_sentences + copy_sentences)
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return "Desordenar frases", similarity_score * 100

def detect_time_change(original_text, copy_text):
    past_words = {'was', 'were', 'had', 'did'}
    present_words = {'is', 'are', 'has', 'does'}

    original_tokens = set(word_tokenize(original_text.lower()))
    copy_tokens = set(word_tokenize(copy_text.lower()))

    original_past_count = sum(word in past_words for word in original_tokens)
    copy_past_count = sum(word in past_words for word in copy_tokens)
    original_present_count = sum(word in present_words for word in original_tokens)
    copy_present_count = sum(word in present_words for word in copy_tokens)

    original_past_percentage = original_past_count / max(len(original_tokens), 1) * 100
    copy_past_percentage = copy_past_count / max(len(copy_tokens), 1) * 100

    if abs(original_past_percentage - copy_past_percentage) > 10:  # 10% difference threshold
        return "Cambio de tiempo", max(original_past_percentage, copy_past_percentage)
    return None, 0

def detect_voice_change(original_text, copy_text):
    personal_pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they'}

    original_tokens = word_tokenize(original_text.lower())
    copy_tokens = word_tokenize(copy_text.lower())

    original_personal_count = sum(token in personal_pronouns for token in original_tokens)
    copy_personal_count = sum(token in personal_pronouns for token in copy_tokens)

    original_personal_percentage = original_personal_count / max(len(original_tokens), 1) * 100
    copy_personal_percentage = copy_personal_count / max(len(copy_tokens), 1) * 100

    if abs(original_personal_percentage - copy_personal_percentage) > 10:  # 10% difference threshold
        return "Cambio de voz", max(original_personal_percentage, copy_personal_percentage)
    return None, 0

def detect_inserted_phrases(original_text, copy_text):
    original_tokens = set(word_tokenize(original_text.lower()))
    copy_tokens = set(word_tokenize(copy_text.lower()))

    common_tokens = original_tokens.intersection(copy_tokens)
    if len(original_tokens) == 0 or len(copy_tokens) == 0:
        return None, 0

    original_percentage = len(common_tokens) / len(original_tokens)
    copy_percentage = len(common_tokens) / len(copy_tokens)

    # Considerar significativo sólo si más del 50% de los tokens coinciden
    if original_percentage > 0.5 and copy_percentage > 0.5:
        return "Insertar o reemplazar frases", max(original_percentage, copy_percentage) * 100
    return None, 0


def detect_plagiarism_type(original_text, copy_text):
    types = []
    percentages = []

    # Asegúrate de que cada función devuelva un porcentaje como un flotante
    # Por ejemplo, si detect_disordered_phrases devuelve una tupla, asegúrate de extraer solo el valor del porcentaje.
    disordered_phrases_score = detect_disordered_phrases(original_text, copy_text)
    if disordered_phrases_score[1] is not None:  # Asumiendo que la función devuelve una tupla (tipo, porcentaje)
        types.append(disordered_phrases_score[0])
        percentages.append(disordered_phrases_score[1])

    time_change_type, time_change_percentage = detect_time_change(original_text, copy_text)
    if time_change_type:
        types.append(time_change_type)
        percentages.append(time_change_percentage)

    voice_change_type, voice_change_percentage = detect_voice_change(original_text, copy_text)
    if voice_change_type:
        types.append(voice_change_type)
        percentages.append(voice_change_percentage)

    inserted_phrases_type, inserted_phrases_percentage = detect_inserted_phrases(original_text, copy_text)
    if inserted_phrases_type:
        types.append(inserted_phrases_type)
        percentages.append(inserted_phrases_percentage)

    paraphrasing_type, paraphrasing_percentage = detect_paraphrasing(original_text, copy_text)
    if paraphrasing_type:
        types.append(paraphrasing_type)
        percentages.append(paraphrasing_percentage)

    if percentages:
        max_percentage_index = percentages.index(max(percentages))
        return types[max_percentage_index], percentages[max_percentage_index]
    return "Ninguno", 0


def main():
    originals_path = "data/original"
    copies_path = "data/copias"
    originals = load_originals(originals_path)
    copies = load_copies(copies_path)

    all_documents = originals + copies

    preprocessed_documents = [preprocess_text(text) for _, text in all_documents]
    features = extract_features(preprocessed_documents)

    num_originals = len(originals)
    features_originals = features[:num_originals]
    features_copies = features[num_originals:]

    model = train_model(features_originals, features_copies, None)
    predictions = detect_plagiarism(model, features_copies)

    plagiarism_results = []
    df = pd.DataFrame(columns=['original_name', 'original_text', 'copy_name', 'copy_text', 'is_copy', 'copy_type', 'percentage'])

    for i, copy_doc in enumerate(copies):
        copy_name, copy_text = copy_doc
        plagiarism_results_copy = []
        for j, (original_name, original_text) in enumerate(originals):
            plagiarism_type, plagiarism_percentage = detect_plagiarism_type(original_text, copy_text)
            plagiarism_found = predictions[i] == 1
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
            print(f"Porcentaje de plagio: {round(plagiarism_percentage, 2)}%")
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
                file.write(f"Porcentaje de plagio: {round(plagiarism_percentage, 2)}%\n")
                file.write("------------------------------------------\n")

    # Guardar resultados en un archivo Excel
    df.to_excel('plagiarism_results.xlsx', index=False)

if __name__ == "__main__":
    main()
