import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, f1_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
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


def train_model(features_originals, features_copies):
    features_combined = scipy.sparse.vstack([features_originals, features_copies])
    num_originals = features_originals.shape[0]
    num_copies = features_copies.shape[0]
    labels_combined = [0] * num_originals + [1] * num_copies
    X_train, X_test, y_train, y_test = train_test_split(features_combined, labels_combined, test_size=0.25, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)  # Get decision scores for ROC
    return model, y_test, y_pred, y_scores

def calculate_metrics(y_test, y_pred, y_scores):
    auc_score = roc_auc_score(y_test, y_scores)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return auc_score, recall, f1

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_test, y_scores):
    RocCurveDisplay.from_predictions(y_test, y_scores)
    plt.title('ROC Curve')
    plt.show()

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
    paraphrasing_percentage = min(paraphrasing_percentage, 100)

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
    disordered_phrases_score = detect_disordered_phrases(original_text, copy_text)
    if disordered_phrases_score[1] is not None:
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
    originals_path = "src/data/otros1"
    copies_path = "src/data/plagioProfes/Final Testing"
    originals = load_originals(originals_path)
    copies = load_copies(copies_path)

    all_documents = originals + copies
    preprocessed_documents = [preprocess_text(text) for _, text in all_documents]
    features = extract_features(preprocessed_documents)

    num_originals = len(originals)
    features_originals = features[:num_originals]
    features_copies = features[num_originals:]

    # Entrenando el modelo y obteniendo predicciones
    start_time = time.time()
    model, y_test, y_pred, y_scores = train_model(features_originals, features_copies)
    
    # Calcular métricas
    auc_score = calculate_metrics(y_test, y_pred, y_scores)
    print(f"AUC: {auc_score}")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_scores)
    end_time = time.time()
    print(f"Tiempo de ejecución: {end_time - start_time} segundos")
    # Detección de plagio individual y escribir resultados
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
        # Ordenar los resultados por porcentaje de similitud y seleccionar los cinco primeros
        plagiarism_results_copy.sort(key=lambda x: x[3], reverse=True)
        top_5_results = plagiarism_results_copy[:5]
        plagiarism_results.append((copy_name, top_5_results))

    with open('src/results/plagiarism_results.txt', 'w') as file:
        for result in plagiarism_results:
            copy_name, top_5_results = result
            file.write(f'\n************* Archivo copia: {copy_name} **************\n')
            for original_name, plagiarism_found, plagiarism_type, plagiarism_percentage in top_5_results:
                file.write(f'Archivo original: {original_name}\n')
                file.write(f"¿Es plagio?: {'Si' if plagiarism_found else 'No'}\n")
                file.write(f"Tipo de plagio: {'Ninguno' if plagiarism_found == 'No' else plagiarism_type}\n")
                file.write(f"Porcentaje de plagio: {round(plagiarism_percentage, 2)}%\n")
                file.write("------------------------------------------\n")

    df.to_excel('src/results/plagiarism_results.xlsx', index=False)

if __name__ == "__main__":
    main()
