from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer



def calculate_similarity(text1, text2):
    # Aquí implementamos la función para calcular la similitud entre dos textos.
    # Por ahora, usaremos la similitud de coseno como ejemplo.
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score