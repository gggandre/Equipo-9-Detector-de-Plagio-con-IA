from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(texts):
    tfidf_vectorizer = TfidfVectorizer()
    features = tfidf_vectorizer.fit_transform(texts)
    return features
