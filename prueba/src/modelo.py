import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Cargar datos
data = pd.read_excel('data/resultado.xlsx')  # Asegúrate de cambiar esto por la ruta correcta de tu archivo

# Función de preprocesamiento de texto
def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    tokens = nltk.word_tokenize(text)  # Tokenizar
    tokens = [token for token in tokens if token.isalpha()]  # Remover puntuación
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remover stopwords
    return " ".join(tokens)

# Preprocesar textos de original y copia
data['original_text_processed'] = data['original_text'].apply(preprocess_text)
data['copy_text_processed'] = data['copy_text'].apply(preprocess_text)

# Combinar los textos originales y plagiados para entrenamiento
data['combined_text'] = data['original_text_processed'] + " " + data['copy_text_processed']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data['combined_text'], data['copy_type'], test_size=0.2, random_state=42)

# Vectorización de los textos
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar un modelo de clasificación
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluar el modelo
predictions = model.predict(X_test_tfidf)
print(classification_report(y_test, predictions))

# Función para predecir el tipo de plagio de un nuevo texto
def predict_plagiarism_type(original_text, copy_text):
    original_text_processed = preprocess_text(original_text)
    copy_text_processed = preprocess_text(copy_text)
    combined_text = original_text_processed + " " + copy_text_processed
    combined_text_tfidf = vectorizer.transform([combined_text])
    return model.predict(combined_text_tfidf)[0]

# Ejemplo de uso
original_text_example = "How would people distribute risks of autonomous vehicles (AVs) in everyday road traffic? The rich literature on the ethics of autonomous vehicles (AVs) revolves around moral judgments in unavoidable collision scenarios. We argue for extending the debate to driving behaviors in everyday road traffic where ubiquitous ethical questions arise due to the permanent redistribution of risk among road users. This distribution of risks raises ethically relevant questions that cannot be evaded by simple heuristics such as “hitting the brakes.” Using an interactive, graphical representation of different traffic situations, we measured participants’ preferences on driving maneuvers of AVs in a representative survey in Germany. Our participants’ preferences deviated significantly from mere collision avoidance. Interestingly, our participants were willing to take risks themselves for the benefit of other road users, suggesting that the social dilemma of AVs may be mitigated in risky environments. Our research might build a bridge between engineers and philosophers to discuss the ethics of AVs more constructively."
copy_text_example = """How would the distribution of risks associated with autonomous vehicles (AVs) in everyday road traffic be approached by people? The extensive body of literature on the ethics of AVs primarily focuses on moral judgments in situations of inevitable collisions. We advocate for broadening the discourse to encompass driving behaviors in daily road traffic, where ethical dilemmas arise due to the ongoing redistribution of risk among road users. This redistribution of risks prompts pertinent ethical inquiries that cannot be sidestepped with simplistic heuristics such as "applying the brakes forcefully." Through the utilization of an interactive, graphical representation of various traffic scenarios, participants' preferences regarding AV driving maneuvers were gauged in a representative survey conducted in Germany. Notably, the preferences of participants exhibited significant deviation from mere collision avoidance strategies. Intriguingly, participants demonstrated a willingness to assume risks themselves for the benefit of other road users, suggesting a potential mitigation of the social dilemma associated with AVs in environments characterized by heightened risk. Our research endeavors to facilitate a constructive dialogue between engineers and philosophers regarding the ethical dimensions of AVs."""
predicted_type = predict_plagiarism_type(original_text_example, copy_text_example)
print("Tipo de plagio predicho:", predicted_type)
