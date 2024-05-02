from sklearn.svm import SVC
import scipy.sparse


def train_model(features_originals, features_copies, labels):
    # Imprimir la forma de las características

    # Concatenar las características de los documentos originales y de copia
    features_combined = scipy.sparse.vstack([features_originals, features_copies])

    # Crear las etiquetas combinadas
    num_originals = features_originals.shape[0]
    num_copies = features_copies.shape[0]
    labels_combined = [0] * num_originals + [1] * num_copies

    # Crear y entrenar el modelo SVM
    model = SVC(kernel='linear')
    model.fit(features_combined, labels_combined)

    return model


def detect_plagiarism(model, features_copies):
    # Usar el modelo entrenado para predecir la clase de los documentos de copia
    predictions = model.predict(features_copies)
    return predictions
