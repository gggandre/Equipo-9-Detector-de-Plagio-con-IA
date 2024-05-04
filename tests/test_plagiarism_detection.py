# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado

import unittest
import sys
import os
from unittest.mock import patch, mock_open
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Añadir el directorio src al path para poder importar los módulos necesarios
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from plagiarism_detection import (preprocess_text, calculate_similarity, train_model, detect_plagiarism,
                                  detect_paraphrasing, detect_disordered_phrases, detect_time_change, detect_voice_change,
                                  detect_inserted_phrases, detect_plagiarism_type, load_originals, load_copies)
import numpy as np
import scipy.sparse as sp

class TestPlagiarismDetection(unittest.TestCase):
    def setUp(self):
        """Método que se ejecuta antes de cada test para configurar las condiciones iniciales."""
        self.text1 = "This is a simple test text."
        self.text2 = "This is a simple test text with some extra words."
        self.text3 = "Completely different content here."

    def train_model(self, features_originals, features_copies):
        """Entrena un modelo SVM utilizando datos de entrada como características originales y copias.
        Asegura que haya suficientes datos para una división significativa y retorna el modelo entrenado junto con los conjuntos de prueba y sus predicciones."""
        features_combined = sp.vstack([features_originals, features_copies])
        labels_combined = [0] * features_originals.shape[0] + [1] * features_copies.shape[0]
        total_samples = features_combined.shape[0]
        test_size = min(2, total_samples)  # Limita el tamaño del test a 2 o el total de muestras disponibles
        X_train, X_test, y_train, y_test = train_test_split(features_combined, labels_combined, test_size=test_size, random_state=42)
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_scores = model.decision_function(X_test)
        return model, y_test, y_pred, y_scores

    def test_train_model_and_detect_plagiarism(self):
        """Verifica que el modelo entrenado pueda predecir correctamente y que el tamaño del conjunto de pruebas sea el esperado."""
        features_originals = sp.csr_matrix(np.array([[0, 0, 1], [1, 1, 0]]))
        features_copies = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 1]]))
        model, y_test, y_pred, y_scores = self.train_model(features_originals, features_copies)
        expected_test_size = min(2, features_originals.shape[0] + features_copies.shape[0])
        self.assertEqual(len(y_test), expected_test_size)
        predictions = detect_plagiarism(model, features_copies)
        self.assertEqual(len(predictions), expected_test_size)

    def test_preprocess_text(self):
        """Prueba la función de preprocesamiento de texto para asegurar que elimine stopwords y realice stemming correctamente."""
        result = preprocess_text("This is an example with the stopwords")
        expected = "exampl stopword"
        self.assertEqual(result, expected)

    def test_calculate_similarity(self):
        """Prueba la función de cálculo de similitud para verificar que mide correctamente la similitud entre dos textos."""
        similarity = calculate_similarity(self.text1, self.text1)
        self.assertAlmostEqual(similarity, 1.0, places=7)  # Permite un pequeño margen de error
        similarity = calculate_similarity(self.text1, self.text3)
        self.assertTrue(similarity < 0.1)

    @patch('os.listdir', return_value=['file1.txt', 'file2.txt'])
    @patch('builtins.open', new_callable=mock_open, read_data='data')
    def test_load_originals(self, mock_file, mock_os_list):
        """Prueba la función de carga de archivos originales simulando la lectura de archivos."""
        result = load_originals('path')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], 'data')

    @patch('os.listdir', return_value=['file1.txt', 'file2.txt'])
    @patch('builtins.open', new_callable=mock_open, read_data='data')
    def test_load_copies(self, mock_file, mock_os_list):
        """Prueba la función de carga de archivos de copias simulando la lectura de archivos."""
        result = load_copies('path')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], 'data')

    def test_detect_paraphrasing(self):
        """Verifica que la función de detección de parafraseo identifique correctamente el parafraseo entre dos textos."""
        type_detected, percentage = detect_paraphrasing(self.text1, self.text2)
        self.assertIsNotNone(type_detected)
        self.assertTrue(percentage > 0)

    def test_detect_disordered_phrases(self):
        """Prueba la capacidad de la función para detectar frases desordenadas entre dos textos."""
        result, percentage = detect_disordered_phrases(self.text1, self.text2)
        self.assertIsInstance(result, str)
        self.assertIsInstance(percentage, float)

    def test_detect_time_change(self):
        """Evalúa la función de detección de cambio de tiempo entre dos textos para asegurarse de que funciona correctamente."""
        result, percentage = detect_time_change(self.text1, self.text3)
        self.assertIsNone(result)
        self.assertEqual(percentage, 0)

    def test_detect_voice_change(self):
        """Prueba la función de detección de cambio de voz para verificar su eficacia."""
        result, percentage = detect_voice_change(self.text1, self.text2)
        self.assertIsNone(result)
        self.assertEqual(percentage, 0)

    def test_detect_inserted_phrases(self):
        """Evalúa la detección de frases insertadas o reemplazadas entre dos textos."""
        result, percentage = detect_inserted_phrases(self.text1, self.text3)
        self.assertIsNone(result)
        self.assertEqual(percentage, 0)

    def test_detect_plagiarism_type(self):
        """Prueba la función que identifica el tipo de plagio entre dos textos."""
        plagiarism_type, percentage = detect_plagiarism_type(self.text1, self.text2)
        self.assertIsInstance(plagiarism_type, str)
        self.assertIsInstance(percentage, float)

if __name__ == '__main__':
    unittest.main()
