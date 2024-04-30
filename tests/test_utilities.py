# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado

import unittest
from unittest import mock
from unittest.mock import mock_open, patch
import sys
import os
# Esto agrega la carpeta 'src' al path para que puedas importar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utilities

class TestUtilities(unittest.TestCase):

    def test_load_document(self):
        # Test para verificar que la función load_document carga correctamente el contenido de un archivo.
        with patch('builtins.open', mock_open(read_data="test content")) as mocked_file:
            result = utilities.load_document("fake_path.txt")
            mocked_file.assert_called_once_with("fake_path.txt", 'r', encoding='latin1')
            self.assertEqual(result, "test content")

    def test_load_stopwords(self):
        # Test para verificar que la función load_stopwords carga correctamente un conjunto de stopwords.
        stopwords_data = "stopword1\nstopword2\nstopword3"
        with patch('builtins.open', mock_open(read_data=stopwords_data)) as mocked_file:
            result = utilities.load_stopwords("fake_path.txt")
            mocked_file.assert_called_once_with("fake_path.txt", 'r', encoding='utf-8')
            self.assertEqual(result, {"stopword1", "stopword2", "stopword3"})

    def test_save_results_to_txt(self):
        # Test para verificar que se llama correctamente a la función de guardado en texto.
        results = [(("Doc1", "Doc2", 0.9),), (("Doc3", "Doc4", 0.75),)]
        max_plagiarism = "Doc1 vs Doc2: 90.00% similar\n"
        with patch('builtins.open', mock_open()) as mocked_file:
            utilities.save_results_to_txt(results, max_plagiarism, "fake_path.txt")
            mocked_file.assert_called_once_with("fake_path.txt", 'w', encoding='utf-8')
            handle = mocked_file()
            handle.write.assert_called()  # Verifica que se llamó al menos una vez

    def test_save_results_to_excel(self):
        # Test para verificar que se llama correctamente a la función de guardado en Excel.
        # Preparando una lista de listas de tuplas, como espera la función.
        results = [
            [("Doc1", "Doc2", 0.9), ("Doc3", "Doc4", 0.75)],
            [("Doc5", "Doc6", 0.85)]
        ]
        with mock.patch('pandas.DataFrame.to_excel') as mocked_to_excel:
            utilities.save_results_to_excel(results, "fake_path.xlsx")
            mocked_to_excel.assert_called_once_with("fake_path.xlsx", index=False)
        # Verificar que todos los elementos en cada sublista de results son tuplas y cada tupla tiene tres elementos
        for sub_list in results:
            for result in sub_list:
                self.assertIsInstance(result, tuple, "Cada resultado debe ser una tupla.")
                self.assertEqual(len(result), 3, "Cada tupla debe contener exactamente tres elementos.")
                
    def test_evaluate_results(self):
        # Datos de entrada simulados
        plagiarism_results = ['plagiarism', 'genuine', 'plagiarism', 'genuine']
        real_labels = ['plagiarism', 'plagiarism', 'genuine', 'genuine']
        scores = [0.9, 0.2, 0.6, 0.1]
        # Resultados esperados
        expected_output = {
            'TP': 1,
            'FP': 1,
            'TN': 1,
            'FN': 1,
            'AUC': 0.75  # Valor hipotético
        }
        # Mocking roc_auc_score
        with mock.patch('utilities.roc_auc_score', return_value=0.75) as mocked_auc:
            result = utilities.evaluate_results(plagiarism_results, real_labels, scores)
            # Asegurar que roc_auc_score sea llamado una vez
            mocked_auc.assert_called_once_with([1, 1, 0, 0], scores)
        # Prueba de los resultados
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()