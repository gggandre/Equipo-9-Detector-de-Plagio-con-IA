# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado

import unittest
from src.text_comparison import jaccard_similarity

class TestTextComparison(unittest.TestCase):        

    def test_jaccard_similarity_identical(self):
        # Prueba de similitud entre dos vectores idénticos
        vec1 = {"this": 1, "is": 1, "a": 1, "test": 1}
        vec2 = {"this": 1, "is": 1, "a": 1, "test": 1}
        self.assertEqual(jaccard_similarity(vec1, vec2), 1.0)

    def test_jaccard_similarity_completely_different(self):
        # Prueba de similitud entre dos vectores completamente diferentes
        vec1 = {"apple": 1, "banana": 1, "orange": 1}
        vec2 = {"car": 1, "house": 1, "tree": 1}
        self.assertEqual(jaccard_similarity(vec1, vec2), 0.0)

    def test_jaccard_similarity_partial_overlap(self):
        # Prueba de similitud entre dos vectores con solapamiento parcial
        vec1 = {"apple": 1, "banana": 1, "orange": 1}
        vec2 = {"banana": 1, "orange": 1, "kiwi": 1}
        # El solapamiento es {banana, orange}, la unión es {apple, banana, orange, kiwi}
        # Por lo tanto, la similitud de Jaccard es 2/4 = 0.5
        self.assertEqual(jaccard_similarity(vec1, vec2), 0.5)

    def test_jaccard_similarity_empty_vectors(self):
        # Prueba de similitud entre dos vectores vacíos
        vec1 = {}
        vec2 = {}
        # La similitud de Jaccard de dos conjuntos vacíos es 1.0 por convención
        self.assertEqual(jaccard_similarity(vec1, vec2), 1.0)
        
    def test_jaccard_similarity_one_empty_vector(self):
        # Prueba de similitud entre un vector vacío y un vector no vacío
        vec1 = {}
        vec2 = {"apple": 1, "banana": 1, "orange": 1}
        # La similitud de Jaccard de un vector vacío y un vector no vacío es 0.0
        self.assertEqual(jaccard_similarity(vec1, vec2), 0.0)

    def test_jaccard_similarity_different_lengths(self):
        # Prueba de similitud entre dos vectores de diferentes longitudes
        vec1 = {"apple": 1, "banana": 1}
        vec2 = {"apple": 1, "banana": 1, "orange": 1}
        # La similitud de Jaccard entre dos vectores de diferentes longitudes se calcula
        # considerando la intersección y la unión de las claves en ambos vectores
        # En este caso, la intersección es {apple, banana} y la unión es {apple, banana, orange}
        # Por lo tanto, la similitud de Jaccard es 2/3 ≈ 0.67
        self.assertAlmostEqual(jaccard_similarity(vec1, vec2), 2/3, places=2)

if __name__ == '__main__':
    unittest.main()
