# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado

import unittest
from collections import Counter
from src.feature_extraction import build_feature_vector

class TestFeatureExtraction(unittest.TestCase):
    def test_build_feature_vector(self):
        # Casos de prueba para la función build_feature_vector
        self.assertEqual(build_feature_vector([]), {})  # Prueba con una lista vacía
        self.assertEqual(build_feature_vector(["this", "is", "a", "test"]), {"this": 1, "is": 1, "a": 1, "test": 1})  # Prueba con palabras únicas
        self.assertEqual(build_feature_vector(["this", "is", "a", "test", "this"]), {"this": 2, "is": 1, "a": 1, "test": 1})  # Prueba con palabras repetidas
        self.assertEqual(build_feature_vector(["this", "is", "a", "test", "this", "this"]), {"this": 3, "is": 1, "a": 1, "test": 1})  # Prueba con una palabra repetida varias veces
        self.assertEqual(build_feature_vector(["this", "is", "a", "test", "another", "test"]), {"this": 1, "is": 1, "a": 1, "test": 2, "another": 1})  # Prueba con diferentes palabras
        self.assertEqual(build_feature_vector(["apple", "banana", "orange", "banana", "apple", "kiwi"]), {"apple": 2, "banana": 2, "orange": 1, "kiwi": 1})  # Prueba con palabras no relacionadas
        self.assertEqual(build_feature_vector(["apple", "banana", "apple", "banana", "apple", "banana"]), {"apple": 3, "banana": 3})  # Prueba con solo dos palabras


    def test_build_feature_vector_empty_list(self):
        # Prueba con una lista vacía
        self.assertEqual(build_feature_vector([]), {})  

    def test_build_feature_vector_unique_words(self):
        # Prueba con palabras únicas
        self.assertEqual(build_feature_vector(["this", "is", "a", "test"]), {"this": 1, "is": 1, "a": 1, "test": 1})  

    def test_build_feature_vector_repeated_words(self):
        # Prueba con palabras repetidas
        self.assertEqual(build_feature_vector(["this", "is", "a", "test", "this"]), {"this": 2, "is": 1, "a": 1, "test": 1})  

    def test_build_feature_vector_repeated_words_multiple_times(self):
        # Prueba con una palabra repetida varias veces
        self.assertEqual(build_feature_vector(["this", "is", "a", "test", "this", "this"]), {"this": 3, "is": 1, "a": 1, "test": 1})  

    def test_build_feature_vector_different_words(self):
        # Prueba con diferentes palabras
        self.assertEqual(build_feature_vector(["this", "is", "a", "test", "another", "test"]), {"this": 1, "is": 1, "a": 1, "test": 2, "another": 1})  

    def test_build_feature_vector_unrelated_words(self):
        # Prueba con palabras no relacionadas
        self.assertEqual(build_feature_vector(["apple", "banana", "orange", "banana", "apple", "kiwi"]), {"apple": 2, "banana": 2, "orange": 1, "kiwi": 1})  

    def test_build_feature_vector_only_two_words(self):
        # Prueba con solo dos palabras
        self.assertEqual(build_feature_vector(["apple", "banana", "apple", "banana", "apple", "banana"]), {"apple": 3, "banana": 3})  

    def test_build_feature_vector_numbers(self):
        # Prueba con números como palabras
        self.assertEqual(build_feature_vector(["one", "two", "three", "four", "five", "six", "six"]), {"one": 1, "two": 1, "three": 1, "four": 1, "five": 1, "six": 2})

    def test_build_feature_vector_mixed_case(self):
        # Prueba con palabras en mayúsculas y minúsculas
        self.assertEqual(build_feature_vector(["apple", "APPLE", "Banana", "banana", "OrAnGe", "orange"]), {"apple": 1, "APPLE": 1, "Banana": 1, "banana": 1, "OrAnGe": 1, "orange": 1})

    def test_build_feature_vector_special_characters(self):
        # Prueba con palabras que contienen caracteres especiales
        self.assertEqual(build_feature_vector(["!@#$", "%^&", "*()"]), {"!@#$": 1, "%^&": 1, "*()": 1})

        def test_build_feature_vector_empty_strings(self):
        # Prueba con cadenas vacías en la lista. 
        # La función cuenta todas las cadenas pasadas, incluyendo cadenas vacías.
            self.assertEqual(build_feature_vector(["", "", ""]), Counter({'': 3}))


if __name__ == '__main__':
    unittest.main()
