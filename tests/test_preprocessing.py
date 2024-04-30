# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado

import unittest
from src.preprocessing import tokenize, remove_stopwords, stem_words, preprocess_text

class TestPreprocessing(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("special_characters!@#$%^&*()"), ["special_characters"])

    def test_remove_stopwords(self):
        # Casos de prueba para la función remove_stopwords
        # Prueba básica con algunas stopwords comunes
        self.assertEqual(remove_stopwords(["this", "is", "a", "test", "of", "stopwords"]), ["test", "of", "stopwords"])
        # Prueba con un array vacío, debe retornar un array vacío
        self.assertEqual(remove_stopwords([]), [])
        # Prueba donde todas las palabras son stopwords
        self.assertEqual(remove_stopwords(["the", "an", "a", "by", "on", "at", "of"]), [])
        # Prueba con palabras que no son stopwords
        self.assertEqual(remove_stopwords(["python", "code", "programming", "language"]), ["python", "code", "programming", "language"])
        # Prueba con mezcla de mayúsculas y minúsculas
        self.assertEqual(remove_stopwords(["This", "is", "The", "Python", "language"]), ["Python", "language"])
        # Prueba con repetición de stopwords y no-stopwords
        self.assertEqual(remove_stopwords(["this", "test", "test", "this", "is", "is", "good"]), ["test", "test", "good"])
        # Prueba con contracciones y palabras compuestas
        self.assertEqual(remove_stopwords(["it's", "the", "children's", "books"]), ["it's", "children's", "books"])
        # Prueba con números y stopwords, asumiendo que los números no son tratados como stopwords
        self.assertEqual(remove_stopwords(["one", "two", "three", "of", "four", "five"]), ["one", "two", "three", "four", "five"])

    def test_stem_words(self):
        # Casos de prueba para la función stem_words
        self.assertEqual(stem_words([]), [])  # Prueba con una lista vacía
        self.assertEqual(stem_words(["running", "walked"]), ["run", "walk"])  # Prueba con palabras verbales
        self.assertEqual(stem_words(["running", "walking", "walked"]), ["run", "walk", "walk"])  # Prueba con formas verbales diferentes
        self.assertEqual(stem_words(["running", "runs", "run"]), ["run", "run", "run"])  # Prueba con diferentes formas verbales del mismo verbo
        self.assertEqual(stem_words(["run", "runner", "running"]), ["run", "runner", "run"])  # Prueba con formas verbales diferentes y el mismo verbo
        self.assertEqual(stem_words(["happiness", "happy", "happier"]), ["happi", "happi", "happier"])  # Prueba con adjetivos
        self.assertEqual(stem_words(["eat", "eats", "eating"]), ["eat", "eat", "eat"])  # Prueba con formas verbales diferentes del verbo "eat"

    def test_preprocess_text(self):
        # Preprocess_text debe devolver una lista de tokens
        self.assertEqual(preprocess_text("This test contains numbers like 123."), ["test", "contain", "number", "like", "123"])

    def test_tokenize_empty_string(self):
        # Prueba con cadena vacía
        self.assertEqual(tokenize(""), [])  

    def test_tokenize_simple_string(self):
        # Prueba con una cadena simple
        self.assertEqual(tokenize("Hello World"), ["hello", "world"])  

    def test_tokenize_string_with_punctuation(self):
        # Prueba con puntuación
        self.assertEqual(tokenize("This is a test."), ["this", "is", "a", "test"])  

    def test_tokenize_with_numbers(self):
        # Confirmar que los números son tratados como tokens si no se especifica lo contrario
        self.assertEqual(tokenize("1 2 3"), ["1", "2", "3"])

    def test_tokenize_with_special_characters(self):
        # Prueba con caracteres especiales
        self.assertEqual(tokenize("special_characters!@#$%^&*()"), ["special_characters"])

    def test_tokenize_with_leading_and_trailing_spaces(self):
        # El test debe reflejar que las cadenas con guiones bajos no son divididas por tokenize
        self.assertEqual(tokenize("   leading_and_trailing_spaces   "), ["leading_and_trailing_spaces"])

    def test_tokenize_with_non_ascii_characters(self):
        # Prueba con caracteres no ASCII
        self.assertEqual(tokenize("¿Cómo estás?"), ["cómo", "estás"])

    def test_tokenize_string_with_leading_and_trailing_spaces(self):
        # Prueba con espacios al principio y al final
        self.assertEqual(tokenize("   leading_and_trailing_spaces   "), ["leading_and_trailing_spaces"])  

    def test_remove_stopwords(self):
        # Prueba con stopwords
        self.assertEqual(remove_stopwords(["this", "is", "a", "test", "of", "stopwords"]), ["test", "stopwords"])


    def test_stem_words_verbs(self):
        # Prueba con palabras verbales
        self.assertEqual(stem_words(["running", "walked"]), ["run", "walk"])  

    def test_stem_words_adjectives(self):
        # Prueba con adjetivos
        self.assertEqual(stem_words(["happiness", "happy", "happier"]), ["happi", "happi", "happier"])  

    def test_preprocess_text_string_with_new_lines(self):
        # Prueba con saltos de línea
        self.assertEqual(preprocess_text("This text has\nnew\nlines"), ["text", "new", "line"])  

    def test_preprocess_text_mixed_case_string(self):
        # Prueba con mayúsculas y minúsculas mixtas
        self.assertEqual(preprocess_text("ThiS iS a TesT"), ["test"])  

if __name__ == '__main__':
    unittest.main()
