# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado
import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
import sys
import os

# Esto agrega la carpeta 'src' al path para que se pueda importar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from main import PlagiarismCheckerApp, load_original_documents

class TestPlagiarismCheckerApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()  # Oculta la ventana principal

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()  # Destruye la ventana principal después de ejecutar todas las pruebas

    def setUp(self):
        # Crea una nueva instancia de PlagiarismCheckerApp para cada test
        self.app = PlagiarismCheckerApp(self.root)

    def tearDown(self):
        # Limpia la ventana después de cada test para evitar problemas con Tkinter
        for widget in self.root.winfo_children():
            widget.destroy()

    def test_init(self):
        # Verifica que el título de la ventana sea el correcto
        self.assertEqual(self.app.root.title(), "Plagiarism Detection System")
        # Fuerza la actualización de la ventana para asegurar que se aplique la geometría
        self.app.root.update_idletasks()
        # Ahora obtén la geometría y comprueba que sea 800x600
        geometry = self.app.root.geometry().split('+')[0]
        self.assertEqual(geometry, "800x600")

    @patch('tkinter.filedialog.askopenfilenames')
    @patch('main.load_document', side_effect=lambda x: f"Contents of {x}")
    def test_load_suspicious_files(self, mocked_load_document, mocked_askopenfilenames):
        # Simula la selección de archivos
        mocked_askopenfilenames.return_value = ['path/to/doc1.txt', 'path/to/doc2.txt']
        self.app.load_suspicious_files()
        self.assertEqual(self.app.suspicious_documents, ["Contents of path/to/doc1.txt", "Contents of path/to/doc2.txt"])
        # Verifica que el texto esperado esté en el área de texto
        self.assertIn("Loaded 2 suspicious documents.", self.app.text_area.get('1.0', tk.END))

    @patch('tkinter.messagebox.showwarning')  # Cambia showinfo a showwarning
    def test_check_plagiarism_without_documents(self, mocked_showwarning):
        self.app.original_documents = []
        self.app.suspicious_documents = []
        self.app.check_plagiarism()
        mocked_showwarning.assert_called_once_with("Warning", "Please load both suspicious and original documents.")

    @patch('main.save_results_to_txt')
    @patch('main.save_results_to_excel')
    def test_save_results(self, mocked_save_to_excel, mocked_save_to_txt):
        # Configura el estado necesario antes de llamar a check_plagiarism
        self.app.all_results = [('Doc1', 'Doc2', 0.9)]
        # Asegúrate de que hay documentos para que el método proceda a guardar
        self.app.original_documents = ["Dummy content"]
        self.app.suspicious_documents = ["Dummy content"]
        self.app.check_plagiarism()
        # Verifica que save_results_to_txt fue llamada
        mocked_save_to_txt.assert_called_once()

    def test_clear_results(self):
        # Verifica que el área de texto se limpie correctamente
        self.app.text_area.insert(tk.END, "Some results")
        self.app.clear_results()
        # Verifica que el área de texto esté vacía después de limpiar
        self.assertEqual(self.app.text_area.get('1.0', tk.END), '\n')

# Test para cargar documentos originales
class TestLoadOriginalDocuments(unittest.TestCase):

    @patch('os.listdir')
    @patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    @patch('main.load_document', side_effect=lambda x: f"Contents of {x}")
    def test_load_original_documents(self, mocked_load_document, mocked_join, mocked_listdir):
        # Simula que hay documentos en el directorio
        mocked_listdir.return_value = ['doc1.txt', 'doc2.txt']
        # Simula la carga de documentos
        result = load_original_documents('/path/to/originals')
        # Verifica que los documentos se carguen correctamente
        self.assertEqual(result, ['Contents of /path/to/originals/doc1.txt', 'Contents of /path/to/originals/doc2.txt'])

if __name__ == '__main__':
    unittest.main()
