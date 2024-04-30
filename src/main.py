import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
from transformers import BertTokenizer, BertModel
import os
import difflib
import threading
from utilities import load_document, save_results_to_txt, save_results_to_excel
from feature_extraction import BertFeatureExtractor
from text_comparison import jaccard_similarity

class PlagiarismCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de Plagio")
        self.root.geometry("800x600")
        self.frame = tk.Frame(self.root, bg="white")
        self.frame.pack(padx=20, pady=20)

        self.load_btn = tk.Button(self.frame, text="Cargar Documentos Sospechosos", command=self.load_suspicious_files, bg="lightgray")
        self.check_btn = tk.Button(self.frame, text="Verificar Plagio", command=self.check_plagiarism, bg="lightblue")
        self.clear_btn = tk.Button(self.frame, text="Limpiar Resultados", command=self.clear_results, bg="lightgreen")

        self.load_btn.pack(pady=5, fill=tk.X)
        self.check_btn.pack(pady=5, fill=tk.X)
        self.clear_btn.pack(pady=5, fill=tk.X)

        self.text_area = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD)
        self.text_area.pack(fill=tk.BOTH, expand=True)

        # Inicializar BERT tokenizer y modelo
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.feature_extractor = BertFeatureExtractor()

        self.original_documents = {}
        self.suspicious_documents = []
        self.results = []

        # Define el umbral de similitud para considerar un documento como plagio
        self.plagiarism_threshold = 0.75

        # Cargar documentos originales automáticamente
        self.load_original_documents("data/original")

    def load_suspicious_files(self):
        file_paths = filedialog.askopenfilenames(title="Selecciona Documentos Sospechosos", filetypes=(("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")))
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            content = load_document(file_path)
            self.suspicious_documents.append((file_name, content))
            self.text_area.insert(tk.END, f"Documento sospechoso cargado: {file_name}\n")

    def load_original_documents(self, directory):
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    file_path = os.path.join(directory, filename)
                    content = load_document(file_path)
                    self.original_documents[filename] = content
            print("Documentos originales cargados correctamente.")
        except Exception as e:
            print(f"Error al cargar documentos originales: {e}")

    def check_plagiarism(self):
        # Primero, mostramos un mensaje en la interfaz de usuario indicando que el proceso ha iniciado.
        self.text_area.insert(tk.END, "Iniciando verificación de plagio...\n")
        # Deshabilitamos el botón para evitar verificaciones múltiples simultáneas
        self.check_btn.config(state=tk.DISABLED)

        # Usamos threading para procesar en segundo plano
        plagiarism_thread = threading.Thread(target=self.run_plagiarism_check, daemon=True)
        plagiarism_thread.start()

    def run_plagiarism_check(self):
        if not self.suspicious_documents:
            messagebox.showwarning("Advertencia", "Por favor, carga documentos sospechosos.")
            return
        elif not self.original_documents:
            messagebox.showwarning("Advertencia", "No hay documentos originales cargados.")
            return

        for susp_name, susp_content in self.suspicious_documents:
            features_susp = self.feature_extractor.build_feature_vector(susp_content)
            for orig_name, orig_content in self.original_documents.items():
                features_orig = self.feature_extractor.build_feature_vector(orig_content)
                similarity = jaccard_similarity(features_susp, features_orig).item()  # Asegúrate de obtener el valor numérico con .item()
                
                # Decide si es copia basándote en el umbral
                is_copy = "Sí" if similarity > self.plagiarism_threshold else "No"
                
                # Si es una copia, intenta determinar el tipo de plagio (esto es un placeholder)
                type_of_plagiarism = self.determine_plagiarism_type(susp_content, orig_content) if is_copy == "Sí" else "Ninguno"

                result = {
                    "Documento Sospechoso": susp_name,
                    "Copia": is_copy,
                    "Documento Plagiado": orig_name if is_copy == "Sí" else "Ninguno",
                    "Tipo de Plagio": type_of_plagiarism,
                    "Similitud": similarity
                }
                self.results.append(result)
                self.root.after(0, self.display_result, result)

        # Reanudamos el botón y mostramos un mensaje de finalización.
        self.root.after(0, self.finish_plagiarism_check)

    def determine_plagiarism_type(susp_content, orig_content):
        """
        Determina el tipo de plagio de un documento sospechoso.

        Args:
        - susp_content (str): Contenido del documento sospechoso.
        - orig_content (str): Contenido del documento original.

        Returns:
        - str: Tipo de plagio detectado.
        """
        
        # Convertir a listas de frases para una comparación más fácil
        susp_sentences = susp_content.split('.')
        orig_sentences = orig_content.split('.')
        
        # Diferencia entre documentos
        sm = difflib.SequenceMatcher(None, orig_sentences, susp_sentences)
        differences = sm.get_opcodes()

        for tag, i1, i2, j1, j2 in differences:
            if tag == 'insert':
                return "Insertar o reemplazar frases"
            elif tag == 'delete':
                return "Desordenar las frases"
            elif tag == 'replace':
                return "Parafraseo"
            # Aquí se podrían añadir más reglas y refinamientos.

        # Si no se detectan diferencias significativas, no hay plagio.
        return "Ninguno"

    def display_result(self, result):
            # Aquí actualizamos la UI con los resultados individuales
        self.text_area.insert(tk.END, f"{result}\n")

    def finish_plagiarism_check(self):
        # Habilitar el botón de nuevo
        self.check_btn.config(state=tk.NORMAL)
        # Indicar que la verificación ha terminado
        self.text_area.insert(tk.END, "Verificación de plagio completada.\n")

    def clear_results(self):
        self.text_area.delete(1.0, tk.END)
        self.results.clear()

    def save_results(self):
        if self.results:
            df = pd.DataFrame(self.results)
            save_results_to_txt(self.results, "resultados_plagio.txt")
            save_results_to_excel(df, "resultados_plagio.xlsx")

def main():
    root = tk.Tk()
    app = PlagiarismCheckerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
