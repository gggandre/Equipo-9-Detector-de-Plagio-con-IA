import os
import pandas as pd
from difflib import SequenceMatcher

def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio() * 100

def detect_copy(original_text, copy_text):
    similarity_percentage = calculate_similarity(original_text, copy_text)
    
    if similarity_percentage == 100:
        copy_type = "Reemplazo completo"
    elif similarity_percentage > 80:
        copy_type = "Cambio de tiempo"
    elif similarity_percentage > 50:
        copy_type = "Parafraseo"
    else:
        copy_type = "Desconocido"
    
    return similarity_percentage, copy_type

def main():
    data = {
        "original_name": [],
        "original_text": [],
        "copy_name": [],
        "copy_text": [],
        "is_copy": [],
        "copy_type": [],
        "percentage": []
    }
    
    original_folder = "data/original"
    copy_folder = "data/copias"
    
    original_files = os.listdir(original_folder)
    copy_files = os.listdir(copy_folder)
    
    for original_file in original_files:
        if original_file.endswith(".txt"):
            original_identifier = original_file.split("-")[1].split(".")[0]
            with open(os.path.join(original_folder, original_file), "rb") as f:
                original_text = f.read().decode("utf-8", errors="ignore")

                
            for copy_file in copy_files:
                if copy_file.endswith(".txt") and original_identifier in copy_file:
                    with open(os.path.join(copy_folder, copy_file), "r", encoding="utf-8") as f2:
                        copy_text = f2.read()
                    
                    
                    # Siempre es una copia, entonces is_copy es 1
                    is_copy = 1
                    
                    # Analizar el nombre del archivo de copia para determinar el porcentaje de plagio
                    plagiarism_percentage = int(copy_file.split("-")[0][2:])  # Porcentaje de plagio
                    
                    # Mapear el código de copia a una descripción
                    copy_type_mapping = {
                        "P": "Parafraseo",
                        "D": "Desordenar las frases",
                        "CV": "Cambio de voz",
                        "CT": "Cambio de tiempo",
                        "IR": "Insertar o reemplazar frases"
                    }
                    
                    copy_type = copy_type_mapping.get(copy_file.split("-")[0][:2], "Desconocido")
                    
                    data["original_name"].append(original_file)
                    data["original_text"].append(original_text)
                    data["copy_name"].append(copy_file)
                    data["copy_text"].append(copy_text)
                    data["is_copy"].append(is_copy)
                    data["copy_type"].append(copy_type)
                    data["percentage"].append(plagiarism_percentage)
                    break  # Salir del bucle una vez que se haya encontrado la correspondencia
    
    df = pd.DataFrame(data)
    df.to_excel("resultado.xlsx", index=False)

if __name__ == "__main__":
    main()
