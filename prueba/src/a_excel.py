import os
import pandas as pd

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
    
    original_folder = "data/otros1"
    copy_folder = "data/copias"
    
    original_files = os.listdir(original_folder)
    copy_files = os.listdir(copy_folder)
    
    for original_file in original_files:
        if original_file.endswith(".txt"):
            original_identifier = original_file.split("-")[1].split(".")[0]
            with open(os.path.join(original_folder, original_file), "r", encoding="latin1") as f:
                original_text = f.read()

            for copy_file in copy_files:
                if copy_file.endswith(".txt") and original_identifier in copy_file:
                    with open(os.path.join(copy_folder, copy_file), "r", encoding="utf-8") as f2:
                        copy_text = f2.read()
                    
                    is_copy = 1  # Siempre es una copia
                    
                    # Obtener el tipo de plagio del nombre del archivo de copia
                    copy_type_code = copy_file.split("-")[0][:2]  # Código de tipo de plagio
                    copy_type_mapping = {
                        "P": "Parafraseo",
                        "D": "Desordenar las frases",
                        "CV": "Cambio de voz",
                        "CT": "Cambio de tiempo",
                        "IR": "Insertar o reemplazar frases"
                    }
                    
                    copy_type = copy_type_mapping.get(copy_type_code, "Desconocido")
                    print(copy_file)
                    # Obtener el porcentaje de plagio del nombre del archivo de copia
                    plagiarism_percentage = int(copy_file.split("-")[0][2:])  # Porcentaje de plagio
                    
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
