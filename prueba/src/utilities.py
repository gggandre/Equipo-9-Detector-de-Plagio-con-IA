import os
import pandas as pd

def load_originals(originals_path):
    originals = []
    original_labels = []
    
    # Cargar datos de los documentos originales
    for filename in os.listdir(originals_path):
        with open(os.path.join(originals_path, filename), 'r', encoding='latin-1') as file:
            originals.append(file.read())
        original_labels.append(0)  # Etiqueta 0 para documentos originales
    
    return originals, original_labels

def load_copies(copies_path):
    copies = []
    copy_labels = []
    
    # Cargar datos de los documentos de copia
    for filename in os.listdir(copies_path):
        with open(os.path.join(copies_path, filename), 'r', encoding='latin-1') as file:
            copies.append(file.read())
        copy_labels.append(1)  # Etiqueta 1 para documentos de copia
    
    return copies, copy_labels

def load_data(originals_path, copies_path):
    originals, original_labels = load_originals(originals_path)
    copies, copy_labels = load_copies(copies_path)
    
    # Combinar datos originales y de copia
    all_documents = originals + copies
    all_labels = original_labels + copy_labels
    
    return all_documents, all_labels

def save_results(predictions, txt_path, xlsx_path):
    with open(txt_path, 'w') as txt_file:
        for prediction in predictions:
            txt_file.write(str(prediction) + '\n')
    
    df = pd.DataFrame({'Predictions': predictions})
    df.to_excel(xlsx_path, index=False)

import pandas as pd

def save_results(predictions, output_txt, output_excel, originals, copies, plagiarism_types, plagiarism_percentages):
    results = {
        'Archivo Original': originals,
        'Archivo Copia': copies,
        '¿Es plagio?': predictions,
        'Tipo de plagio': plagiarism_types,
        'Porcentaje de plagio': plagiarism_percentages
    }

    # Guardar los resultados en un archivo de texto
    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        for original, copy, prediction, p_type, p_percentage in zip(originals, copies, predictions, plagiarism_types, plagiarism_percentages):
            txt_file.write('-----------------------------------------------------\n')
            txt_file.write(f"Archivo original: {original}\n")
            txt_file.write(f"Archivo de copia: {copy}\n")
            txt_file.write("¿Es plagio?: Sí\n" if prediction == 1 else "¿Es plagio?: No\n")
            txt_file.write(f"Tipo de plagio: {p_type}\n")
            txt_file.write(f"Porcentaje de plagio: {p_percentage}%\n")
            txt_file.write('-----------------------------------------------------\n\n')

    # Guardar los resultados en un archivo Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
