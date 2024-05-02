from feature_extraction import extract_features
from preprocessing import preprocess_text
from model import train_model, detect_plagiarism
from utilities import load_data, save_results, load_originals, load_copies
from text_comparison import calculate_similarity
import os

def main():
    # Cargar datos
    originals_path = "data/original"
    copies_path = "data/copias"
    all_documents, all_labels = load_data(originals_path, copies_path)
    
    # Preprocesamiento de textos
    preprocessed_documents = [preprocess_text(text) for text in all_documents]
    
    # Extracción de características
    features = extract_features(preprocessed_documents)

    # Separar datos originales y de copia
    num_originals = len(load_originals(originals_path)[0])
    features_originals = features[:num_originals]
    features_copies = features[num_originals:]
    
    # Entrenamiento del modelo
    model = train_model(features_originals, features_copies, all_labels)
    
    # Detección de plagio
    predictions = detect_plagiarism(model, features_copies)
    
    # Analizar los tipos de plagio
    originals = all_documents[:num_originals]
    copies = all_documents[num_originals:]
    plagiarism_types = []
    plagiarism_percentages = []
    for copy in copies:
        for original in originals:
            similarity_score = calculate_similarity(original, copy)
            if similarity_score >= 0.75:  # Ajusta el umbral según tus necesidades
                p_type, p_percentage = detect_plagiarism_type(original, copy)
                plagiarism_types.append(p_type)
                plagiarism_percentages.append(p_percentage)
                break
        else:
            # Si no se encuentra ninguna coincidencia con ningún original, agregamos valores vacíos
            plagiarism_types.append('')
            plagiarism_percentages.append('')
    
    # Guardar resultados
    save_results(predictions, "results/similarity.txt", "results/similarity.xlsx", originals, copies, plagiarism_types, plagiarism_percentages)



def analyze_plagiarism_types(originals, predictions, num_original, threshold=0.75):

    contador = 0
    for original, prediction in zip(originals, predictions):
        contador = contador + 1
        #print(original, prediction)
        for copy in originals[num_original:]:
            similarity_score = calculate_similarity(original, copy)
            
            if similarity_score >= threshold:
                plagiarism_type, plagiarism_percentage = detect_plagiarism_type(original, copy)
                print(contador)
                print('-----------------------------------------------------')
                print(f"Archivo original: {original.encode('utf-8', 'ignore')} \n")
                print(f"Archivo de copia: {copy.encode('utf-8', 'ignore')}\n")
                print(f"¿Es plagio?: Sí")
                print(f"Tipo de plagio: {plagiarism_type}")
                print(f"Porcentaje de plagio: {plagiarism_percentage}%")
                print('-----------------------------------------------------')
                print("\n")
                break 
        

def detect_plagiarism_type(original_text, copy_text):
    # Aquí implementarás la lógica para detectar los diferentes tipos de plagio.
    # Por ahora, haremos una detección simple basada en la similitud de coseno entre los documentos.
    # Puedes agregar lógica adicional para identificar patrones específicos según los tipos de plagio.
    similarity_score = calculate_similarity(original_text, copy_text)
    plagiarism_percentage = round(similarity_score * 100, 2)

    # Aquí puedes agregar más lógica para identificar los tipos de plagio basados en patrones específicos.

    # Por ejemplo, si el porcentaje de similitud es alto, podrías clasificarlo como parafraseo.
    # Si el porcentaje de similitud es menor, podrías clasificarlo como inserción o reemplazo de frases.

    if plagiarism_percentage >= 90:
        plagiarism_type = "Parafraseo"
    else:
        plagiarism_type = "Insertar o reemplazar frases"  # Este es solo un ejemplo, debes agregar más lógica

    return plagiarism_type, plagiarism_percentage

if __name__ == "__main__":
    main()