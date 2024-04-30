# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado

import pandas as pd

def load_document(filename):
    """
    Carga un documento desde un archivo con manejo de diferentes codificaciones.

    Args:
    - filename (str): Nombre del archivo a cargar.

    Returns:
    - str: Contenido del documento.
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']  # Lista de codificaciones comunes
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar el archivo {filename} con las codificaciones comunes.")

def save_results_to_txt(results, filename):
    """
    Guarda los resultados en un archivo de texto.
    Args:
        results (list): Lista de resultados a guardar.
        filename (str): Nombre del archivo de salida.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(str(result) + '\n')

def save_results_to_excel(dataframe, filename):
    """
    Guarda los resultados en un archivo Excel.
    Args:
        dataframe (pandas.DataFrame): DataFrame que contiene los resultados.
        filename (str): Nombre del archivo de salida.
    """
    dataframe.to_excel(filename, index=False)
