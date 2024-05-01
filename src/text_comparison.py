# text_comparison.py

# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado

# import torch

# def jaccard_similarity(features1, features2):
#     """
#     Calcula la similitud de Jaccard entre dos vectores de características utilizando BERT.
#     Args:
#         features1 (torch.Tensor): Tensor de características del primer texto.
#         features2 (torch.Tensor): Tensor de características del segundo texto.
#     Returns:
#         float: Similitud de Jaccard entre los dos conjuntos de características.
#     """
#     intersection = torch.min(features1, features2).sum()
#     union = torch.max(features1, features2).sum()
#     return intersection / union


# def jaccard_similarity(features1, features2):
#     """
#     Calcula la similitud de Jaccard entre dos vectores de características utilizando BERT.
#     Args:
#         features1 (torch.Tensor): Tensor de características del primer texto.
#         features2 (torch.Tensor): Tensor de características del segundo texto.
#     Returns:
#         float: Similitud de Jaccard entre los dos conjuntos de características.
#     """
#     # Aplicar ReLU para asegurarse de que todos los valores sean no negativos
#     features1 = torch.relu(features1)
#     features2 = torch.relu(features2)

#     # Calcular la intersección y la unión
#     intersection = torch.min(features1, features2).sum()
#     union = torch.max(features1, features2).sum()

#     # Calcular la similitud de Jaccard
#     return intersection / union

# import torch
# import torch.nn.functional as F

# def cosine_similarity(features1, features2):
#     """
#     Calcula la similitud coseno entre dos vectores de características utilizando BERT.
#     Args:
#         features1 (torch.Tensor): Tensor de características del primer texto.
#         features2 (torch.Tensor): Tensor de características del segundo texto.
#     Returns:
#         float: Similitud coseno entre los dos conjuntos de características.
#     """
#     dot_product = torch.mm(features1, features2.t())
#     norm1 = torch.norm(features1, dim=1, keepdim=True)
#     norm2 = torch.norm(features2, dim=1, keepdim=True)
#     return dot_product / (norm1 * norm2.t())

# def euclidean_distance(features1, features2):
#     """
#     Calcula la distancia euclidiana entre dos vectores de características utilizando BERT.
#     Args:
#         features1 (torch.Tensor): Tensor de características del primer texto.
#         features2 (torch.Tensor): Tensor de características del segundo texto.
#     Returns:
#         float: Distancia euclidiana entre los dos conjuntos de características.
#     """
#     return torch.norm(features1 - features2, dim=1)

# def similarity_measure(features1, features2, method="cosine"):
#     """
#     Calcula la similitud entre dos vectores de características utilizando el método especificado.
#     Args:
#         features1 (torch.Tensor): Tensor de características del primer texto.
#         features2 (torch.Tensor): Tensor de características del segundo texto.
#         method (str): Método de similitud a utilizar ("cosine" para similitud coseno, "euclidean" para distancia euclidiana).
#     Returns:
#         float: Medida de similitud entre los dos conjuntos de características.
#     """
#     if method == "cosine":
#         return cosine_similarity(features1, features2)
#     elif method == "euclidean":
#         return euclidean_distance(features1, features2)
#     else:
#         raise ValueError("Método de similitud no válido. Utilice 'cosine' o 'euclidean'.")


# text_comparison.py

# import torch

# def cosine_similarity(features1, features2):
#     """
#     Calcula la similitud coseno entre dos vectores de características utilizando BERT.
#     Args:
#         features1 (torch.Tensor): Tensor de características del primer texto.
#         features2 (torch.Tensor): Tensor de características del segundo texto.
#     Returns:
#         float: Similitud coseno entre los dos conjuntos de características.
#     """
#     dot_product = torch.mm(features1, features2.t())
#     norm1 = torch.norm(features1, dim=1, keepdim=True)
#     norm2 = torch.norm(features2, dim=1, keepdim=True)
#     return dot_product / (norm1 * norm2.t())

# def similarity_measure(features1, features2, method="cosine"):
#     """
#     Calcula la similitud entre dos vectores de características utilizando el método especificado.
#     Args:
#         features1 (torch.Tensor): Tensor de características del primer texto.
#         features2 (torch.Tensor): Tensor de características del segundo texto.
#         method (str): Método de similitud a utilizar ("cosine" para similitud coseno).
#     Returns:
#         float: Medida de similitud entre los dos conjuntos de características.
#     """
#     if method == "cosine":
#         return cosine_similarity(features1, features2)
#     else:
#         raise ValueError("Método de similitud no válido. Utilice 'cosine'.")

import torch

def cosine_similarity(features1, features2):
    """
    Calcula la similitud coseno entre dos vectores de características utilizando BERT.
    Args:
        features1 (torch.Tensor): Tensor de características del primer texto.
        features2 (torch.Tensor): Tensor de características del segundo texto.
    Returns:
        float: Similitud coseno entre los dos conjuntos de características.
    """
    dot_product = torch.mm(features1, features2.t())
    norm1 = torch.norm(features1, dim=1, keepdim=True)
    norm2 = torch.norm(features2, dim=1, keepdim=True)
    return dot_product / (norm1 * norm2.t())

def similarity_measure(features1, features2, method="cosine"):
    """
    Calcula la similitud entre dos vectores de características utilizando el método especificado.
    Args:
        features1 (torch.Tensor): Tensor de características del primer texto.
        features2 (torch.Tensor): Tensor de características del segundo texto.
        method (str): Método de similitud a utilizar ("cosine" para similitud coseno).
    Returns:
        float: Medida de similitud entre los dos conjuntos de características.
    """
    if method == "cosine":
        return cosine_similarity(features1, features2)
    else:
        raise ValueError("Método de similitud no válido. Utilice 'cosine'.")
