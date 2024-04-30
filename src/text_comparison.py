# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado

import torch

def jaccard_similarity(features1, features2):
    """
    Calcula la similitud de Jaccard entre dos vectores de características utilizando BERT.
    Args:
        features1 (torch.Tensor): Tensor de características del primer texto.
        features2 (torch.Tensor): Tensor de características del segundo texto.
    Returns:
        float: Similitud de Jaccard entre los dos conjuntos de características.
    """
    intersection = torch.min(features1, features2).sum()
    union = torch.max(features1, features2).sum()
    return intersection / union
