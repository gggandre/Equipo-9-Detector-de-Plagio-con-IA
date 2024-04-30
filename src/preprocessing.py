# Autores: A01745312 - Paula Sophia Santoyo Arteaga
#          A01753176 - Gilberto André García Gaytán
#          A01379299 - Ricardo Ramírez Condado

from transformers import BertTokenizer

def preprocess_text(text, tokenizer):
    """
    Preprocesa el texto para ser compatible con el modelo BERT.
    Args:
        text (str): Texto a preprocesar.
        tokenizer (BertTokenizer): Instancia de BertTokenizer.
    
    Returns:
        list: Lista de tokens ID según BERT.
    """
    return tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
