# Equipo-9-Detector-de-Plagio-con-IA

Este proyecto implementa un sistema de detección de plagio utilizando técnicas de procesamiento de texto y aprendizaje automático para identificar similitudes entre documentos y detectar diferentes tipos de plagio.

## Autores

- Paula Sophia Santoyo Arteaga - A01745312
- Gilberto André García Gaytán - A01753176
- Ricardo Ramírez Condado - A01379299

## Tecnologías Utilizadas

- **Python:** Lenguaje de programación utilizado para el desarrollo del proyecto.
- **NLTK (Natural Language Toolkit):** Utilizado para el procesamiento del lenguaje natural, incluyendo la tokenización y eliminación de stopwords.
- **Scikit-learn:** Utilizado para funciones de aprendizaje automático como TF-IDF, SVC y métricas de rendimiento.
- **Pandas:** Utilizado para la manipulación y análisis de datos.
- **Matplotlib:** Utilizado para generar visualizaciones de datos como matrices de confusión y curvas ROC.

## Funcionalidades Principales

1. **Preprocesamiento de Texto**: Eliminación de stopwords, stemming y lematización para preparar los textos para análisis.
2. **Extracción de Características**: Utilización de TF-IDF para convertir texto a un formato numérico que pueda ser procesado por modelos de machine learning.
3. **Entrenamiento de Modelo de Machine Learning**: Uso de un modelo SVM para clasificar documentos como originales o copias.
4. **Evaluación del Modelo**: Uso de métricas como AUC, recall y F1 para evaluar la efectividad del modelo.
5. **Detección de Plagio**: Identificación de documentos plagiados utilizando el modelo entrenado.
6. **Detección de Parafraseo y Otros Tipos de Plagio**: Identificación de parafraseo, cambio de tiempo, cambio de voz, y otras formas de alteraciones entre documentos.

## Cómo Funciona

El sistema procesa los documentos de entrada realizando primero un preprocesamiento que incluye la eliminación de stopwords, stemming y lematización. Luego, utiliza TF-IDF para extraer características de los textos. Estas características alimentan un modelo SVM entrenado que clasifica cada documento como plagio o no. El sistema evalúa su rendimiento utilizando varias métricas y proporciona visualizaciones como la matriz de confusión y la curva ROC.

## Ejecución del Programa

Para ejecutar el programa principal que realiza la detección de plagio, utiliza el siguiente comando en la consola:

```python -m unittest src/plagiarism_detection.py```
![image](https://github.com/gggandre/Equipo-9-Detector-de-Plagio-con-IA/assets/84719490/ee12f352-5b1a-4ccb-b2b3-0838a4c61c1f)

Este comando ejecutará el script principal, procesando los documentos, entrenando el modelo, y evaluando los resultados. El programa está diseñado para guardar los resultados de la detección de plagio de la siguiente manera:

### Guardado de Resultados
Una vez completada la ejecución del programa:
![image](https://github.com/gggandre/Equipo-9-Detector-de-Plagio-con-IA/assets/84719490/fbb3926f-ae2d-4df4-a256-cd0c99e0155c)
**Resultados en Texto:** Los resultados de la detección se guardarán en un archivo .txt dentro de la carpeta results. Este archivo contendrá detalles sobre cada documento analizado, incluyendo si se detectó plagio y el tipo de plagio identificado.
**Resultados en Excel:** Adicionalmente, los resultados se guardarán en un archivo .xlsx dentro de la misma carpeta sresults. Este archivo de Excel facilita una visualización y análisis más detallado, permitiendo filtrar y ordenar los datos según sea necesario.
Estos archivos proporcionan una documentación completa y detallada de todos los análisis realizados, permitiendo un seguimiento fácil de las instancias detectadas de plagio y otros detalles relevantes del análisis.

## Ejecución de Tests

El proyecto incluye pruebas unitarias para cada función principal, asegurando que cada componente funcione correctamente de manera aislada. Para ejecutar los tests:

```python -m unittest tests/test_plagiarism_detection.py```

![image](https://github.com/gggandre/Equipo-9-Detector-de-Plagio-con-IA/assets/84719490/3f40682f-4db7-4089-a472-02c0be72a525)
