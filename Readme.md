# Tarea 03: Aplicación de Filtros sobre Imágenes y Desarrollo de Chatbot

## Descripción General

Este proyecto se divide en dos partes principales: la aplicación de filtros a imágenes utilizando OpenCV y el desarrollo de un chatbot basado en embeddings de texto utilizando la biblioteca spaCy.

### Parte A: Aplicación de Filtros a Imágenes

En esta parte del proyecto, se aplican diversos filtros a imágenes del Dataset Covid-19 Image Dataset disponible en Kaggle. Los filtros utilizados son:

1. **Blur Filter**
2. **Sobel Edge Detection**
3. **Canny Edge Detection**
4. **Color Map**

El objetivo de esta sección es explorar cómo estos filtros afectan las características visuales de las imágenes y mejorar la capacidad de los clasificadores al eliminar ruido o destacar bordes. Los resultados obtenidos y las comparaciones entre las imágenes originales y filtradas se presentan en el informe adjunto.

### Parte B: Desarrollo de Chatbot Basado en Embeddings

Esta sección del proyecto involucra la creación de un chatbot diseñado para responder a preguntas relacionadas con una pizzería. El chatbot utiliza la similitud de coseno entre embeddings de texto para seleccionar la respuesta más apropiada a una consulta dada.

**Pasos implementados:**
1. **Definición de propósito y respuestas del chatbot**: Se definieron 10 respuestas posibles que el chatbot puede ofrecer.
2. **Transformación de respuestas en embeddings**: Utilizando la biblioteca spaCy, se convirtieron las respuestas en vectores de embeddings.
3. **Procesamiento de preguntas del usuario**: Las preguntas ingresadas por el usuario se limpian eliminando palabras no significativas y se convierten en embeddings.
4. **Comparación de similitud de coseno**: Se calcula la similitud de coseno entre el embedding de la pregunta del usuario y las respuestas predefinidas para seleccionar la respuesta más adecuada.

El código de ambas partes está desarrollado en Jupyter Notebooks, y se incluye un informe detallado en formato PDF que sigue la plantilla de artículos científicos de IEEE.

## Estructura del Proyecto

- `ParteA.ipynb`: Contiene el código para la aplicación de filtros a las imágenes.
- `ParteB.ipynb`: Contiene el código para el desarrollo del chatbot basado en embeddings.
- `Informe.pdf`: Informe detallado que describe los métodos utilizados, los resultados obtenidos y el análisis correspondiente.
- `Instrucciones.pdf`: Documento de referencia de la tarea que incluye la rúbrica y los pasos a seguir.

## Requisitos

- **Python 3.x**
- **Jupyter Notebook**
- **Bibliotecas**:
  - OpenCV
  - spaCy
  - NumPy
  - Scikit-learn

## Instrucciones para Ejecutar

1. Clone o descargue este repositorio.
2. Asegúrese de tener instaladas todas las bibliotecas requeridas.
3. Abra los archivos `.ipynb` en Jupyter Notebook.
4. Ejecute cada celda en el orden provisto para replicar los resultados.

## Referencias

- [Edge Detection using OpenCV](https://learnopencv.com/edge-detection-using-opencv/)
- [Word Embeddings and Semantic Similarity using spaCy](https://ashutoshtripathi.com/2020/09/04/word2vec-and-semantic-similarity-using-spacy-nlp-spacy-series-part-7/)
- [Covid-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
- [Understanding Word Embeddings](https://www.freecodecamp.org/news/understanding-word-embeddings-the-building-blocks-of-nlp-and-gpts/)

## Autor

- **Bryan Campos Castro** - Alajuela, Costa Rica - [bryancampos20@gmail.com](mailto:bryancampos20@gmail.com)
- **Miguel David Sánchez Sánchez** - Heredia, Costa Rica - [miguelsanchez712000@gmail.com](mailto:miguelsanchez712000@gmail.com)
