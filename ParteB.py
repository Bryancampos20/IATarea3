import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar el modelo de spaCy para generar embeddings
nlp = spacy.load('en_core_web_md')

# Definir las respuestas
respuestas_larga = [
    "We are open from 10 AM to 10 PM.",
    "You can pay with credit card, cash, or mobile payment.",
    "Our pizzas come in three sizes: small, medium, and large.",
    "We offer delivery services from 11 AM to 9 PM.",
    "Yes, we have gluten-free pizza options.",
    "You can call us at 123-456-7890.",
    "Our most popular pizza is the Margherita.",
    "We are located at 123 Main Street.",
    "You can find our menu online on our website.",
    "We do offer vegan cheese for our pizzas."
]

# Frases para embeddings
respuestas_enveding = [
    "open close schedule",
    "pay credit card money",
    "sizes",
    "delivery deliver",
    "gluten-free",
    "call contact",
    "popular best-selling",
    "location ubication",
    "menu online",
    "vegan"
]

# Generar embeddings para las respuestas y normalizarlas
embeddings_respuestas = np.array([nlp(respuesta).vector for respuesta in respuestas_enveding])
norms_respuestas = np.linalg.norm(embeddings_respuestas, axis=1, keepdims=True)
embeddings_respuestas = embeddings_respuestas / norms_respuestas

# Palabras "basura" a eliminar
palabras_basura = ["what", "how", "where", "when", "do", "you", "can", "I", "with", "a", "your", "pizza", "pizzas", "have"]

# Definir las preguntas del usuario
preguntas_usuario = [
    "What time do you close?",
    "Can I pay with a credit card?",
    "What sizes do your pizzas come in?",
    "Do you deliver?",
    "Do you offer gluten-free pizzas?",
    "How can I contact you?",
    "What is your best-selling pizza?",
    "Where are you located?",
    "Where can I see your menu?",
    "Do you have vegan options?"
]

# Procesar cada pregunta
for pregunta in preguntas_usuario:
    # Limpiar la pregunta
    pregunta_limpia = ' '.join([palabra for palabra in pregunta.lower().split() if palabra not in palabras_basura])
    
    # Generar embedding para la pregunta limpia
    embedding_pregunta = nlp(pregunta_limpia).vector
    embedding_pregunta = embedding_pregunta / np.linalg.norm(embedding_pregunta)  # Normalizar el embedding de la pregunta
    
    # Calcular la similitud de coseno con todas las respuestas
    similitudes = cosine_similarity([embedding_pregunta], embeddings_respuestas).flatten()
    
    # Encontrar el índice de la respuesta más similar
    indice_mejor_respuesta = np.argmax(similitudes)
    mejor_similitud = similitudes[indice_mejor_respuesta]
    
    # Verificar si la mejor similitud cumple con el umbral mínimo
    umbral_similitud = 0.1  # Ajusta este valor según sea necesario
    if mejor_similitud >= umbral_similitud:
        respuesta_seleccionada = respuestas_larga[indice_mejor_respuesta]
    else:
        respuesta_seleccionada = "I'm sorry, I don't understand your question. Could you rephrase it?"

    print(f"Pregunta: {pregunta}")
    print(f"Respuesta seleccionada: {respuesta_seleccionada} (Similitud: {mejor_similitud:.2f})\n")
