import numpy as np

# Predefined responses with simple word embeddings
# Here, we're using very simplified embeddings for demonstration purposes
responses = {
    "Hello! How can I help you?": np.array([1, 0, 0, 0, 0]),
    "I'm doing well, thank you!": np.array([0, 1, 0, 0, 0]),
    "I don't understand what you mean.": np.array([0, 0, 1, 0, 0]),
    "Can you provide more details?": np.array([0, 0, 0, 1, 0]),
    "Goodbye, have a great day!": np.array([0, 0, 0, 0, 1])
}

# Example word embeddings for a small vocabulary
# In a real-world scenario, use pre-trained embeddings like Word2Vec or GloVe
word_embeddings = {
    "hello": np.array([1, 0, 0, 0, 0]),
    "help": np.array([0.8, 0.2, 0, 0, 0]),
    "well": np.array([0, 1, 0, 0, 0]),
    "thank": np.array([0, 0.9, 0.1, 0, 0]),
    "understand": np.array([0, 0, 1, 0, 0]),
    "details": np.array([0, 0, 0, 1, 0]),
    "goodbye": np.array([0, 0, 0, 0, 1])
}

def text_to_vector(text):
    words = text.lower().split()
    vector = np.zeros(5)  # Size of the embedding
    for word in words:
        if word in word_embeddings:
            vector += word_embeddings[word]
    return vector

def get_response(user_input):
    user_vector = text_to_vector(user_input)
    
    # Compute cosine similarity between user input and predefined responses
    similarities = {}
    user_vector_norm = np.linalg.norm(user_vector)
    for response, vector in responses.items():
        vector_norm = np.linalg.norm(vector)
        if user_vector_norm == 0 or vector_norm == 0:
            similarity = 0
        else:
            similarity = np.dot(user_vector, vector) / (user_vector_norm * vector_norm)
        similarities[response] = similarity
    
    # Get the response with the highest similarity
    best_response = max(similarities, key=similarities.get)
    return best_response

# Chatbot loop
print("Chatbot: Hello! Type 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_input)
    print(f"Chatbot: {response}")
