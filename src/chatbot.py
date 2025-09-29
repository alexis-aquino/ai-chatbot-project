import random
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from preprocessing import preprocesstext
from vectorizer import pad_texts

# Paths
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_chatbot_model.keras")
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
INTENTS_PATH = "data/intents.json"

# Load model (prefer best model)
if os.path.exists(BEST_MODEL_PATH):
    print("Loading best model...")
    model = load_model(BEST_MODEL_PATH)
else:
    print("Best model not found, loading fallback...")
    model = load_model(FALLBACK_MODEL_PATH)

# Load tokenizer + label encoder
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Load intents
import json
with open(INTENTS_PATH, "r") as f:
    intents = json.load(f)["intents"]

# Helper: get response
def get_response(tag):
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand."

# Chat function
def chatbot_response(user_input, threshold=0.6):
    # Preprocess input
    processed = " ".join(preprocesstext(user_input))
    seq = pad_texts(tokenizer, [processed])

    # Predict
    preds = model.predict(seq, verbose=0)
    confidence = np.max(preds)
    tag = le.inverse_transform([np.argmax(preds)])[0]

    # Low confidence handling
    if confidence < threshold:
        return "Sorry, can you rephrase that?"

    return get_response(tag)

# Run chatbot
if __name__ == "__main__":
    print("Chatbot is running! (type 'quit' to exit)\n")
    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye! 👋")
            break
        response = chatbot_response(msg)
        print(f"Bot: {response}")
