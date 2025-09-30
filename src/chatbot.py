import random
import pickle
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from preprocessing import preprocesstext
from vectorizer import pad_texts
from context_manager import ContextManager

# ================================
# Paths
# ================================
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_chatbot_model.keras")
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
INTENTS_PATH = "data/intents.json"

# ================================
# Load Model
# ================================
if os.path.exists(BEST_MODEL_PATH):
    print("Loading best model...")
    model = load_model(BEST_MODEL_PATH)
else:
    print("Best model not found, loading fallback...")
    model = load_model(FALLBACK_MODEL_PATH)

# ================================
# Load Tokenizer + Label Encoder
# ================================
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# ================================
# Load Intents
# ================================
with open(INTENTS_PATH, "r") as f:
    intents = json.load(f)["intents"]

# ================================
# Context Manager
# ================================
context = ContextManager()

# ================================
# Helper: Get Response
# ================================
def get_response(tag):
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand."

# ================================
# Chatbot Response Function
# ================================
def chatbot_response(user_input, threshold=0.6):
    # Check for context-based replies
    if user_input.lower() in ["yes", "yeah", "yep", "sure"]:
        last_intent = context.get_intent()
        if last_intent == "study_help":
            return "Awesome! What programming language are you learning?"
        elif last_intent == "tech_support":
            return "Great, tell me more about the issue with your device."
        elif last_intent == "smalltalk_creator":
            return "Haha, I’m glad you’re curious about who made me!"
        else:
            return "Yes to what? Can you clarify?"

    if user_input.lower() in ["no", "nope", "nah"]:
        last_intent = context.get_intent()
        if last_intent == "study_help":
            return "No worries! Let me know if you need help later."
        elif last_intent == "tech_support":
            return "Alright, I’ll stay out of your devices 😅"
        else:
            return "Got it, thanks for letting me know."

    # Normal prediction flow
    processed = " ".join(preprocesstext(user_input))
    seq = pad_texts(tokenizer, [processed])

    preds = model.predict(seq, verbose=0)
    confidence = np.max(preds)
    tag = le.inverse_transform([np.argmax(preds)])[0]

    if confidence < threshold:
        return "Sorry, can you rephrase that?"

    # Save context
    context.set_intent(tag)

    return get_response(tag)

# ================================
# Run Chatbot
# ================================
if __name__ == "__main__":
    print("Chatbot is running! (type 'quit' to exit)\n")
    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye! 👋")
            break
        response = chatbot_response(msg)
        print(f"Bot: {response}")
