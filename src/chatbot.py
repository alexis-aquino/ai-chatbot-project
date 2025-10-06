import random
import pickle
import numpy as np
import os
import json
from tensorflow.keras.models import load_model

from preprocessing import preprocesstext
from vectorizer import pad_texts
from context_manager import ContextManager
from ner import extract_entities   # src/ner.py -> extract_entities(text)

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
with open(INTENTS_PATH, "r") as f:
    intents = json.load(f)["intents"]

# Context manager
context = ContextManager()

# Helper: get generic response for tag
def get_response(tag):
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand."

# Booking helper: assemble booking reply from entities
def booking_reply(entities):
    # look for place in GPE or LOC
    dest = None
    for key in ("GPE", "LOC", "FAC", "ORG"):
        if key in entities and entities[key]:
            dest = entities[key][0]
            break

    date = entities.get("DATE", [None])[0] if "DATE" in entities else None

    if dest and date:
        return f"Got it! Booking flight to {dest} on {date}."
    if dest:
        return f"Got it! Booking flight to {dest}. When would you like to travel?"
    if date:
        return f"Got it! Booking flight on {date}. Where would you like to go?"
    return "Sure — where do you want to fly to, and when?"

# Main response function
def chatbot_response(user_input, threshold=0.6, debug=False):
    user_input_lower = user_input.lower().strip()

    # quick yes/no context handling
    if user_input_lower in ["yes", "yeah", "yep", "sure"]:
        last_intent = context.get_intent()
        if last_intent == "study_help":
            return "Awesome! What programming language are you learning?"
        if last_intent == "tech_support":
            return "Great — can you describe the issue in more detail?"
        if last_intent == "book_flight":
            return "Do you want me to proceed with booking? If yes, tell me destination and date."
        return "Yes to what? Can you clarify?"

    if user_input_lower in ["no", "nah", "nope"]:
        last_intent = context.get_intent()
        if last_intent:
            context.clear()   # optionally clear context when user declines
        return "Okay, no problem. Let me know if you need anything else."

    # Normal flow: preprocess -> predict intent
    processed = " ".join(preprocesstext(user_input))
    seq = pad_texts(tokenizer, [processed])

    preds = model.predict(seq, verbose=0)
    intent_idx = int(np.argmax(preds))
    confidence = float(preds[0][intent_idx])
    tag = le.inverse_transform([intent_idx])[0]

    # Debug
    if debug:
        entities_dbg = extract_entities(user_input)
        print(f"[DEBUG] intent={tag} conf={confidence:.3f} entities={entities_dbg}")

    # Low confidence handling
    if confidence < threshold:
        return "Sorry, can you rephrase that?"

    # Extract entities (intent-specific usage)
    entities = extract_entities(user_input)

    # Intent-specific behavior
    if tag == "book_flight":
        reply = booking_reply(entities)
        # Save context for follow-ups like "yes"
        context.set_intent(tag)
        return reply

    # Default: store intent and return generic response
    context.set_intent(tag)

    # If entities are present but not a special intent you handle, you can mention them optionally:
    # e.g., user asked "tell me about Manila" (fun_fact + GPE). Here we keep it simple.
    return get_response(tag)

# Run chatbot loop
if __name__ == "__main__":
    print("Chatbot is running with intent+entity extraction (type 'quit' to exit)\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye!")
            break
        resp = chatbot_response(msg)
        print("Bot:", resp)
