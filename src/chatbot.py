import random
import pickle
import numpy as np
import os
import json
import requests
from tensorflow.keras.models import load_model

from preprocessing import preprocesstext
from vectorizer import pad_texts
from context_manager import ContextManager
from ner import extract_entities   # src/ner.py -> extract_entities(text)

# ========================
# CONFIG & PATHS
# ========================

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_chatbot_model.keras")
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
INTENTS_PATH = "data/intents.json"
CONFIG_PATH = "config.json"

# Load config (for API keys, etc.)
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing {CONFIG_PATH}. Please create one with your API key.")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

WEATHER_API_KEY = config.get("openweather_api_key", "").strip()
if not WEATHER_API_KEY:
    raise ValueError("Missing 'openweather_api_key' in config.json")

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# ========================
# LOAD MODEL & COMPONENTS
# ========================

if os.path.exists(BEST_MODEL_PATH):
    print("Loading best model...")
    model = load_model(BEST_MODEL_PATH)
else:
    print("Best model not found, loading fallback...")
    model = load_model(FALLBACK_MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

with open(INTENTS_PATH, "r") as f:
    intents = json.load(f)["intents"]

context = ContextManager()

# ========================
# HELPERS
# ========================

def get_response(tag):
    """Return a random response for the given intent tag."""
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand."


def booking_reply(entities):
    """Assemble booking response based on extracted entities."""
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


# 🌦️ WEATHER HELPER
def get_weather(city):
    """Fetch weather info for a city using OpenWeatherMap API."""
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
    try:
        res = requests.get(WEATHER_URL, params=params, timeout=5)

        if res.status_code == 401:
            return "API key rejected (401). Try regenerating it or waiting a bit — sometimes new keys take up to an hour to activate."
        if res.status_code == 404:
            return f"Couldn't find weather data for '{city}'. Try another city name."
        if res.status_code != 200:
            return f"Error fetching weather: {res.status_code} ({res.reason})"

        data = res.json()
        if "main" not in data:
            return f"Sorry, couldn't find weather info for '{city}'."

        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].capitalize()
        feels = data["main"].get("feels_like", temp)
        humidity = data["main"].get("humidity", "N/A")

        return f"The weather in {city.title()} is {desc} with {temp}°C (feels like {feels}°C), humidity at {humidity}%."
    except Exception as e:
        return f"Error fetching weather: {e}"


# 😂 JOKE HELPER
def get_joke():
    """Fetch a random joke from the Official Joke API."""
    try:
        res = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=5)
        if res.status_code != 200:
            return "Couldn't fetch a joke right now, sorry!"
        data = res.json()
        return f"{data['setup']} ... {data['punchline']}"
    except Exception as e:
        return f"Error fetching joke: {e}"


# 💬 QUOTE HELPER
def get_quote():
    """Fetch a random quote from Quotable API."""
    try:
        res = requests.get("https://api.quotable.io/random", timeout=5)
        if res.status_code != 200:
            return "Couldn't fetch a quote at the moment, sorry!"
        data = res.json()
        return f"“{data['content']}” — {data['author']}"
    except Exception as e:
        return f"Error fetching quote: {e}"

# ========================
# MAIN RESPONSE FUNCTION
# ========================

def chatbot_response(user_input, threshold=0.6, debug=False):
    user_input_lower = user_input.lower().strip()
    ctx = context.get_context()

    # 🌦️ Follow-up: user mentions another city after weather query
    if ctx["last_api_type"] == "weather":
        entities = extract_entities(user_input)
        city = None
        for key in ("GPE", "LOC", "FAC"):
            if key in entities and entities[key]:
                city = entities[key][0]
                break

        # fallback if city not recognized but user typed one word
        if not city:
            tokens = user_input_lower.split()

            if tokens:
                possible_city = tokens[-1].strip("?.1,")

                skip_words = {"about","what","how","the","is","and"}
                if possible_city not in skip_words and len(possible_city) > 2:
                    city = possible_city

        if city:
            context.set_context(entity=city)  # update only entity
            return get_weather(city)
        else:
            return "I didn’t catch the city name. Can you repeat it?"

    # ✅ Handle yes/no quick responses
    if user_input_lower in ["yes", "yeah", "yep", "sure"]:
        last_intent = ctx["last_intent"]
        if last_intent == "study_help":
            return "Awesome! What programming language are you learning?"
        if last_intent == "tech_support":
            return "Great — can you describe the issue in more detail?"
        if last_intent == "book_flight":
            return "Do you want me to proceed with booking? If yes, tell me destination and date."
        return "Yes to what? Can you clarify?"

    if user_input_lower in ["no", "nah", "nope"]:
        context.clear_context()
        return "Okay, no problem. Let me know if you need anything else."

    # 🧠 Preprocess → predict intent
    processed = " ".join(preprocesstext(user_input))
    seq = pad_texts(tokenizer, [processed])

    preds = model.predict(seq, verbose=0)
    intent_idx = int(np.argmax(preds))
    confidence = float(preds[0][intent_idx])
    tag = le.inverse_transform([intent_idx])[0]

    if confidence < threshold:
        return "Sorry, can you rephrase that?"

    entities = extract_entities(user_input)
    context.set_context(intent=tag)

    # 🌦️ Weather intent
    if tag == "weather":
        city = None
        for key in ("GPE", "LOC", "FAC"):
            if key in entities and entities[key]:
                city = entities[key][0]
                break

        # fallback if entity extractor doesn't detect city
        if not city:
            tokens = user_input_lower.split()
            if "in" in tokens:
                idx = tokens.index("in")
                if idx + 1 < len(tokens):
                    city = tokens[idx + 1]

        if city:
            context.set_context()
            return get_weather(city)
        else:
            context.set_context(api_type="weather")
            return "Sure — which city's weather would you like to know?"

    # ✈️ Booking intent
    if tag == "book_flight":
        context.set_context(api_type="booking")
        return booking_reply(entities)

    # 😂 Joke intent
    if tag in ["tell_joke", "jokes"]:
        context.set_context(api_type="joke")
        return get_joke()

    # 💬 Quote intent
    if tag == "motivate_me":
        context.set_context(api_type="quote")
        return get_quote()

    # 🗣️ Default fallback
    return get_response(tag)

# ========================
# CHAT LOOP
# ========================

if __name__ == "__main__":
    print("Chatbot is running with deep model + NER + context-aware API integration (type 'quit' to exit)\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye!")
            break
        resp = chatbot_response(msg, debug=True)
        print("Bot:", resp)

# chatbot.py
import random

def get_response(user_input):
    return "You said: " + user_input
