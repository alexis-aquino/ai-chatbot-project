import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 2. Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 3. Load trained model
model = load_model("models/chatbot_model.h5")

# 4. Load intents (to pick responses)
with open("data/intents.json", "r") as f:
    intents = json.load(f)

# Helper function: predict intent
def predict_intent(text, max_len=20):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")
    prediction = model.predict(padded, verbose=0)
    intent_index = np.argmax(prediction)
    intent_tag = label_encoder.inverse_transform([intent_index])[0]
    return intent_tag

# Helper function: get response
def get_response(intent_tag):
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return np.random.choice(intent["responses"])
    return "Sorry, I don’t understand."

# 5. Try some test inputs
while True:
    msg = input("You: ")
    if msg.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye!")
        break
    tag = predict_intent(msg, max_len=model.input_shape[1])
    response = get_response(tag)
    print(f"Bot ({tag}): {response}")
