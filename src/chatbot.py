import json
import random
import pickle

# Load trained model + vectorizer
with open("model.pkl", "rb") as f:
    vectorizer, clf = pickle.load(f)

# Load intents
with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# Helper: find response
def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn’t get that."

# Chat loop
print("Chatbot is running! Type 'quit' to exit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    # Vectorize user input
    X = vectorizer.transform([user_input])

    # Predict intent
    predicted_tag = clf.predict(X)[0]

    # Get response
    response = get_response(predicted_tag)
    print("Bot:", response)
