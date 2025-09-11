import json
import os

def load_intents(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    intents = load_intents("data/intents.json")

    for intent in intents["intents"][:3]:
        print(f"Tag: {intent['tag']}")
        print(f"Patterns: {intent['patterns']}")
        print(f"Responses {intent['responses']}")
        print("---------")