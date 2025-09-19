import json
import os

def load_intents(filepath):
    """Load raw intents JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_dataset(filepath):
    """Load patterns (X) and labels (y) from intents.json."""
    data = load_intents(filepath)
    X = []
    y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            X.append(pattern)
            y.append(intent["tag"])
    return X, y   # ✅ only two things returned

if __name__ == "__main__":
    X, y = load_dataset("data/intents.json")
    print("First patterns:", X[:5])
    print("First labels:", y[:5])
