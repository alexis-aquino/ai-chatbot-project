import json
from typing import List, Tuple

def load_intents(json_path: str = "data/intents.json") -> Tuple[List[str], List[str]]:
    """
    Loads intents.json and returns patterns (X) and labels (y).
    
    Args:
        json_path (str): Path to the JSON file containing intents.

    Returns:
        X (list of str): List of training patterns (sentences).
        y (list of str): List of intent labels (tags).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        intents = json.load(f)

    X = []
    y = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            X.append(pattern)
            y.append(intent["tag"])

    return X, y


# Debugging / test run
if __name__ == "__main__":
    X, y = load_intents()
    print("First 5 patterns:", X[:5])
    print("First 5 labels:", y[:5])
