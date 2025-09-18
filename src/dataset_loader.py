import json
from sklearn.model_selection import train_test_split

def load_dataset(filepath="data/intents.json"):
    with open(filepath, "r") as f:
        data = json.load(f)

    X = []
    y = []


    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            X.append(pattern)
            y.append(intent["tag"])


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    print("Training set size:", len(X_train))
    print("Testing set size:", len(X_test))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
