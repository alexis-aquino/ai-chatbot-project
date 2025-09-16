import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Starting training...")

# Load dataset
with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)
print("Loaded intents.json")

X = []
y = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:   
        X.append(pattern)
        y.append(intent["tag"])

print("Dataset built")
print("Samples:", X[:5])
print("Labels:", y[:5])
print("Total samples:", len(X))

# Vectorize using TF-IDF (better than BoW)
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
print("Vectorized dataset, shape:", X_vec.shape)

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)
print("Split into train/test")

# Train classifier
clf = LogisticRegression(max_iter=1000)  # more iterations for convergence
clf.fit(X_train, y_train)
print("Model trained")

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model + vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, clf), f)

print("Model saved as model.pkl")

if __name__ == "__main__":
    print("Finished training run.")
