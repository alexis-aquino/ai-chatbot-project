from dataset_loader import load_dataset
from preprocessing import preprocesstext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Starting pipeline demo...")

# Load dataset
X, y = load_dataset("data/intents.json")
print(f"Loaded dataset: {len(X)} samples")

# Preprocess → join tokens back into one string
X_processed = [" ".join(preprocesstext(text)) for text in X]
print("First 5 processed:", X_processed[:5])

# Vectorize
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X_processed)
print("Vectorized shape:", X_vec.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
