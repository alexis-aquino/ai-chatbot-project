from dataset_loader import load_dataset
from preprocessing import preprocesstext
from vectorizer import create_tokenizer, pad_texts
from label_encoder import encode_labels
from model import create_model
from tensorflow.keras.utils import to_categorical
import pickle
import os

# Make sure models folder exists
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
X, y = load_dataset("data/intents.json")

print(f"Dataset loaded: {len(X)} samples")

# Preprocess text
X_processed = [" ".join(preprocesstext(text)) for text in X]

# Tokenizer + padding
print("Tokenizing and padding text...")
tokenizer, X_padded = create_tokenizer(X_processed)
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Encode labels
print("Encoding labels...")
y_encoded, le = encode_labels(y)
y_categorical = to_categorical(y_encoded)

# Model params
vocab_size = len(tokenizer.word_index) + 1
max_len = X_padded.shape[1]
num_classes = y_categorical.shape[1]

# Build model
print("Building model...")
model = create_model(vocab_size, max_len, num_classes)

# Train model
print("Training model...")
history = model.fit(
    X_padded,
    y_categorical,
    epochs=200,
    batch_size=8,
    verbose=1
)
print("Saving model and label encoder...")

# save in legacy HDF5 format
model.save("models/chatbot_model.h5")

# save in new Keras format
model.save("models/chatbot_model.keras")

# save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Training complete! Model saved in /models")


# Save model + encoder
print("Saving model and label encoder...")
model.save("models/chatbot_model.h5")
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Training complete! Model saved in /models")
