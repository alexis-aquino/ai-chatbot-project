from dataset_loader import load_dataset
from preprocessing import preprocesstext
from vectorizer import create_tokenizer, pad_texts
from label_encoder import encode_labels
from model import create_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
X, y = load_dataset("data/intents.json")
print(f"Dataset loaded: {len(X)} samples, {len(set(y))} intents")

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

# Train/validation split (stratified to balance classes)
X_train, X_val, y_train, y_val = train_test_split(
    X_padded, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Model params
vocab_size = len(tokenizer.word_index) + 1
max_len = X_padded.shape[1]
num_classes = y_categorical.shape[1]

# Build model
print("Building model...")
model = create_model(vocab_size, max_len, num_classes)

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/best_chatbot_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Train model
print("Training model...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=8,
    verbose=1,
    callbacks=[early_stop, checkpoint]
)

# Save final artifacts
print("Saving tokenizer, model, and label encoder...")
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

model.save("models/chatbot_model.h5")      # legacy HDF5
model.save("models/chatbot_model.keras")   # full model (final)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Training complete! Best model saved as /models/best_chatbot_model.keras")
