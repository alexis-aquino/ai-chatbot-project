# src/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten

def create_model(vocab_size, max_len, num_classes, embedding_dim=64):
    """
    Build and return a compiled Sequential model for intent classification.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Flatten(),  # flatten embeddings into a vector
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    # ✅ compile before returning
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model
