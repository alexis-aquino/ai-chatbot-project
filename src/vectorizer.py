import json
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load dataset
with open("data/intents.json", "r") as f:
    data = json.load(f)

# Collect all patterns
patterns = []
for intent in data["intents"]:
    patterns.extend(intent["patterns"])

print(f"Loaded {len(patterns)} patterns")

# 2. Initialize tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)

# 3. Convert to sequences
sequences = tokenizer.texts_to_sequences(patterns)

# 4. Pad sequences
padded = pad_sequences(sequences, padding="post")

# 5. Print sample
sample = "hello"
sample_seq = tokenizer.texts_to_sequences([sample])
sample_pad = pad_sequences(sample_seq, maxlen=padded.shape[1])

print("Input:", sample)
print("Tokenized:", sample_seq[0])
print("Padded:", sample_pad[0].tolist())

# 6. Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved to tokenizer.pkl")
