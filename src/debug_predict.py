# src/debug_predict.py
import pickle, json, numpy as np, os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "models/chatbot_model.h5"       # or .keras
TOKEN_PATH = "models/tokenizer.pkl"
LE_PATH    = "models/label_encoder.pkl"
INTENTS    = "data/intents.json"

print("Files mtimes:")
for p in [MODEL_PATH, TOKEN_PATH, LE_PATH]:
    try:
        print(p, "=>", os.path.getmtime(p))
    except Exception as e:
        print(p, "=> NOT FOUND", e)

# load
with open(TOKEN_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LE_PATH, "rb") as f:
    le = pickle.load(f)

model = load_model(MODEL_PATH)

with open(INTENTS, "r", encoding="utf-8") as f:
    intents = json.load(f)

print("\nLabelEncoder classes (index -> tag):")
for idx, cls in enumerate(le.classes_):
    print(idx, cls)

# test input
text = "who created you?"
seq = tokenizer.texts_to_sequences([text])
max_len = model.input_shape[1]
padded = pad_sequences(seq, maxlen=max_len, padding="post")
print("\nInput seq:", seq, "padded len:", padded.shape)

pred = model.predict(padded, verbose=0)[0]
print("\nPred probs (top 5):")
top = np.argsort(pred)[::-1][:5]
for i in top:
    print(i, le.classes_[i], round(pred[i], 4))

pred_idx = int(np.argmax(pred))
print("\nArgmax index:", pred_idx)
print("Decoded with label_encoder.inverse_transform:", le.inverse_transform([pred_idx])[0])

# show responses for that tag
tag = le.inverse_transform([pred_idx])[0]
for it in intents["intents"]:
    if it["tag"] == tag:
        print("\nResponses for predicted tag:", it["responses"])
        break
else:
    print("\nNo responses found for tag:", tag)
