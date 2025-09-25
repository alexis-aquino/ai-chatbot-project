import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1️⃣ Load trained model
model = load_model("models/chatbot_model.h5")

# 2️⃣ Load tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 3️⃣ Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# 4️⃣ Get max_len from model input
max_len = model.input_shape[1]

# 5️⃣ Predict intent
def predict_intent(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(seq, verbose=0)[0]
    
    intent_index = pred.argmax()
    confidence = pred[intent_index]
    
    intent = le.classes_[intent_index]
    return intent, confidence

# 6️⃣ Test loop
if __name__ == "__main__":
    while True:
        sentence = input("You: ")
        if sentence.lower() in ["quit", "exit"]:
            break
        intent, conf = predict_intent(sentence)
        print(f"Predicted intent: {intent} (confidence: {conf:.2f})")
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1️⃣ Load trained model
model = load_model("models/chatbot_model.h5")

# 2️⃣ Load tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 3️⃣ Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# 4️⃣ Get max_len from model input
max_len = model.input_shape[1]

# 5️⃣ Predict intent
def predict_intent(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(seq, verbose=0)[0]
    
    intent_index = pred.argmax()
    confidence = pred[intent_index]
    
    intent = le.classes_[intent_index]
    return intent, confidence

# 6️⃣ Test loop
if __name__ == "__main__":
    while True:
        sentence = input("You: ")
        if sentence.lower() in ["quit", "exit"]:
            break
        intent, conf = predict_intent(sentence)
        print(f"Predicted intent: {intent} (confidence: {conf:.2f})")
