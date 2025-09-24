import pickle
from sklearn.preprocessing import LabelEncoder

def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # save encoder so we can decode predictions later
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    return y_encoded, le