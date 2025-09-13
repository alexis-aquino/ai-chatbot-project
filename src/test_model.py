import pickle

with open("model.pkl", "rb") as f:
    vectorizer, clf = pickle.load(f)

sample = ["hello there"]
sample_vec = vectorizer.transform(sample)
prediction = clf.predict(sample_vec)

print("Input:", sample[0])
print("Predicted intent: ", prediction[0])