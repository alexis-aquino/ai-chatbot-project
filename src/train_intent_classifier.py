import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#niloload ang dataset
with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)
X = []
y = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(intent["tag"])

print("Samples:", X[:5])
print("Labels:", y[:5])

#we vectorize the text, meaning text to numbers
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

#train and split test
X_train, X_test, y_train, y_test = train_test_split(X_vec, y , test_size=0.2,random_state=42)

#train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

#evaluate na
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

#save yung model and vectorize
with open("model.pkl","wb") as f:
    pickle.dump((vectorizer,clf),f)

print("Model saved and updated to model.pkl")