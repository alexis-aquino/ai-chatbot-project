import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("stopwords")

import string
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocesstext(text: str):
    text = text.lower()
    tokens = word_tokenize(text)

    tokens = [
        w for w in tokens
        if w not in string.punctuation and w not in stop_words
    ]

    token = [stemmer.stem(w) for w in tokens]

    return tokens

if __name__ == "__main__":
    sample = "Hello, I am running late!"
    print("Raw:", sample)
    print("Processed:", preprocesstext(sample))