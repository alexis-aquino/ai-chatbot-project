import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocesstext(text: str):
    # Lowercase
    text = text.lower()

    # 🧠 Normalize elongated words (e.g., "heyyyy" → "hey", "noooo" → "no")
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    tokens = [
        w for w in tokens
        if w not in string.punctuation and w not in stop_words
    ]

    # Stemming
    token = [stemmer.stem(w) for w in tokens]

    return tokens  # ⚠️ keep your original return unchanged

if __name__ == "__main__":
    sample = "Hiiiii, I am runnnning late!!!"
    print("Raw:", sample)
    print("Processed:", preprocesstext(sample))
