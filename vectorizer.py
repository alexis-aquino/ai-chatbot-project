from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 20

def pad_texts(tokenizer, texts):
    # texts = list of strings
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post")
