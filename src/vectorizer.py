from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_tokenizer(texts, oov_token="<OOV>"):
    """
    Fit a tokenizer on the given texts and return (tokenizer, padded_sequences).
    
    Args:
        texts (list[str]): List of input strings to fit on.
        oov_token (str): Token for out-of-vocabulary words.

    Returns:
        tokenizer (Tokenizer): Fitted tokenizer object.
        padded (ndarray): Padded integer sequences.
    """
    tokenizer = Tokenizer(oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, padding="post")
    return tokenizer, padded

def pad_texts(tokenizer, texts, max_len=None):
    """
    Convert new texts into padded sequences using an existing tokenizer.

    Args:
        tokenizer (Tokenizer): Pre-trained tokenizer object.
        texts (list[str]): List of new input strings.
        max_len (int): Maximum sequence length (use same as training).

    Returns:
        padded (ndarray): Padded sequences for new texts.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post")
    return padded
