import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities from text using spaCy.
    Returns list of (entity_text, entity_label).
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

if __name__ == "__main__":
    # Example
    test_text = "Book a flight to Manila tomorrow"
    entities = extract_entities(test_text)
    print("Entities:", entities)
