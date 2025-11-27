# Minimal Named Entity Recognition placeholder
def extract_entities(text):
    # Returns dict like {"GPE": ["Paris"], "DATE": ["tomorrow"]}
    entities = {}
    words = text.split()
    for w in words:
        if w.istitle():
            entities.setdefault("GPE", []).append(w)
        if any(d in w for d in ["today","tomorrow","Monday","Tuesday"]):
            entities.setdefault("DATE", []).append(w)
    return entities
