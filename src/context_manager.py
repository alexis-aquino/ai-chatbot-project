# Simple context manager for chatbot

class ContextManager:
    def __init__(self):
        self.last_intent = None

    def set_intent(self, intent):
        self.last_intent = intent

    def get_intent(self):
        return self.last_intent

    def clear(self):
        self.last_intent = None
