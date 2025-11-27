class ContextManager:
    def __init__(self):
        self.context = {"last_intent": None, "last_api_type": None, "entity": None}

    def get_context(self):
        return self.context

    def set_context(self, intent=None, api_type=None, entity=None):
        if intent:
            self.context["last_intent"] = intent
        if api_type:
            self.context["last_api_type"] = api_type
        if entity:
            self.context["entity"] = entity

    def clear_context(self):
        self.context = {"last_intent": None, "last_api_type": None, "entity": None}
