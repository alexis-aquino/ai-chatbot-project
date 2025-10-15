# src/context_manager.py
# =====================
# ContextManager: stores conversation context for follow-up queries
# Example use:
#   context.set_context(intent="weather", entity="Manila", api_type="weather")
#   ctx = context.get_context()

class ContextManager:
    def __init__(self):
        self.context = {
            "last_intent": None,
            "last_entity": None,
            "last_api_type": None,
        }

    # ✅ Set or update any part of the context
    def set_context(self, intent=None, entity=None, api_type=None):
        if intent:
            self.context["last_intent"] = intent
        if entity:
            self.context["last_entity"] = entity
        if api_type:
            self.context["last_api_type"] = api_type

    # ✅ Get the full context as a dictionary
    def get_context(self):
        return self.context

    # ✅ Clear all stored context (start fresh)
    def clear_context(self):
        self.context = {
            "last_intent": None,
            "last_entity": None,
            "last_api_type": None,
        }

    # ✅ (Optional) individual getters for convenience
    def get_intent(self):
        return self.context["last_intent"]

    def get_entity(self):
        return self.context["last_entity"]

    def get_api_type(self):
        return self.context["last_api_type"]
