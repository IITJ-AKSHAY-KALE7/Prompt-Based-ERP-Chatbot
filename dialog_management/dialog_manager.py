class DialogManager:
    def __init__(self):
        self.rules = {
            "data_analysis": "I'll analyze the data for you.",
            "run_workflow": "I'll start the workflow for you.",
            "greeting": "Hello! How can I assist you today?",
            "farewell": "Goodbye! Have a great day!",
            "thanks": "You're welcome!",
        }

    def handle_message(self, message):
        intent = self.parse_intent(message)
        return self.rules.get(intent, "I'm not sure how to respond to that.")

    def parse_intent(self, message):
        # This should be replaced with actual intent parsing logic
        return "data_analysis"  # Default intent for demonstration