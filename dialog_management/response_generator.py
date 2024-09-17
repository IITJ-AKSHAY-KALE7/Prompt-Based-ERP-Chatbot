import random

class ResponseGenerator:
    def __init__(self):
        self.responses = {
            "LABEL_0": ["Hello!", "Hi there!", "Greetings!"],
            "LABEL_1": ["Goodbye!", "See you later!", "Take care!"],
            "LABEL_2": ["You're welcome!", "My pleasure!", "Glad I could help!"],
            "default": ["I'm not sure how to respond to that.", "Could you please rephrase that?"]
        }

    def generate_response(self, intent):
        if intent in self.responses:
            return random.choice(self.responses[intent])
        return random.choice(self.responses["default"])