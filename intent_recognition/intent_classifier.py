from transformers import pipeline

class IntentClassifier:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="distilbert-base-uncased")

    def predict(self, text):
        result = self.classifier(text)
        label = result[0]['label']
        if 'analysis' in text.lower():
            return 'data_analysis'
        elif 'workflow' in text.lower():
            return 'run_workflow'
        elif 'hello' in text.lower() or 'hi' in text.lower():
            return 'greeting'
        elif 'bye' in text.lower() or 'goodbye' in text.lower():
            return 'farewell'
        elif 'thank' in text.lower():
            return 'thanks'
        else:
            return 'data_analysis'  # Default to data analysis