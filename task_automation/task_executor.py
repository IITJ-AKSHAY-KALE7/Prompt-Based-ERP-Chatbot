import pandas as pd
from pandasai import SmartDataframe
from transformers import pipeline
from utils.helpers import StreamlitResponse

class CustomLLM:
    def __init__(self):
        self.pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

    def generate(self, prompt):
        return self.pipeline(prompt, max_length=100)[0]['generated_text']

class TaskExecutor:
    def __init__(self):
        self.llm = CustomLLM()

    def execute_task(self, task, data):
        df = pd.DataFrame(data)
        smart_df = SmartDataframe(df, config={"llm": self.llm, "response_parser": StreamlitResponse})
        result = smart_df.chat(task)
        return result