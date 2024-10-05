import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TaskExecutor:
    def __init__(self):
        self.df = None
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def prepare_data(self, data):
        self.df = pd.DataFrame(data)
        self.df['combined_text'] = self.df.astype(str).agg(' '.join, axis=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])

    def execute_task(self, query, data=None):
        if self.df is None and data is not None:
            self.prepare_data(data)
        
        if self.df is None:
            return "No data available. Please prepare the data first."

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_5_indices = similarities.argsort()[-5:][::-1]
        
        relevant_data = self.df.iloc[top_5_indices]
        
        response = self.generate_response(query, relevant_data)
        return response

    def generate_response(self, query, data):
        response = f"Based on the query '{query}', here's what I found:\n\n"

        if 'average' in query.lower() or 'mean' in query.lower():
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            averages = data[numeric_columns].mean()
            response += "Average values:\n"
            response += averages.to_string()
            response += "\n\n"

        if 'highest' in query.lower() or 'max' in query.lower():
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            max_values = data[numeric_columns].max()
            response += "Highest values:\n"
            response += max_values.to_string()
            response += "\n\n"

        if 'lowest' in query.lower() or 'min' in query.lower():
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            min_values = data[numeric_columns].min()
            response += "Lowest values:\n"
            response += min_values.to_string()
            response += "\n\n"

        response += "Here's a summary of the most relevant data:\n"
        response += data.to_markdown()

        return response