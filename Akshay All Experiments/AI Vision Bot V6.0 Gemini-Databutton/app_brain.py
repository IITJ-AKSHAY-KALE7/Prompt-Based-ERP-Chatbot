import databutton as db
import streamlit as st
import pandas as pd
import re
import google.generativeai as genai

class GeminiChatbot:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def handle_gemini_query(self, df, column_names):
        query = st.text_area(
            "Enter your Prompt:",
            placeholder="Prompt tips: Use plotting related keywords such as 'Plots' or 'Charts'.",
            help="How an ideal prompt should look like..."
        )

        if st.button("Get Answer"):
            if query and query.strip() != "":
                prompt_content = f"""
                The dataset is ALREADY loaded into a DataFrame named 'df'. DO NOT load the data again.
                The DataFrame has the following columns: {column_names}
                Prepare the data and provide a single code block solution to the following query:
                {query}
                """

                response = self.model.generate_content(prompt_content)

                if response and response.text:
                    self.execute_gemini_code(response.text, df, query)

    def execute_gemini_code(self, response_text: str, df: pd.DataFrame, query):
        code = self.extract_code_from_markdown(response_text)

        if code:
            try:
                exec(code)
                st.pyplot()
            except Exception as e:
                error_message = str(e)
                st.error(f"📟 Apologies, failed to execute the code due to the error: {error_message}")
                st.warning("Check the error message and the code executed above to investigate further.")
        else:
            st.write(response_text)

    def extract_code_from_markdown(self, md_text):
        code_blocks = re.findall(r"```(python)?(.*?)```", md_text, re.DOTALL)
        return "\n".join([block[1].strip() for block in code_blocks])
