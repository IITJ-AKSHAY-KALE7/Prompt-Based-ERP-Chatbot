import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.responses.response_parser import ResponseParser
from openai import OpenAI  # Replace with your actual NVIDIA client if necessary

# Load environment variables from .env file
load_dotenv()

# Retrieve the NVIDIA API key from environment variables
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

if NVIDIA_API_KEY is None:
    raise ValueError("API key not found. Please set the NVIDIA_API_KEY environment variable.")

class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

# Make the app wider
st.set_page_config(layout='wide')

# Set the title
st.title("ChatBot: Prompt Based Data Analysis and Visualization")
st.markdown('---')

# File uploader using Streamlit
upload_csv_file = st.file_uploader("Upload Your CSV file for data analysis and visualization", type=["csv"])

# If statement to make sure the data is uploaded
if upload_csv_file is not None:
    data = pd.read_csv(upload_csv_file)
    data.columns = data.columns.str.upper()  # Convert the columns to uppercase
    st.table(data.head(5))
    st.write('Data Uploaded Successfully!')

st.markdown('---')

st.write('### Enter Your Analysis or Visualization Request')
query = st.text_area("Enter your prompt")

# Initialize NVIDIA client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Create a custom LLM class that wraps the NVIDIA client
class CustomLLM:
    def __init__(self, client):
        self.client = client

    def chat(self, messages):
        completion = self.client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=messages,
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response

# Create an instance of CustomLLM
llm_instance = CustomLLM(client)

if st.button("Submit"):
    if query:
        with st.spinner("Loading, please wait..."):
            st.write('### OUTPUT:')
            st.markdown('---')
            query_engine = SmartDataframe(data, config={'llm': llm_instance, "response_parser": StreamlitResponse})

            # Call the chat method from CustomLLM
            response = query_engine.chat([{"role": "user", "content": query}])
            st.write(response)

    else:
        st.warning("Please enter a prompt")
