import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from data_processing import load_data
from nlp_utils import process_query_with_spacy, process_query_with_nltk
from visualization import process_user_prompt, call_llama_api

# Set page configuration as the first Streamlit command
st.set_page_config(layout='wide', page_title='ERP ChatBot For Data Analysis', page_icon='🤖')

# Load environment variables from .env file
load_dotenv()

# Sidebar for file upload and options
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-subtext">Upload your file and customize settings.</p>', unsafe_allow_html=True)

# File uploader in the sidebar
upload_csv_file = st.sidebar.file_uploader("Upload Your CSV File", type=["csv"])

# Initialize the data variable
data = None

if upload_csv_file is not None:
    data = load_data(upload_csv_file)
    if data is not None:
        st.sidebar.success('Data Uploaded Successfully!')
        st.sidebar.write(data.head(10))
    else:
        st.sidebar.warning("The uploaded file is empty. Please upload a valid CSV file.")

# Main Title
st.markdown('<div class="main-title">ERP ChatBot: Prompt-Based Data Analysis and Visualization</div>', unsafe_allow_html=True)

# Input area for user prompt
st.markdown('<div class="sub-title" style="color: #003366; font-size: 20px;">Enter Your Analysis or Visualization Request</div>', unsafe_allow_html=True)
query = st.text_area("Enter your prompt")

# Submit button
if st.button("Submit", key='submit_button', help='Click to submit your query'):
    if data is None:
        st.error("Please upload a CSV file before submitting a query.")
    elif not query.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            with st.spinner("Loading, please wait..."):
                # 1. Print the query sent to LLaMA
                llama_query = f"Process the following request: {query}"
                print(f"Query sent to LLaMA: {llama_query}")  # Debug statement

                # 2. Call LLaMA API
                llama_response = call_llama_api(llama_query)
                print(f"LLaMA Response: {llama_response}")  # Debug statement

                # 3. Process the original query
                visualization_result = process_user_prompt(query, data)

                if visualization_result is not None:
                    if isinstance(visualization_result, str):  # If it's a string message
                        st.write(visualization_result)
                    elif visualization_result is None:  # Check for None to catch visualization errors
                        print("No visualization to display.")
                    else:
                        st.dataframe(visualization_result)  # If it's a DataFrame

        except Exception as e:
            st.error(f"Error processing the query: {e}")

else:
    st.info("Upload a CSV file from the sidebar and enter a prompt to get started.")