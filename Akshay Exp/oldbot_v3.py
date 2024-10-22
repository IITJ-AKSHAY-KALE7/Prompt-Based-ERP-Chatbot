import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Set page configuration as the first Streamlit command
st.set_page_config(layout='wide', page_title='ChatBot Data Analysis', page_icon='🤖')

# Load environment variables from .env file
load_dotenv()

# Retrieve the NVIDIA API key from environment variables
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

if NVIDIA_API_KEY is None:
    raise ValueError("API key not found. Please set the NVIDIA_API_KEY environment variable.")

class StreamlitResponse:
    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

# Initialize the OpenAI client with NVIDIA API settings
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Custom CSS for sidebar and main page styling with shades of blue
st.markdown(""" 
    <style>
        .main-title {
            font-size: 36px;
            color: #FFFFFF;
            background: linear-gradient(to right, #003366, #0078D4);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #0078D4, #004578);
            color: white;
        }
        .sidebar-title {
            color: #FFFFFF;
            font-weight: bold;
            font-size: 22px;
            margin-top: 20px;
            text-align: center;
        }
        .sidebar-subtext {
            color: #D0E7FF;
            font-size: 16px;
        }
        .stButton button {
            color: #FFFFFF;
            background-color: #0063B1;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #005A9E;
        }
        .stTextArea textarea {
            color: #003366;
            font-size: 16px;
            background-color: #E6F0FF;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for file upload and options
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-subtext">Upload your file and customize settings.</p>', unsafe_allow_html=True)

# File uploader in the sidebar
upload_csv_file = st.sidebar.file_uploader("Upload Your CSV File", type=["csv"])

# Initialize the data variable
data = None

if upload_csv_file is not None:
    try:
        data = pd.read_csv(upload_csv_file)
        if data.empty:
            st.sidebar.warning("The uploaded file is empty. Please upload a valid CSV file.")
        else:
            data.columns = data.columns.str.upper()  # Convert the columns to uppercase
            st.sidebar.success('Data Uploaded Successfully!')
            st.sidebar.write(data.head(10))
    except Exception as e:
        st.sidebar.error(f"Error reading the file: {e}")

# Main Title
st.markdown('<div class="main-title">ERP ChatBot: Prompt-Based Data Analysis and Visualization</div>', unsafe_allow_html=True)

# Input area for user prompt
st.markdown('<div class="sub-title" style="color: #003366; font-size: 20px;">Enter Your Analysis or Visualization Request</div>', unsafe_allow_html=True)
query = st.text_area("Enter your prompt")

# Function to call NVIDIA LLaMA API
def call_nvidia_api(prompt):
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )

        response_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response_content += chunk.choices[0].delta.content

        return response_content

    except Exception as e:
        st.error(f"Error calling NVIDIA API: {e}")
        return None

# Submit button with custom style
if st.button("Submit", key='submit_button', help='Click to submit your query'):
    if data is None:
        st.error("Please upload a CSV file before submitting a query.")
    elif not query.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            with st.spinner("Loading, please wait..."):
                st.write('### OUTPUT:')
                st.markdown('---')
                
                # Call the NVIDIA API
                response = call_nvidia_api(query)
                
                if response:
                    st.write(response)

        except Exception as e:
            st.error(f"Error processing the query: {e}")
else:
    st.info("Upload a CSV file from the sidebar and enter a prompt to get started.")