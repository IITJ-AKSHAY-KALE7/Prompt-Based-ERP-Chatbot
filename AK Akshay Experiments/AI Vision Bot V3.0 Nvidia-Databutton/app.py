import os
from dotenv import load_dotenv
import databutton as db
import streamlit as st
from utils import get_data
from app_brain import NvidiaChatbot

# Load environment variables from .env file
load_dotenv()

# Set the background color to white
st.markdown(
    """
    <style>
    .reportview-container {
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Retrieve the NVIDIA API key from environment variables
API_KEY = os.getenv("NVIDIA_API_KEY")

# Check if the API key is set
if not API_KEY:
    st.error("NVIDIA API key is not set in the .env file.")
else:
    # Create an instance of the NvidiaChatbot
    chatbot = NvidiaChatbot(api_key=API_KEY)

    # Cache the header of the app to prevent re-rendering on each load
    @st.cache_resource
    def display_app_header():
        st.title("AI Vision Bot for ERP Data Visualization  📊 ")
        st.markdown("***Prompt about your data, and see it visualized** ✨ This app runs on the power of your prompting. As here in Databutton HQ, we envision, '**Prompting is the new programming.**'*")

    # Display the header of the app
    display_app_header()

    options = st.radio("Data Usage", options=["Upload file", "Use Data in Storage"], horizontal=True)
    if options == "Upload file":
        df = get_data()
    else:
        df = db.storage.dataframes.get(key="car_price_dataset-csv")

    if df is not None:
        with st.expander("Show data"):
            st.write(df)

        column_names = ", ".join(df.columns)

        if not df.empty:
            chatbot.handle_nvidia_query(df, column_names)
        else:
            st.warning("The given data is empty.")