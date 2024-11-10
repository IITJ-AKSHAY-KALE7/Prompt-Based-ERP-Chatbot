import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
import base64
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')  # Load your Google API key

# Check if the API key is set
if not gemini_api_key:
    st.error("GEMINI_API_KEY environment variable not set.")
    st.stop()

# Configure the Google Generative AI
genai.configure(api_key=gemini_api_key)

# Function to convert base64 string to an image
def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# Streamlit menu
menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Question based Graph", "Chat", "Story Generation"])

if menu == "Summarize":
    st.subheader("Summarization of your Data")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())

        # Generate a summary using Google Generative AI
        summary_response = genai.GenerativeModel("gemini-1.5-flash").generate_content(f"Summarize the contents of {path_to_save}.")
        summary = summary_response.text
        st.write(summary)

elif menu == "Question based Graph":
    st.subheader("Query your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename1.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        
        text_area = st.text_area("Query your Data to Generate Graph", height=200)
        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)

                # Generate summary again for the new query
                summary_response = genai.GenerativeModel("gemini-1.5-flash").generate_content(f"Summarize the contents of {path_to_save}.")
                summary = summary_response.text

                # Display summary and a placeholder for visualization logic
                st.write("Summary:")
                st.write(summary)
                st.write("Visualization logic would go here.")

elif menu == "Chat":
    st.subheader("Chat with AI")
    
    user_message = st.text_input("Your message:")
    
    if st.button("Send"):
        if user_message:
            # Get response from the model
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(user_message)
            st.write(response.text)

elif menu == "Story Generation":
    st.subheader("Generate a Story")
    user_prompt = st.text_input("Enter a prompt for your story:")
    
    if st.button("Generate Story"):
        if user_prompt:
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(user_prompt)
            st.write(response.text)
