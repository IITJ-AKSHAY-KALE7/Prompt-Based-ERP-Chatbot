import pandas as pd
import streamlit as st
import requests
import os

def process_user_prompt(user_input, csv_data):
    # Basic prompt handling
    if "unique" in user_input and "company" in user_input:
        return "Please provide the column name for unique company names."

    elif "histogram" in user_input and "price" in user_input:
        return "Please provide the column name for the price."

    # Fallback response
    return "Sorry, I didn't understand your prompt."

def call_llama_api(query):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    payload = {
        "model": "nvidia/llama-3.1-nemotron-70b-instruct",
        "max_tokens": 1024,
        "stream": False,
        "temperature": 0.5,
        "top_p": 1,
        "stop": None,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "seed": 0,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {os.getenv('LLAMA_API_KEY')}",
        "accept": "application/json",
        "content-type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        
        if response.status_code == 200:
            return response_data.get("choices", [])
        else:
            print(f"Error from LLaMA API: {response_data}")
            return None
    except Exception as e:
        print(f"Exception when calling LLaMA API: {e}")
        return None

# Streamlit app logic here (if applicable)
