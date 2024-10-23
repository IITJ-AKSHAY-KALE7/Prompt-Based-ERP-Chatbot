import databutton as db
import streamlit as st
import pandas as pd

def get_data():
    file_types = ["csv", "xlsx", "xls"]
    data_upload = st.file_uploader("Upload a data file", type=file_types)
    
    if data_upload:
        if data_upload.name.endswith('.csv'):
            df = pd.read_csv(data_upload)
        elif data_upload.name.endswith('.xlsx') or data_upload.name.endswith('.xls'):
            df = pd.read_excel(data_upload)
        else:
            df = None
        return df
    
    return None
