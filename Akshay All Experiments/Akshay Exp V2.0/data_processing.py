import pandas as pd

def load_data(file):
    try:
        data = pd.read_csv(file)
        if data.empty:
            return None
        data.columns = data.columns.str.upper()  # Convert the columns to uppercase
        return data
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None
