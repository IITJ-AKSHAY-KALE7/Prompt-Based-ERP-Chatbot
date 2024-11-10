import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime
import os
import re
import google.generativeai as genai
import matplotlib.pyplot as plt  # Ensure plt is imported

# Load data function
@st.cache_data
def load_data():
    try:
        xls = pd.ExcelFile('ERP_project_simplified.xlsx')
        data = {
            'MaintenanceData': pd.read_excel(xls, 'MaintenanceData'),
            'UserData': pd.read_excel(xls, 'UserData')
        }

        # Convert date columns to datetime
        data['MaintenanceData']['Date'] = pd.to_datetime(data['MaintenanceData']['Date'], errors='coerce')

        # Initialize AuditLogs if not present
        if 'AuditLogs' not in xls.sheet_names:
            data['AuditLogs'] = pd.DataFrame(columns=[
                'ActivityID', 'Timestamp', 'UserID', 'UserRole',
                'ActionType', 'ActionDetails', 'SuccessFlag', 'Comments'
            ])
        else:
            data['AuditLogs'] = pd.read_excel(xls, 'AuditLogs')

        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Save data function
def save_data():
    try:
        excel_file_path = os.path.join(os.getcwd(), 'ERP_project_simplified.xlsx')
        with pd.ExcelWriter(excel_file_path, mode='a', if_sheet_exists='replace') as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        st.success("Data saved successfully.")
        load_data.clear()  # Clear the cache after saving data
    except Exception as e:
        import traceback
        st.error(f"Error saving data: {e}")
        st.text("Traceback:")
        st.text(traceback.format_exc())

# Authentication function
def authenticate_user(user_id, password):
    users = data['UserData']
    user_row = users[(users['UserID'] == user_id) & (users['Password'] == password)]
    if not user_row.empty:
        return user_row.iloc[0]['JobProfile']
    else:
        return None

# Login function
def login():
    st.title("ERP System Login")
    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        role = authenticate_user(user_id, password)
        if role:
            st.session_state['user_id'] = user_id
            st.session_state['role'] = role
            st.success(f"Welcome {user_id}!")
            log_user_activity(user_id, 'Login', 'Success')
            st.rerun()  # Use st.rerun()
        else:
            st.error("Invalid User ID or password")
            log_user_activity(user_id, 'Login', 'Failure')

# User activity logging function
def log_user_activity(user_id, action_type, success_flag, action_details=''):
    st.write(f"Logging activity: {action_type} by {user_id}")
    new_log = {
        'ActivityID': len(data['AuditLogs']) + 1,
        'Timestamp': datetime.datetime.now(),
        'UserID': user_id,
        'UserRole': st.session_state.get('role', ''),
        'ActionType': action_type,
        'ActionDetails': action_details,
        'SuccessFlag': success_flag,
        'Comments': ''
    }
    data['AuditLogs'] = pd.concat(
        [data['AuditLogs'], pd.DataFrame([new_log])],
        ignore_index=True
    )
    save_data()

# Access control decorator
def require_role(allowed_roles):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'role' not in st.session_state:
                st.error("Please log in to access this feature.")
                return
            if st.session_state.get('role') in allowed_roles:
                return func(*args, **kwargs)
            else:
                st.error("Access Denied: You do not have permission to access this feature.")
        return wrapper
    return decorator

# Dashboard function
@require_role(['Admin', 'Manager', 'Technician'])
def show_dashboard():
    st.title("ERP System Dashboard")

    st.subheader("Maintenance Types Distribution")
    maintenance_counts = data['MaintenanceData']['MaintenanceType'].value_counts()
    fig = px.pie(
        values=maintenance_counts.values,
        names=maintenance_counts.index,
        title="Maintenance Types"
    )
    st.plotly_chart(fig)

    st.subheader("Cost Over Time")
    cost_over_time = data['MaintenanceData'].groupby('Date')['Cost'].sum().reset_index()
    fig = px.line(cost_over_time, x='Date', y='Cost', title='Maintenance Cost Over Time')
    st.plotly_chart(fig)

# Maintenance input function
@require_role(['Admin', 'Technician'])
def maintenance_input():
    st.title("Maintenance Data Input")
    with st.form("maintenance_form"):
        maintenance_id = st.text_input("Maintenance ID")
        date = st.date_input("Date")
        days_since_last = st.number_input("Days Since Last Maintenance", min_value=0)
        maintenance_type = st.selectbox("Maintenance Type", ["Preventive", "Corrective"])
        temperature = st.number_input("Equipment Temperature")
        cost = st.number_input("Maintenance Cost", min_value=0.0)
        technician_id = st.text_input("Technician ID")
        submit = st.form_submit_button("Submit")

    if submit:
        if not maintenance_id or not technician_id:
            st.error("Please fill all required fields.")
        else:
            new_record = {
                'MaintenanceID': maintenance_id,
                'Date': date,
                'DaysSinceLastMaintenance': days_since_last,
                'MaintenanceType': maintenance_type,
                'Temperature': temperature,
                'Cost': cost,
                'TechnicianID': technician_id
            }
            data['MaintenanceData'] = pd.concat(
                [data['MaintenanceData'], pd.DataFrame([new_record])],
                ignore_index=True
            )
            save_data()
            st.success("Maintenance record added successfully.")
            log_user_activity(
                st.session_state['user_id'], 'Maintenance Input', 'Success'
            )

# Predictive maintenance function
@require_role(['Admin', 'Manager'])
def predictive_maintenance():
    st.title("Predictive Maintenance")
    try:
        maintenance_data = data['MaintenanceData'].dropna()
        X = maintenance_data[['DaysSinceLastMaintenance', 'Temperature']]
        y = (maintenance_data['MaintenanceType'] == 'Corrective').astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.subheader("Model Performance")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Maintenance Prediction")
        days_since_last = st.number_input("Days Since Last Maintenance", min_value=0)
        temperature = st.number_input("Equipment Temperature")

        if st.button("Predict"):
            prediction = model.predict([[days_since_last, temperature]])
            if prediction[0] == 1:
                st.warning("Corrective maintenance may be needed soon.")
            else:
                st.success("No immediate maintenance needed.")
    except Exception as e:
        st.error(f"Error in predictive maintenance: {str(e)}")

# Helper function to extract code from markdown
def extract_code_from_markdown(md_text):
    code_blocks = re.findall(r"```(python)?(.*?)```", md_text, re.DOTALL)
    if code_blocks:
        return "\n".join([block[1].strip() for block in code_blocks])
    else:
        # If no code blocks found, try to extract the code directly
        return md_text.strip()

# NLP chatbot function using Gemini
@require_role(['Admin', 'Manager', 'Technician'])
def nlp_chatbot():
    st.title("AI Chatbot")
    
    # Load the Gemini API key from environment variable or let user input it
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        API_KEY = st.text_input("Enter your Gemini API key:", type="password")
    
    if API_KEY:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        user_query = st.text_area(
            "Ask me anything about the maintenance data:",
            placeholder="For example: '1] show the bar chart of the maintenance cost for last 90 days '",
            help="You can ask questions about maintenance types, total cost, average temperature, etc."
        )
        if st.button("Get Answer"):
            if user_query and user_query.strip() != "":
                column_names = ", ".join(data['MaintenanceData'].columns)
                prompt_content = f"""
You are provided with a DataFrame named 'df' that contains maintenance data.
The DataFrame has the following columns: {column_names}.
Write Python code to answer the following query:
{user_query}
Provide only the code; do not include any explanations.
"""
                try:
                    response = model.generate_content(prompt_content)
                   # st.write("Debug: Response from Gemini model:", response)
                except Exception as e:
                    st.error(f"Error from Gemini model: {e}")
                    return
                
                if response and response.text:
                    code = extract_code_from_markdown(response.text)
                  #  st.write("Debug: Extracted code:", code)
                    # Adjust the code formatting
                    code_lines = code.strip().splitlines()
                    code_lines = [line.strip() for line in code_lines if line.strip()]
                    code = '\n'.join(code_lines)
                  #  st.write("Debug: Adjusted code:", code)
                    if code:
                        try:
                            # Prepare the DataFrame 'df'
                            df = data['MaintenanceData']
                            # Add necessary imports
                            local_vars = {
                                'df': df,
                                'st': st,
                                'px': px,
                                'pd': pd,
                                'np': np,
                                'plt': plt,
                            }
                            exec(code, {'__builtins__': __builtins__}, local_vars)
                            # After code execution, display the plot if plt is used
                            if 'plt' in code:
                                st.pyplot(plt.gcf())
                                plt.clf()  # Clear the figure
                            action_details = f"Executed code for query: {user_query}"
                        except Exception as e:
                            import traceback
                            error_message = str(e)
                            st.error(f"Error executing code: {error_message}")
                            st.code(code, language='python')
                            st.text("Traceback:")
                            st.text(traceback.format_exc())
                            action_details = f"Failed to execute code for query: {user_query}"
                    else:
                        st.write(response.text)
                        action_details = f"Received response: {response.text}"
                else:
                    st.error("No response from the Gemini model.")
                    action_details = "No response from the Gemini model"
                
                log_user_activity(
                    st.session_state['user_id'], 'NLP Chatbot', 'Success', action_details
                )
    else:
        st.warning("Please provide a Gemini API key to use the chatbot.")

# Audit logs function
@require_role(['Admin'])
def audit_logs():
    st.title("Audit Logs")
    st.dataframe(data['AuditLogs'])

# Main function
def main():
    st.sidebar.title("Navigation")
    role = st.session_state.get('role', None)
    if role:
        menu = ["Dashboard", "Maintenance Input", "Predictive Maintenance", "AI Chatbot"]
        if role == 'Admin':
            menu.append("Audit Logs")
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Dashboard":
            show_dashboard()
        elif choice == "Maintenance Input":
            maintenance_input()
        elif choice == "Predictive Maintenance":
            predictive_maintenance()
        elif choice == "AI Chatbot":
            nlp_chatbot()
        elif choice == "Audit Logs":
            audit_logs()
    else:
        login()

if __name__ == '__main__':
    data = load_data()
    if data is not None:
        if 'user_id' not in st.session_state:
            login()
        else:
            main()
    else:
        st.error("Failed to load data. Please check the data source.")
