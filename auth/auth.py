import streamlit as st
import datetime
import pandas as pd

def authenticate_user(username, password, data):
    users = data['UserData']
    user_row = users[(users['UserName'] == username) & (users['PasswordHash'] == password)]
    return user_row.iloc[0]['UserRole'] if not user_row.empty else None

def login(data):
    st.title("ERP System Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        role = authenticate_user(username, password, data)
        if role:
            st.session_state['username'] = username
            st.session_state['role'] = role
            log_user_activity(data, username, 'Login', 'Success')
            st.success(f"Welcome {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
            log_user_activity(data, username, 'Login', 'Failure')

def log_user_activity(data, user_name, action_type, success_flag, action_details=''):
    new_log = {
        'ActivityID': len(data['UserActivityData']) + 1,
        'Timestamp': datetime.datetime.now(),
        'UserID': user_name,
        'UserName': user_name,
        'UserRole': st.session_state.get('role', ''),
        'ActionType': action_type,
        'ActionDetails': action_details,
        'IPAddress': '',
        'SuccessFlag': success_flag,
        'Comments': ''
    }
    data['UserActivityData'] = pd.concat(
        [data['UserActivityData'], pd.DataFrame([new_log])],
        ignore_index=True
    )

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