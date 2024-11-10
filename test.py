import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime
import spacy
from statsmodels.tsa.arima.model import ARIMA

# Load data function
@st.cache_data
def load_data():
    try:
        xls = pd.ExcelFile('ERP_project.xlsx')
        sheets = [
            'MaintenanceData', 'InventoryData', 'OperationalData', 'UserActivityData',
            'MaintenanceLogs', 'InventoryTransactions', 'EquipmentDetails', 'SupplierData',
            'UserData', 'ShiftData'
        ]
        data = {sheet: pd.read_excel(xls, sheet) for sheet in sheets}

        # Convert date columns to datetime
        date_columns = {
            'MaintenanceData': ['Date', 'NextScheduledMaintenance'],
            'OperationalData': ['Date'],
            'InventoryTransactions': ['Date'],
            'MaintenanceLogs': ['Date'],
            'UserActivityData': ['Timestamp'],
            'EquipmentDetails': ['InstallationDate', 'WarrantyExpiryDate', 'LastMaintenanceDate', 'NextMaintenanceDue'],
            'UserData': ['DateCreated', 'LastLoginDate'],
            'ShiftData': ['StartTime', 'EndTime']
        }
        for sheet, columns in date_columns.items():
            for col in columns:
                data[sheet][col] = pd.to_datetime(data[sheet][col], errors='coerce')

        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Save data function
def save_data():
    try:
        with pd.ExcelWriter('ERP_project.xlsx', mode='a', if_sheet_exists='replace') as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        st.success("Data saved successfully.")
        load_data.clear()  # Clear the cache after saving data
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")

# Authentication function
def authenticate_user(username, password):
    users = data['UserData']
    user_row = users[(users['UserName'] == username) & (users['PasswordHash'] == password)]
    if not user_row.empty:
        return user_row.iloc[0]['UserRole']
    else:
        return None

# Login function
def login():
    st.title("ERP System Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        role = authenticate_user(username, password)
        if role:
            st.session_state['username'] = username
            st.session_state['role'] = role
            st.success(f"Welcome {username}!")
            log_user_activity(username, 'Login', 'Success')
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
            log_user_activity(username, 'Login', 'Failure')

# User activity logging function
def log_user_activity(user_name, action_type, success_flag, action_details=''):
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

# Main function
def main():
    st.sidebar.title("Navigation")
    role = st.session_state.get('role', None)
    if role:
        menu = [
            "Dashboard", "Maintenance Input", "Predictive Maintenance",
            "Inventory Optimization", "NLP Chatbot", "Supplier Management"
        ]
        if role == 'Admin':
            menu.append("Audit Logs")
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Dashboard":
            show_dashboard()
        elif choice == "Maintenance Input":
            maintenance_input()
        elif choice == "Predictive Maintenance":
            predictive_maintenance()
        elif choice == "Inventory Optimization":
            inventory_optimization()
        elif choice == "NLP Chatbot":
            nlp_chatbot()
        elif choice == "Audit Logs":
            audit_logs()
        elif choice == "Supplier Management":
            supplier_management()
    else:
        login()

# Dashboard function
@require_role(['Admin', 'Manager', 'Technician', 'Employee'])
def show_dashboard():
    st.title("ERP System Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Maintenance Overview")
        maintenance_counts = data['MaintenanceData']['MaintenanceType'].value_counts()
        fig = px.pie(
            values=maintenance_counts.values,
            names=maintenance_counts.index,
            title="Maintenance Types"
        )
        st.plotly_chart(fig)

    with col2:
        st.subheader("Inventory Levels")
        inventory_data = data['InventoryData'].sort_values(
            'QuantityOnHand', ascending=False
        ).head(10)
        fig = px.bar(
            inventory_data,
            x='ItemName',
            y='QuantityOnHand',
            title="Top 10 Items by Quantity"
        )
        st.plotly_chart(fig)

    st.subheader("Recent Maintenance Logs")
    st.dataframe(data['MaintenanceLogs'].tail(5))

# Maintenance input function
@require_role(['Admin', 'Technician'])
def maintenance_input():
    st.title("Maintenance Data Input")
    with st.form("maintenance_form"):
        maintenance_id = st.text_input("Maintenance ID")
        date = st.date_input("Date")
        equipment_id = st.selectbox(
            "Equipment ID", data['EquipmentDetails']['EquipmentID'].unique()
        )
        equipment_type = st.text_input("Equipment Type")
        technician_id = st.text_input("Technician ID")
        maintenance_type = st.selectbox("Maintenance Type", ["Preventive", "Corrective"])
        issue_description = st.text_area("Issue Description")
        actions_taken = st.text_area("Actions Taken")
        downtime_hours = st.number_input("Downtime Hours", min_value=0.0)
        maintenance_cost = st.number_input("Maintenance Cost", min_value=0.0)
        next_scheduled_maintenance = st.date_input("Next Scheduled Maintenance")
        sensor_temp = st.number_input("Sensor Reading Temperature")
        sensor_vibration = st.number_input("Sensor Reading Vibration")
        comments = st.text_area("Comments")
        submit = st.form_submit_button("Submit")

    if submit:
        if not maintenance_id or not equipment_id or not technician_id \
           or not issue_description or not actions_taken:
            st.error("Please fill all required fields.")
        else:
            new_record = {
                'MaintenanceID': maintenance_id,
                'Date': date,
                'EquipmentID': equipment_id,
                'EquipmentType': equipment_type,
                'TechnicianID': technician_id,
                'MaintenanceType': maintenance_type,
                'IssueDescription': issue_description,
                'ActionsTaken': actions_taken,
                'DowntimeHours': downtime_hours,
                'MaintenanceCost': maintenance_cost,
                'NextScheduledMaintenance': next_scheduled_maintenance,
                'SensorReading_Temperature': sensor_temp,
                'SensorReading_Vibration': sensor_vibration,
                'Comments': comments
            }
            data['MaintenanceData'] = pd.concat(
                [data['MaintenanceData'], pd.DataFrame([new_record])],
                ignore_index=True
            )
            save_data()
            st.success("Maintenance record added successfully.")
            log_user_activity(
                st.session_state['username'], 'Maintenance Input', 'Success'
            )

# Predictive maintenance function
@require_role(['Admin', 'Manager'])
def predictive_maintenance():
    st.title("Predictive Maintenance")
    try:
        maintenance_data = data['MaintenanceData']
        maintenance_data['Date'] = pd.to_datetime(maintenance_data['Date'])
        maintenance_data.sort_values(['EquipmentID', 'Date'], inplace=True)
        maintenance_data['DaysSinceLastMaintenance'] = maintenance_data.groupby('EquipmentID')['Date'].diff().dt.days.fillna(0)

        X = maintenance_data[['DaysSinceLastMaintenance', 'SensorReading_Temperature', 'SensorReading_Vibration']]
        y = (maintenance_data['MaintenanceType'] == 'Corrective').astype(int)

        X = X.fillna(0)

        if X.empty or y.empty:
            st.error("Not enough data to build the model.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.subheader("Model Performance")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Maintenance Prediction")
        equipment_id = st.selectbox(
            "Select Equipment ID", data['EquipmentDetails']['EquipmentID'].unique()
        )
        days_since_last = st.number_input("Days Since Last Maintenance", min_value=0)
        temp = st.number_input("Current Temperature Reading")
        vibration = st.number_input("Current Vibration Reading")

        if st.button("Predict"):
            try:
                prediction = model.predict([[days_since_last, temp, vibration]])
                if prediction[0] == 1:
                    st.warning("Corrective maintenance may be needed soon.")
                else:
                    st.success("No immediate maintenance needed.")
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
    except Exception as e:
        st.error(f"Error in predictive maintenance: {str(e)}")

# Inventory optimization function
@require_role(['Admin', 'Manager'])
def inventory_optimization():
    st.title("Inventory Optimization")
    try:
        item_id = st.selectbox("Select Item", data['InventoryData']['ItemID'].unique())
        item_data = data['InventoryTransactions'][data['InventoryTransactions']['ItemID'] == item_id]
        item_data['Date'] = pd.to_datetime(item_data['Date'])
        item_data = item_data.set_index('Date')
        item_data = item_data.resample('D')['Quantity'].sum().fillna(0)

        if item_data.empty:
            st.error("No transaction data available for this item.")
            return

        # Fit ARIMA model
        try:
            model = ARIMA(item_data, order=(1, 1, 1))
            results = model.fit()

            forecast = results.forecast(steps=30)

            st.subheader("30-Day Forecast")
            fig = px.line(
                x=forecast.index, y=forecast.values,
                title=f"Forecasted Demand for Item {item_id}"
            )
            st.plotly_chart(fig)

            current_stock = data['InventoryData'][
                data['InventoryData']['ItemID'] == item_id
            ]['QuantityOnHand'].values[0]
            st.info(f"Current stock: {current_stock}")
            st.info(f"Forecasted demand (next 30 days): {forecast.sum():.2f}")

            if forecast.sum() > current_stock:
                recommended_order = forecast.sum() - current_stock
                st.warning(f"Reorder recommended. Suggested order quantity: {recommended_order:.2f}")
            else:
                st.success("Current stock level is sufficient for forecasted demand.")
        except Exception as e:
            st.error(f"Error fitting ARIMA model: {str(e)}")
    except Exception as e:
        st.error(f"Error in inventory optimization: {str(e)}")

# NLP chatbot function
@require_role(['Admin', 'Manager', 'Technician', 'Employee'])
def nlp_chatbot():
    st.title("AI Chatbot")
    nlp = spacy.load('en_core_web_sm')

    user_query = st.text_input("Ask me anything about the ERP system:")
    if user_query:
        doc = nlp(user_query.lower())
        action_details = "Unrecognized query"

        if "inventory levels" in user_query.lower():
            fig = px.bar(
                data['InventoryData'], x='ItemName', y='QuantityOnHand',
                title='Inventory Levels'
            )
            st.plotly_chart(fig)
            action_details = "Displayed inventory levels"
        elif "production volume graph" in user_query.lower():
            operational_data = data['OperationalData']
            operational_data['Date'] = pd.to_datetime(operational_data['Date'])
            production_volume = operational_data.groupby('Date')['ProductionVolume'].sum()
            fig = px.line(
                x=production_volume.index, y=production_volume.values,
                title='Production Volume Over Time'
            )
            st.plotly_chart(fig)
            action_details = "Displayed production volume graph"
        elif "maintenance logs for" in user_query.lower():
            try:
                equipment_id = user_query.lower().split("for")[-1].strip()
                logs = data['MaintenanceLogs']
                equipment_logs = logs[logs['EquipmentID'].str.lower() == equipment_id.lower()]
                if not equipment_logs.empty:
                    st.write(equipment_logs)
                else:
                    st.write(f"No logs found for equipment ID: {equipment_id}")
                action_details = f"Displayed maintenance logs for {equipment_id}"
            except Exception as e:
                st.error(f"Error retrieving maintenance logs: {str(e)}")
        else:
            st.write("I'm sorry, I didn't understand that query. You can ask about inventory levels, production volume, or maintenance logs for specific equipment.")

        log_user_activity(
            st.session_state['username'], 'NLP Chatbot', 'Success', action_details
        )

# Audit logs function
@require_role(['Admin'])
def audit_logs():
    st.title("Audit Logs")
    st.dataframe(data['UserActivityData'])

# Supplier management function
@require_role(['Admin', 'Manager'])
def supplier_management():
    st.title("Supplier Management")
    suppliers = data['SupplierData']
    st.dataframe(suppliers)

    if st.session_state['role'] == 'Admin':
        st.subheader("Add New Supplier")
        with st.form("supplier_form"):
            supplier_id = st.text_input("Supplier ID")
            supplier_name = st.text_input("Supplier Name")
            contact_person = st.text_input("Contact Person")
            phone_number = st.text_input("Phone Number")
            email = st.text_input("Email")
            address = st.text_area("Address")
            city = st.text_input("City")
            state = st.text_input("State")
            postal_code = st.text_input("Postal Code")
            country = st.text_input("Country")
            preferred_items = st.text_input("Preferred Items")
            payment_terms = st.text_input("Payment Terms")
            comments = st.text_area("Comments")
            submit = st.form_submit_button("Add Supplier")

        if submit:
            if not supplier_id or not supplier_name:
                st.error("Supplier ID and Supplier Name are required.")
            else:
                new_supplier = {
                    'SupplierID': supplier_id,
                    'SupplierName': supplier_name,
                    'ContactPerson': contact_person,
                    'PhoneNumber': phone_number,
                    'Email': email,
                    'Address': address,
                    'City': city,
                    'State': state,
                    'PostalCode': postal_code,
                    'Country': country,
                    'PreferredItems': preferred_items,
                    'PaymentTerms': payment_terms,
                    'Comments': comments
                }
                data['SupplierData'] = pd.concat(
                    [data['SupplierData'], pd.DataFrame([new_supplier])],
                    ignore_index=True
                )
                save_data()
                st.success("Supplier added successfully.")
                log_user_activity(
                    st.session_state['username'], 'Add Supplier', 'Success'
                )

    st.subheader("Update Supplier")
    selected_supplier = st.selectbox(
        "Select Supplier to Update", suppliers['SupplierID'].tolist(), key="update_supplier"
    )
    if selected_supplier:
        supplier_data = suppliers[suppliers['SupplierID'] == selected_supplier]
        if not supplier_data.empty:
            supplier_data = supplier_data.iloc[0]
            with st.form("update_supplier_form"):
                updated_name = st.text_input("Supplier Name", value=supplier_data['SupplierName'])
                updated_contact = st.text_input("Contact Person", value=supplier_data['ContactPerson'])
                updated_phone = st.text_input("Phone Number", value=supplier_data['PhoneNumber'])
                updated_email = st.text_input("Email", value=supplier_data['Email'])
                updated_address = st.text_area("Address", value=supplier_data['Address'])
                updated_city = st.text_input("City", value=supplier_data['City'])
                updated_state = st.text_input("State", value=supplier_data['State'])
                updated_postal = st.text_input("Postal Code", value=supplier_data['PostalCode'])
                updated_country = st.text_input("Country", value=supplier_data['Country'])
                updated_items = st.text_input("Preferred Items", value=supplier_data['PreferredItems'])
                updated_terms = st.text_input("Payment Terms", value=supplier_data['PaymentTerms'])
                updated_comments = st.text_area("Comments", value=supplier_data['Comments'])
                update_submit = st.form_submit_button("Update Supplier")

            if update_submit:
                data['SupplierData'].loc[
                    data['SupplierData']['SupplierID'] == selected_supplier, :
                ] = {
                    'SupplierID': selected_supplier,
                    'SupplierName': updated_name,
                    'ContactPerson': updated_contact,
                    'PhoneNumber': updated_phone,
                    'Email': updated_email,
                    'Address': updated_address,
                    'City': updated_city,
                    'State': updated_state,
                    'PostalCode': updated_postal,
                    'Country': updated_country,
                    'PreferredItems': updated_items,
                    'PaymentTerms': updated_terms,
                    'Comments': updated_comments
                }
                save_data()
                st.success("Supplier updated successfully.")
                log_user_activity(
                    st.session_state['username'], 'Update Supplier', 'Success'
                )
        else:
            st.error("Selected supplier not found.")

    if st.session_state['role'] == 'Admin':
        st.subheader("Delete Supplier")
        supplier_to_delete = st.selectbox(
            "Select Supplier to Delete", suppliers['SupplierID'].tolist(), key="delete_supplier"
        )
        if st.button("Delete Supplier"):
            confirm = st.checkbox("Confirm deletion")
            if confirm:
                data['SupplierData'] = data['SupplierData'][
                    data['SupplierData']['SupplierID'] != supplier_to_delete
                ]
                save_data()
                st.success(f"Supplier {supplier_to_delete} deleted successfully.")
                log_user_activity(
                    st.session_state['username'], 'Delete Supplier', 'Success'
                )
            else:
                st.warning("Please confirm deletion by checking the box.")

if __name__ == '__main__':
    data = load_data()  # Load data at the start of the application
    if data is not None:
        if 'username' not in st.session_state:
            login()
        else:
            main()
    else:
        st.error("Failed to load data. Please check the data source.")


