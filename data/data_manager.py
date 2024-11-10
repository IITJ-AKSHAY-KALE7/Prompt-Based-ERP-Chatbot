import pandas as pd
import streamlit as st

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

def save_data(data):
    try:
        with pd.ExcelWriter('ERP_project.xlsx', mode='a', if_sheet_exists='replace') as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        st.success("Data saved successfully.")
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")