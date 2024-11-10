import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objs as go

from auth.auth import require_role, log_user_activity
from data.data_manager import save_data

@require_role(['Admin', 'Technician'])
def maintenance_input(data):
    st.title("Maintenance Data Input")
    
    with st.form("maintenance_form"):
        # Basic Information
        col1, col2 = st.columns(2)
        with col1:
            maintenance_id = st.text_input("Maintenance ID", 
                                           value=f"MAINT-{len(data['MaintenanceData'])+1:04d}")
            date = st.date_input("Date")
            equipment_id = st.selectbox(
                "Equipment ID", 
                data['EquipmentDetails']['EquipmentID'].unique()
            )
            equipment_type = st.text_input("Equipment Type")
            technician_id = st.text_input("Technician ID")
        
        with col2:
            maintenance_type = st.selectbox(
                "Maintenance Type", 
                ["Preventive", "Corrective", "Predictive", "Emergency"]
            )
            downtime_hours = st.number_input("Downtime Hours", min_value=0.0, step=0.5)
            maintenance_cost = st.number_input("Maintenance Cost", min_value=0.0, step=10.0)
            next_scheduled_maintenance = st.date_input("Next Scheduled Maintenance")
        
        # Detailed Information
        issue_description = st.text_area("Issue Description")
        actions_taken = st.text_area("Actions Taken")
        
        # Sensor Readings
        col3, col4 = st.columns(2)
        with col3:
            sensor_temp = st.number_input("Sensor Reading Temperature (°C)")
        with col4:
            sensor_vibration = st.number_input("Sensor Reading Vibration")
        
        comments = st.text_area("Additional Comments")
        
        submit = st.form_submit_button("Submit Maintenance Record")

    if submit:
        # Validation
        if not all([maintenance_id, equipment_id, technician_id, 
                    issue_description, actions_taken]):
            st.error("Please fill all required fields.")
            return

        # Prepare new maintenance record
        new_record = {
            'MaintenanceID': maintenance_id,
            'Date': pd.to_datetime(date),
            'EquipmentID': equipment_id,
            'EquipmentType': equipment_type,
            'TechnicianID': technician_id,
            'MaintenanceType': maintenance_type,
            'IssueDescription': issue_description,
            'ActionsTaken': actions_taken,
            'DowntimeHours': downtime_hours,
            'MaintenanceCost': maintenance_cost,
            'NextScheduledMaintenance': pd.to_datetime(next_scheduled_maintenance),
            'SensorReading_Temperature': sensor_temp,
            'SensorReading_Vibration': sensor_vibration,
            'Comments': comments
        }

        # Add to MaintenanceData
        data['MaintenanceData'] = pd.concat([
            data['MaintenanceData'], 
            pd.DataFrame([new_record])
        ], ignore_index=True)

        # Save data
        save_data(data)

        # Log activity
        log_user_activity(
            data, 
            st.session_state['username'], 
            'Maintenance Input', 
            'Success',
            f"Added Maintenance Record: {maintenance_id}"
        )

        st.success(f"Maintenance Record {maintenance_id} Added Successfully!")

@require_role(['Admin', 'Manager'])
def predictive_maintenance(data):
    st.title("Predictive Maintenance")
    
    try:
        # Prepare maintenance data
        maintenance_data = data['MaintenanceData'].copy()
        maintenance_data['Date'] = pd.to_datetime(maintenance_data['Date'])
        maintenance_data.sort_values(['EquipmentID', 'Date'], inplace=True)
        
        # Calculate days since last maintenance
        maintenance_data['DaysSinceLastMaintenance'] = maintenance_data.groupby('EquipmentID')['Date'].diff().dt.days.fillna(0)
        
        # Prepare features for prediction
        features = [
            'DaysSinceLastMaintenance', 
            'SensorReading_Temperature', 
            'SensorReading_Vibration'
        ]
        
        # Prepare target variable (1 for Corrective, 0 for Preventive)
        maintenance_data['MaintenanceRisk'] = (maintenance_data['MaintenanceType'] == 'Corrective').astype(int)
        
        # Handle missing values
        X = maintenance_data[features].fillna(0)
        y = maintenance_data['MaintenanceRisk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Model Performance
    
        st.subheader("Model Performance")
        performance_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Performance Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{performance_report['accuracy']:.2%}")
            st.metric("Precision", f"{performance_report['weighted avg']['precision']:.2%}")
        
        with col2:
            st.metric("Recall", f"{performance_report['weighted avg']['recall']:.2%}")
            st.metric("F1-Score", f"{performance_report['weighted avg']['f1-score']:.2%}")
        
        # Prediction Interface
        st.subheader("Maintenance Risk Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days_since_last = st.number_input("Days Since Last Maintenance", min_value=0)
        
        with col2:
            current_temp = st.number_input("Current Temperature Reading")
        
        with col3:
            current_vibration = st.number_input("Current Vibration Reading")
        
        if st.button("Predict Maintenance Risk"):
            # Prepare input for prediction
            input_data = scaler.transform([[days_since_last, current_temp, current_vibration]])
            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data)
            
            if prediction[0] == 1:
                st.warning(f"High Maintenance Risk Detected! Probability: {proba[0][1]:.2%}")
            else:
                st.success(f"Low Maintenance Risk. Probability: {proba[0][0]:.2%}")
        
        # Equipment Maintenance Trend
        st.subheader("Equipment Maintenance Trend")
        trend_data = maintenance_data.groupby(['EquipmentID', 'MaintenanceType']).size().reset_index(name='Count')
        
        fig = px.bar(
            trend_data, 
            x='EquipmentID', 
            y='Count', 
            color='MaintenanceType',
            title='Maintenance Types by Equipment'
        )
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error in Predictive Maintenance: {str(e)}")

def maintenance_dashboard(data):
    st.title("Maintenance Dashboard")
    
    # Maintenance Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Maintenance Type Distribution")
        maintenance_type_counts = data['MaintenanceData']['MaintenanceType'].value_counts()
        fig_pie = px.pie(
            values=maintenance_type_counts.values,
            names=maintenance_type_counts.index,
            title="Maintenance Types"
        )
        st.plotly_chart(fig_pie)
    
    with col2:
        st.subheader("Maintenance Cost Analysis")
        cost_by_equipment = data['MaintenanceData'].groupby('EquipmentID')['MaintenanceCost'].sum()
        fig_bar = px.bar(
            x=cost_by_equipment.index, 
            y=cost_by_equipment.values,
            title="Total Maintenance Cost by Equipment"
        )
        st.plotly_chart(fig_bar)
    
    # Downtime Analysis
    st.subheader("Downtime Analysis")
    downtime_by_type = data['MaintenanceData'].groupby('MaintenanceType')['DowntimeHours'].sum()
    fig_downtime = px.bar(
        x=downtime_by_type.index, 
        y=downtime_by_type.values,
        title="Total Downtime Hours by Maintenance Type"
    )
    st.plotly_chart(fig_downtime)

# Export functions for use in main application
__all__ = [
    'maintenance_input', 
    'predictive_maintenance', 
    'maintenance_dashboard'
]