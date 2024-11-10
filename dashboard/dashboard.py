import streamlit as st
import plotly.express as px
from  auth.auth import require_role

@require_role(['Admin', 'Manager', 'Technician', 'Employee'])
def show_dashboard(data):
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