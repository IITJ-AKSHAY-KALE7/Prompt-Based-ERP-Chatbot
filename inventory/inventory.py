import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from auth.auth import require_role, log_user_activity
from data.data_manager import save_data

@require_role(['Admin', 'Manager'])
def inventory_optimization(data):
    st.title("Inventory Optimization")
    
    # Sidebar for Inventory Analysis Options
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type", 
        [
            "Overview", 
            "Item Details", 
            "Demand Forecasting", 
            "Stock Classification",
            "Reorder Point Analysis"
        ]
    )

    if analysis_type == "Overview":
        inventory_overview(data)
    elif analysis_type == "Item Details":
        item_details(data)
    elif analysis_type == "Demand Forecasting":
        demand_forecasting(data)
    elif analysis_type == "Stock Classification":
        stock_classification(data)
    elif analysis_type == "Reorder Point Analysis":
        reorder_point_analysis(data)

def inventory_overview(data):
    st.subheader("Inventory Overview")
    
    # Total Inventory Value
    data['InventoryData']['TotalValue'] = data['InventoryData']['QuantityOnHand'] * data['InventoryData']['UnitPrice']
    total_inventory_value = data['InventoryData']['TotalValue'].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Inventory Value", f"${total_inventory_value:,.2f}")
    
    with col2:
        st.metric("Total Unique Items", len(data['InventoryData']))
    
    with col3:
        low_stock_items = data['InventoryData'][data['InventoryData']['QuantityOnHand'] < data['InventoryData']['ReorderLevel']]
        st.metric("Low Stock Items", len(low_stock_items))
    
    # Inventory Distribution
    st.subheader("Inventory Distribution by Category")
    category_inventory = data['InventoryData'].groupby('Category')['TotalValue'].sum()
    
    fig = px.pie(
        values=category_inventory.values, 
        names=category_inventory.index, 
        title="Inventory Value by Category"
    )
    st.plotly_chart(fig)

def item_details(data):
    st.subheader("Item Details and Transaction History")
    
    # Item Selection
    item_id = st.selectbox(
        "Select Item", 
        data['InventoryData']['ItemID'].unique()
    )
    
    # Item Details
    item_details = data['InventoryData'][data['InventoryData']['ItemID'] == item_id].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Stock", item_details['QuantityOnHand'])
    
    with col2:
        st.metric("Unit Price", f"${item_details['UnitPrice']:.2f}")
    
    with col3:
        st.metric("Reorder Level", item_details['ReorderLevel'])
    
    # Transaction History
    item_transactions = data['InventoryTransactions'][
        data['InventoryTransactions']['ItemID'] == item_id
    ]
    
    st.subheader("Transaction History")
    st.dataframe(item_transactions)
    
    # Transaction Trend
    st.subheader("Transaction Trend")
    item_transactions['Date'] = pd.to_datetime(item_transactions['Date'])
    transaction_trend = item_transactions.groupby(
        pd.Grouper(key='Date', freq='M')
    )['Quantity'].sum()
    
    fig = px.line(
        x=transaction_trend.index, 
        y=transaction_trend.values, 
        title="Monthly Transaction Quantity"
    )
    st.plotly_chart(fig)

def demand_forecasting(data):
    st.subheader("Demand Forecasting")
    
    item_id = st.selectbox(
        "Select Item for Forecasting", 
        data['InventoryData']['ItemID'].unique()
    )
    
    # Prepare time series data
    item_transactions = data['InventoryTransactions'][
        data['InventoryTransactions']['ItemID'] == item_id
    ]
    item_transactions['Date'] = pd.to_datetime(item_transactions['Date'])
    monthly_demand = item_transactions.groupby(
        pd.Grouper(key='Date', freq='M')
    )['Quantity'].sum()
    
    # ARIMA Forecasting
    try:
        model = ARIMA(monthly_demand, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=6)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_demand.index, 
            y=monthly_demand.values, 
            mode='lines', 
            name='Historical Demand'
        ))
        fig.add_trace(go.Scatter(
            x=pd.date_range(monthly_demand.index[-1], periods=7)[1:],
            y=forecast,
            mode='lines',
            name='Forecasted Demand',
            line=dict(dash='dot')
        ))
        
        st.plotly_chart(fig)
        
        st.subheader("Forecast Summary")
        forecast_df = pd.DataFrame({
            'Month': pd.date_range(monthly_demand.index[-1], periods=7)[1:],
            'Forecasted Demand': forecast
        })
        st.dataframe(forecast_df)
        
    except Exception as e:
        st.error(f"Forecasting Error: {str(e)}")

def stock_classification(data):
    st.subheader("Stock Classification (ABC-XYZ Analysis)")
    
    # Prepare data for clustering
    inventory_data = data['InventoryData'].copy()
    inventory_data['AnnualConsumption'] = inventory_data['QuantityOnHand'] * inventory_data['UnitPrice']
    
    # Normalize data
    scaler = StandardScaler()
    features = ['AnnualConsumption', 'QuantityOnHand']
    X = scaler.fit_transform(inventory_data[features])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    inventory_data['StockClass'] = kmeans.fit_predict(X)
    
    # Visualization
    fig = px.scatter(
        inventory_data, 
        x='AnnualConsumption', 
        y='QuantityOnHand', 
        color='StockClass',
        title='Stock Classification'
    )
    st.plotly_chart(fig)
    
    # Detailed Classification
    st.subheader("Stock Classification Details")
    classification_summary = inventory_data.groupby('StockClass').agg({
        'ItemID': 'count',
        'AnnualConsumption': 'sum',
        'QuantityOnHand': 'mean'
    })
    st.dataframe(classification_summary)

def reorder_point_analysis(data):
    st.subheader("Reorder Point Analysis")
    
    # Calculate Reorder Metrics
    inventory_data = data['InventoryData'].copy()
    inventory_data['DaysUntilReorder'] = (
        inventory_data['QuantityOnHand'] / 
        (inventory_data['AverageDailyUsage'] + 1)
    )
    
    # Identify Items Needing Reorder
    reorder_items = inventory_data[
        inventory_data['QuantityOnHand'] <= inventory_data['ReorderLevel']
    ]
    
    st.subheader("Items Requiring Reorder")
    st.dataframe(reorder_items[['ItemID', 'ItemName', 'QuantityOnHand', 'ReorderLevel', 'DaysUntilReorder']])
    
    # Visualization
    fig = px.bar(
        reorder_items, 
        x='ItemName', 
        y='DaysUntilReorder', 
        title='Days Until Reorder by Item'
    )
    st.plotly_chart(fig)

def inventory_input(data):
    st.title("Inventory Data Input")
    
    with st.form("inventory_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            item_id = st.text_input("Item ID")
            item_name = st.text_input("Item Name")
            category = st.selectbox("Category", ["Raw Material", "Finished Goods", "Spare Parts"])
            
        with col2:
            quantity_on_hand = st.number_input("Quantity on Hand", min_value=0)
            unit_price = st.number_input("Unit Price", min_value=0.0, step=0.01)
            reorder_level = st.number_input("Reorder Level", min_value=0)
        
        description = st.text_area("Description")
        submit = st.form_submit_button("Add Inventory Item")
    
    if submit:
        new_item = {
            'ItemID': item_id,
            'ItemName': item_name,
            'Category': category,
            'QuantityOnHand': quantity_on_hand,
            'UnitPrice': unit_price,
            'ReorderLevel': reorder_level,
            'Description': description
        }
        
        data['InventoryData'] = pd.concat([
            data['InventoryData'], 
            pd.DataFrame([new_item])
        ], ignore_index=True)
        
        save_data(data)
        
        log_user_activity(
            data, 
            st.session_state['username'], 
            'Inventory Input', 
            'Success',
            f"Added Inventory Item: {item_id}"
        )
        
        st.success(f"Inventory Item {item_id} Added Successfully!")

# Export functions for use in main application
__all__ = [
    'inventory_optimization', 
    'inventory_input'
]