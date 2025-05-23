import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configure page
st.set_page_config(
    page_title="AI + IoT Monitoring Dashboard",
    page_icon="🛢️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.gauge-container {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
}
.alert-badge {
    background-color: #ffcccc;
    padding: 5px 10px;
    border-radius: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# 1. Load Production Data
# ----------------------
@st.cache_data
def load_production_data():
    prod_data = pd.read_excel('Production_data.xlsx')
    prod_data['Date'] = pd.to_datetime(prod_data['Date'], format='%Y-%m-%d')
    return prod_data.sort_values('Date')

# ----------------------
# 2. Load ESP Monitoring Data
# ----------------------
@st.cache_data
def load_esp_data():
    # Replace with your actual loading logic
    main_df3 = pd.read_excel('NEW_ESP_DATA.xlsx', sheet_name=None)
    monitor_dfs = list(main_df3.values())
    monitor_df = pd.concat(monitor_dfs, ignore_index=True)
    monitor_df = monitor_df.drop(columns=['Remark'], errors='ignore')
    monitor_df['DateTime'] = pd.to_datetime(monitor_df['Date'], errors='coerce')
    return monitor_df.sort_values('DateTime')

# ----------------------
# Data Processing
# ----------------------
def preprocess_esp_data(df):
    # Replace invalid entries and convert to numeric
    to_numeric_cols = ['Freq (Hz)', 'Current (Amps)', 'Intake Press psi', 'Motor Temp (F)']
    df.replace('-', np.nan, inplace=True)
    for col in to_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ----------------------
# Anomaly Detection
# ----------------------
def detect_anomalies(df):
    features = df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi', 'Motor Temp (F)']].dropna()
    
    # Train multiple anomaly detection models
    iso = IsolationForest(contamination=0.05, random_state=42)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    svm = OneClassSVM(nu=0.05)
    
    df['anomaly_iso'] = np.nan
    df['anomaly_lof'] = np.nan
    df['anomaly_svm'] = np.nan
    
    df.loc[features.index, 'anomaly_iso'] = iso.fit_predict(features)
    df.loc[features.index, 'anomaly_lof'] = lof.fit_predict(features)
    df.loc[features.index, 'anomaly_svm'] = svm.fit_predict(features)
    
    # Combined anomaly score (0-3)
    df['anomaly_score'] = (
        (df['anomaly_iso'] == -1).astype(int) + 
        (df['anomaly_lof'] == -1).astype(int) + 
        (df['anomaly_svm'] == -1).astype(int)
    )
    
    return df

# ----------------------
# Predictive Modeling
# ----------------------
def train_predictive_models(df):
    features = df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi']].dropna()
    target_temp = df['Motor Temp (F)'].dropna()
    
    # Align indices
    common_idx = features.index.intersection(target_temp.index)
    features = features.loc[common_idx]
    target_temp = target_temp.loc[common_idx]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target_temp, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.session_state.model_mse = mse
    
    # Make predictions on full dataset
    df['Motor Temp Predicted (F)'] = model.predict(
        df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi']].fillna(features.mean())
    )
    
    return df, model

# ----------------------
# Load and Process Data
# ----------------------
prod_data = load_production_data()
esp_data = load_esp_data()
esp_data = preprocess_esp_data(esp_data)
esp_data = detect_anomalies(esp_data)
esp_data, temp_model = train_predictive_models(esp_data)

# Get last recorded values
last_reading = esp_data.iloc[100]
has_anomaly = last_reading['anomaly_score'] >= 2

# ----------------------
# Dashboard Layout
# ----------------------
st.title("🛢️ AI + IoT Monitoring Dashboard")

# Status Overview
st.header("Current System Status")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Frequency (Hz)", f"{20:.1f}")

with col2:
    st.metric("Motor Current (Amps)", f"{2.2:.1f}")

with col3:
    anomaly_status = "⚠️ ALERT" if has_anomaly else "✅ Normal"
    st.metric("System Status", anomaly_status)

with col4:
    st.metric("Prediction MSE", f"{st.session_state.model_mse:.2f}")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Production Data", "ESP Monitoring", "Anomaly Analysis"])

with tab1:
    st.header("Production Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            prod_data, x='Date', y='Gross Act (BBL)',
            title='Gross Production (BBL)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(
            prod_data, x='Date', y='BSW',
            title='Basic Sediment & Water (%)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            prod_data, x='Date', y='Gas Produced (MMSCFD)',
            title='Gas Production (MMSCFD)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(
            prod_data, x='Date', y='Hrs of Production',
            title='Production Hours',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("ESP Monitoring Data")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            esp_data, x='DateTime', y='Freq (Hz)',
            title='Pump Frequency (Hz)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(
            esp_data, x='DateTime', y='Intake Press psi',
            title='Intake Pressure (psi)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            esp_data, x='DateTime', y='Current (Amps)',
            title='Motor Current (Amps)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(
            esp_data, x='DateTime', 
            y=['Motor Temp (F)', 'Motor Temp Predicted (F)'],
            title='Motor Temperature: Actual vs Predicted',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Anomaly Detection Analysis")
    
    st.subheader("Anomaly Distribution")
    anomaly_counts = esp_data['anomaly_score'].value_counts().sort_index()
    fig = px.bar(
        anomaly_counts, 
        title="Count of Anomalies by Severity Level",
        labels={'index': 'Anomaly Score', 'value': 'Count'},
        color=anomaly_counts.index,
        color_continuous_scale='reds'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Anomaly Events")
    anomaly_events = esp_data[esp_data['anomaly_score'] > 0]
    
    if not anomaly_events.empty:
        col1, col2 = st.columns(2)
        # with col1:
        #     fig = px.scatter(
        #         anomaly_events, x='DateTime', y='Freq (Hz)',
        #         color='anomaly_score',
        #         size='anomaly_score',
        #         title='Anomalies in Frequency',
        #         color_continuous_scale='reds',
        #         range_color=[1, 3]
        #     )
        #     st.plotly_chart(fig, use_container_width=True)
        
        # with col2:
        #     fig = px.scatter(
        #         anomaly_events, x='DateTime', y='Motor Temp (F)',
        #         color='anomaly_score',
        #         size='anomaly_score',
        #         title='Anomalies in Motor Temperature',
        #         color_continuous_scale='reds',
        #         range_color=[1, 3]
        #     )
        #     st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Anomaly Records")
        st.dataframe(
            anomaly_events[
                ['DateTime', 'Freq (Hz)', 'Current (Amps)', 
                 'Intake Press psi', 'Motor Temp (F)', 'anomaly_score']
            ].sort_values('DateTime', ascending=False),
            height=300
        )
    else:
        st.success("No anomalies detected in the dataset")

# Footer
st.markdown("---")
st.markdown("""
**Dashboard Features:**
- Production performance trends
- ESP sensor monitoring
- Multi-model anomaly detection (Isolation Forest, LOF, One-Class SVM)
- Motor temperature prediction (Random Forest)
""")
