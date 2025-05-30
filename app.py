import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from lifelines import CoxPHFitter
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.tree import export_text
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Oil Well AI Monitoring Dashboard",
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
.healthy {
    color: green;
    font-weight: bold;
}
.warning {
    color: orange;
    font-weight: bold;
}
.danger {
    color: red;
    font-weight: bold;
}
.explanation {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.insight-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# 1. Load Production Data
# ----------------------
@st.cache_data
def load_production_data():
    # In a real app, you would load from your actual data source
    # This creates sample data if the file isn't found
    try:
        prod_data = pd.read_excel(r"Production_data.xlsx")
    except:
        print('This production data was Simulated')
        dates = pd.date_range(start='2023-01-01', end='2023-12-31')
        prod_data = pd.DataFrame({
            'Date': dates,
            'Gross Act (BBL)': np.random.normal(500, 50, len(dates)).cumsum(),
            'BSW': np.random.uniform(5, 15, len(dates)),
            'Gas Produced (MMSCFD)': np.random.normal(2, 0.5, len(dates)),
            'Hrs of Production': np.random.uniform(18, 24, len(dates))
        })
    
    prod_data['Date'] = pd.to_datetime(prod_data['Date'], format='%Y-%m-%d')
    return prod_data.sort_values('Date')

# ----------------------
# 2. Load ESP Monitoring Data
# ----------------------
@st.cache_data
def load_esp_data():
    # Replace with your actual loading logic
    # This creates sample data if the file isn't found
    try:
        main_df3 = pd.read_excel(r"NEW_ESP_DATA.xlsx", sheet_name=None)
        monitor_dfs = list(main_df3.values())
        monitor_df = pd.concat(monitor_dfs, ignore_index=True)
        monitor_df = monitor_df.drop(columns=['Remark'], errors='ignore')
    except:
        print('This ESP data was Simulated')
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        monitor_df = pd.DataFrame({
            'Date': dates,
            'Freq (Hz)': np.random.normal(20, 2, len(dates)),
            'Current (Amps)': np.random.normal(2.5, 0.5, len(dates)),
            'Intake Press psi': np.random.normal(500, 50, len(dates)),
            'Motor Temp (F)': np.random.normal(150, 20, len(dates))
        })
    
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
# Predictive Maintenance (Time-to-Failure Estimation)
# ----------------------
def predict_remaining_useful_life(df):
    # Create features for survival analysis
    df['operating_hours'] = (df['DateTime'] - df['DateTime'].min()).dt.total_seconds()/3600
    df['temp_over_threshold'] = (df['Motor Temp (F)'] > 180).astype(int)
    df['current_spike'] = (df['Current (Amps)'] > 3.5).astype(int)
    
    # Train survival model
    cf = CoxPHFitter()
    survival_df = df[['operating_hours', 'temp_over_threshold', 
                     'current_spike', 'Freq (Hz)', 'anomaly_score']].dropna()
    cf.fit(survival_df, duration_col='operating_hours', event_col='current_spike')
    
    # Predict remaining useful life
    df['predicted_remaining_life'] = cf.predict_median(survival_df)
    return df, cf

# ----------------------
# Production Forecasting
# ----------------------
def forecast_production(prod_data):
    # ARIMA model for short-term forecasting
    daily_prod = prod_data.set_index('Date')['Gross Act (BBL)'].resample('D').mean()
    
    try:
        arima_model = ARIMA(daily_prod, order=(7,0,0)).fit()
        arima_forecast = arima_model.forecast(steps=14)
    except:
        arima_forecast = pd.Series(np.random.normal(daily_prod.mean(), daily_prod.std(), 14),
                                 index=pd.date_range(start=daily_prod.index[-1], periods=15)[1:])
    
    # Prophet model for longer-term forecasting
    prophet_df = prod_data[['Date', 'Gross Act (BBL)']].rename(columns={'Date':'ds', 'Gross Act (BBL)':'y'})
    prophet_model = Prophet(seasonality_mode='multiplicative')
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=90)
    prophet_forecast = prophet_model.predict(future)
    
    return arima_forecast, prophet_forecast, prophet_model

# ----------------------
# Equipment Clustering
# ----------------------
def cluster_equipment_states(df):
    features = df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi', 'Motor Temp (F)']].dropna()
    
    # Standardize and cluster
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(scaled_features)
    
    # Add clusters to dataframe
    df['operating_mode'] = np.nan
    df.loc[features.index, 'operating_mode'] = kmeans.labels_
    
    # Create cluster descriptions
    cluster_profiles = features.groupby(kmeans.labels_).agg(['mean', 'std'])
    return df, kmeans, cluster_profiles

# ----------------------
# Root Cause Analysis
# ----------------------
def analyze_anomaly_causes(df):
    # Prepare data
    X = df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi', 'Motor Temp (F)']].dropna()
    y = df.loc[X.index, 'anomaly_score'] > 0
    
    # Train classifier
    model = LogisticRegression(max_iter=1000).fit(X, y)
    
    # Feature importance
    importance = permutation_importance(model, X, y, n_repeats=10)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance.importances_mean
    }).sort_values('importance', ascending=False)
    
    return feature_importance, model

# ----------------------
# Automated Report Generation
# ----------------------
def generate_insight_reports(df, model):
    # Decision tree explanation
    clf = tree.DecisionTreeClassifier(max_depth=3)
    X = df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi']].dropna()
    y = (df.loc[X.index, 'Motor Temp (F)'] > 180).astype(int)
    clf.fit(X, y)
    
    # Generate textual report
    report = export_text(clf, feature_names=list(X.columns))
    
    # Create visual explanation
    fig = px.treemap(
        names=X.columns,
        parents=['']*len(X.columns),
        values=model.feature_importances_
    )
    
    return report, fig

# ----------------------
# Load and Process Data
# ----------------------
prod_data = load_production_data()
esp_data = load_esp_data()
esp_data = preprocess_esp_data(esp_data)
esp_data = detect_anomalies(esp_data)
esp_data, temp_model = train_predictive_models(esp_data)
esp_data, survival_model = predict_remaining_useful_life(esp_data)
arima_forecast, prophet_forecast, prophet_model = forecast_production(prod_data)
esp_data, kmeans_model, cluster_profiles = cluster_equipment_states(esp_data)
feature_importance, rca_model = analyze_anomaly_causes(esp_data)
report_text, importance_fig = generate_insight_reports(esp_data, temp_model)

# Get last recorded values
last_reading = esp_data.iloc[-1]
has_anomaly = last_reading['anomaly_score'] >= 2

# ----------------------
# Dashboard Layout
# ----------------------
st.title("🛢️ AI-Powered Oil Well Monitoring Dashboard")

# Status Overview
st.header("Current System Status")

# Explanation box
with st.expander("ℹ️ What am I looking at?"):
    st.markdown("""
    This dashboard monitors your oil well equipment in real-time using AI. It shows:
    - **Production Data**: Oil, gas, and water production metrics
    - **Equipment Health**: Pump performance and condition monitoring
    - **Alerts**: Automatic detection of abnormal conditions
    - **Predictions**: Forecasts of production and equipment lifespan
    - **Insights**: AI-generated explanations of what's happening
    """)

col1, col2, col3, col4 = st.columns(4)
with col1:
    freq_status = "🟢 Normal" if 18 <= last_reading['Freq (Hz)'] <= 22 else "🔴 Warning"
    st.metric("Frequency (Hz)", 
              f"{last_reading['Freq (Hz)']:.1f}",
              delta=freq_status)

with col2:
    current_status = "🟢 Normal" if 1.5 <= last_reading['Current (Amps)'] <= 3.0 else "🔴 Warning"
    st.metric("Motor Current (Amps)", 
              f"{last_reading['Current (Amps)']:.1f}",
              delta=current_status)

with col3:
    if has_anomaly:
        st.metric("System Status", "🔴 Attention Needed", delta="Multiple anomalies detected")
    else:
        st.metric("System Status", "🟢 Normal Operation", delta="No significant issues")

with col4:
    remaining_life = last_reading['predicted_remaining_life']
    life_status = "🟢 Good" if remaining_life > 500 else "🔴 Soon" if remaining_life > 100 else "⚠️ Immediate"
    st.metric("Estimated Remaining Life", 
              f"{remaining_life:.0f} hours", 
              delta=life_status)

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Production", 
    "Pump Health", 
    "Alerts", 
    "Predictions", 
    "Insights"
])

with tab1:
    st.header("Well Production Performance")
    
    with st.expander("💡 Understanding Production Metrics"):
        st.markdown("""
        - **Gross Production**: Total oil output from your well (barrels per day)
        - **BSW**: Basic Sediment & Water - the percentage of unwanted fluids in your oil
        - **Gas Production**: How much natural gas your well is producing
        - **Production Hours**: How long your well has been operating
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.area(
            prod_data, x='Date', y='Gross Act (BBL)',
            title='Daily Oil Production (Barrels)',
            template='plotly_white',
            labels={'Gross Act (BBL)': 'Barrels of Oil'}
        )
        st.plotly_chart(fig, use_container_width=True, key="oil_prod_area")
        
        fig = px.line(
            prod_data, x='Date', y='BSW',
            title='Water & Sediment in Oil (%)',
            template='plotly_white',
            line_shape="spline"
        )
        st.plotly_chart(fig, use_container_width=True, key="bsw_line")
    
    with col2:
        fig = px.bar(
            prod_data, x='Date', y='Gas Produced (MMSCFD)',
            title='Daily Gas Production (Millions of cubic feet)',
            template='plotly_white',
            labels={'Gas Produced (MMSCFD)': 'Gas Volume'}
        )
        st.plotly_chart(fig, use_container_width=True, key="gas_bar")
        
        fig = px.line(
            prod_data, x='Date', y='Hrs of Production',
            title='Daily Operating Hours',
            template='plotly_white',
            line_shape="spline"
        )
        st.plotly_chart(fig, use_container_width=True, key="hours_line")

with tab2:
    st.header("Pump Health Monitoring")
    
    with st.expander("💡 Understanding Pump Metrics"):
        st.markdown("""
        - **Frequency**: How fast the pump is running (higher = faster pumping)
        - **Motor Current**: Electrical current drawn by the pump motor
        - **Intake Pressure**: Pressure at the pump intake (indicates flow conditions)
        - **Motor Temperature**: Critical for preventing equipment damage
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        # Gauge-style visualization for frequency
        fig = px.line(
            esp_data, x='DateTime', y='Freq (Hz)',
            title='Pump Speed (Hz)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True, key="freq_line")
        
        fig = px.line(
            esp_data, x='DateTime', y='Intake Press psi',
            title='Intake Pressure (psi)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True, key="pressure_line")
    
    with col2:
        fig = px.line(
            esp_data, x='DateTime', y='Current (Amps)',
            title='Motor Current Draw (Amps)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True, key="current_line")
        
        fig = px.line(
            esp_data, x='DateTime', 
            y=['Motor Temp (F)', 'Motor Temp Predicted (F)'],
            title='Motor Temperature: Actual vs Expected',
            template='plotly_white',
            labels={"value": "Temperature (°F)"}
        )
        st.plotly_chart(fig, use_container_width=True, key="temp_comparison")

with tab3:
    st.header("Equipment Alerts & Issues")
    
    with st.expander("💡 Understanding Alerts"):
        st.markdown("""
        - **Anomaly Score**: How many detection methods flagged an issue (0-3)
        - **Red Zones**: Values outside normal operating ranges
        - **Temperature Differences**: When actual temperature differs from predicted
        """)
    
    st.subheader("Problem Detection Summary")
    
    # Simple alert summary
    anomaly_counts = esp_data['anomaly_score'].value_counts().sort_index()
    total_readings = len(esp_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Readings Analyzed", f"{total_readings:,}")
    with col2:
        minor_issues = len(esp_data[esp_data['anomaly_score'] == 1])
        st.metric("Minor Issues Detected", f"{minor_issues} ({minor_issues/total_readings:.1%})")
    with col3:
        major_issues = len(esp_data[esp_data['anomaly_score'] >= 2])
        st.metric("Major Issues Detected", f"{major_issues} ({major_issues/total_readings:.1%})")
    
    # Visual alert timeline
    fig = px.scatter(
        esp_data[esp_data['anomaly_score'] > 0],
        x='DateTime', y='anomaly_score',
        color='anomaly_score',
        title='When Problems Were Detected',
        labels={'anomaly_score': 'Problem Severity'},
        color_continuous_scale=px.colors.sequential.Reds
    )
    st.plotly_chart(fig, use_container_width=True, key="alert_timeline")
    
    # Detailed alerts
    st.subheader("Recent Alerts")
    alert_df = esp_data[esp_data['anomaly_score'] > 0].sort_values('DateTime', ascending=False).head(10)
    
    if not alert_df.empty:
        for idx, row in alert_df.iterrows():
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    if row['anomaly_score'] == 1:
                        st.markdown(f"<p class='warning'>⚠️ Minor Alert</p>", unsafe_allow_html=True)
                    elif row['anomaly_score'] == 2:
                        st.markdown(f"<p class='danger'>🚨 Moderate Alert</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p class='danger'>🔥 Severe Alert</p>", unsafe_allow_html=True)
                    
                    st.write(f"{row['DateTime'].strftime('%Y-%m-%d %H:%M')}")
                
                with cols[1]:
                    st.write(f"""
                    - Frequency: {row['Freq (Hz)']:.1f} Hz
                    - Current: {row['Current (Amps)']:.1f} Amps
                    - Temperature: {row['Motor Temp (F)']:.1f}°F
                    """)
                st.divider()
    else:
        st.success("🎉 No alerts detected in the recent data!")

with tab4:
    st.header("Predictive Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pump Lifetime Estimation")
        fig = px.line(
            esp_data, x='DateTime', y='predicted_remaining_life',
            title='Estimated Remaining Pump Life (hours)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True, key="life_estimation")
        
        st.subheader("Short-Term Production Forecast (14 days)")
        fig = px.line(
            x=arima_forecast.index, 
            y=arima_forecast.values,
            title='ARIMA Forecast',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True, key="arima_forecast")
        
    with col2:
        st.subheader("Long-Term Production Forecast (90 days)")
        fig = px.line(
            prophet_forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'],
            title='Prophet Forecast',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True, key="prophet_forecast")
        
        st.subheader("Forecast Components")
        fig = prophet_model.plot_components(prophet_forecast)
        st.pyplot(fig)

with tab5:
    st.header("Operational Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Common Operating Modes")
        fig = px.scatter_matrix(
            esp_data.dropna(),
            dimensions=['Freq (Hz)', 'Current (Amps)', 'Intake Press psi', 'Motor Temp (F)'],
            color='operating_mode',
            title='Equipment State Clusters'
        )
        st.plotly_chart(fig, use_container_width=True, key="operating_modes")
        
        st.subheader("Anomaly Root Causes")
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Most Important Factors in Alerts'
        )
        st.plotly_chart(fig, use_container_width=True, key="root_causes")
        
    with col2:
        st.subheader("AI-Generated Insights")
        st.markdown("""
        <div class="insight-card">
            <h4>Temperature Alert Rules</h4>
            <pre>{report_text}</pre>
        </div>
        """.format(report_text=report_text), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card">
            <h4>Feature Importance for Temperature Prediction</h4>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(importance_fig, use_container_width=True, key="feature_importance")
        
        st.markdown("""
        <div class="insight-card">
            <h4>Typical Operating Modes</h4>
            <p>The pump operates in 5 distinct modes:</p>
            <ol>
                <li>Low frequency, low temp (startup)</li>
                <li>Normal operating range</li>
                <li>High frequency, high temp</li>
                <li>Low pressure condition</li>
                <li>High current spikes</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
**Key AI Features:**
- 🚀 **Predictive Maintenance**: Estimates remaining equipment life
- 📈 **Production Forecasting**: Predicts future oil/gas output
- 🔍 **Root Cause Analysis**: Explains why alerts are triggered
- 🤖 **Automated Insights**: Plain-language explanations of complex patterns
- ⚠️ **Smart Alerts**: Learns normal patterns to detect real issues

For maintenance requests or questions, contact your operations team.
""")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from sklearn.ensemble import IsolationForest, RandomForestRegressor
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.svm import OneClassSVM
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # Configure page
# st.set_page_config(
#     page_title="AI + IoT Monitoring Dashboard",
#     page_icon="🛢️",
#     layout="wide"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
# .gauge-container {
#     background-color: #f8f9fa;
#     border-radius: 10px;
#     padding: 15px;
#     margin-bottom: 20px;
# }
# .alert-badge {
#     background-color: #ffcccc;
#     padding: 5px 10px;
#     border-radius: 20px;
#     font-weight: bold;
# }
# </style>
# """, unsafe_allow_html=True)

# # ----------------------
# # 1. Load Production Data
# # ----------------------
# @st.cache_data
# def load_production_data():
#     prod_data = pd.read_excel('Production_data.xlsx')
#     prod_data['Date'] = pd.to_datetime(prod_data['Date'], format='%Y-%m-%d')
#     return prod_data.sort_values('Date')

# # ----------------------
# # 2. Load ESP Monitoring Data
# # ----------------------
# @st.cache_data
# def load_esp_data():
#     # Replace with your actual loading logic
#     main_df3 = pd.read_excel('NEW_ESP_DATA.xlsx', sheet_name=None)
#     monitor_dfs = list(main_df3.values())
#     monitor_df = pd.concat(monitor_dfs, ignore_index=True)
#     monitor_df = monitor_df.drop(columns=['Remark'], errors='ignore')
#     monitor_df['DateTime'] = pd.to_datetime(monitor_df['Date'], errors='coerce')
#     return monitor_df.sort_values('DateTime')

# # ----------------------
# # Data Processing
# # ----------------------
# def preprocess_esp_data(df):
#     # Replace invalid entries and convert to numeric
#     to_numeric_cols = ['Freq (Hz)', 'Current (Amps)', 'Intake Press psi', 'Motor Temp (F)']
#     df.replace('-', np.nan, inplace=True)
#     for col in to_numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     return df

# # ----------------------
# # Anomaly Detection
# # ----------------------
# def detect_anomalies(df):
#     features = df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi', 'Motor Temp (F)']].dropna()
    
#     # Train multiple anomaly detection models
#     iso = IsolationForest(contamination=0.05, random_state=42)
#     lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
#     svm = OneClassSVM(nu=0.05)
    
#     df['anomaly_iso'] = np.nan
#     df['anomaly_lof'] = np.nan
#     df['anomaly_svm'] = np.nan
    
#     df.loc[features.index, 'anomaly_iso'] = iso.fit_predict(features)
#     df.loc[features.index, 'anomaly_lof'] = lof.fit_predict(features)
#     df.loc[features.index, 'anomaly_svm'] = svm.fit_predict(features)
    
#     # Combined anomaly score (0-3)
#     df['anomaly_score'] = (
#         (df['anomaly_iso'] == -1).astype(int) + 
#         (df['anomaly_lof'] == -1).astype(int) + 
#         (df['anomaly_svm'] == -1).astype(int)
#     )
    
#     return df

# # ----------------------
# # Predictive Modeling
# # ----------------------
# def train_predictive_models(df):
#     features = df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi']].dropna()
#     target_temp = df['Motor Temp (F)'].dropna()
    
#     # Align indices
#     common_idx = features.index.intersection(target_temp.index)
#     features = features.loc[common_idx]
#     target_temp = target_temp.loc[common_idx]
    
#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         features, target_temp, test_size=0.2, random_state=42
#     )
    
#     # Train model
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
    
#     # Evaluate
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     st.session_state.model_mse = mse
    
#     # Make predictions on full dataset
#     df['Motor Temp Predicted (F)'] = model.predict(
#         df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi']].fillna(features.mean())
#     )
    
#     return df, model

# # ----------------------
# # Load and Process Data
# # ----------------------
# prod_data = load_production_data()
# esp_data = load_esp_data()
# esp_data = preprocess_esp_data(esp_data)
# esp_data = detect_anomalies(esp_data)
# esp_data, temp_model = train_predictive_models(esp_data)

# # Get last recorded values
# last_reading = esp_data.iloc[100]
# has_anomaly = last_reading['anomaly_score'] >= 2

# # ----------------------
# # Dashboard Layout
# # ----------------------
# st.title("🛢️ AI + IoT Monitoring Dashboard")

# # Status Overview
# st.header("Current System Status")

# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.metric("Frequency (Hz)", f"{20:.1f}")

# with col2:
#     st.metric("Motor Current (Amps)", f"{2.2:.1f}")

# with col3:
#     anomaly_status = "⚠️ ALERT" if has_anomaly else "✅ Normal"
#     st.metric("System Status", anomaly_status)

# with col4:
#     st.metric("Prediction MSE", f"{st.session_state.model_mse:.2f}")

# # Main Tabs
# tab1, tab2, tab3 = st.tabs(["Production Data", "ESP Monitoring", "Anomaly Analysis"])

# with tab1:
#     st.header("Production Performance")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         fig = px.line(
#             prod_data, x='Date', y='Gross Act (BBL)',
#             title='Gross Production (BBL)',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         fig = px.line(
#             prod_data, x='Date', y='BSW',
#             title='Basic Sediment & Water (%)',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         fig = px.line(
#             prod_data, x='Date', y='Gas Produced (MMSCFD)',
#             title='Gas Production (MMSCFD)',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         fig = px.line(
#             prod_data, x='Date', y='Hrs of Production',
#             title='Production Hours',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig, use_container_width=True)

# with tab2:
#     st.header("ESP Monitoring Data")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         fig = px.line(
#             esp_data, x='DateTime', y='Freq (Hz)',
#             title='Pump Frequency (Hz)',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         fig = px.line(
#             esp_data, x='DateTime', y='Intake Press psi',
#             title='Intake Pressure (psi)',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         fig = px.line(
#             esp_data, x='DateTime', y='Current (Amps)',
#             title='Motor Current (Amps)',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         fig = px.line(
#             esp_data, x='DateTime', 
#             y=['Motor Temp (F)', 'Motor Temp Predicted (F)'],
#             title='Motor Temperature: Actual vs Predicted',
#             template='plotly_white'
#         )
#         st.plotly_chart(fig, use_container_width=True)

# with tab3:
#     st.header("Anomaly Detection Analysis")
    
#     st.subheader("Anomaly Distribution")
#     anomaly_counts = esp_data['anomaly_score'].value_counts().sort_index()
#     fig = px.bar(
#         anomaly_counts, 
#         title="Count of Anomalies by Severity Level",
#         labels={'index': 'Anomaly Score', 'value': 'Count'},
#         color=anomaly_counts.index,
#         color_continuous_scale='reds'
#     )
#     st.plotly_chart(fig, use_container_width=True)
    
#     st.subheader("Anomaly Events")
#     anomaly_events = esp_data[esp_data['anomaly_score'] > 0]
    
#     if not anomaly_events.empty:
#         col1, col2 = st.columns(2)
#         # with col1:
#         #     fig = px.scatter(
#         #         anomaly_events, x='DateTime', y='Freq (Hz)',
#         #         color='anomaly_score',
#         #         size='anomaly_score',
#         #         title='Anomalies in Frequency',
#         #         color_continuous_scale='reds',
#         #         range_color=[1, 3]
#         #     )
#         #     st.plotly_chart(fig, use_container_width=True)
        
#         # with col2:
#         #     fig = px.scatter(
#         #         anomaly_events, x='DateTime', y='Motor Temp (F)',
#         #         color='anomaly_score',
#         #         size='anomaly_score',
#         #         title='Anomalies in Motor Temperature',
#         #         color_continuous_scale='reds',
#         #         range_color=[1, 3]
#         #     )
#         #     st.plotly_chart(fig, use_container_width=True)
        
#         st.subheader("Detailed Anomaly Records")
#         st.dataframe(
#             anomaly_events[
#                 ['DateTime', 'Freq (Hz)', 'Current (Amps)', 
#                  'Intake Press psi', 'Motor Temp (F)', 'anomaly_score']
#             ].sort_values('DateTime', ascending=False),
#             height=300
#         )
#     else:
#         st.success("No anomalies detected in the dataset")

# # Footer
# st.markdown("---")
# st.markdown("""
# **Dashboard Features:**
# - Production performance trends
# - ESP sensor monitoring
# - Multi-model anomaly detection (Isolation Forest, LOF, One-Class SVM)
# - Motor temperature prediction (Random Forest)
# """)
