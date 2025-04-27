
# 🛢️ AI + IoT Dashboard (Light Theme)

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------
# 1. Load Production Data
# ----------------------
prod_data = pd.read_excel(r'Production_data.xlsx')
prod_data['Date'] = pd.to_datetime(prod_data['Date'], format='%Y-%m-%d')
prod_data = prod_data.sort_values('Date')

# ----------------------
# 2. Load ESP Monitoring Data
# ----------------------
main_df3 = pd.read_excel(
    r'NEW_ESP_DATA.xlsx', sheet_name=None
)
monitor_dfs = list(main_df3.values())
monitor_df = pd.concat(monitor_dfs, ignore_index=True)
monitor_df = monitor_df.drop(columns=['Remark'], errors='ignore')
monitor_df['DateTime'] = pd.to_datetime(monitor_df['Date'], errors='coerce')
monitor_df = monitor_df.sort_values('DateTime')

# ----------------------
# 3. Clean ESP Monitoring Data
# ----------------------
# Replace invalid entries and convert to numeric
to_numeric_cols = ['Freq (Hz)', 'Current (Amps)', 'Intake Press psi', 'Motor Temp (F)']
monitor_df.replace('-', np.nan, inplace=True)
for col in to_numeric_cols:
    monitor_df[col] = pd.to_numeric(monitor_df[col], errors='coerce')

# Extract features for ML
features = monitor_df[to_numeric_cols].dropna()

# ----------------------
# 4. Machine Learning Models
# ----------------------
# 4.1 Anomaly Detection (Isolation Forest)
iso_model = IsolationForest(contamination=0.05, random_state=42)
features['anomaly'] = iso_model.fit_predict(features)
monitor_df['anomaly'] = np.nan
monitor_df.loc[features.index, 'anomaly'] = features['anomaly']

# 4.2 Predict Motor Temperature (Linear Regression)
X = features[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi']]
y = features['Motor Temp (F)']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr_model = LinearRegression().fit(X_train, y_train)
monitor_df['Motor Temp Predicted (F)'] = lr_model.predict(
    monitor_df[['Freq (Hz)', 'Current (Amps)', 'Intake Press psi']].fillna(0)
)

# Optional: print test error
mse = mean_squared_error(y_test, lr_model.predict(X_test))
print(f"Motor Temp Prediction MSE: {mse:.2f}")

# ----------------------
# 5. Dash App Layout
# ----------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "AI + IoT Dashboard"

app.layout = dbc.Container([
    html.H1(
        "🛢️ AI + IoT Production Monitoring", 
        className="text-center text-primary mb-4 mt-4"
    ),

    # Production Section
    dbc.Row([
        dbc.Col(
            html.H2(
                "⛽ Daily Production Summary", 
                style={"textAlign": "center", "color": "orange"}
            ), 
            width=12
        )
    ], className="mb-2"),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure=px.line(
                    prod_data, x='Date', y='Gross Act (BBL)',
                    title='Gross Production Over Time',
                    template='plotly_white'
                ).update_traces(
                    line_color='royalblue', line_width=3
                )
            ), width=6
        ),
        dbc.Col(
            dcc.Graph(
                figure=px.line(
                    prod_data, x='Date', y='Gas Produced (MMSCFD)',
                    title='Gas Production Over Time',
                    template='plotly_white'
                ).update_traces(
                    line_color='seagreen', line_width=3
                )
            ), width=6
        ),
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure=px.line(
                    prod_data, x='Date', y='BSW',
                    title='Basic Sediment & Water (BSW)',
                    template='plotly_white'
                ).update_traces(
                    line_color='crimson', line_width=3
                )
            ), width=6
        ),
        dbc.Col(
            dcc.Graph(
                figure=px.line(
                    prod_data, x='Date', y='Hrs of Production',
                    title='Production Hours Per Day',
                    template='plotly_white'
                ).update_traces(
                    line_color='darkorange', line_width=3
                )
            ), width=6
        ),
    ], className="mb-4"),

    # ESP Monitoring Section
    dbc.Row([
        dbc.Col(
            html.H2(
                "⚙️ ESP Monitoring Data", 
                style={"textAlign": "center", "color": "teal"}
            ), width=12
        )
    ], className="mt-4 mb-2"),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure=px.scatter(
                    monitor_df, x='DateTime', y='Freq (Hz)',
                    title='Pump Frequency (Hz)',
                    template='plotly_white',
                    color=monitor_df['anomaly'].map({-1:'red',1:'blue',np.nan:'blue'})
                ).update_traces(marker=dict(size=6))
            ), width=6
        ),
        dbc.Col(
            dcc.Graph(
                figure=px.scatter(
                    monitor_df, x='DateTime', y='Current (Amps)',
                    title='Motor Current (Amps)',
                    template='plotly_white',
                    color=monitor_df['anomaly'].map({-1:'red',1:'blue',np.nan:'blue'})
                ).update_traces(marker=dict(size=6))
            ), width=6
        ),
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure=px.scatter(
                    monitor_df, x='DateTime', y='Intake Press psi',
                    title='Intake Pressure (psi)',
                    template='plotly_white',
                    color=monitor_df['anomaly'].map({-1:'red',1:'blue',np.nan:'blue'})
                ).update_traces(marker=dict(size=6))
            ), width=6
        ),
        dbc.Col(
            dcc.Graph(
                figure=px.line(
                    monitor_df, x='DateTime',
                    y=['Motor Temp (F)', 'Motor Temp Predicted (F)'],
                    title='Motor Temperature: Actual vs Predicted',
                    template='plotly_white'
                ).update_traces(line=dict(width=3))
            ), width=6
        ),
    ], className="mb-4"),

    html.Div(
        "Dashboard built with Dash + Plotly for showcasing AI + IoT monitoring in oil production.",
        style={"textAlign": "center", "color": "gray", "fontStyle": "italic"}
    ),
], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)
