import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
import time

st.set_page_config(page_title="GOLD AI Dashboard", layout="wide", page_icon="📈")

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(script_dir, "..", "features", "xauusd_features.csv")
model_path = os.path.join(script_dir, "..", "models", "xgb_ict_model.pkl")

# Helper functions
@st.cache_data(ttl=60)
def load_data():
    if os.path.exists(features_path):
        df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        return df.tail(500)
    return None

@st.cache_resource
def load_ml_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Sidebar
st.sidebar.title("🤖 GOLD AI System")
st.sidebar.markdown("---")
st.sidebar.markdown("**Strategy:** ICT + XGBoost")
st.sidebar.markdown("**Asset:** XAUUSD (Gold)")
st.sidebar.markdown("**Timeframe:** 1 Minute")

df = load_data()
model = load_ml_model()

if df is not None and model is not None:
    st.success("✅ System Online - ICT Model Loaded")
    
    # Get latest data
    latest_data = df.iloc[[-1]]
    latest_price = latest_data['Close'].values[0]
    
    # Features
    features = ['Bullish_FVG', 'Bearish_FVG', 'Bullish_Sweep', 'Bearish_Sweep',
                'EQH', 'EQL', 'BOS_Up', 'BOS_Down', 'Bullish_OB', 'Bearish_OB',
                'In_Discount', 'In_Premium', 'London_Killzone', 'NY_Killzone', 
                'Silver_Bullet', 'RSI', 'ATR', 'Trend_Up']
    
    features = [f for f in features if f in df.columns]
    X_latest = latest_data[features].fillna(0)
    
    # Predict
    prediction = model.predict(X_latest)[0]
    prob = model.predict_proba(X_latest)[0]
    confidence = max(prob) * 100
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Price", f"${latest_price:,.2f}")
    with col2:
        if prediction == 1:
            st.metric("Signal", "🟢 BUY", "Bullish ICT")
        elif prediction == 2:
            st.metric("Signal", "🔴 SELL", "Bearish ICT")
        else:
            st.metric("Signal", "⚪ WAIT", "No Setup")
    with col3:
        st.metric("Confidence", f"{confidence:.1f}%")
    with col4:
        trend = "📈 UP" if latest_data['Trend_Up'].values[0] else "📉 DOWN"
        st.metric("Market Trend", trend)

    # Secondary Metrics
    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("RSI (14)", f"{latest_data['RSI'].values[0]:.2f}")
    with col6:
        st.metric("Volatility (ATR)", f"{latest_data['ATR'].values[0]:.4f}")
    with col7:
        kz = "NY" if latest_data['NY_Killzone'].values[0] else ("London" if latest_data['London_Killzone'].values[0] else "None")
        st.metric("Active Killzone", kz)

    st.markdown("---")
    
    # Chart
    st.subheader("ICT Market Structure & Technicals")
    chart_df = df.tail(100)
    
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price & OBs', 'RSI'), 
                        row_width=[0.3, 0.7])

    # Candlestick
    fig.add_trace(go.Candlestick(x=chart_df.index,
                    open=chart_df['Open'], high=chart_df['High'],
                    low=chart_df['Low'], close=chart_df['Close'], name='Price'), row=1, col=1)
    
    # EMA 50
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_50'], name='EMA 50', line=dict(color='yellow', width=1)), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI'], name='RSI', line=dict(color='cyan', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Highlight FVGs and OBs
    for i, row in chart_df.iterrows():
        if row['Bullish_OB']:
            fig.add_annotation(x=i, y=row['Low'], text="OB", showarrow=True, arrowhead=1, color="green", row=1, col=1)
        if row['Bearish_OB']:
            fig.add_annotation(x=i, y=row['High'], text="OB", showarrow=True, arrowhead=1, color="red", row=1, col=1)
            
    fig.update_layout(template='plotly_dark', height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Active Signals Table
    st.subheader("Detected ICT Signals")
    signals = chart_df[chart_df[features].sum(axis=1) > 0][features].tail(5)
    st.table(signals)

else:
    st.warning("Data or Model missing. Running pipeline recommended.")
    
