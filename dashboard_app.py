# dashboard_app.py (Version 3.0.0 - Decoupled Architecture)
import streamlit as st
import json
import os
from PIL import Image
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import datetime

# --- Application Configuration ---
APP_VERSION = "3.0.0"
IMAGE_DIR = 'test_images'
DATA_FILE = 'dashboard_data.json'

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Poaching Detection Dashboard", layout="wide")

# --- Data Loading Function ---
@st.cache_data(ttl=3) # Cache for only 3 seconds to get frequent updates
def load_data():
    """Load, parse, and prepare event data from the JSON file."""
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_json(DATA_FILE, convert_dates=['timestamp'])
        if df.empty:
            return pd.DataFrame()
        df['date'] = df['timestamp'].dt.date
        return df.sort_values('timestamp', ascending=False)
    except Exception:
        return pd.DataFrame()

# --- Initialize and Run ---
st_autorefresh(interval=2000, limit=None, key="dashboard_refresher") # Refresh every 2 seconds
df_events = load_data()

# --- Sidebar ---
with st.sidebar:
    st.header("Poaching Detection System")
    st.write(f"Version: {APP_VERSION}")
    page = st.radio("Navigation Menu", ["Dashboard", "Alerts Log", "Monitoring", "Settings"])
    st.divider()
    if not df_events.empty:
        last_update = df_events['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
        st.info(f"Last Event: {last_update}")
        st.metric("Total Events Processed", len(df_events))
    else:
        st.info("Awaiting data...")

# (The rest of the rendering functions are identical to the previous version and do not need changes)
def render_dashboard():
    st.title("Live Operations Dashboard")
    if df_events.empty:
        st.warning("Waiting for data. Please ensure all backend services are running and the producer is sending images.")
        return
    alert_threshold = st.session_state.get('alert_threshold', 0.75)
    high_risk_alerts = df_events[df_events['poaching_score'] >= alert_threshold]
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Images Processed", len(df_events))
    kpi2.metric("Total High-Risk Alerts", len(high_risk_alerts))
    kpi3.metric("Average Risk Score", f"{df_events['poaching_score'].mean():.2%}")
    st.divider()
    st.header("Most Recent Analysis")
    latest_event = df_events.iloc[0].to_dict()
    col1, col2 = st.columns([2, 3])
    with col1:
        filename = latest_event.get('filename', '')
        image_path = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(image_path): st.image(Image.open(image_path), caption=f"Analyzed Image: {filename}", use_column_width=True)
        else: st.error(f"Image file not found: {filename}")
    with col2:
        score = latest_event.get('poaching_score', 0.0)
        detections = latest_event.get('detections', [])
        if score < 0.5: color = "#28a745"
        elif score < alert_threshold: color = "#ffc107"
        else: color = "#dc3545"
        st.markdown(f"**Poaching Confidence Score**")
        st.markdown(f"<h1 style='color:{color};'>{score:.1%}</h1>", unsafe_allow_html=True)
        st.progress(score)
        st.subheader("Detections")
        det_col1, det_col2, det_col3 = st.columns(3)
        det_col1.metric("Vehicles", sum(1 for d in detections if d['class'] in ['car']))
        det_col2.metric("Water Sources", sum(1 for d in detections if d['class'] in ['water', 'water_body']))
        det_col3.metric("Campfires", sum(1 for d in detections if d['class'] in ['wildfire']))

def render_alerts():
    st.title("High-Risk Alerts Log")
    alert_threshold = st.session_state.get('alert_threshold', 0.75)
    st.markdown(f"Displaying events with a poaching score of **{alert_threshold:.0%}** or higher.")
    if df_events.empty or 'poaching_score' not in df_events.columns:
        st.success("No high-risk alerts have been detected.")
        return
    high_risk_alerts = df_events[df_events['poaching_score'] >= alert_threshold].copy()
    if high_risk_alerts.empty:
        st.success("No high-risk alerts have been detected.")
        return
    st.dataframe(high_risk_alerts[['timestamp', 'filename', 'poaching_score']], use_container_width=True, column_config={"timestamp": "Timestamp", "filename": "Image File", "poaching_score": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0, max_value=1)}, hide_index=True)

def render_monitoring():
    st.title("Historical Monitoring and Trends")
    if df_events.empty or len(df_events) < 2:
        st.info("Insufficient data to display trends.")
        return
    date_range = st.date_input("Select Date Range", value=(df_events['date'].min(), df_events['date'].max()), min_value=df_events['date'].min(), max_value=df_events['date'].max())
    if len(date_range) == 2:
        df_filtered = df_events[(df_events['date'] >= date_range[0]) & (df_events['date'] <= date_range[1])]
    else: df_filtered = df_events
    st.subheader("Poaching Score Over Time")
    fig_score = px.line(df_filtered.sort_values('timestamp'), x='timestamp', y='poaching_score', title='Poaching Score Trend', markers=True, template="plotly_white")
    st.plotly_chart(fig_score, use_container_width=True)

def render_settings():
    st.title("System Settings")
    with st.form("settings_form"):
        st.subheader("Alert Configuration")
        alert_threshold = st.slider("High-Risk Alert Threshold", 0.0, 1.0, st.session_state.get('alert_threshold', 0.75), 0.05)
        submitted = st.form_submit_button("Save Settings")
        if submitted:
            st.session_state.alert_threshold = alert_threshold
            st.success("Settings saved successfully.")
    if st.button("Clear All Historical Data", type="secondary"):
        if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
        st.rerun()

# --- Main App Router ---
if page == "Dashboard": render_dashboard()
elif page == "Alerts Log": render_alerts()
elif page == "Monitoring": render_monitoring()
elif page == "Settings": render_settings()
