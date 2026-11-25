import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
from pathlib import Path
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak
)
from reportlab.lib.units import inch

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="DDoS Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}
[data-testid="stSidebar"] * {
    color: #e5e7eb;
}
.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 5px;
}
.subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 25px;
}
.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 20px;
    margin-bottom: 15px;
}
.metric-card {
    background: linear-gradient(145deg, #172338, #111827);
    border: 1px solid #29384e;
    border-radius: 13px;
    padding: 20px;
    min-height: 115px;
}
.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
}
.metric-value {
    color: #f8fafc;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}
.metric-note {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}
.info-card {
    background: #172338;
    border: 1px solid #29384e;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
}
div.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}
.stDownloadButton > button {
    background: #172338;
    border: 1px solid #3b82f6;
    color: #e5e7eb;
    border-radius: 8px;
    width: 100%;
}
hr {
    border-color: #29384e;
}
[data-testid="stDataFrame"] {
    border: 1px solid #29384e;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL FILES
# =====================================================
BASE_DIR = Path("/content")
MODEL_PATH = BASE_DIR / "ddos_random_forest_model.pkl"
FEATURE_PATH = BASE_DIR / "ddos_features.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_PATH}")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features

try:
    model, selected_features = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("# 🛡️ CyberGuard")
    st.caption("NETWORK INTRUSION DETECTION SYSTEM")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Upload & Detection",
            "Prediction Results",
            "Analytics",
            "Export Reports",
            "About Project"
        ]
    )
    st.divider()
    st.markdown("### Model Information")
    st.write("**Algorithm:** Random Forest")
    st.write("**Classification:** Binary")
    st.write("**Classes:** Benign / DDoS")
    st.write(f"**Features:** {len(selected_features)}")
    st.divider()
    st.caption("DDoS Detection System | CICIoT2023")

# =====================================================
# SESSION STATE
# =====================================================
if "results" not in st.session_state:
    st.session_state.results = None
if "original_data" not in st.session_state:
    st.session_state.original_data = None

# =====================================================
# REUSABLE METRIC CARD
# =====================================================
def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# PDF REPORT GENERATOR
# =====================================================
def create_pdf_report(results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    styles = getSampleStyleSheet()
    elements = []
    total = len(results)
    ddos_count = (results["Predicted Label"] == "DDoS").sum()
    benign_count = (results["Predicted Label"] == "Benign").sum()
    attack_percentage = ddos_count / total * 100 if total else 0
    avg_confidence = results["Confidence (%)"].mean()

    elements.append(Paragraph("<b>DDOS ATTACK DETECTION REPORT</b>", styles["Title"]))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Random Forest Based Network Traffic Classification", styles["Normal"]))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>1. Detection Summary</b>", styles["Heading2"]))

    summary_data = [
        ["Metric", "Result"],
        ["Total Records", f"{total:,}"],
        ["Benign Records", f"{benign_count:,}"],
        ["DDoS Records", f"{ddos_count:,}"],
        ["DDoS Percentage", f"{attack_percentage:.2f}%"],
        ["Average Prediction Confidence", f"{avg_confidence:.2f}%"]
    ]
    table = Table(summary_data, colWidths=[250, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 25))

    elements.append(Paragraph("<b>2. Selected Model Features</b>", styles["Heading2"]))
    for feature in selected_features:
        elements.append(Paragraph(f"• {feature}", styles["Normal"]))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>3. Prediction Details</b>", styles["Heading2"]))
    details = [["Record", "Prediction", "Confidence"]]
    for index, row in results.head(100).iterrows():
        details.append([
            str(index + 1),
            str(row["Predicted Label"]),
            f'{row["Confidence (%)"]:.2f}%'
        ])
    detail_table = Table(details, colWidths=[90, 160, 160], repeatRows=1)
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    elements.append(detail_table)
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Note: The complete prediction results are available in the CSV export.", styles["Italic"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# PAGE 1: DASHBOARD
# =====================================================
if page == "Dashboard":
    st.markdown(
        '<div class="main-title">🛡️ DDoS Detection Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Real-time network traffic classification using a trained Random Forest machine learning model.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.results is None:
        st.info("Upload a network traffic CSV file from the 'Upload & Detection' section to begin detection.")
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("MODEL", "Random Forest", "Machine Learning")
        with col2:
            metric_card("FEATURES", str(len(selected_features)), "Selected features")
        with col3:
            metric_card("CLASSES", "2", "Benign / DDoS")
    else:
        results = st.session_state.results
        total = len(results)
        ddos = (results["Predicted Label"] == "DDoS").sum()
        benign = (results["Predicted Label"] == "Benign").sum()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("TOTAL RECORDS", f"{total:,}", "Analyzed traffic")
        with col2:
            metric_card("DDoS DETECTED", f"{ddos:,}", "Potential attacks")
        with col3:
            metric_card("BENIGN TRAFFIC", f"{benign:,}", "Normal traffic")
        with col4:
            metric_card("ATTACK RATE", f"{ddos / total * 100:.2f}%" if total else "0%", "Percentage of predicted DDoS")

        st.markdown('<div class="section-title">Detection Overview</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(names=["Benign", "DDoS"], values=[benign, ddos], hole=0.55, title="Traffic Classification", color_discrete_sequence=["#22c55e", "#ef4444"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Bar(x=["Benign", "DDoS"], y=[benign, ddos], marker_color=["#22c55e", "#ef4444"]))
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE 2: UPLOAD AND DETECTION
# =====================================================
elif page == "Upload & Detection":
    st.markdown('<div class="main-title">📂 Upload Network Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose your network traffic CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset uploaded successfully!")
            missing_features = [f for f in selected_features if f not in df.columns]
            if missing_features:
                st.error("Missing required model features: " + ", ".join(missing_features))
            else:
                if st.button("🚀 Start DDoS Detection", use_container_width=True):
                    with st.spinner("Analyzing network traffic..."):
                        X = df[selected_features].copy()
                        for col in selected_features:
                            X[col] = pd.to_numeric(X[col], errors="coerce")
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)
                        classes = list(model.classes_)
                        ddos_index = classes.index("DDoS")
                        benign_index = classes.index("Benign")
                        output = df.copy()
                        output["Predicted Label"] = predictions
                        output["Confidence (%)"] = probabilities.max(axis=1) * 100
                        st.session_state.results = output
                        st.success("Detection completed successfully!")
                        st.rerun()
        except Exception as e:
            st.error(f"Unable to read CSV file: {str(e)}")

# =====================================================
# PAGE 3: PREDICTION RESULTS
# =====================================================
elif page == "Prediction Results":
    st.markdown('<div class="main-title">🔍 Prediction Results</div>', unsafe_allow_html=True)
    results = st.session_state.results
    if results is None:
        st.warning("Please upload a dataset and run detection first.")
    else:
        st.dataframe(results.head(100), use_container_width=True)

# =====================================================
# PAGE 4: ANALYTICS
# =====================================================
elif page == "Analytics":
    st.markdown('<div class="main-title">📊 Detection Analytics</div>', unsafe_allow_html=True)
    results = st.session_state.results
    if results is None:
        st.warning("Run detection to view analytics.")
    else:
        st.metric("Average Confidence", f'{results["Confidence (%)"].mean():.2f}%')

# =====================================================
# PAGE 5: EXPORT REPORTS
# =====================================================
elif page == "Export Reports":
    st.markdown('<div class="main-title">📥 Export Detection Results</div>', unsafe_allow_html=True)
    results = st.session_state.results
    if results is None:
        st.warning("No prediction results available.")
    else:
        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(label="⬇️ Download Complete CSV", data=csv_data, file_name="ddos_detection_results.csv", mime="text/csv")
        pdf_data = create_pdf_report(results)
        st.download_button(label="⬇️ Download PDF Report", data=pdf_data, file_name="ddos_detection_report.pdf", mime="application/pdf")

# =====================================================
# PAGE 6: ABOUT PROJECT
# =====================================================
elif page == "About Project":
    st.markdown('<div class="main-title">ℹ️ About the Project</div>', unsafe_allow_html=True)
    st.write("This project implements a machine learning based network traffic classification system to identify DDoS attacks.")

st.markdown("---")
st.markdown('<div style="text-align:center; color:#64748b; padding:10px;">🛡️ DDoS Detection System</div>', unsafe_allow_html=True)
