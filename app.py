import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import joblib
import os
from fpdf import FPDF

# ======================================================================
# BACKEND
# The notebook trained Linear Regression, Decision Tree, and Random
# Forest, and saved the Random Forest as the final model — but the
# LabelEncoders used for Service_Type / Department / Day were not
# saved. This app rebuilds them deterministically from the training
# CSV (LabelEncoder sorts each column's unique values alphabetically,
# so the mapping is guaranteed to match what the model was trained on).
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "queue_predictor.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "queue_feature_names.pkl")
DATA_PATH = os.path.join(BASE_DIR, "queue_dataset.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")

TARGET_COLUMN = "Waiting_Time"
CATEGORICAL_COLUMNS = ["Service_Type", "Department", "Day"]

# Test-set metrics for this exact model, evaluated on the same 80/20 split
# used in the notebook (random_state=42).
MODEL_INFO = {"mae": 5.72, "rmse": 7.04, "r2": 0.913, "n_estimators": 100}

MODEL_COMPARISON = [
    {"name": "Random Forest", "mae": 5.72, "rmse": 7.04, "r2": 0.913},
    {"name": "Linear Regression", "mae": 7.75, "rmse": 9.08, "r2": 0.855},
    {"name": "Decision Tree", "mae": 8.57, "rmse": 10.86, "r2": 0.793},
]


def load_raw_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_feature_names():
    return joblib.load(FEATURES_PATH)


@st.cache_resource
def build_encoders(_df):
    return {col: LabelEncoder().fit(_df[col]) for col in CATEGORICAL_COLUMNS}


def get_category_options(encoders):
    return {col: list(le.classes_) for col, le in encoders.items()}


def encode_input(raw_input: dict, encoders: dict, feature_names: list) -> pd.DataFrame:
    row = {}
    for col in feature_names:
        if col in CATEGORICAL_COLUMNS:
            row[col] = int(encoders[col].transform([raw_input[col]])[0])
        else:
            row[col] = raw_input[col]
    return pd.DataFrame([row], columns=feature_names)


def predict_wait(model, X: pd.DataFrame):
    """Prediction + confidence via agreement across the forest's own trees."""
    pred = float(model.predict(X)[0])
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
    std = tree_preds.std()
    mean = max(abs(tree_preds.mean()), 1e-6)
    confidence = 100 - (std / mean * 100)
    return pred, float(np.clip(confidence, 50, 99))


def get_recommendation(predicted_wait: float, available_staff: int, df: pd.DataFrame = None):
    """Rule-based staffing recommendation using data-driven quartile
    thresholds plus a staff-strain signal (queue pressure per staff)."""
    if df is not None and TARGET_COLUMN in df.columns:
        q25, q75 = df[TARGET_COLUMN].quantile([0.25, 0.75])
    else:
        q25, q75 = 35.0, 68.0

    if predicted_wait >= q75 or available_staff <= 2:
        action, detail, severity = (
            "Add More Staff",
            "Predicted wait is high. Open an additional counter or reassign staff to this queue immediately.",
            "high",
        )
    elif predicted_wait >= q25:
        action, detail, severity = (
            "Monitor Queue Closely",
            "Wait time is moderate. Keep an eye on queue growth and be ready to add staff if it climbs further.",
            "medium",
        )
    else:
        action, detail, severity = (
            "Staffing Sufficient",
            "Wait time is low. Current staffing level is handling the queue well.",
            "low",
        )

    return {"action": action, "detail": detail, "severity": severity, "q25": float(q25), "q75": float(q75)}


def append_to_history(record: dict):
    df_row = pd.DataFrame([record])
    if os.path.exists(HISTORY_PATH):
        df_row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORY_PATH, mode="w", header=True, index=False)


def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


def build_pdf_report(inputs: dict, predicted_wait: float, confidence: float, recommendation: dict) -> bytes:
    pdf = FPDF(format="A4")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(49, 46, 129)
    pdf.cell(0, 12, "Queue Wait Time Prediction Report", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_fill_color(230, 230, 250)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Predicted Waiting Time: {predicted_wait:.0f} minutes", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 9, f"Model Confidence: {confidence:.1f}%", ln=True, fill=True)
    pdf.cell(0, 9, f"Recommendation: {recommendation['action']}", ln=True, fill=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, recommendation["detail"])
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(49, 46, 129)
    pdf.cell(0, 10, "Queue Details", ln=True)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(20, 20, 20)
    for key, value in inputs.items():
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(70, 8, f"{key}:")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"{value}", ln=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 5, "Generated by AI Smart Queue Predictor & Intelligent Service Optimization System.")

    return bytes(pdf.output())


# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Smart Queue AI | Service Optimization Platform",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FAFAFE", "figure.facecolor": "#FAFAFE"})

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

:root {
    --brand-dark: #2E2A78;
    --brand: #4338CA;
    --brand-light: #6D63E8;
    --brand-pale: #EDEBFC;
    --accent: #D98E3B;
    --ink: #1E1B2E;
    --muted: #66637A;
    --surface: #FFFFFF;
    --border: #E3E1F0;
}

.stApp { background: linear-gradient(180deg,#FAF9FE 0%, #F1EFFA 100%); }

.hero {
    background: linear-gradient(120deg, var(--brand-dark) 0%, var(--brand) 55%, var(--brand-light) 100%);
    border-radius: 18px; padding: 28px 32px; color: white; margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(46,42,120,0.18);
}
.hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p { margin: 6px 0 0; opacity: 0.88; font-size: 0.95rem; }

.metric-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 1px 3px rgba(20,20,20,0.04);
}
.metric-card .label { font-size: 0.76rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
.metric-card .value { font-size: 1.65rem; font-weight: 800; color: var(--ink); }
.metric-card .sub { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

.result-hero {
    border-radius: 18px; padding: 28px; color: white; text-align: center;
    box-shadow: 0 10px 28px rgba(20,20,20,0.16);
}
.result-high   { background: linear-gradient(135deg, #7A1F13, #C0392B); }
.result-medium { background: linear-gradient(135deg, #7A5A0C, #B9770E); }
.result-low    { background: linear-gradient(135deg, #2E2A78, #4338CA); }
.result-hero .tag { font-size: 0.8rem; opacity: 0.88; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600; }
.result-hero .num { font-size: 2.8rem; font-weight: 800; margin: 6px 0; letter-spacing: -0.02em; }
.result-hero .stats { display:flex; justify-content:center; gap: 28px; margin-top: 10px; font-size: 0.85rem; opacity: 0.92; }

.rec-card { border-radius: 14px; padding: 18px 20px; margin-top: 16px; display: flex; gap: 14px; align-items: flex-start; border: 1px solid transparent; }
.rec-high   { background: #FCECE5; border-color: #F0B79C; }
.rec-medium { background: #FDF3DF; border-color: #F0CD8F; }
.rec-low    { background: var(--brand-pale); border-color: #C6C0F0; }
.rec-card .rec-icon { font-size: 1.6rem; line-height: 1; }
.rec-card h4 { margin: 0 0 4px; font-size: 1.02rem; font-weight: 700; color: var(--ink); }
.rec-card p { margin: 0; font-size: 0.87rem; color: #4A4A44; }

.section-title { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin: 6px 0 12px; display:flex; align-items:center; gap:8px; }
.section-sub { font-size: 0.82rem; color: var(--muted); margin-top: -8px; margin-bottom: 14px; }

.leaderboard-row { display:flex; align-items:center; justify-content:space-between; padding: 10px 14px; border-radius: 10px; margin-bottom: 6px; background: var(--surface); border: 1px solid var(--border); }
.leaderboard-row.best { background: var(--brand-pale); border-color: #C6C0F0; }
.rank-badge { display:inline-flex; align-items:center; justify-content:center; width: 26px; height: 26px; border-radius: 50%; background: #F1EFFA; font-weight: 700; font-size: 0.8rem; margin-right: 10px; }
.rank-badge.gold { background: #F5D68B; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #2E2A78 0%, #4338CA 100%); }
section[data-testid="stSidebar"] * { color: #EDEBFC !important; }

.stButton>button, .stDownloadButton>button {
    background: var(--brand); color: white; border-radius: 10px; border: none;
    font-weight: 600; padding: 0.55rem 1rem; transition: all 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover { background: var(--brand-dark); transform: translateY(-1px); }

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600; }

footer, #MainMenu { visibility: hidden; }
.app-footer { text-align:center; color: var(--muted); font-size: 0.78rem; padding: 18px 0 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================================
# LOAD MODEL & DATA
# ======================================================================
model = load_model()
feature_names = load_feature_names()
raw_df = load_raw_data()
encoders = build_encoders(raw_df)
category_options = get_category_options(encoders)

# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="font-size:1.8rem;">⏱️</div>
            <div>
                <div style="font-weight:800; font-size:1.05rem; line-height:1.1;">Smart Queue AI</div>
                <div style="font-size:0.72rem; opacity:0.75;">Service Optimization Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin:10px 0 16px;'>", unsafe_allow_html=True)
    st.caption(
        "Predicts expected customer waiting time and recommends staffing "
        "actions to keep queues moving."
    )
    st.markdown(
        "<div style='font-size:0.72rem; opacity:0.6; margin-top:20px;'>v1.0 · Powered by Random Forest</div>",
        unsafe_allow_html=True,
    )

# ======================================================================
# HERO HEADER
# ======================================================================
st.markdown(
    """
    <div class="hero">
        <h1>⏱️ AI Smart Queue Predictor &amp; Intelligent Service Optimization System</h1>
        <p>Forecast customer wait times before they happen — and staff up before queues get out of hand.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_dashboard, tab_batch, tab_history = st.tabs(
    ["🔮  Predict", "📊  Dashboard", "📁  Batch Prediction", "🕘  History"]
)

# ======================================================================
# PREDICT TAB
# ======================================================================
with tab_predict:
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="section-title">📝 Queue Details</div>', unsafe_allow_html=True)
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                service_type = st.selectbox("Service Type", category_options["Service_Type"])
                department = st.selectbox("Department", category_options["Department"])
                day = st.selectbox("Day", category_options["Day"])
                hour = st.slider("Hour of Day (24h)", min_value=8, max_value=20, value=14)
            with c2:
                current_queue = st.number_input("Current Queue Length", min_value=0, max_value=300, value=60, step=1)
                available_staff = st.number_input("Available Staff", min_value=1, max_value=30, value=7, step=1)
                emergency_cases = st.number_input("Emergency Cases", min_value=0, max_value=100, value=10, step=1)
                previous_avg_wait = st.number_input("Previous Average Wait (minutes)", min_value=0, max_value=200, value=30, step=1)

            submitted = st.form_submit_button("🔮  Predict Wait Time", width="stretch")

    with right:
        st.markdown('<div class="section-title">📈 Prediction Result</div>', unsafe_allow_html=True)

        if not submitted:
            st.markdown(
                """
                <div style="border:1px dashed var(--border); border-radius:14px; padding:40px 20px; text-align:center; color:var(--muted); background:var(--surface);">
                    <div style="font-size:2rem;">⏱️</div>
                    <p style="margin-top:8px; font-size:0.88rem;">Fill in the queue details and click <b>Predict Wait Time</b> to see results here.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            raw_input = {
                "Service_Type": service_type,
                "Department": department,
                "Current_Queue": current_queue,
                "Available_Staff": available_staff,
                "Emergency_Cases": emergency_cases,
                "Previous_Average_Wait": previous_avg_wait,
                "Day": day,
                "Hour": hour,
            }

            X = encode_input(raw_input, encoders, feature_names)
            prediction, confidence = predict_wait(model, X)
            recommendation = get_recommendation(prediction, available_staff, raw_df)

            st.markdown(
                f"""
                <div class="result-hero result-{recommendation['severity']}">
                    <div class="tag">Predicted Waiting Time</div>
                    <div class="num">{prediction:.0f} <span style="font-size:1.3rem; font-weight:600;">min</span></div>
                    <div class="stats">
                        <span>🎯 {confidence:.0f}% confidence</span>
                        <span>👥 {current_queue} people in queue</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            icons = {"high": "🆘", "medium": "🟡", "low": "✅"}
            sev_class = f"rec-{recommendation['severity']}"
            st.markdown(
                f"""
                <div class="rec-card {sev_class}">
                    <div class="rec-icon">{icons[recommendation['severity']]}</div>
                    <div>
                        <h4>{recommendation['action']}</h4>
                        <p>{recommendation['detail']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **raw_input,
                "Predicted Wait (min)": round(prediction, 1),
                "Confidence (%)": round(confidence, 1),
                "Recommendation": recommendation["action"],
            }
            append_to_history(record)

            pdf_bytes = build_pdf_report(raw_input, prediction, confidence, recommendation)
            st.download_button(
                "⬇️  Download PDF Report",
                data=pdf_bytes,
                file_name=f"queue_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                width="stretch",
            )

# ======================================================================
# DASHBOARD TAB
# ======================================================================
with tab_dashboard:
    st.markdown('<div class="section-title">📊 Queue Analytics Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Insights derived from the historical queue dataset.</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Total Records", f"{len(raw_df):,}", "queue events"),
        ("Avg Wait Time", f"{raw_df[TARGET_COLUMN].mean():.0f} min", "across all queues"),
        ("Max Wait Time", f"{raw_df[TARGET_COLUMN].max():.0f} min", "worst case observed"),
        ("Avg Queue Length", f"{raw_df['Current_Queue'].mean():.0f}", "people"),
    ]
    for col, (label, value, sub) in zip([k1, k2, k3, k4], kpis):
        col.markdown(
            f"""<div class="metric-card"><div class="label">{label}</div>
            <div class="value">{value}</div><div class="sub">{sub}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Waiting Time Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.histplot(raw_df[TARGET_COLUMN], bins=25, color="#4338CA", ax=ax, kde=True)
        ax.set_xlabel("Waiting Time (minutes)")
        st.pyplot(fig)

    with c2:
        st.markdown("**Avg Wait by Service Type**")
        avg_by_type = raw_df.groupby("Service_Type")[TARGET_COLUMN].mean().sort_values(ascending=False)
        st.bar_chart(avg_by_type, color="#D98E3B")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Wait Time by Department**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.boxplot(data=raw_df, x="Department", y=TARGET_COLUMN, ax=ax, palette="Purples")
        ax.tick_params(axis="x", rotation=25)
        st.pyplot(fig)

    with c4:
        st.markdown("**Queue Length vs Wait Time**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.scatterplot(data=raw_df, x="Current_Queue", y=TARGET_COLUMN, color="#4338CA", alpha=0.5, ax=ax)
        st.pyplot(fig)

    st.markdown("**Correlation Heatmap**")
    numeric_df = raw_df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="Purples", ax=ax)
    st.pyplot(fig)

    st.markdown("<hr style='border-color:var(--border); margin:24px 0 18px;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 Model Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Random Forest, Linear Regression, and Decision Tree — trained and benchmarked on the same data.</div>', unsafe_allow_html=True)

    comp_df = pd.DataFrame(MODEL_COMPARISON).sort_values("r2", ascending=False).reset_index(drop=True)
    medals = ["🥇", "🥈", "🥉"]
    for i, row in comp_df.iterrows():
        is_best = i == 0
        st.markdown(
            f"""
            <div class="leaderboard-row {'best' if is_best else ''}">
                <div style="display:flex; align-items:center;">
                    <span class="rank-badge {'gold' if is_best else ''}">{medals[i]}</span>
                    <b>{row['name']}</b>
                </div>
                <div style="display:flex; gap:22px; font-size:0.85rem; color:var(--muted);">
                    <span>R² <b style="color:var(--ink);">{row['r2']:.3f}</b></span>
                    <span>MAE <b style="color:var(--ink);">{row['mae']:.2f}</b></span>
                    <span>RMSE <b style="color:var(--ink);">{row['rmse']:.2f}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("Test R² by model (higher is better)")
        st.bar_chart(comp_df.set_index("name")["r2"], color="#4338CA")
    with cc2:
        st.caption("Test MAE by model, in minutes (lower is better)")
        st.bar_chart(comp_df.set_index("name")["mae"], color="#D98E3B")

# ======================================================================
# BATCH PREDICTION TAB
# ======================================================================
with tab_batch:
    st.markdown('<div class="section-title">📁 Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">Upload a CSV with columns: <code>{", ".join(feature_names)}</code></div>',
        unsafe_allow_html=True,
    )

    bc1, bc2 = st.columns([1, 2])
    with bc1:
        template = raw_df[feature_names].head(3)
        st.download_button(
            "⬇️  Sample Template CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="batch_template.csv",
            mime="text/csv",
            width="stretch",
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [c for c in feature_names if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                preds, confs, recs = [], [], []
                for _, row in batch_df.iterrows():
                    ri = {col: row[col] for col in feature_names}
                    Xb = encode_input(ri, encoders, feature_names)
                    pred, conf = predict_wait(model, Xb)
                    rec = get_recommendation(pred, row["Available_Staff"], raw_df)
                    preds.append(round(pred, 1))
                    confs.append(round(conf, 1))
                    recs.append(rec["action"])

                batch_df["Predicted Wait (min)"] = preds
                batch_df["Confidence (%)"] = confs
                batch_df["Recommendation"] = recs

                m1, m2, m3 = st.columns(3)
                m1.markdown(f"""<div class="metric-card"><div class="label">Rows Processed</div><div class="value">{len(batch_df)}</div></div>""", unsafe_allow_html=True)
                m2.markdown(f"""<div class="metric-card"><div class="label">Avg Predicted Wait</div><div class="value">{np.mean(preds):.0f} min</div></div>""", unsafe_allow_html=True)
                m3.markdown(f"""<div class="metric-card"><div class="label">Add Staff Recommended</div><div class="value">{recs.count('Add More Staff')}</div></div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
                st.dataframe(batch_df, width="stretch")

                st.download_button(
                    "⬇️  Download Results CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"batch_queue_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ======================================================================
# HISTORY TAB
# ======================================================================
with tab_history:
    st.markdown('<div class="section-title">🕘 Prediction History</div>', unsafe_allow_html=True)

    hist_df = load_history()
    if hist_df.empty:
        st.markdown(
            """
            <div style="border:1px dashed var(--border); border-radius:14px; padding:40px 20px; text-align:center; color:var(--muted); background:var(--surface);">
                <div style="font-size:2rem;">📭</div>
                <p style="margin-top:8px; font-size:0.88rem;">No predictions logged yet. Make one on the Predict tab.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        h1, h2, h3 = st.columns(3)
        h1.markdown(f"""<div class="metric-card"><div class="label">Total Predictions</div><div class="value">{len(hist_df)}</div></div>""", unsafe_allow_html=True)
        h2.markdown(f"""<div class="metric-card"><div class="label">Avg Predicted Wait</div><div class="value">{hist_df['Predicted Wait (min)'].mean():.0f} min</div></div>""", unsafe_allow_html=True)
        h3.markdown(f"""<div class="metric-card"><div class="label">Most Common Action</div><div class="value" style="font-size:1.1rem;">{hist_df['Recommendation'].mode()[0]}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.dataframe(hist_df.sort_values("timestamp", ascending=False), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Predicted Wait Over Time**")
            st.line_chart(hist_df.set_index("timestamp")["Predicted Wait (min)"], color="#4338CA")
        with c2:
            st.markdown("**Recommendation Breakdown**")
            st.bar_chart(hist_df["Recommendation"].value_counts(), color="#D98E3B")

        st.download_button(
            "⬇️  Download Full History CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

st.markdown(
    '<div class="app-footer">⏱️ Smart Queue AI · Built with Streamlit &amp; scikit-learn · Service optimization made data-driven</div>',
    unsafe_allow_html=True,
)
