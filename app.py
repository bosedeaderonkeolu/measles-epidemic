import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import io
import requests
import time
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, GRU, Dense
from tensorflow.keras.optimizers import Adam
# ─── Optional heavy imports ────────────────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model as keras_load
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Measles Outbreak Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --card: #1a2233;
    --accent: #e63946;
    --accent2: #f4a261;
    --text: #e8eaf0;
    --muted: #8892a4;
    --border: #2a3448;
    --success: #2ec4b6;
    --warn: #f4a261;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1400px; }

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a0a0a 50%, #0d1b2a 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(230,57,70,0.15) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; line-height: 1.1;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(90deg, #fff 0%, #e63946 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: 0.95rem; color: var(--muted); font-weight: 300; margin: 0; }
.hero-badge {
    display: inline-block; background: rgba(230,57,70,0.15); border: 1px solid rgba(230,57,70,0.35);
    color: var(--accent); font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; padding: 0.2rem 0.7rem; border-radius: 20px; margin-bottom: 0.8rem;
}
.section-label {
    font-family: 'Syne', sans-serif; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--accent);
    margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.result-outbreak {
    background: linear-gradient(135deg, rgba(230,57,70,0.12), rgba(230,57,70,0.04));
    border: 2px solid var(--accent); border-radius: 16px; padding: 2rem; text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, rgba(46,196,182,0.12), rgba(46,196,182,0.04));
    border: 2px solid var(--success); border-radius: 16px; padding: 2rem; text-align: center;
}
.result-title { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; margin: 0.4rem 0; }
.result-icon { font-size: 2.5rem; margin-bottom: 0.4rem; }

.info-box {
    background: rgba(46,196,182,0.08); border: 1px solid rgba(46,196,182,0.25);
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; font-size: 0.88rem;
}
.warn-box {
    background: rgba(244,162,97,0.08); border: 1px solid rgba(244,162,97,0.3);
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; font-size: 0.88rem;
}

/* Table styling */
.dataframe { font-size: 0.82rem !important; }

/* Streamlit overrides */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] input {
    background: var(--surface) !important; border-color: var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important;
}
.stButton > button {
    background: var(--accent) !important; color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important; letter-spacing: 0.05em !important;
    padding: 0.65rem 2rem !important; width: 100% !important; transition: all 0.2s ease !important;
}
.stButton > button:hover { background: #c1121f !important; transform: translateY(-1px) !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; color: var(--text); border-right: 1px solid var(--border) !important; }
div[data-testid="metric-container"] {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; padding: 0.8rem !important;
}
hr { border-color: var(--border) !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif; font-weight: 600; color: var(--muted) !important; }
.stTabs [aria-selected="true"] { background: var(--card) !important; color: var(--text) !important; border-radius: 8px; }

/* Download button */
.stDownloadButton > button {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important; font-size: 0.85rem !important;
    width: auto !important; padding: 0.4rem 1rem !important;
}
.stDownloadButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* Progress */
.stProgress > div > div { background: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ─────────────────────────────────────────────────────────────────
RFE_FEATURES = [
    "Measles_deaths", "MCV1_coverage (%)", "MCV2_coverage (%)", "DTP3_coverage (%)",
    "Urbanization (%)", "Birth_rate (per 1000)", "Proportion_under5 (%)", "GDP_per_capita (US$)",
    "Hospital_beds_per_1000", "Physicians_per_10000", "GoogleTrends_index",
    "Measles_incidence_rate (per million)", "SIAs_in_past_3yrs", "Internet_penetration (%)",
    "Routine_immunization_dropout (%)", "Avg_Annual_Temperature", "Avg_Annual_Humidity",
    "Rainy_Season_Length", "Temp_Seasonality", "Dry_Season_Length", "Extreme_Rain_Days"
]

MI_FEATURES = [
    "Measles_incidence_rate (per million)", "GoogleTrends_index",
    "Suspected_measles_cases", "Reported_measles_cases", "Measles_deaths",
    "MCV1_coverage (%)", "Dropout_rate (MCV1-MCV2)",
    "Air_travel_volume (million passengers)", "MCV2_coverage (%)",
    "Population", "DTP3_coverage (%)", "Population_density (people/km2)",
    "Urbanization (%)", "Health_expenditure_per_capita (US$)",
    "GDP_per_capita (US$)", "Proportion_under5 (%)", "Birth_rate (per 1000)",
    "Hospital_beds_per_1000", "Physicians_per_10000",
    "SIAs_in_past_3yrs", "Internet_penetration (%)",
]

PCA_PC1_FEATURES = [
    "Population", "Reported_measles_cases", "Suspected_measles_cases",
    "Total_Annual_Rainfall", "Measles_incidence_rate (per million)"
]
PCA_PC2_FEATURES = [
    "Reported_measles_cases", "Measles_incidence_rate (per million)",
    "Suspected_measles_cases", "Total_Annual_Rainfall",
    "Health_expenditure_per_capita (US$)"
]
PCA_PC3_FEATURES = [
    "Suspected_measles_cases", "Measles_deaths", "Health_expenditure_per_capita (US$)",
    "Total_Annual_Rainfall", "Reported_measles_cases"
]
PCA_PC4_FEATURES = [
    "Health_expenditure_per_capita (US$)", "Measles_incidence_rate (per million)",
    "Suspected_measles_cases", "Total_Annual_Rainfall", "Internet_penetration (%)"
]
PCA_PC5_FEATURES = [
    "Total_Annual_Rainfall", "Measles_incidence_rate (per million)",
    "Health_expenditure_per_capita (US$)", "Extreme_Rain_Days", "Avg_Annual_Humidity"
]

FEATURE_SETS = {
    "Mutual Information (21 features)": MI_FEATURES,
    "Recursive Feature Elimination (21 features)": RFE_FEATURES,
    "PCA Component 1 (Top 5)": PCA_PC1_FEATURES,
    "PCA Component 2 (Top 5)": PCA_PC2_FEATURES,
    "PCA Component 3 (Top 5)": PCA_PC3_FEATURES,
    "PCA Component 4 (Top 5)": PCA_PC4_FEATURES,
    "PCA Component 5 (Top 5)": PCA_PC5_FEATURES,
}

MODEL_DIR_MAP = {
    "Mutual Information (21 features)": "MI",
    "Recursive Feature Elimination (21 features)": "RFE",
    "PCA Component 1 (Top 5)": "PCA",
    "PCA Component 2 (Top 5)": "PCA",
    "PCA Component 3 (Top 5)": "PCA",
    "PCA Component 4 (Top 5)": "PCA",
    "PCA Component 5 (Top 5)": "PCA",
}

FEATURE_CONFIGS = {
    "Reported_measles_cases":                {"min": 0,     "max": 500000,    "default": 500,    "step": 1},
    "Suspected_measles_cases":               {"min": 0,     "max": 500000,    "default": 1000,   "step": 1},
    "Measles_deaths":                        {"min": 0,     "max": 10000,     "default": 5,      "step": 1},
    "MCV1_coverage (%)":                     {"min": 0.0,   "max": 100.0,     "default": 80.0,   "step": 0.1},
    "MCV2_coverage (%)":                     {"min": 0.0,   "max": 100.0,     "default": 70.0,   "step": 0.1},
    "DTP3_coverage (%)":                     {"min": 0.0,   "max": 100.0,     "default": 78.0,   "step": 0.1},
    "Measles_incidence_rate (per million)":  {"min": 0.0,   "max": 5000.0,    "default": 10.0,   "step": 0.1},
    "GoogleTrends_index":                    {"min": 0,     "max": 100,       "default": 20,     "step": 1},
    "Dropout_rate (MCV1-MCV2)":              {"min": -50.0, "max": 100.0,     "default": 5.0,    "step": 0.1},
    "Air_travel_volume (million passengers)":{"min": 0.0,   "max": 1000.0,    "default": 10.0,   "step": 0.1},
    "Population":                            {"min": 0,     "max": 2000000000,"default": 5000000,"step": 1000},
    "Population_density (people/km2)":       {"min": 0.0,   "max": 50000.0,   "default": 100.0,  "step": 0.1},
    "Urbanization (%)":                      {"min": 0.0,   "max": 100.0,     "default": 55.0,   "step": 0.1},
    "Health_expenditure_per_capita (US$)":   {"min": 0.0,   "max": 15000.0,   "default": 200.0,  "step": 1.0},
    "GDP_per_capita (US$)":                  {"min": 0.0,   "max": 150000.0,  "default": 3000.0, "step": 10.0},
    "Proportion_under5 (%)":                 {"min": 0.0,   "max": 25.0,      "default": 10.0,   "step": 0.1},
    "Birth_rate (per 1000)":                 {"min": 0.0,   "max": 60.0,      "default": 20.0,   "step": 0.1},
    "Hospital_beds_per_1000":                {"min": 0.0,   "max": 20.0,      "default": 2.0,    "step": 0.1},
    "Physicians_per_10000":                  {"min": 0.0,   "max": 100.0,     "default": 5.0,    "step": 0.1},
    "SIAs_in_past_3yrs":                     {"min": 0,     "max": 3,         "default": 1,      "step": 1},
    "Internet_penetration (%)":              {"min": 0.0,   "max": 100.0,     "default": 40.0,   "step": 0.1},
    "Total_Annual_Rainfall":                 {"min": 0.0,   "max": 10000.0,   "default": 100.0,  "step": 0.1},
    "Extreme_Rain_Days":                     {"min": 0.0,   "max": 366.0,     "default": 10.0,   "step": 1.0},
    "Avg_Annual_Humidity":                   {"min": 0.0,   "max": 100.0,     "default": 60.0,   "step": 0.1},
    "Routine_immunization_dropout (%)":      {"min": 0.0,   "max": 100.0,     "default": 5.0,    "step": 0.1},
}

# World Bank indicator codes
WB_INDICATORS = {
    "Population":                          "SP.POP.TOTL",
    "Population_density (people/km2)":     "EN.POP.DNST",
    "Urbanization (%)":                    "SP.URB.TOTL.IN.ZS",
    "GDP_per_capita (US$)":                "NY.GDP.PCAP.CD",
    "Birth_rate (per 1000)":               "SP.DYN.CBRT.IN",
    "Proportion_under5 (%)":               "SP.POP.0004.TO.ZS",
    "Internet_penetration (%)":            "IT.NET.USER.ZS",
    "Hospital_beds_per_1000":              "SH.MED.BEDS.ZS",
    "Physicians_per_10000":                "SH.MED.PHYS.ZS",
    "Health_expenditure_per_capita (US$)": "SH.XPD.CHEX.PC.CD",
}

# WHO immunization indicators (from WHO GHO API)
WHO_INDICATORS = {
    "MCV1_coverage (%)": "WHS4_543",
    "MCV2_coverage (%)": "WHS4_544",
    "DTP3_coverage (%)": "WHS4_100",
}

COUNTRIES = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO",
    "Argentina": "ARG", "Australia": "AUS", "Bangladesh": "BGD", "Brazil": "BRA",
    "Cameroon": "CMR", "Canada": "CAN", "Chad": "TCD", "China": "CHN",
    "Colombia": "COL", "Congo (DRC)": "COD", "Egypt": "EGY", "Ethiopia": "ETH",
    "France": "FRA", "Germany": "DEU", "Ghana": "GHA", "India": "IND",
    "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ", "Italy": "ITA",
    "Japan": "JPN", "Kazakhstan": "KAZ", "Kenya": "KEN", "Malaysia": "MYS",
    "Mexico": "MEX", "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR",
    "Nepal": "NPL", "Nigeria": "NGA", "Pakistan": "PAK", "Philippines": "PHL",
    "Russia": "RUS", "Saudi Arabia": "SAU", "Senegal": "SEN", "Somalia": "SOM",
    "South Africa": "ZAF", "Sudan": "SDN", "Tanzania": "TZA", "Thailand": "THA",
    "Turkey": "TUR", "Uganda": "UGA", "Ukraine": "UKR", "United Kingdom": "GBR",
    "United States": "USA", "Venezuela": "VEN", "Vietnam": "VNM", "Yemen": "YEM",
    "Zambia": "ZMB", "Zimbabwe": "ZWE",
}


# ─── Model Loader ───────────────────────────────────────────────────────────────
@st.cache_resource

def _build(arch, n_features):
    m = Sequential()
    if arch == "LSTM":
        m.add(LSTM(64, input_shape=(1, n_features), activation='tanh'))
    elif arch == "BiLSTM":
        m.add(Bidirectional(LSTM(64, activation='tanh'), input_shape=(1, n_features)))
    elif arch == "GRU":
        m.add(GRU(64, input_shape=(1, n_features), activation='tanh'))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return m

@st.cache_resource
def load_models(feature_method: str):
    model_subdir = MODEL_DIR_MAP.get(feature_method, "MI")
    base_dir = os.path.join("models", model_subdir)
    n_features = len(FEATURE_SETS.get(feature_method, MI_FEATURES))
    models = {}

    for name in ["LSTM", "BiLSTM", "GRU"]:
        path = os.path.join(base_dir, f"{name}.weights.h5")
        if os.path.exists(path):
            m = _build(name, n_features)
            m.load_weights(path)
            models[name] = m

    meta_path = os.path.join(base_dir, "MetaLearner_LogisticRegression.pkl")
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else None
    return models, meta


# ─── Prediction Core ────────────────────────────────────────────────────────────
def predict_row(X_row: np.ndarray, feature_method: str):
    """Run stacked ensemble on a single row. Returns (final_pred, final_prob, model_results)."""
    models, meta = load_models(feature_method)
    if not models:
        return None, None, {}

    X_rnn = np.expand_dims(X_row.reshape(1, -1), axis=1)
    base_preds = []
    model_results = {}

    for name, model in models.items():
        prob = float(model.predict(X_rnn, verbose=0)[0][0])
        cls = int(prob > 0.5)
        base_preds.append(cls)
        model_results[name] = {"prob": prob, "class": cls}

    if meta and len(base_preds) > 0:
        meta_input = np.array([base_preds])
        final_pred = int(meta.predict(meta_input)[0])
        try:
            final_prob = float(meta.predict_proba(meta_input)[0][1])
        except:
            final_prob = float(sum(base_preds) / len(base_preds))
    else:
        final_pred = int(sum(base_preds) >= len(base_preds) / 2)
        final_prob = float(sum(base_preds) / max(len(base_preds), 1))

    return final_pred, final_prob, model_results


def predict_dataframe(df: pd.DataFrame, active_features: list, feature_method: str):
    """Predict for a whole DataFrame. Returns DataFrame with results appended."""
    results = []
    probs = []
    for _, row in df.iterrows():
        X = np.array([float(row.get(f, 0)) for f in active_features], dtype=np.float32)
        pred, prob, _ = predict_row(X, feature_method)
        results.append("Outbreak" if pred == 1 else "Non-Outbreak")
        probs.append(round(prob, 4) if prob is not None else None)

    df_out = df.copy()
    df_out["Predicted_Outbreak"] = results
    df_out["Outbreak_Probability"] = probs
    return df_out


# ─── World Bank / WHO Data Fetch ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_world_bank(iso3: str, year: int):
    """Fetch World Bank indicators for a country/year."""
    data = {}
    base = "https://api.worldbank.org/v2/country/{iso}/indicator/{ind}?date={yr}&format=json&per_page=1"
    for col, ind in WB_INDICATORS.items():
        try:
            url = base.format(iso=iso3, ind=ind, yr=year)
            r = requests.get(url, timeout=8)
            j = r.json()
            if len(j) > 1 and j[1] and j[1][0]["value"] is not None:
                data[col] = float(j[1][0]["value"])
            else:
                # Try previous year as fallback
                url2 = base.format(iso=iso3, ind=ind, yr=year - 1)
                r2 = requests.get(url2, timeout=8)
                j2 = r2.json()
                if len(j2) > 1 and j2[1] and j2[1][0]["value"] is not None:
                    data[col] = float(j2[1][0]["value"])
                else:
                    data[col] = None
        except Exception:
            data[col] = None
        time.sleep(0.05)
    return data


@st.cache_data(ttl=3600)
def fetch_who_immunization(iso3: str, year: int):
    """Fetch WHO GHO immunization coverage data."""
    data = {}
    for col, ind in WHO_INDICATORS.items():
        try:
            url = f"https://ghoapi.azureedge.net/api/{ind}?$filter=SpatialDim eq '{iso3}' and TimeDim eq {year}"
            r = requests.get(url, timeout=8)
            j = r.json()
            vals = j.get("value", [])
            if vals:
                data[col] = float(vals[0].get("NumericValue", 0) or 0)
            else:
                # try year-1
                url2 = f"https://ghoapi.azureedge.net/api/{ind}?$filter=SpatialDim eq '{iso3}' and TimeDim eq {year-1}"
                r2 = requests.get(url2, timeout=8)
                j2 = r2.json()
                vals2 = j2.get("value", [])
                data[col] = float(vals2[0].get("NumericValue", 0)) if vals2 else None
        except Exception:
            data[col] = None
        time.sleep(0.05)
    return data


@st.cache_data(ttl=3600)
def fetch_who_measles(iso3: str, year: int):
    """Fetch WHO measles surveillance data (cases, deaths, incidence)."""
    data = {}
    # Reported cases
    try:
        url = f"https://ghoapi.azureedge.net/api/WHS3_45?$filter=SpatialDim eq '{iso3}' and TimeDim eq {year}"
        r = requests.get(url, timeout=8)
        j = r.json()
        vals = j.get("value", [])
        if vals:
            data["Reported_measles_cases"] = float(vals[0].get("NumericValue", 0) or 0)
    except Exception:
        data["Reported_measles_cases"] = None

    # Incidence rate
    try:
        url = f"https://ghoapi.azureedge.net/api/WHS3_46?$filter=SpatialDim eq '{iso3}' and TimeDim eq {year}"
        r = requests.get(url, timeout=8)
        j = r.json()
        vals = j.get("value", [])
        if vals:
            data["Measles_incidence_rate (per million)"] = float(vals[0].get("NumericValue", 0) or 0)
    except Exception:
        data["Measles_incidence_rate (per million)"] = None

    return data


# ─── UI Helpers ────────────────────────────────────────────────────────────────
def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def render_result(final_pred, final_prob):
    if final_pred == 1:
        st.markdown(f"""
        <div class="result-outbreak">
            <div class="result-icon">🚨</div>
            <div class="result-title" style="color:#e63946;">OUTBREAK DETECTED</div>
            <div style="color:#8892a4;margin-top:0.4rem;">
                The ensemble model predicts elevated measles outbreak risk
                {f'<br><strong style="color:#e63946;font-size:1.1rem;">Confidence: {final_prob:.1%}</strong>' if final_prob is not None else ''}
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <div class="result-icon">✅</div>
            <div class="result-title" style="color:#2ec4b6;">NO OUTBREAK</div>
            <div style="color:#8892a4;margin-top:0.4rem;">
                The ensemble model predicts low measles outbreak risk
                {f'<br><strong style="color:#2ec4b6;font-size:1.1rem;">Confidence: {final_prob:.1%}</strong>' if final_prob is not None else ''}
            </div>
        </div>""", unsafe_allow_html=True)


def render_model_breakdown(model_results, final_pred=None, final_prob=None):
    if not model_results:
        return
    st.markdown("<br>", unsafe_allow_html=True)
    section("📈 Individual Model Predictions")

    # Base models + Meta Learner column
    all_cols = st.columns(len(model_results) + 1)

    for col, (name, res) in zip(all_cols, model_results.items()):
        with col:
            st.metric(
                label=name,
                value="Outbreak" if res["class"] == 1 else "Non-Outbreak",
                delta=f"prob: {res['prob']:.3f}"
            )

    # Meta Learner as final column
    with all_cols[-1]:
        if final_pred is not None:
            st.metric(
                label="Meta Learner (Final)",
                value="Outbreak" if final_pred == 1 else "Non-Outbreak",
                delta=f"prob: {final_prob:.3f}" if final_prob is not None else "—"
            )

def generate_template_csv(feature_set: str, n_rows: int = 3):
    feats = FEATURE_SETS.get(feature_set, MI_FEATURES)
    rows = []
    for i in range(n_rows):
        row = {f: FEATURE_CONFIGS.get(f, {}).get("default", 0) for f in feats}
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Epidemiological AI</div>
  <h1 class="hero-title">Measles Outbreak<br>Prediction System</h1>
  <p class="hero-sub">Stacked ensemble · LSTM · BiLSTM · GRU · Logistic Regression meta-learner</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Configuration")
    st.markdown("---")

    feature_method = st.selectbox(
        "Feature selection method",
        options=list(FEATURE_SETS.keys()),
        index=list(FEATURE_SETS.keys()).index("Mutual Information (21 features)")
    )

    active_features = FEATURE_SETS.get(feature_method, MI_FEATURES)
    model_group = MODEL_DIR_MAP.get(feature_method, "MI")

    st.markdown(f"**Features used:** `{len(active_features)}`")
    st.markdown(f"**Method:** {feature_method}")
    st.markdown(f"**Model source folder:** `models/{model_group}`")
    st.markdown("""
    <div style='margin:0.5rem 0;'>
      <span style='background:#1a2233;border:1px solid #2ec4b6;color:#ffffff;font-size:0.75rem;padding:0.2rem 0.6rem;border-radius:20px;margin:0.2rem;display:inline-block;'>✓ LSTM</span>
      <span style='background:#1a2233;border:1px solid #2ec4b6;color:#ffffff;font-size:0.75rem;padding:0.2rem 0.6rem;border-radius:20px;margin:0.2rem;display:inline-block;'>✓ BiLSTM</span>
      <span style='background:#1a2233;border:1px solid #2ec4b6;color:#ffffff;font-size:0.75rem;padding:0.2rem 0.6rem;border-radius:20px;margin:0.2rem;display:inline-block;'>✓ GRU</span>
      <span style='background:#1a2233;border:1px solid #2ec4b6;color:#ffffff;font-size:0.75rem;padding:0.2rem 0.6rem;border-radius:20px;margin:0.2rem;display:inline-block;'>✓ Meta LR</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Modes**")
    st.markdown("① **Manual** — enter values row by row  \n② **Bulk CSV** — upload many rows at once  \n③ **Country Lookup** — fetch live data by country & year")
    st.markdown("---")
    st.caption("Data sources: WHO GHO · World Bank · Google Trends")
    st.caption("For research use only")


# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Manual Prediction",
    "Bulk CSV Upload",
    "Country & Year Lookup"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MANUAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    section("📋 Enter Feature Values")

    input_values = {}
    col_groups = [active_features[i:i+3] for i in range(0, len(active_features), 3)]
    for group in col_groups:
        cols = st.columns(len(group))
        for c, feat in zip(cols, group):
            cfg = FEATURE_CONFIGS.get(feat, {"min": 0.0, "max": 1000.0, "default": 0.0, "step": 0.1})
            with c:
                input_values[feat] = st.number_input(
                    feat, min_value=float(cfg["min"]), max_value=float(cfg["max"]),
                    value=float(cfg["default"]), step=float(cfg["step"]),
                    key=f"manual_{feat}", help=f"Range: {cfg['min']} – {cfg['max']}"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Outbreak Prediction", key="btn_manual"):
        section("Prediction Result")
        X_row = np.array([input_values[f] for f in active_features], dtype=np.float32)
        final_pred, final_prob, model_results = predict_row(X_row, feature_method)
        if final_pred is None:
            st.warning(f"⚠️ No trained models found at `models/`. Train and save your models first.")
        else:
            render_result(final_pred, final_prob)
            render_model_breakdown(model_results, final_pred, final_prob)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BULK CSV
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns([2, 1])
    with c1:
        section("📂 Upload CSV for Batch Prediction")
        st.markdown("""
        <div class="info-box">
        Upload a CSV file with one row per observation. Columns must match the feature names 
        for the selected feature method. Download the template below to get started.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop your CSV here", type=["csv"],
            help="Must contain the required feature columns for the selected method"
        )

    with c2:
        section("📥 Download Templates")
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        template_csv = generate_template_csv(feature_method)
        st.download_button(
            label=f"⬇  {feature_method} Template ({len(active_features)} features)",
            data=template_csv,
            file_name=f"measles_template_{feature_method.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
            mime="text/csv",
            key="dl_template"
        )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box" style='font-size:0.8rem;'>
        💡 Templates include 3 sample rows with default values. 
        Add/edit rows before uploading. Keep column names exactly as shown.
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown("---")
            section(f"📊 Preview — {len(df_upload)} rows detected")
            st.dataframe(df_upload.head(5), use_container_width=True)

            # Check columns
            missing_cols = [f for f in active_features if f not in df_upload.columns]
            if missing_cols:
                st.error(f"❌ Missing columns for {feature_method} method: `{'`, `'.join(missing_cols)}`")
                st.info("Download and use the correct template above, then re-upload.")
            else:
                if st.button(f"🚀 Classify All {len(df_upload)} Rows", key="btn_bulk"):
                    progress = st.progress(0)
                    status = st.empty()

                    # Process in batches for progress feedback
                    results_list = []
                    probs_list = []
                    models_loaded, meta_loaded = load_models(feature_method)

                    if not models_loaded:
                        st.warning(f"⚠️ No trained models found at `saved_models/{feature_method}/`.")
                    else:
                        for i, (_, row) in enumerate(df_upload.iterrows()):
                            X = np.array([float(row.get(f, 0)) for f in active_features], dtype=np.float32)
                            X_rnn = np.expand_dims(X.reshape(1, -1), axis=1)

                            base_preds = []
                            for name, model in models_loaded.items():
                                prob = float(model.predict(X_rnn, verbose=0)[0][0])
                                base_preds.append(int(prob > 0.5))

                            if meta_loaded:
                                meta_input = np.array([base_preds])
                                final_pred = int(meta_loaded.predict(meta_input)[0])
                                try:
                                    final_prob = float(meta_loaded.predict_proba(meta_input)[0][1])
                                except:
                                    final_prob = sum(base_preds) / len(base_preds)
                            else:
                                final_pred = int(sum(base_preds) >= len(base_preds) / 2)
                                final_prob = sum(base_preds) / max(len(base_preds), 1)

                            results_list.append("Outbreak" if final_pred == 1 else "Non-Outbreak")
                            probs_list.append(round(final_prob, 4))

                            pct = (i + 1) / len(df_upload)
                            progress.progress(pct)
                            status.markdown(f"Processing row **{i+1}** of **{len(df_upload)}**...")

                        progress.progress(1.0)
                        status.success(f"✅ Classification complete — {len(df_upload)} rows processed.")

                        df_result = df_upload.copy()
                        df_result["Predicted_Outbreak"] = results_list
                        df_result["Outbreak_Probability"] = probs_list

                        # Summary stats
                        n_outbreak = results_list.count("Outbreak")
                        n_safe = results_list.count("Non-Outbreak")

                        st.markdown("<br>", unsafe_allow_html=True)
                        section("📊 Summary")
                        sm1, sm2, sm3 = st.columns(3)
                        sm1.metric("Total Rows", len(df_upload))
                        sm2.metric("🚨 Outbreak", n_outbreak,
                                   delta=f"{n_outbreak/len(df_upload):.1%}" if df_upload is not None else "")
                        sm3.metric("✅ Non-Outbreak", n_safe,
                                   delta=f"{n_safe/len(df_upload):.1%}" if df_upload is not None else "")

                        section("📋 Results Table")

                        def color_result(val):
                            if val == "Outbreak":
                                return "background-color: rgba(230,57,70,0.15); color: #e63946; font-weight: bold;"
                            return "background-color: rgba(46,196,182,0.10); color: #2ec4b6;"

                        styled = df_result.style.applymap(color_result, subset=["Predicted_Outbreak"])
                        st.dataframe(styled, use_container_width=True, height=400)

                        # Download results
                        csv_out = df_result.to_csv(index=False)
                        st.download_button(
                            label="⬇  Download Results CSV",
                            data=csv_out,
                            file_name=f"measles_predictions_{feature_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="dl_results"
                        )

        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COUNTRY & YEAR LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    section("Live Data Lookup by Country & Year")
    st.markdown("""
    <div class="info-box">
    Select a country and year to automatically fetch real epidemiological and demographic data 
    from <strong>WHO GHO</strong> and the <strong>World Bank API</strong>. The fetched values 
    are then run through the outbreak prediction model. You can review and edit any values before predicting.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        country_name = st.selectbox("Select Country", sorted(COUNTRIES.keys()), key="country_sel")
    with col_b:
        year_sel = st.number_input("Year", min_value=2000, max_value=2024, value=2022, step=1, key="year_sel")
    with col_c:
        st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
        fetch_clicked = st.button("🌐 Fetch Live Data", key="btn_fetch")

    # Session state for live data
    if "live_data" not in st.session_state:
        st.session_state.live_data = {}
    if "fetch_country" not in st.session_state:
        st.session_state.fetch_country = None

    if fetch_clicked:
        iso3 = COUNTRIES[country_name]
        with st.spinner(f"Fetching data for **{country_name}** ({iso3}) — year **{year_sel}** from WHO & World Bank..."):
            wb_data   = fetch_world_bank(iso3, year_sel)
            who_imm   = fetch_who_immunization(iso3, year_sel)
            who_meas  = fetch_who_measles(iso3, year_sel)

        fetched = {}
        fetched.update(wb_data)
        fetched.update(who_imm)
        fetched.update(who_meas)

        # Fill defaults for anything not fetched
        for feat in active_features:
            if feat not in fetched or fetched[feat] is None:
                fetched[feat] = FEATURE_CONFIGS.get(feat, {}).get("default", 0.0)

        st.session_state.live_data = fetched
        st.session_state.fetch_country = country_name

        # Show fetch status
        fetched_keys = [k for k, v in fetched.items() if v is not None and v != FEATURE_CONFIGS.get(k, {}).get("default", 0)]
        st.success(f"Retrieved {len(fetched_keys)} data points. Fields not found from APIs use default values — please review and edit below.")

    if st.session_state.live_data:
        st.markdown("---")
        section(f"Review & Edit — {st.session_state.fetch_country or country_name} {year_sel}")
        st.markdown("""
        <div class="warn-box" style='font-size:0.82rem;'>
        Some fields (Suspected cases, GoogleTrends index, SIAs, Dropout rate, Air travel) are not available via public APIs. 
        Please fill these in manually. Fields with API data are pre-populated.
        </div>
        """, unsafe_allow_html=True)

        live_inputs = {}
        col_groups2 = [active_features[i:i+3] for i in range(0, len(active_features), 3)]
        for group in col_groups2:
            cols = st.columns(len(group))
            for c, feat in zip(cols, group):
                cfg = FEATURE_CONFIGS.get(feat, {"min": 0.0, "max": 1000.0, "default": 0.0, "step": 0.1})
                prefill = st.session_state.live_data.get(feat, cfg["default"])
                with c:
                    live_inputs[feat] = st.number_input(
                        feat,
                        min_value=float(cfg["min"]),
                        max_value=float(cfg["max"]),
                        value=float(min(max(prefill, cfg["min"]), cfg["max"])),
                        step=float(cfg["step"]),
                        key=f"live_{feat}",
                        help=f"Range: {cfg['min']} – {cfg['max']}"
                    )

        # Show data source summary
        with st.expander("Data Sources Used", expanded=False):
            source_rows = []
            for feat in active_features:
                if feat in WB_INDICATORS:
                    src = f"World Bank ({WB_INDICATORS[feat]})"
                elif feat in WHO_INDICATORS:
                    src = f"WHO GHO ({WHO_INDICATORS[feat]})"
                elif feat in ["Reported_measles_cases", "Measles_incidence_rate (per million)"]:
                    src = "WHO GHO (measles surveillance)"
                else:
                    src = "Manual entry required"
                val = live_inputs.get(feat, "—")
                source_rows.append({"Feature": feat, "Value": val, "Source": src})
            st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f" Predict Outbreak Risk for {country_name}", key="btn_live_predict"):
            section("Prediction Result")
            X_row = np.array([live_inputs[f] for f in active_features], dtype=np.float32)
            final_pred, final_prob, model_results = predict_row(X_row, feature_method)
            if final_pred is None:
                st.warning(f"No trained models found at `saved_models/{feature_method}/`.")
            else:
                # Country context header
                st.markdown(f"""
                <div style='font-family:Syne,sans-serif;font-size:0.8rem;color:#8892a4;margin-bottom:0.75rem;'>
                     {country_name} · {year_sel} · Feature method: {feature_method}
                </div>""", unsafe_allow_html=True)
                render_result(final_pred, final_prob)
                render_model_breakdown(model_results, final_pred, final_prob)

                # Export this row
                row_data = live_inputs.copy()
                row_data["Country"] = country_name
                row_data["Year"] = year_sel
                row_data["Predicted_Outbreak"] = "Outbreak" if final_pred == 1 else "Non-Outbreak"
                row_data["Outbreak_Probability"] = round(final_prob, 4) if final_prob else None
                export_df = pd.DataFrame([row_data])
                st.download_button(
                    label="⬇  Export This Result as CSV",
                    data=export_df.to_csv(index=False),
                    file_name=f"measles_{country_name}_{year_sel}.csv",
                    mime="text/csv",
                    key="dl_live"
                )

    elif not fetch_clicked:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;color:#3d4a5c;'>
            <div style='font-size:3rem;margin-bottom:0.5rem;'>🌍</div>
            <div style='font-family:Syne,sans-serif;font-size:1rem;'>Select a country and year above, then click <strong>Fetch Live Data</strong></div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#3d4a5c;font-size:0.72rem;'>"
    "Measles Outbreak Prediction · Stacked Deep Learning Ensemble · "
    "Data: WHO GHO · World Bank · For research use only"
    "</center>",
    unsafe_allow_html=True
)



