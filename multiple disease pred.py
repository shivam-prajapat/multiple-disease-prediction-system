# -*- coding: utf-8 -*-
"""
MediPredict AI - Multiple Disease Prediction System
Author: Shivam Prajapat
Version: 1.0.0
Description: AI-powered screening tool for Diabetes, Heart Disease, and Parkinson's.
             Built with Streamlit and scikit-learn ML models.
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="MediPredict AI - Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# Custom CSS Styling
# =======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* ── Dark gradient background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a192f 100%);
        min-height: 100vh;
    }

    /* ── Hide Streamlit watermark ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1a2744 100%);
        border-right: 1px solid rgba(99, 179, 237, 0.15);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* ── Page title ── */
    .page-header {
        text-align: center;
        padding: 2.5rem 0 2rem;
        margin-bottom: 2rem;
    }
    .page-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #63b3ed, #76e4f7, #b794f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .page-header p {
        color: rgba(255,255,255,0.55);
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* ── Card container ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,179,237,0.18);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: rgba(99,179,237,0.8);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(99,179,237,0.15);
    }

    /* ── Input fields ── */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(99,179,237,0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.25s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(99,179,237,0.7) !important;
        box-shadow: 0 0 0 3px rgba(99,179,237,0.12) !important;
        background: rgba(255,255,255,0.08) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.25) !important;
    }
    .stTextInput label {
        color: rgba(255,255,255,0.75) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.2px !important;
    }

    /* ── Predict button ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3182ce, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(49,130,206,0.35) !important;
        letter-spacing: 0.3px !important;
        margin-top: 1.5rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(49,130,206,0.5) !important;
        background: linear-gradient(135deg, #2c5282, #6d28d9) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Result boxes ── */
    .result-positive {
        background: linear-gradient(135deg, rgba(229,62,62,0.15), rgba(197,48,48,0.1));
        border: 1px solid rgba(229,62,62,0.4);
        border-left: 4px solid #e53e3e;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: slideUp 0.4s ease;
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(56,161,105,0.15), rgba(47,133,90,0.1));
        border: 1px solid rgba(56,161,105,0.4);
        border-left: 4px solid #38a169;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: slideUp 0.4s ease;
    }
    .result-icon {
        font-size: 2.2rem;
        flex-shrink: 0;
    }
    .result-content h3 {
        margin: 0 0 0.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
    }
    .result-content p {
        margin: 0;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Warning alert ── */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(246,173,85,0.35) !important;
        background: rgba(246,173,85,0.08) !important;
    }

    /* ── Metric pills ── */
    .accuracy-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99,179,237,0.1);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: rgba(99,179,237,0.9);
        margin-bottom: 1.5rem;
    }

    /* ── Columns gap ── */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }

    /* ── Success empty box hide ── */
    div[data-testid="stText"] { display: none; }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid rgba(99,179,237,0.12);
        margin: 2rem 0;
    }

    /* ── Sidebar menu style overrides ── */
    .nav-link {
        border-radius: 10px !important;
        margin: 2px 0 !important;
    }
    .nav-link.active {
        background: linear-gradient(135deg, #3182ce22, #7c3aed22) !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
    }

    /* ── Tooltip-style info boxes ── */
    .info-pill {
        background: rgba(118,228,247,0.07);
        border: 1px solid rgba(118,228,247,0.2);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.8rem;
        color: rgba(118,228,247,0.85);
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# =======================
# Load Saved Models
# =======================
@st.cache_resource
def load_models():
    diab_bundle     = pickle.load(open("diabetes_model.sav", "rb"))
    heart_bundle    = pickle.load(open("trained_model.sav", "rb"))
    park_bundle     = pickle.load(open("parkinsons_model.sav", "rb"))
    return diab_bundle, heart_bundle, park_bundle

diab_bundle, heart_bundle, park_bundle = load_models()

diabetes_model   = diab_bundle['model']
diabetes_scaler  = diab_bundle['scaler']

heart_model      = heart_bundle['model']
heart_scaler     = heart_bundle['scaler']

parkinsons_model = park_bundle['model']
parkinsons_scaler = park_bundle['scaler']


# =======================
# Sidebar Navigation
# =======================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:2.8rem; margin-bottom:0.4rem;'>🏥</div>
        <div style='font-size:1.15rem; font-weight:700; color:white; letter-spacing:-0.3px;'>MediPredict AI</div>
        <div style='font-size:0.72rem; color:rgba(255,255,255,0.4); margin-top:0.2rem;'>Intelligent Disease Screening</div>
    </div>
    <hr style='border-color:rgba(99,179,237,0.15); margin: 1rem 0;'/>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['droplet-fill', 'heart-pulse-fill', 'person-fill'],
        default_index=0,
        styles={
            "container": {"background-color": "transparent", "padding": "0"},
            "icon": {"color": "#63b3ed", "font-size": "16px"},
            "nav-link": {
                "font-size": "0.88rem",
                "font-weight": "500",
                "color": "rgba(255,255,255,0.7)",
                "border-radius": "10px",
                "margin": "3px 0",
                "padding": "10px 14px",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, rgba(49,130,206,0.25), rgba(124,58,237,0.2))",
                "color": "white",
                "font-weight": "600",
                "border": "1px solid rgba(99,179,237,0.3)",
            },
        }
    )

    st.markdown("""
    <hr style='border-color:rgba(99,179,237,0.15); margin: 1.5rem 0 1rem;'/>
    <div style='padding: 0.8rem; background: rgba(255,255,255,0.03); border-radius:10px;
                border:1px solid rgba(99,179,237,0.12); font-size:0.73rem;
                color:rgba(255,255,255,0.4); line-height:1.6;'>
        ⚠️ <b style='color:rgba(255,255,255,0.55);'>Disclaimer:</b> This tool is for
        educational purposes only and does not replace professional medical advice.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)


# =======================
# Helper: result display
# =======================
def show_result(positive: bool, disease: str):
    if positive:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-icon">⚠️</div>
            <div class="result-content">
                <h3>High Risk Detected</h3>
                <p>The analysis suggests this person <b>may have {disease}</b>.
                   Please consult a healthcare professional immediately.</p>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-icon">✅</div>
            <div class="result-content">
                <h3>Low Risk — Looks Healthy</h3>
                <p>The analysis suggests this person is <b>unlikely to have {disease}</b>.
                   Continue maintaining a healthy lifestyle.</p>
            </div>
        </div>""", unsafe_allow_html=True)


def input_error(e):
    st.warning(f"⚠️ Please fill in all fields with valid numeric values. ({e})")


# =======================
# DIABETES PREDICTION
# =======================
if selected == 'Diabetes Prediction':

    st.markdown("""
    <div class="page-header">
        <h1>🩸 Diabetes Prediction</h1>
        <p>Enter the patient's clinical parameters to assess diabetes risk using a Support Vector Machine model.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="accuracy-badge">🎯 Model Accuracy &nbsp;≈ 77%&nbsp; (Test Set)</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-pill">💡 <b>Tip:</b> All fields are required. Use values from a standard blood test report. Insulin of 0 means not measured.</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section-label">🔬 Patient Clinical Data</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.text_input('Pregnancies', placeholder='e.g. 2', help='Number of times pregnant')
        with col2:
            Glucose = st.text_input('Glucose (mg/dL)', placeholder='e.g. 120', help='Plasma glucose concentration (2-hr OGTT)')
        with col3:
            BloodPressure = st.text_input('Blood Pressure (mmHg)', placeholder='e.g. 70', help='Diastolic blood pressure')

        col1, col2, col3 = st.columns(3)
        with col1:
            SkinThickness = st.text_input('Skin Thickness (mm)', placeholder='e.g. 20', help='Triceps skin fold thickness')
        with col2:
            Insulin = st.text_input('Insulin (μU/mL)', placeholder='e.g. 0', help='2-Hour serum insulin (0 if not measured)')
        with col3:
            BMI = st.text_input('BMI (kg/m²)', placeholder='e.g. 27.5', help='Body mass index')

        col1, col2, col3 = st.columns(3)
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', placeholder='e.g. 0.45', help='Genetic diabetes risk function (0.08–2.42)')
        with col2:
            Age = st.text_input('Age (years)', placeholder='e.g. 35', help='Age in years')
        with col3:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if st.button('🔍 Run Diabetes Analysis', key='diab_btn'):
        try:
            features = np.array([[
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]])
            scaled = diabetes_scaler.transform(features)
            prediction = diabetes_model.predict(scaled)
            show_result(prediction[0] == 1, "Diabetes")
        except Exception as e:
            input_error(e)


# =======================
# HEART DISEASE PREDICTION
# =======================
elif selected == 'Heart Disease Prediction':

    st.markdown("""
    <div class="page-header">
        <h1>❤️ Heart Disease Prediction</h1>
        <p>Assess cardiovascular risk using a Logistic Regression model trained on the Cleveland Heart Disease dataset.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="accuracy-badge">🎯 Model Accuracy &nbsp;≈ 79%&nbsp; (Test Set)</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-pill">💡 <b>Chest Pain Types:</b> 0 = Typical Angina &nbsp;|&nbsp; 1 = Atypical Angina &nbsp;|&nbsp; 2 = Non-Anginal Pain &nbsp;|&nbsp; 3 = Asymptomatic</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section-label">👤 Demographics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.text_input('Age (years)', placeholder='e.g. 55', key='h_age')
        with col2:
            sex = st.text_input('Sex (1=Male, 0=Female)', placeholder='e.g. 1', key='h_sex')
        with col3:
            cp = st.text_input('Chest Pain Type (0–3)', placeholder='e.g. 2', key='h_cp')

        st.markdown('<div class="section-label">🩺 Cardiovascular Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            trestbps = st.text_input('Resting Blood Pressure (mmHg)', placeholder='e.g. 130', key='h_tbp')
        with col2:
            chol = st.text_input('Serum Cholesterol (mg/dL)', placeholder='e.g. 240', key='h_chol')
        with col3:
            fbs = st.text_input('Fasting Blood Sugar >120 (1=Yes, 0=No)', placeholder='e.g. 0', key='h_fbs')

        col1, col2, col3 = st.columns(3)
        with col1:
            restecg = st.text_input('Resting ECG (0/1/2)', placeholder='e.g. 1', key='h_ecg', help='0=Normal, 1=ST-T abnormality, 2=LV hypertrophy')
        with col2:
            thalach = st.text_input('Max Heart Rate Achieved', placeholder='e.g. 150', key='h_thal')
        with col3:
            exang = st.text_input('Exercise-Induced Angina (1=Yes, 0=No)', placeholder='e.g. 0', key='h_exang')

        col1, col2, col3 = st.columns(3)
        with col1:
            oldpeak = st.text_input('ST Depression (Oldpeak)', placeholder='e.g. 1.5', key='h_op')
        with col2:
            slope = st.text_input('Slope of Peak Exercise ST (0/1/2)', placeholder='e.g. 1', key='h_slope')
        with col3:
            ca = st.text_input('Major Vessels Colored by Fluoroscopy (0–4)', placeholder='e.g. 0', key='h_ca')

        col1, col2, col3 = st.columns(3)
        with col1:
            thal = st.text_input('Thalassemia (1=Normal, 2=Fixed, 3=Reversible)', placeholder='e.g. 2', key='h_thal2')

    if st.button('🔍 Run Heart Disease Analysis', key='heart_btn'):
        try:
            features = np.array([[
                float(age), float(sex), float(cp), float(trestbps),
                float(chol), float(fbs), float(restecg),
                float(thalach), float(exang), float(oldpeak),
                float(slope), float(ca), float(thal)
            ]])
            scaled = heart_scaler.transform(features)
            prediction = heart_model.predict(scaled)
            show_result(prediction[0] == 1, "Heart Disease")
        except Exception as e:
            input_error(e)


# =======================
# PARKINSONS PREDICTION
# =======================
elif selected == 'Parkinsons Prediction':

    st.markdown("""
    <div class="page-header">
        <h1>🧠 Parkinson's Prediction</h1>
        <p>Detect early-stage Parkinson's Disease from voice biomarkers using a Support Vector Machine model.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="accuracy-badge">🎯 Model Accuracy &nbsp;≈ 87%&nbsp; (Test Set)</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-pill">💡 These parameters are extracted from sustained phonation recordings. Values can be obtained through clinical voice analysis tools.</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section-label">🎙️ Fundamental Frequency Measures</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = st.text_input('MDVP:Fo (Hz) — Avg vocal freq.', placeholder='e.g. 119.992', key='p_fo')
        with col2:
            fhi = st.text_input('MDVP:Fhi (Hz) — Max vocal freq.', placeholder='e.g. 157.302', key='p_fhi')
        with col3:
            flo = st.text_input('MDVP:Flo (Hz) — Min vocal freq.', placeholder='e.g. 74.997', key='p_flo')

        st.markdown('<div class="section-label">📊 Jitter (Frequency Variation)</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            jitter_pct = st.text_input('MDVP:Jitter (%)', placeholder='e.g. 0.00784', key='p_jp')
        with col2:
            jitter_abs = st.text_input('MDVP:Jitter (Abs)', placeholder='e.g. 0.00007', key='p_ja')
        with col3:
            rap = st.text_input('MDVP:RAP', placeholder='e.g. 0.00370', key='p_rap')

        col1, col2, col3 = st.columns(3)
        with col1:
            ppq = st.text_input('MDVP:PPQ', placeholder='e.g. 0.00554', key='p_ppq')
        with col2:
            ddp = st.text_input('Jitter:DDP', placeholder='e.g. 0.01109', key='p_ddp')

        st.markdown('<div class="section-label">📈 Shimmer (Amplitude Variation)</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            shimmer = st.text_input('MDVP:Shimmer', placeholder='e.g. 0.04374', key='p_sh')
        with col2:
            shimmer_db = st.text_input('MDVP:Shimmer (dB)', placeholder='e.g. 0.426', key='p_shdb')
        with col3:
            apq3 = st.text_input('Shimmer:APQ3', placeholder='e.g. 0.02182', key='p_apq3')

        col1, col2, col3 = st.columns(3)
        with col1:
            apq5 = st.text_input('Shimmer:APQ5', placeholder='e.g. 0.03130', key='p_apq5')
        with col2:
            apq = st.text_input('MDVP:APQ', placeholder='e.g. 0.02971', key='p_apq')
        with col3:
            dda = st.text_input('Shimmer:DDA', placeholder='e.g. 0.06545', key='p_dda')

        st.markdown('<div class="section-label">🔬 Noise &amp; Nonlinear Dynamics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            nhr = st.text_input('NHR — Noise-to-Harmonics Ratio', placeholder='e.g. 0.02211', key='p_nhr')
        with col2:
            hnr = st.text_input('HNR — Harmonics-to-Noise Ratio', placeholder='e.g. 21.033', key='p_hnr')
        with col3:
            rpde = st.text_input('RPDE', placeholder='e.g. 0.41478', key='p_rpde')

        col1, col2, col3 = st.columns(3)
        with col1:
            dfa = st.text_input('DFA — Signal Fractal Scaling', placeholder='e.g. 0.81528', key='p_dfa')
        with col2:
            spread1 = st.text_input('Spread1', placeholder='e.g. -4.81303', key='p_sp1')
        with col3:
            spread2 = st.text_input('Spread2', placeholder='e.g. 0.26648', key='p_sp2')

        col1, col2, col3 = st.columns(3)
        with col1:
            d2 = st.text_input('D2 — Correlation Dimension', placeholder='e.g. 2.30144', key='p_d2')
        with col2:
            ppe = st.text_input('PPE — Pitch Period Entropy', placeholder='e.g. 0.28465', key='p_ppe')

    if st.button('🔍 Run Parkinson\'s Analysis', key='park_btn'):
        try:
            features = np.array([[
                float(fo), float(fhi), float(flo),
                float(jitter_pct), float(jitter_abs), float(rap), float(ppq), float(ddp),
                float(shimmer), float(shimmer_db), float(apq3), float(apq5), float(apq), float(dda),
                float(nhr), float(hnr), float(rpde), float(dfa),
                float(spread1), float(spread2), float(d2), float(ppe)
            ]])
            scaled = parkinsons_scaler.transform(features)
            prediction = parkinsons_model.predict(scaled)
            show_result(prediction[0] == 1, "Parkinson's Disease")
        except Exception as e:
            input_error(e)
