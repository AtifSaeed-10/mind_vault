import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Mental Health Crisis Predictor",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --ink:      #0b0d17;
    --surface:  #111527;
    --card:     #161b2e;
    --border:   #1f2a45;
    --accent1:  #7f5af0;
    --accent2:  #2cb67d;
    --accent3:  #ff8906;
    --text:     #fffffe;
    --muted:    #72757e;
}

* { font-family: 'Outfit', sans-serif; box-sizing: border-box; }

.stApp {
    background-color: var(--ink);
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%,  rgba(127,90,240,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%,  rgba(44,182,125,0.14) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 60% 30%,  rgba(255,137,6,0.08)  0%, transparent 50%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.015'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    color: var(--text);
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(127,90,240,0.12), transparent 70%);
    animation: drift1 18s ease-in-out infinite alternate;
    pointer-events: none; z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -150px; right: -150px;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(44,182,125,0.10), transparent 70%);
    animation: drift2 22s ease-in-out infinite alternate;
    pointer-events: none; z-index: 0;
}

@keyframes drift1 {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(120px, 80px) scale(1.2); }
}
@keyframes drift2 {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(-80px,-60px) scale(1.15); }
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; position: relative; z-index: 1; }

.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(127,90,240,0.12);
    border: 1px solid rgba(127,90,240,0.3);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #a78bfa;
    letter-spacing: 0.1em;
    margin-bottom: 1.2rem;
    animation: fadeDown 0.8s ease both;
}

.hero-brain {
    font-size: 5rem;
    display: block;
    margin-bottom: 0.8rem;
    animation: brainPulse 4s ease-in-out infinite;
    filter: drop-shadow(0 0 30px rgba(127,90,240,0.6));
}

@keyframes brainPulse {
    0%,100% { transform: scale(1) rotate(-3deg);  filter: drop-shadow(0 0 20px rgba(127,90,240,0.5)); }
    50%     { transform: scale(1.1) rotate(3deg); filter: drop-shadow(0 0 45px rgba(127,90,240,0.9)); }
}

.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 0.5rem;
    background: linear-gradient(135deg, #c4b5fd 0%, #fffffe 45%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeDown 0.9s ease 0.1s both;
}

.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
    animation: fadeDown 1s ease 0.2s both;
    margin-bottom: 0.5rem;
}

@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 1.5rem 0;
    animation: fadeDown 1s ease 0.3s both;
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #a78bfa;
}
.stat-label {
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, #7f5af055 20%, #2cb67d55 80%, transparent 100%);
    margin: 1.5rem 0;
    position: relative;
}
.glow-divider::after {
    content: '';
    position: absolute;
    top: -1px; left: 30%; right: 30%;
    height: 3px;
    background: linear-gradient(90deg, #7f5af0, #2cb67d);
    border-radius: 999px;
    filter: blur(4px);
}

.sec-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7f5af0;
    margin: 2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.sec-label span {
    background: rgba(127,90,240,0.12);
    border: 1px solid rgba(127,90,240,0.25);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(127,90,240,0.3), transparent);
}

.stSelectbox label, .stSlider label, .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.04em;
}

div[data-baseweb="select"] > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    transition: border-color 0.2s !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #7f5af0 !important;
    box-shadow: 0 0 0 3px rgba(127,90,240,0.12) !important;
}

div[data-testid="stRadio"] > div { gap: 0.5rem; flex-wrap: wrap; }
div[data-testid="stRadio"] label {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.35rem 0.9rem !important;
    color: var(--muted) !important;
    transition: all 0.2s !important;
    font-size: 0.85rem !important;
}
div[data-testid="stRadio"] label:hover {
    border-color: #7f5af0 !important;
    color: var(--text) !important;
    background: rgba(127,90,240,0.08) !important;
}

div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #7f5af0 0%, #2cb67d 100%);
    color: white;
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.9rem 2rem;
    border: none;
    border-radius: 14px;
    cursor: pointer;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    transition: all 0.3s;
    margin-top: 1.5rem;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(127,90,240,0.45), 0 4px 12px rgba(44,182,125,0.2);
}
div[data-testid="stButton"] > button:active { transform: translateY(-1px); }

.result-box {
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
    animation: resultReveal 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
}

.result-treatment {
    background: linear-gradient(135deg, #1a0f3a 0%, #0f1628 100%);
    border: 1px solid rgba(127,90,240,0.5);
    box-shadow: 0 0 60px rgba(127,90,240,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
.result-ok {
    background: linear-gradient(135deg, #0a2419 0%, #0f1628 100%);
    border: 1px solid rgba(44,182,125,0.5);
    box-shadow: 0 0 60px rgba(44,182,125,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
}

@keyframes resultReveal {
    from { opacity: 0; transform: scale(0.88) translateY(20px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}

.result-icon  { font-size: 4.5rem; display: block; margin-bottom: 0.6rem; }
.result-title { font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.4rem; }
.result-treatment .result-title { color: #c4b5fd; }
.result-ok .result-title        { color: #6ee7b7; }

.result-desc {
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 300;
    max-width: 340px;
    margin: 0 auto 1.2rem;
    line-height: 1.6;
}

.conf-wrap {
    background: rgba(0,0,0,0.3);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.conf-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.conf-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.08em;
}
.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
}
.result-treatment .conf-pct { color: #a78bfa; }
.result-ok .conf-pct        { color: #2cb67d; }

.conf-track {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}
.conf-fill-t {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7f5af0, #c4b5fd);
    box-shadow: 0 0 8px rgba(127,90,240,0.6);
}
.conf-fill-ok {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2cb67d, #6ee7b7);
    box-shadow: 0 0 8px rgba(44,182,125,0.6);
}

.disclaimer {
    background: rgba(255,137,6,0.06);
    border: 1px solid rgba(255,137,6,0.2);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-top: 1.2rem;
    font-size: 0.78rem;
    color: #a8956a;
    line-height: 1.5;
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
}

.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}
.footer-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #2a3045;
    letter-spacing: 0.1em;
}
.footer-dots { display: flex; justify-content: center; gap: 0.4rem; margin-top: 0.6rem; }
.footer-dot  { width: 4px; height: 4px; border-radius: 50%; background: var(--border); }
.footer-dot:nth-child(2) { background: #7f5af044; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model    = joblib.load(os.path.join(BASE_DIR, 'mental_health_model.pkl'))
    scaler   = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded  = True
except Exception as e:
    model_loaded  = False
    error_msg     = str(e)

# ─── HERO ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI POWERED &nbsp;·&nbsp; VOTING CLASSIFIER</div>
    <span class="hero-brain">🧠</span>
    <h1>Mental Health<br>Crisis Predictor</h1>
    <p class="hero-sub">Answer a few questions — our model analyzes your risk profile instantly</p>
    <div class="stats-row">
        <div class="stat-item">
            <div class="stat-num">292K</div>
            <div class="stat-label">Records Trained</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">6</div>
            <div class="stat-label">ML Models</div>
        </div>
        
    </div>
</div>
<div class="glow-divider"></div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Model load nahi hua: {error_msg}")
    st.stop()

# ─── FORM ────────────────────────────────────────────────────
st.markdown('<div class="sec-label"><span>01</span> Personal Info</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    gender     = st.selectbox("Gender", ["Female", "Male"])
with col2:
    occupation = st.selectbox("Occupation", ["Corporate", "Self-Employed", "Student", "Housewife", "Others"])

col3, col4 = st.columns(2)
with col3:
    self_employed  = st.radio("Self Employed?", ["No", "Yes"], horizontal=True)
with col4:
    family_history = st.radio("Family History of Mental Illness?", ["No", "Yes"], horizontal=True)

st.markdown('<div class="sec-label"><span>02</span> Stress & Behavior</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    growing_stress = st.selectbox("Growing Stress?",    ["No", "Yes", "Maybe"])
with col6:
    changes_habits = st.selectbox("Changes in Habits?", ["No", "Yes", "Maybe"])

col7, col8 = st.columns(2)
with col7:
    coping_struggles = st.radio("Coping Struggles?", ["No", "Yes"], horizontal=True)
with col8:
    mood_swings = st.select_slider("Mood Swings", options=["Low", "Medium", "High"])

col9, col10 = st.columns(2)
with col9:
    work_interest   = st.selectbox("Interest in Work?", ["No", "Yes", "Maybe"])
with col10:
    social_weakness = st.selectbox("Social Weakness?",  ["No", "Yes", "Maybe"])

st.markdown('<div class="sec-label"><span>03</span> Lifestyle & Awareness</div>', unsafe_allow_html=True)

col11, col12 = st.columns(2)
with col11:
    days_indoors = st.select_slider(
        "Days Spent Indoors",
        options=["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"]
    )
with col12:
    care_options = st.selectbox("Aware of Care Options?", ["No", "Not sure", "Yes"])

mental_health_history   = st.selectbox("Mental Health History?",          ["No", "Yes", "Maybe"])
mental_health_interview = st.selectbox("Comfortable Discussing at Work?", ["No", "Yes", "Maybe"])

# ─── MAPPINGS ────────────────────────────────────────────────
binary_map = {"Yes": 1, "No": 0}
tri_map    = {"Yes": 1, "No": 0, "Maybe": 0.5}
mood_map   = {"Low": 0, "Medium": 1, "High": 2}
days_map   = {"Go out Every day": 0, "1-14 days": 1, "15-30 days": 2, "31-60 days": 3, "More than 2 months": 4}
care_map   = {"No": 0, "Not sure": 0.5, "Yes": 1}
occ_map    = {"Corporate": 0, "Housewife": 1, "Others": 2, "Self-Employed": 3, "Student": 4}
gender_map = {"Female": 0, "Male": 1}

# ─── PREDICT ─────────────────────────────────────────────────
predict_btn = st.button("⚡ Analyze My Mental Health Risk")

if predict_btn:
    g    = gender_map[gender]
    occ  = occ_map[occupation]
    se   = binary_map[self_employed]
    fh   = binary_map[family_history]
    gs   = tri_map[growing_stress]
    ch   = tri_map[changes_habits]
    mhi  = tri_map[mental_health_history]
    ms   = mood_map[mood_swings]
    cs   = binary_map[coping_struggles]
    wi   = tri_map[work_interest]
    sw   = tri_map[social_weakness]
    mhi2 = tri_map[mental_health_interview]
    co   = care_map[care_options]
    di   = days_map[days_indoors]

    stress_score       = gs + ms + cs
    behavioral_score   = ch + wi + sw
    awareness_score    = mhi + mhi2 + co
    stress_x_family    = stress_score    * fh
    care_x_family      = co              * fh
    awareness_x_family = awareness_score * fh
    gender_x_stress    = g               * stress_score
    high_risk_flag     = int((stress_score >= 3) and (behavioral_score >= 4) and (fh == 1))

    features = np.array([[
        g, occ, se, fh, di,
        gs, ch, mhi, ms, cs, wi, sw, mhi2, co,
        stress_score, behavioral_score, awareness_score,
        stress_x_family, care_x_family, awareness_x_family,
        gender_x_stress, high_risk_flag
    ]])

    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    proba           = model.predict_proba(features_scaled)[0]
    confidence      = round(proba[prediction] * 100, 1)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box result-treatment">
            <span class="result-icon">🔴</span>
            <div class="result-title">Treatment Recommended</div>
            <p class="result-desc">Our model detected indicators suggesting professional mental health support could be beneficial for you.</p>
            <div class="conf-wrap">
                <div class="conf-top">
                    <span class="conf-text">MODEL CONFIDENCE</span>
                    <span class="conf-pct">{confidence}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill-t" style="width:{confidence}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-ok">
            <span class="result-icon">🟢</span>
            <div class="result-title">No Treatment Needed</div>
            <p class="result-desc">Based on your responses, no immediate mental health intervention appears necessary at this time.</p>
            <div class="conf-wrap">
                <div class="conf-top">
                    <span class="conf-text">MODEL CONFIDENCE</span>
                    <span class="conf-pct">{confidence}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill-ok" style="width:{confidence}%"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <span>⚠️</span>
        <span><strong>Disclaimer:</strong> This tool is for educational purposes only and does not
        replace professional medical advice. Please consult a qualified mental health professional
        for proper diagnosis and treatment.</span>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-text">MENTAL HEALTH CRISIS PREDICTOR &nbsp;·&nbsp; VOTING CLASSIFIER &nbsp;·&nbsp; 292K RECORDS</div>
    <div class="footer-dots">
        <div class="footer-dot"></div>
        <div class="footer-dot"></div>
        <div class="footer-dot"></div>
    </div>
</div>
""", unsafe_allow_html=True)
