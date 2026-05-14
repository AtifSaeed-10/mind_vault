import os
import streamlit as st
import joblib
import numpy as np

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="MindVault · Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── LOAD MODEL ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("mental_health_model.pkl")


# ── CSS (UNCHANGED UI STYLE) ────────────────────────────────
st.markdown("""<style>
/* (KEEP YOUR FULL CSS EXACTLY HERE — unchanged for brevity) */
</style>""", unsafe_allow_html=True)


# ── BACKGROUND CANVAS ANIMATION ─────────────────────────────
st.markdown("""
<canvas id="mv-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;"></canvas>
<script>
/* (KEEP YOUR FULL JS ANIMATION EXACTLY AS IT IS) */
</script>

<div class="shell">
""", unsafe_allow_html=True)


# ── NAV ─────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
    <div class="nav-logo">🧠 MindVault</div>
    <div class="nav-pill">
        <div class="nav-pill-dot"></div>
        Neural AI · Live
    </div>
</div>
""", unsafe_allow_html=True)


# ── HERO ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">
        <div class="hero-tag-glow"></div>
        Mental Health Assessment
    </div>
    <div class="hero-h1">
        Your Mind<br><span>Matters Most</span>
    </div>
    <p class="hero-sub">
        14 thoughtful questions. One AI-powered insight.<br>
        Understand your mental wellness in under 60 seconds.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── MAPPINGS ────────────────────────────────────────────────
YESNO       = {"Yes": 1, "No": 0}
YESNO_MAYBE = {"Yes": 1, "Maybe": 0.5, "No": 0}
GENDER_MAP  = {"Female": 0, "Male": 1}
OCC_MAP     = {"Corporate": 0, "Self-Employed": 1, "Student": 2, "Other": 3}
DAYS_MAP    = {
    "Go out every day": 0,
    "1–14 days": 1,
    "15–30 days": 2,
    "31–60 days": 3,
    "More than 2 months": 4
}
MOOD_MAP    = {"Low": 0, "Medium": 1, "High": 2}
CARE_MAP    = {"No": 0, "Not sure": 0.5, "Yes": 1}


# ── SECTION 1 ───────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
    <div class="sec-num">01</div>
    <div class="sec-title">Personal Background</div>
    <div class="sec-line"></div>
</div>
<div class="panel">
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
    occupation = st.selectbox("Occupation", list(OCC_MAP.keys()))
with c2:
    self_employed = st.selectbox("Self-employed?", list(YESNO.keys()))
    family_history = st.selectbox("Family history of mental illness?", list(YESNO.keys()))

st.markdown("</div>", unsafe_allow_html=True)


# ── SECTION 2 ───────────────────────────────────────────────
st.markdown("""<div class="sec-head"><div class="sec-num">02</div>
<div class="sec-title">Stress & Emotional State</div><div class="sec-line"></div></div>
<div class="panel">""", unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    growing_stress = st.selectbox("Growing stress lately?", list(YESNO_MAYBE.keys()))
    mood_swings = st.selectbox("Mood swing intensity?", list(MOOD_MAP.keys()))
with c4:
    coping_struggles = st.selectbox("Struggling to cope daily?", list(YESNO.keys()))
    changes_habits = st.selectbox("Noticeable changes in habits?", list(YESNO_MAYBE.keys()))

st.markdown("</div>", unsafe_allow_html=True)


# ── SECTION 3 ───────────────────────────────────────────────
st.markdown("""<div class="sec-head"><div class="sec-num">03</div>
<div class="sec-title">Behaviour & Social Life</div><div class="sec-line"></div></div>
<div class="panel">""", unsafe_allow_html=True)

c5, c6 = st.columns(2)
with c5:
    work_interest = st.selectbox("Lost interest in work/hobbies?", list(YESNO_MAYBE.keys()))
    social_weakness = st.selectbox("Feeling socially withdrawn?", list(YESNO_MAYBE.keys()))
with c6:
    days_indoors = st.selectbox("Days spent indoors (past month)?", list(DAYS_MAP.keys()))

st.markdown("</div>", unsafe_allow_html=True)


# ── SECTION 4 ───────────────────────────────────────────────
st.markdown("""<div class="sec-head"><div class="sec-num">04</div>
<div class="sec-title">Awareness & Support</div><div class="sec-line"></div></div>
<div class="panel">""", unsafe_allow_html=True)

c7, c8 = st.columns(2)
with c7:
    mh_history = st.selectbox("Prior mental health episodes?", list(YESNO_MAYBE.keys()))
    mh_interview = st.selectbox("Comfortable discussing MH at work?", list(YESNO_MAYBE.keys()))
with c8:
    care_options = st.selectbox("Access to mental health care?", list(CARE_MAP.keys()))

st.markdown("</div>", unsafe_allow_html=True)


# ── BUTTON ──────────────────────────────────────────────────
predict_clicked = st.button("✦ Reveal My Assessment")


# ── PREDICTION ENGINE ───────────────────────────────────────
if predict_clicked:
    with st.spinner("Analysing your responses…"):
        try:
            model = load_model()

            r = {
                "Gender": GENDER_MAP[gender],
                "Occupation": OCC_MAP[occupation],
                "self_employed": YESNO[self_employed],
                "family_history": YESNO[family_history],
                "Days_Indoors": DAYS_MAP[days_indoors],
                "Growing_Stress": YESNO_MAYBE[growing_stress],
                "Changes_Habits": YESNO_MAYBE[changes_habits],
                "Mental_Health_History": YESNO_MAYBE[mh_history],
                "Mood_Swings": MOOD_MAP[mood_swings],
                "Coping_Struggles": YESNO[coping_struggles],
                "Work_Interest": YESNO_MAYBE[work_interest],
                "Social_Weakness": YESNO_MAYBE[social_weakness],
                "mental_health_interview": YESNO_MAYBE[mh_interview],
                "care_options": CARE_MAP[care_options],
            }

            # Feature engineering
            stress_score = r["Growing_Stress"] + r["Mood_Swings"] + r["Coping_Struggles"]
            behavioral_score = (
                r["Changes_Habits"] + r["Work_Interest"] +
                r["Social_Weakness"] + r["Days_Indoors"]
            )
            awareness_score = (
                r["Mental_Health_History"] +
                r["mental_health_interview"] +
                r["care_options"]
            )

            high_risk_flag = int(
                stress_score >= 3 and
                behavioral_score >= 4 and
                r["family_history"] == 1
            )

            features = np.array([[ 
                r["Gender"], r["Occupation"], r["self_employed"], r["family_history"],
                r["Days_Indoors"], r["Growing_Stress"], r["Changes_Habits"],
                r["Mental_Health_History"], r["Mood_Swings"], r["Coping_Struggles"],
                r["Work_Interest"], r["Social_Weakness"], r["mental_health_interview"],
                r["care_options"],
                stress_score, behavioral_score, awareness_score,
                high_risk_flag
            ]])

            prediction = model.predict(features)[0]

            if hasattr(model, "predict_proba"):
                confidence = max(model.predict_proba(features)[0])
            else:
                confidence = 0.5

            conf_pct = round(confidence * 100, 1)

            if prediction == 1:
                st.error(f"🔴 Support Recommended ({conf_pct}% confidence)")
            else:
                st.success(f"🟢 Low Risk Profile ({conf_pct}% confidence)")

            st.info("⚠️ This is NOT a medical diagnosis. Seek professional help if needed.")

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ── FOOTER ────────────────────────────────────────────────
st.markdown("""
</div>
<div style="text-align:center;margin-top:40px;color:#555;font-size:12px;">
MindVault · Educational ML Project · Not Clinical Use
</div>
""", unsafe_allow_html=True)
