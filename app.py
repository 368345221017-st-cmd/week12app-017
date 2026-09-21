import streamlit as st
import pandas as pd
import joblib
import os

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 60%, #0d2137 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '🚢'; font-size: 5rem;
    position: absolute; top: -10px; right: 20px; opacity: 0.12;
  }
  .hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; color: #e8d5b7; margin: 0 0 0.5rem; line-height: 1.2;
  }
  .hero p { color: #8aafc7; font-size: 0.9rem; margin: 0; }

  .result-survived {
    background: #0d2b1f; border-left: 4px solid #2ecc71;
    border-radius: 8px; padding: 1.2rem 1.5rem;
    color: #a8f0c8; font-size: 1.1rem; font-weight: 600;
  }
  .result-died {
    background: #2b0d0d; border-left: 4px solid #e74c3c;
    border-radius: 8px; padding: 1.2rem 1.5rem;
    color: #f0a8a8; font-size: 1.1rem; font-weight: 600;
  }
  .info-box {
    background: #1a2535; border-radius: 8px;
    padding: 0.8rem 1rem; color: #7fa8c8;
    font-size: 0.82rem; margin-top: 1.5rem;
  }
  div[data-testid="stForm"] {
    background: #f8fafc; border-radius: 12px;
    padding: 1.5rem; border: 1px solid #e2e8f0;
  }
  .stButton > button {
    width: 100%; background: #1a3a5c; color: white;
    font-weight: 600; border: none; border-radius: 8px;
    padding: 0.65rem 0; font-size: 1rem;
  }
  .stButton > button:hover { background: #0d2137; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Titanic Survival Predictor</h1>
  <p>กรอกข้อมูลผู้โดยสาร แล้วให้โมเดล Decision Tree ทำนายโอกาสรอดชีวิต</p>
</div>
""", unsafe_allow_html=True)

# ─── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "titanic_model.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error(f"ไม่พบไฟล์ `{MODEL_PATH}` — วางไฟล์โมเดลในโฟลเดอร์เดียวกับ app.py แล้วรีเฟรช")
    st.stop()

# ─── Input form ────────────────────────────────────────────────────────────────
# โมเดลนี้ใช้ feature: Pclass, Sex_female, Age, Fare, FamilySize
with st.form("predict_form"):
    st.subheader("ข้อมูลผู้โดยสาร")

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("ชั้นโดยสาร (Pclass)", [1, 2, 3],
                              format_func=lambda x: f"ชั้น {x}")
        sex    = st.radio("เพศ (Sex)", ["ชาย", "หญิง"], horizontal=True)
        age    = st.slider("อายุ (Age)", 1, 80, 28)

    with col2:
        sibsp  = st.number_input("พี่น้อง / คู่สมรสบนเรือ (SibSp)", 0, 8, 0)
        parch  = st.number_input("พ่อแม่ / บุตรบนเรือ (Parch)", 0, 6, 0)
        fare   = st.number_input("ราคาตั๋ว (Fare, £)", 0.0, 600.0, 32.2, step=0.5)

    # แสดง FamilySize ที่คำนวณอัตโนมัติ
    family_size = sibsp + parch + 1
    st.caption(f"FamilySize (คำนวณอัตโนมัติ = SibSp + Parch + 1) = **{family_size}**")

    submitted = st.form_submit_button("🔍 ทำนาย")

# ─── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # แปลง Sex → Sex_female (1 = หญิง, 0 = ชาย) ให้ตรงกับที่โมเดลเทรนมา
    sex_female = 1 if sex == "หญิง" else 0

    # สร้าง DataFrame ให้ตรงกับ feature_names_in_ ของโมเดล
    input_df = pd.DataFrame([{
        "Pclass":     pclass,
        "Sex_female": sex_female,
        "Age":        age,
        "Fare":       fare,
        "FamilySize": family_size,
    }])

    try:
        pred  = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        survive_pct = round(proba[1] * 100, 1)
        die_pct     = round(proba[0] * 100, 1)

        st.markdown("---")
        if pred == 1:
            st.markdown(
                f'<div class="result-survived">✅ รอดชีวิต &nbsp;—&nbsp; ความน่าจะเป็น {survive_pct}%</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="result-died">❌ ไม่รอดชีวิต &nbsp;—&nbsp; ความน่าจะเป็น {die_pct}%</div>',
                unsafe_allow_html=True)

        st.markdown("**ความน่าจะเป็น**")
        prob_df = pd.DataFrame({
            "ผลลัพธ์": ["รอดชีวิต", "ไม่รอดชีวิต"],
            "ความน่าจะเป็น (%)": [survive_pct, die_pct]
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

        with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-box">
  <b>Feature ที่โมเดลใช้:</b>
  Pclass · Sex_female (0/1) · Age · Fare · FamilySize (SibSp + Parch + 1)
</div>
""", unsafe_allow_html=True)
