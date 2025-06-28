import streamlit as st
import requests
import plotly.graph_objects as go
import base64
import time

# 🧠 Diccionarios codificados
marital_map = {
    "Single": 1,
    "Married": 2,
    "Divorced": 3,
    "Widowed": 4,
    "Separated": 5,
    "Common-Law Union": 6,
    "Other/Unknown": 7
}
occupation_map = {
    "Public Sector": 1,
    "Private Sector": 2,
    "Self-Employed": 3,
    "Merchant / Retail": 4,
    "Technician / Skilled Labor": 5
}
residence_map = {
    "Owned": 1,
    "Rented": 2,
    "Family": 3,
    "Company-Provided": 4,
    "Other": 5
}

# 🎯 Gauge animado (¡debe ir arriba del uso!)
def animated_gauge(final_value, risk_class):
    ph = st.empty()
    for val in range(0, int(final_value) + 1, 2):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>Risk Level</b><br><span style='font-size:0.8em'>{risk_class}</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#34C759" if val < 35 else "#F4D03F" if val < 70 else "#C0392B"},
                "steps": [
                    {"range": [0, 35], "color": "#1D8348"},
                    {"range": [35, 70], "color": "#F4D03F"},
                    {"range": [70, 100], "color": "#C0392B"},
                ],
            }
        ))
        ph.plotly_chart(fig, use_container_width=True)
        time.sleep(0.05)

# 🌍 Redirección GET
qs = st.query_params
if qs.get("start") == "true":
    st.session_state.show_form = True

st.set_page_config(page_title="RiskCore", layout="wide")
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# 🎨 Estilos
with open("logo_intro.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_of_bin_file("assets/logo.png")

# 🚀 Splash
if not st.session_state.show_form:
    st.markdown(f"""
        <div id="intro-logo-container">
            <img id="intro-logo" src="data:image/png;base64,{img_base64}">
            <a href="?start=true"><button kind="primary">RiskCore Enter</button></a>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# 🔷 Encabezado
st.markdown(f"""
    <div id="header-branding">
        <img src="data:image/png;base64,{img_base64}" class="logo">
        <span class="app-title-gradient">RiskCore</span>
    </div>
""", unsafe_allow_html=True)

# 📋 Formulario
col_inputs, col_output = st.columns([1.4, 1])
with col_inputs:
    st.subheader("📋 Applicant Profile")
    with st.form("user_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 80, 30)
            sex = st.selectbox("Sex", ["M", "F"])
            marital = st.selectbox("Marital Status", list(marital_map.keys()))
            occupation = st.selectbox("Occupation", list(occupation_map.keys()))
            months_res = st.number_input("Months at Current Residence", 0, 240, 36)
            residence_type = st.selectbox("Type of Residence", list(residence_map.keys()))
            flag_phone = st.selectbox("Has Landline Phone?", ["Y", "N"])
            product = st.selectbox("Requested Product", ["Mortgage Loan", "Consumer Credit", "Vehicle Loan", "Credit Card"])
        with c2:
            state_birth = st.selectbox("State of Birth", ["NY", "CA", "IL", "TX", "AZ", "PA"])
            residencial_state = st.selectbox("State of Residence", ["NY", "CA", "IL", "TX", "AZ", "PA"])
            residencial_city = st.selectbox("City of Residence", ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"])
            residencial_borough = st.text_input("Neighborhood", "Downtown")
            phone_area = st.selectbox("Phone Area Code", [212, 213, 312, 713, 602, 215])
            zip3 = st.number_input("Residential ZIP (3 digits)", 100, 999, 110)
            profesional_state = st.selectbox("Workplace State", ["NY", "CA", "IL", "TX", "AZ", "PA"])
            profesional_zip = st.number_input("Workplace ZIP", 100, 999, 111)

        with c2:
            st.markdown("<div class='center-button'>", unsafe_allow_html=True)
            submitted = st.form_submit_button("📊 Evaluate Risk")
            st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.subheader("📈 Credit Risk Assessment")

    if submitted:
        with st.spinner("⏳ Evaluating applicant..."):
            marital_code = marital_map.get(str(marital).strip(), 7)
            occupation_code = occupation_map.get(str(occupation).strip(), 1)
            residence_code = residence_map.get(str(residence_type).strip(), 5)

            payload = {
                "AGE": age,
                "SEX": sex,
                "MARITAL_STATUS": marital_code,
                "OCCUPATION_TYPE": occupation_code,
                "MONTHS_IN_RESIDENCE": months_res,
                "FLAG_RESIDENCIAL_PHONE": flag_phone,
                "STATE_OF_BIRTH": state_birth,
                "RESIDENCIAL_STATE": residencial_state,
                "RESIDENCE_TYPE": residence_code,
                "RESIDENCIAL_CITY": residencial_city,
                "RESIDENCIAL_BOROUGH": residencial_borough,
                "RESIDENCIAL_PHONE_AREA_CODE": phone_area,
                "RESIDENCIAL_ZIP_3": zip3,
                "PROFESSIONAL_STATE": profesional_state,
                "PROFESSIONAL_ZIP_3": profesional_zip,
                "PRODUCT": product
            }

            try:
                res = requests.post("http://localhost:8000/predict", json=payload)
                result = res.json()
                risk_percent = result["risk_percentage"]
                risk_class = result["risk_class"]
                st.success(f"🧾 Predicted Risk: {risk_percent}%  |  Class: {risk_class}")
                animated_gauge(risk_percent, risk_class)
            except Exception as e:
                st.error("❌ Error connecting to the API.")
                st.code(str(e), language="bash")
