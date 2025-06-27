import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import base64

# 🌍 Redirección al formulario usando parámetros GET
qs = st.query_params
if qs.get("start") == "true":
    st.session_state.show_form = True

# ⚙️ Configuración general
st.set_page_config(page_title="RiskCore", layout="wide")

if "show_form" not in st.session_state:
    st.session_state.show_form = False

# 🎨 Estilos
with open("logo_intro.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 🖼️ Logo base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_of_bin_file("assets/logo.png")

# 🚀 Splash con redirección real (solo se muestra si no se ha entrado)
if not st.session_state.show_form:
    st.markdown(f"""
        <div id="intro-logo-container">
            <img id="intro-logo" src="data:image/png;base64,{img_base64}">
            <a href="?start=true">
                <button kind="primary">RiskCore Enter</button>
            </a>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# 🔷 Encabezado animado
st.markdown(f"""
    <div id="header-branding">
        <img src="data:image/png;base64,{img_base64}" class="logo">
        <span class="app-title-gradient">RiskCore</span>
    </div>
""", unsafe_allow_html=True)

# 📋 Formulario y resultado
col_inputs, col_output = st.columns([1.4, 1])

with col_inputs:
    st.subheader("📋 Applicant Profile")
    with st.form("user_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 80, 30)
            sex = st.selectbox("Sex", ["Male", "Female"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Common-Law Union", "Divorced"])
            occupation = st.selectbox("Occupation", ["Public Employee", "Private Employee", "Self-Employed", "Merchant", "Technician", "Technologist"])
            months_res = st.number_input("Months at Current Residence", 0, 240, 36)
            residence_type = st.selectbox("Type of Residence", ["Owned", "Rented", "Family", "Company-Provided"])
            flag_phone = st.selectbox("Has Landline Phone?", [0, 1])
            product = st.selectbox("Requested Product", ["Mortgage Loan", "Consumer Credit", "Vehicle Loan", "Credit Card"])

        with c2:
            state_birth = st.selectbox("State of Birth", ["Bogotá", "Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Tolima", "Atlántico"])
            residencial_state = st.selectbox("State of Residence", ["Bogotá", "Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Tolima", "Atlántico"])
            residencial_city = st.text_input("City of Residence", "Chía")
            residencial_borough = st.text_input("Neighborhood", "Santa María")
            phone_area = st.selectbox("Phone Area Code", [1, 2, 4, 5, 7])
            zip3 = st.number_input("Residential ZIP (3 digits)", 100, 999, 110)
            profesional_state = st.selectbox("Workplace State", ["Bogotá", "Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Tolima", "Atlántico"])
            profesional_zip = st.number_input("Workplace ZIP", 100, 999, 111)

        with c2:
            # Botón centrado relativo a ambas columnas
            st.markdown("<div class='center-button'>", unsafe_allow_html=True)
            submitted = st.form_submit_button("📊 Evaluate Risk")
            st.markdown("</div>", unsafe_allow_html=True)

    
with col_output:
    st.subheader("📈 Credit Risk Assessment")

    if submitted:
        with st.spinner("⏳ Analyzing applicant profile..."):
            payload = {
                "AGE": age,
                "SEX": sex,
                "MARITAL_STATUS": marital,
                "OCCUPATION_TYPE": occupation,
                "MONTHS_IN_RESIDENCE": months_res,
                "FLAG_RESIDENCIAL_PHONE": flag_phone,
                "STATE_OF_BIRTH": state_birth,
                "RESIDENCIAL_STATE": residencial_state,
                "RESIDENCE_TYPE": residence_type,
                "RESIDENCIAL_CITY": residencial_city,
                "RESIDENCIAL_BOROUGH": residencial_borough,
                "RESIDENCIAL_PHONE_AREA_CODE": phone_area,
                "RESIDENCIAL_ZIP_3": zip3,
                "PROFESSIONAL_STATE": profesional_state,
                "PROFESSIONAL_ZIP_3": profesional_zip,
                "PRODUCT": product
            }

            try:
                res =  requests.post("http://api:8000/model/predict", json=payload)
                result = res.json()
                risk_percent = result["risk_percentage"]
                risk_class = result["risk_class"]
                translations = {
                    "Alto": "High",
                    "Medio": "Medium",
                    "Bajo": "Low"
            }
                risk_class = translations.get(risk_class, risk_class)

                st.success(f"🧾 Predicted Risk: {risk_percent}%  |  Class: {risk_class}")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percent,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": f"<b>Risk Level</b><br><span style='font-size:0.8em'>{risk_class}</span>"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#34C759" if risk_percent < 35 else "#F4D03F" if risk_percent < 70 else "#C0392B"},
                        "steps": [
                            {"range": [0, 35], "color": "#1D8348"},
                            {"range": [35, 70], "color": "#F4D03F"},
                            {"range": [70, 100], "color": "#C0392B"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error("❌ Error connecting to the API.")
                st.code(str(e), language="bash")
