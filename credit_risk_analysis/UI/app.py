import streamlit as st
import requests
import plotly.graph_objects as go
import base64
import time
import pandas as pd

# Diccionarios codificados
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
product_map = {
    "Mortgage Loan": 1,
    "Vehicle Loan": 2,
    "Student Loan": 7
}


# Gauge animado
def animated_gauge(final_value, risk_class):
    ph = st.empty()
    bar_color = "#00FF99"
    for val in range(0, int(final_value), 2):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>Risk Level</b><br><span style='font-size:0.8em'>{risk_class}</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 35], "color": "#1D8348"},
                    {"range": [35, 70], "color": "#F4D03F"},
                    {"range": [70, 100], "color": "#C0392B"},
                ],
            }
        ))
        ph.plotly_chart(fig, use_container_width=True)
        time.sleep(0.05)

    final_color = "#1D8348" if final_value < 35 else "#F4D03F" if final_value < 70 else "#C0392B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_value,
        number={
            "valueformat": ".1f",
            "suffix": "%",
            "font": {
                "size": 70,
                "color": final_color,
                "weight": "bold"
            }
        },
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>Risk Level</b><br><span style='font-size:0.8em'>{risk_class}</span>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 35], "color": "#1D8348"},
                {"range": [35, 70], "color": "#F4D03F"},
                {"range": [70, 100], "color": "#C0392B"},
            ],
        }
    ))
    ph.plotly_chart(fig, use_container_width=True)


# Redirección GET
qs = st.query_params
if qs.get("start") == "true":
    st.session_state.show_form = True

st.set_page_config(page_title="RiskCore", layout="wide")
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# Estilos
with open("logo_intro.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


img_base64 = get_base64_of_bin_file("assets/logo.png")

# Splash
if not st.session_state.show_form:
    st.markdown(f"""
        <div id=\"intro-logo-container\">
            <img id=\"intro-logo\" src=\"data:image/png;base64,{img_base64}\">
            <a href=\"?start=true\"><button kind=\"primary\">RiskCore Enter</button></a>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Encabezado
st.markdown(f"""
    <div id=\"header-branding\">
        <img src=\"data:image/png;base64,{img_base64}\" class=\"logo\">
        <span class=\"app-title-gradient\">RiskCore</span>
    </div>
""", unsafe_allow_html=True)

# Formulario
col_inputs, col_output = st.columns([1.4, 1])
with st.form("user_form"):
    with col_inputs:
        st.subheader("\U0001F4CB Applicant Profile")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 80, 30)
            sex = st.selectbox("Sex", ["M", "F"])
            marital = st.selectbox("Marital Status", list(marital_map.keys()))
            occupation = st.selectbox("Occupation", list(occupation_map.keys()))
            months_res = st.number_input("Months at Current Residence", 0, 240, 36)
            residence_type = st.selectbox("Type of Residence", list(residence_map.keys()))
            flag_phone = st.selectbox("Has Landline Phone?", ["Y", "N"])
            product = st.selectbox("Requested Product", ["Mortgage Loan", "Vehicle Loan", "Student Loan"])
        with c2:
            state_birth = st.selectbox("State of Birth", ["SP", "RJ", "MG", "BA", "RS", "PE"])
            residencial_state = st.selectbox("State of Residence", ["SP", "RJ", "MG", "BA", "RS", "PE"])
            residencial_city = st.selectbox("City of Residence",
                                            ["São Paulo", "Rio de Janeiro", "Minas Gerais", "Bahia",
                                             "Rio Grande do Sul", "Pernambuco"])
            residencial_borough = st.text_input("Neighborhood", "Downtown")
            phone_area = st.selectbox("Phone Area Code", [212, 213, 312, 713, 602, 215])
            zip3 = st.number_input("Residential ZIP (3 digits)", 100, 999, 110)
            profesional_state = st.selectbox("Workplace State", ["SP", "RJ", "MG", "BA", "RS", "PE"])
            profesional_zip = st.number_input("Workplace ZIP (3 digits)", 100, 999, 111)

    with col_output:
        st.subheader("\U0001F4C8 Credit Risk Assessment")
        st.markdown("""
            <style>
            .custom-button {
                background-color: #28a745;
                color: white;
                font-size: 1.5rem;
                padding: 0.8rem 2rem;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: background-color 0.3s ease, transform 0.1s ease;
            }
            .custom-button:hover {
                background-color: #218838;
            }
            .custom-button:active {
                transform: scale(0.98);
                box-shadow: 0 2px 6px rgba(0,0,0,0.2) inset;
            }
            </style>
            <div style='text-align:center; margin-top:1rem; margin-bottom:1rem;'>

            </div>
        """, unsafe_allow_html=True)

    submitted = st.form_submit_button("📊 Evaluate Risk", use_container_width=True)

if submitted:
    with col_output:
        with st.spinner("⏳ Evaluating applicant..."):
            try:
                payload = {
                    "AGE": age,
                    "SEX": sex,
                    "MARITAL_STATUS": marital_map.get(marital, 7),
                    "OCCUPATION_TYPE": occupation_map.get(occupation, 1),
                    "MONTHS_IN_RESIDENCE": months_res,
                    "FLAG_RESIDENCIAL_PHONE": flag_phone,
                    "STATE_OF_BIRTH": state_birth,
                    "RESIDENCIAL_STATE": residencial_state,
                    "RESIDENCE_TYPE": residence_map.get(residence_type, 5),
                    "RESIDENCIAL_CITY": residencial_city,
                    "RESIDENCIAL_BOROUGH": residencial_borough,
                    "RESIDENCIAL_PHONE_AREA_CODE": phone_area,
                    "RESIDENCIAL_ZIP_3": zip3,
                    "PROFESSIONAL_STATE": profesional_state,
                    "PROFESSIONAL_ZIP_3": profesional_zip,
                    "PRODUCT": product_map.get(product, 0)
                }

                res = requests.post("http://credit-risk-api:8000/model/predict", json=payload)
                result = res.json()

                risk_percent = result["risk_percentage"]
                risk_class = result["risk_class"]

                with col_output:
                    animated_gauge(risk_percent, risk_class)

            except Exception as e:
                st.error("❌ Error connecting to the API.")
                st.code(str(e), language="bash")

import json

with st.container():
    st.markdown("""
        <style>
        .hist-button {
            background: none;
            border: none;
            color: #555;
            cursor: pointer;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
        }
        .hist-button:hover {
            color: #000;
        }
        .hist-icon {
            margin-right: 6px;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.expander("🕒 Histórico de predicciones", expanded=False):
        try:
            response = requests.get("http://credit-risk-api:8000/model/predictions")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    data = [data]
                rows = []
                for item in data:
                    parsed_json = json.loads(item["request_json"])
                    parsed_json["prediction_date"] = f"**{item['prediction_date']}**"
                    parsed_json["score"] = item["score"]
                    parsed_json["model"] = item["model"]
                    rows.append(parsed_json)
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No se pudo obtener el histórico de predicciones.")
        except Exception as e:
            st.error("❌ Error al obtener el histórico.")
            st.code(str(e), language="bash")
