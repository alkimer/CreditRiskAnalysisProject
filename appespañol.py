import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import base64

# ⚙️ Configuración general
st.set_page_config(page_title="RiskCore", layout="wide")

# 🎨 Carga el CSS personalizado
with open("logo_intro.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 🖼️ Función para convertir la imagen a base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 📁 Carga el logo embebido
img_base64 = get_base64_of_bin_file("assets/logo.png")

# 🔷 Encabezado visual con imagen + texto estilo branding
st.markdown(f"""
    <div id="header-branding">
        <img src="data:image/png;base64,{img_base64}" class="logo">
        <span class="app-title-gradient">RiskCore</span>
    </div>
""", unsafe_allow_html=True)

# 📋 Sección principal
col_inputs, col_output = st.columns([1.4, 1])

with col_inputs:
    st.subheader("📋 Datos del Usuario")
    with st.form("formulario"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Edad", 18, 80, 30)
            sex = st.selectbox("Sexo", ["M", "F"])
            marital = st.selectbox("Estado civil", ["Soltero", "Casado", "Unión libre", "Divorciado"])
            occupation = st.selectbox("Ocupación", ["Empleado público", "Empleado privado", "Independiente", "Comerciante", "Técnico", "Tecnólogo"])
            months_res = st.number_input("Meses en residencia", 0, 240, 36)
            residence_type = st.selectbox("Tipo de residencia", ["Propia", "Arrendada", "Familiar", "Empresa"])
            flag_phone = st.selectbox("¿Tiene teléfono residencial?", [0, 1])
        with c2:
            state_birth = st.selectbox("Departamento de nacimiento", ["Bogotá", "Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Tolima", "Atlántico"])
            residencial_state = st.selectbox("Departamento de residencia", ["Bogotá", "Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Tolima", "Atlántico"])
            residencial_city = st.text_input("Ciudad de residencia", "Chía")
            residencial_borough = st.text_input("Barrio", "Santa María")
            phone_area = st.selectbox("Código área telefónica", [1, 2, 4, 5, 7])
            zip3 = st.number_input("Código postal (3 dígitos)", 100, 999, 110)
            profesional_state = st.selectbox("Departamento profesional", ["Bogotá", "Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Tolima", "Atlántico"])
            profesional_zip = st.number_input("ZIP profesional", 100, 999, 111)
            product = st.selectbox("Producto solicitado", ["Crédito Hipotecario", "Crédito Consumo", "Crédito Vehicular", "Tarjeta de Crédito"])

        submitted = st.form_submit_button("📊 Evaluar Riesgo")

with col_output:
    st.subheader("📈 Resultado de Riesgo")

    if submitted:
        with st.spinner("⏳ Analizando perfil..."):
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
                res = requests.post("http://localhost:8000/predict", json=payload)
                resultado = res.json()
                porcentaje = resultado["riesgo_porcentaje"]
                clase = resultado["riesgo_clase"]
                st.success(f"🧾 Riesgo predicho: {porcentaje}%  |  Clase: {clase}")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=porcentaje,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": f"<b>Nivel de Riesgo</b><br><span style='font-size:0.8em'>{clase}</span>"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#34C759" if porcentaje < 35 else "#F4D03F" if porcentaje < 70 else "#C0392B"},
                        "steps": [
                            {"range": [0, 35], "color": "#1D8348"},
                            {"range": [35, 70], "color": "#F4D03F"},
                            {"range": [70, 100], "color": "#C0392B"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error("❌ Error al conectar con la API.")
                st.code(str(e), language="bash")
