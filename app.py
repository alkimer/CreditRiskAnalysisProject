import streamlit as st
import requests
import plotly.graph_objs as go

# Configuración general
st.set_page_config(layout="wide", page_title="Evaluación de Riesgo Crediticio")

# Aplicar estilo externo (opcional)
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Evaluación de Riesgo Crediticio 🌙")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔍 Datos del Usuario")
    age = st.number_input("Edad", 18, 100)
    income = st.number_input("Ingresos mensuales ($)", 0.0)
    loan_debt = st.number_input("Monto del préstamo / Deuda ($)", 0.0)
    education = st.selectbox("Nivel de educación", ["Bachillerato", "Universitario", "Postgrado", "Otro"])
    credit_type = st.selectbox("Tipo de crédito", ["Auto", "Casa", "Educación", "Tarjeta de Crédito"])

    if st.button("📊 Evaluar Riesgo"):
        input_data = {
            "age": age,
            "income": income,
            "loan_debt": loan_debt,
            "education": education,
            "credit_type": credit_type
        }

        try:
            response = requests.post("http://localhost:8000/predict", json=input_data)
            result = response.json()

            if "error" in result:
                st.error(f"❌ Error en el backend: {result['error']}")
            else:
                risk_score = result["risk_score"]
                risk_label = result["risk_label"]

                with col2:
                    st.header("⚠️ Resultado")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_score,
                        title={'text': f"Nivel de Riesgo: {risk_label}"},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': "white"},
                            'bar': {'color': "#34C759" if risk_score <= 60 else "#FF3B30"},
                            'bgcolor': "#1E1F24",
                            'steps': [
                                {'range': [0, 60], 'color': "#2E8B57"},
                                {'range': [60, 100], 'color': "#FF3B30"}
                            ]
                        },
                        number={'font': {'color': "white"}}
                    ))
                    fig.update_layout(paper_bgcolor="#111217", font_color="white")
                    st.plotly_chart(fig)

                    if "shap_values" in result and result["shap_values"]:
                        st.subheader("🧠 Factores más influyentes")
                        try:
                            # Procesar y ordenar SHAP
                            raw_vals = [float(v[0]) if isinstance(v, list) else float(v) for v in result["shap_values"]]
                            shap_vals = [round(v, 3) for v in raw_vals]
                            sorted_data = sorted(zip(result["feature_names"], shap_vals), key=lambda x: abs(x[1]), reverse=True)
                            features_sorted, shap_sorted = zip(*sorted_data)
                            bar_colors = ['#FF3B30' if val < 0 else '#34C759' for val in shap_sorted]

                            bar_fig = go.Figure(go.Bar(
                                x=features_sorted,
                                y=shap_sorted,
                                text=[str(v) for v in shap_sorted],
                                textposition='auto',
                                marker_color=bar_colors
                            ))
                            bar_fig.update_layout(
                                title="Contribución individual de cada variable",
                                paper_bgcolor="#111217",
                                plot_bgcolor="#111217",
                                font_color="white",
                                height=400
                            )
                            st.plotly_chart(bar_fig)
                        except Exception as e:
                            st.warning(f"SHAP no pudo graficarse correctamente: {e}")
                    else:
                        st.info("ℹ️ Esta predicción aún no incluye explicación por variable.")

        except requests.exceptions.RequestException as e:
            st.error("❌ No se pudo conectar con el backend.")
            st.code(str(e))