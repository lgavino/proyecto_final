from __future__ import annotations
import streamlit as st
import pandas as pd

from src.data_loader import load_default_data, load_uploaded_data, dataset_profile
from src.eda import (
    missing_report,
    numeric_summary,
    target_correlations,
    train_regression_demo,
    build_agent_context,
    train_classification_demo,
    evaluate_classification,
    feature_importance_table,
    build_classification_context,
)
from src.visualizations import (
    histogram, scatter, correlation_heatmap, predicted_vs_real,
    confusion_heatmap, feature_importance_bar,
)
from src.agent import LocalDataAgent, offline_summary

st.set_page_config(page_title="Demo IA + Datos", page_icon="🤖", layout="wide")

st.title("Demo · Explorador de datos con agente local")
st.caption("Streamlit + Scikit-learn + Ollama · ejemplo básico del Proyecto Final Integrador")

with st.sidebar:
    st.header("Configuración")
    uploaded = st.file_uploader("CSV opcional", type=["csv"])
    model_name = st.text_input("Modelo Ollama", value="llama3.2:3b")

try:
    df = load_uploaded_data(uploaded) if uploaded else load_default_data()
except Exception as exc:
    st.error(f"No fue posible cargar el dataset: {exc}")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.error("El demo requiere al menos una variable numérica.")
    st.stop()

target = "KTAS_expert"
if target not in df.columns:
    st.error("El dataset debe incluir la columna KTAS_expert (nivel de triaje que se quiere predecir).")
    st.stop()
st.sidebar.caption("Variable objetivo: KTAS_expert (nivel de triaje, 1 a 5)")

profile = dataset_profile(df)
model_result = None
try:
    model_result = train_regression_demo(df, target)
except Exception:
    pass


@st.cache_data(show_spinner="Entrenando el modelo...")
def entrenar_clasificador(df, target):
    resultado = train_classification_demo(df, target)
    return resultado, evaluate_classification(resultado), feature_importance_table(resultado)


try:
    clf_result, evaluation, importances = entrenar_clasificador(df, target)
except Exception as exc:
    clf_result = evaluation = importances = None
    st.sidebar.error(f"No se pudo entrenar el clasificador: {exc}")

agent = LocalDataAgent(model=model_name)
ollama_ok, installed_models, ollama_detail = agent.status()
with st.sidebar:
    st.markdown("---")
    if ollama_ok:
        st.success("Ollama disponible")
        if installed_models:
            st.caption("Modelos: " + ", ".join(installed_models[:5]))
    else:
        st.warning("Ollama no detectado")
        st.caption("La aplicación continúa funcionando sin el agente generativo.")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", profile["rows"])
c2.metric("Variables", profile["columns"])
c3.metric("Valores nulos", profile["missing"])
c4.metric("Duplicados", profile["duplicates"])

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Datos", "📊 Exploración", "📈 Modelo", "🤖 Agente", "🧩 Arquitectura"
])

with tab1:
    st.subheader("Dataset de trabajo")
    if uploaded is None:
        st.info(
            "Dataset incluido: Emergency Service - Triage Application (KTAS, Kaggle), 1,267 pacientes. "
            "Se usa con fines académicos para clasificar el nivel de triaje (KTAS_expert, 1 a 5)."
        )
    else:
        st.info("Se está utilizando el CSV cargado desde la interfaz.")
    st.dataframe(df.head(25), width="stretch", hide_index=True)

    st.markdown("#### Calidad de datos")
    st.dataframe(missing_report(df), width="stretch", hide_index=True)

    st.markdown("#### Resumen numérico")
    st.dataframe(numeric_summary(df), width="stretch", hide_index=True)

with tab2:
    st.subheader("Exploración visual")
    st.markdown("#### Distribución del nivel de triaje (KTAS_expert)")
    conteo = df[target].value_counts().sort_index()
    tabla_niveles = conteo.rename("pacientes").to_frame()
    tabla_niveles["porcentaje"] = (tabla_niveles["pacientes"] / tabla_niveles["pacientes"].sum() * 100).round(1)

    col_graf, col_tabla = st.columns(2)
    with col_graf:
        st.bar_chart(conteo)
    with col_tabla:
        st.dataframe(tabla_niveles, width="stretch")
    st.caption("En KTAS, el nivel 1 es el más urgente y el 5 el menos urgente.")

    indice_age = numeric_cols.index("Age") if "Age" in numeric_cols else 0
    xcol = st.selectbox("Variable para distribución", numeric_cols, index=indice_age)
    st.plotly_chart(histogram(df, xcol), width="stretch")

    #candidates = [c for c in numeric_cols if c != target]
    #if candidates:
    #    xscatter = st.selectbox("Variable explicativa", candidates, index=0)
    #    st.plotly_chart(scatter(df, xscatter, target), width="stretch")

    st.markdown("#### Correlaciones con la variable objetivo")
    st.dataframe(target_correlations(df, target).head(10), width="stretch", hide_index=True)
    #st.plotly_chart(correlation_heatmap(df), width="stretch")
    variables_matriz = [
        target, "Age", "Sex", "Group", "Arrival mode", "Injury", "Mental", "Pain",
        "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation",
        "Patients number per hour",
    ]
    variables_matriz = [c for c in variables_matriz if c in numeric_cols]

    fig_corr = correlation_heatmap(df[variables_matriz])
    fig_corr.update_layout(height=650)
    fig_corr.update_xaxes(dtick=1)
    fig_corr.update_yaxes(dtick=1)
    st.plotly_chart(fig_corr, width="stretch")
    st.caption(
        "Se muestran solo las variables de entrada del modelo (conocidas en el triaje) y el nivel KTAS_expert. "
        "Las parejas NRS_pain-Pain y SBP-DBP están muy relacionadas entre sí."
    )
with tab3:
    st.subheader("Clasificador RandomForest · nivel de triaje KTAS")
    if clf_result is None:
        st.warning("No fue posible entrenar el modelo con la configuración actual.")
    else:
        m = evaluation["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Acierto (accuracy)", f"{m['accuracy']:.3f}")
        c2.metric("F1 macro", f"{m['f1_macro']:.3f}")
        c3.metric("Acierto balanceado", f"{m['balanced_accuracy']:.3f}")
        c4.metric("Error máx. 1 nivel", f"{m['within_one_level']:.3f}")
        st.caption(f"Entrenamiento: {clf_result['n_train']} pacientes · Prueba: {clf_result['n_test']} pacientes")

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(confusion_heatmap(evaluation["confusion"]), width="stretch")
        with col_b:
            st.plotly_chart(feature_importance_bar(importances), width="stretch")

        st.markdown("#### Reporte por nivel KTAS")
        reporte = evaluation["report"].reset_index().rename(columns={"index": "nivel"})
        st.dataframe(reporte, width="stretch", hide_index=True)
        st.caption("Los niveles con pocos pacientes (1 y 5) producen métricas poco estables.")

with tab4:
    st.subheader("Agente local para interpretación")
    if clf_result is not None:
        context = build_classification_context(df, target, clf_result, evaluation, importances)
    else:
        context = build_agent_context(df, target, None)
    st.caption("El agente recibe un contexto estadístico resumido; no recibe el dataset completo.")

    presets = [
        "Resume los principales hallazgos del dataset.",
        "¿Qué variables son más importantes para predecir el nivel de triaje?",
        "Interpreta las métricas del clasificador y sus limitaciones.",
        "¿Por qué el modelo tiene dificultades con el nivel 5?",
        "¿Qué análisis adicional sería razonable realizar?",
    ]
    choice = st.selectbox("Pregunta sugerida", ["Escribir otra pregunta..."] + presets)
    question = st.text_area(
        "Pregunta",
        value="" if choice == "Escribir otra pregunta..." else choice,
        height=100,
    )

    if st.button("Analizar con el agente", type="primary"):
        if not question.strip():
            st.warning("Escribe una pregunta.")
        elif ollama_ok:
            with st.spinner("Consultando el modelo local..."):
                try:
                    answer = agent.ask(question.strip(), context)
                    st.markdown(answer)
                except Exception as exc:
                    st.error(f"Ollama respondió con un error: {exc}")
                    st.markdown(offline_summary(context))
        else:
            st.markdown(offline_summary(context))
            st.caption("Respuesta de respaldo: Ollama no está disponible en este equipo.")

with tab5:
    st.subheader("Arquitectura modular")
    st.code(
        """CSV / archivo cargado
        ↓
src/data_loader.py
        ↓
src/eda.py
   ↙            ↘
visualizations.py  contexto resumido
   ↓            ↓
app.py ← src/agent.py → src/llm_client.py → Ollama local""",
        language="text",
    )
    st.markdown(
        """
- **Imperativo:** flujo de carga, validaciones y decisiones de la interfaz.
- **Funcional:** funciones de perfilado, métricas, correlaciones y visualización.
- **Orientado a objetos:** clases `LocalDataAgent` (lógica del agente) y `OllamaClient` (comunicación con Ollama).
- **Modularidad:** cada responsabilidad se mantiene en un archivo separado.
        """
    )

st.markdown("---")
st.caption("Demo académico. No constituye una herramienta clínica ni un sistema de decisión real.")
