# Proyecto Final Integrador · Streamlit + Ollama

(Versión de trabajo: partimos del demo del profesor y lo vamos adaptando.)

## Estructura

```text
proyecto_final/
├── app.py                 # interfaz Streamlit
├── requirements.txt
├── README.md
├── .gitignore
├── create_demo_data.py    # genera el CSV del demo (se eliminará al cambiar de dataset)
├── data/
├── assets/
└── src/
    ├── __init__.py
    ├── data_loader.py     # carga de datos y perfil
    ├── eda.py             # análisis, métricas y modelo
    ├── visualizations.py  # gráficos Plotly
    ├── llm_client.py      # comunicación con Ollama
    └── agent.py           # lógica del agente
```

## Ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
python -m streamlit run app.py
```
