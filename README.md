# Explorador de datos de triaje con agente de IA local

Proyecto Final Integrador – *Paradigmas de Programación para Inteligencia Artificial y Análisis de Datos* (C2026 P1)
Maestría en Inteligencia Artificial y Ciencia de Datos – Universidad Casa Grande
Autor: Luis Armando Gavino Armijos

Aplicación modular en **Streamlit** que analiza un dataset de triaje de urgencias (escala KTAS), entrena un modelo **Random Forest** para predecir el nivel de triaje y usa un **agente de IA local** (Llama 3.2 ejecutado con **Ollama**) para explicar los resultados en lenguaje natural.

## Dataset

*Emergency Service – Triage Application* (Kaggle): 1.267 pacientes de dos servicios de urgencias de Corea, 24 variables. La variable objetivo es `KTAS_expert` (nivel 1 = más grave, 5 = menos grave).

- Fuente: https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application
- Estudio original: Moon et al. (2019), *PLOS ONE*. https://doi.org/10.1371/journal.pone.0216972

El archivo ya está incluido en `data/data.csv`.

## Requisitos

- Windows 10/11 (también funciona en macOS/Linux cambiando el comando de activación)
- Python 3.10 o superior (desarrollado con Python 3.14)
- Git
- Ollama: https://ollama.com/download

## Instalación y ejecución local

```powershell
# 1. Clonar el repositorio
git clone https://github.com/lgavino/proyecto_final.git
cd proyecto_final

# 2. Crear y activar el entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Instalar las librerías
python -m pip install -r requirements.txt

# 4. Descargar el modelo de lenguaje (solo la primera vez)
ollama pull llama3.2:3b

# 5. Abrir la aplicación
python -m streamlit run app.py
```

La aplicación se abre en http://localhost:8501.

> Si PowerShell indica que la ejecución de scripts está deshabilitada, ejecute una vez:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

Ollama debe estar en ejecución (aplicación abierta o `ollama serve`). Si no lo está, la aplicación sigue funcionando y, en lugar de la respuesta del agente, muestra un resumen automático con los mismos números.

## Pestañas de la aplicación

| Pestaña | Contenido |
|---|---|
| Datos | Vista del dataset, tipos de variables y datos faltantes |
| Exploración | Distribución del nivel de triaje y correlaciones con la variable objetivo |
| Modelo | Random Forest: acierto, F1 macro, matriz de confusión e importancia de variables |
| Agente | Preguntas en lenguaje natural respondidas por Llama 3.2 a partir de los resultados calculados |
| Arquitectura | Organización del código |

## Estructura del repositorio

```
proyecto_final/
├── app.py                # interfaz Streamlit (5 pestañas)
├── requirements.txt      # librerías necesarias
├── data/
│   └── data.csv          # dataset de triaje
├── assets/
└── src/
    ├── data_loader.py    # carga y limpieza de datos (pandas)
    ├── eda.py            # análisis exploratorio y modelo (scikit-learn)
    ├── visualizations.py # gráficos (Plotly)
    ├── llm_client.py     # comunicación con Ollama
    └── agent.py          # lógica del agente (clase LocalDataAgent)
```

## Nota

El modelo es un ejercicio académico y no debe usarse para clasificar pacientes reales.
