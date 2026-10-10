"""Lógica del agente: construye el prompt con el contexto calculado y usa el cliente LLM."""
from __future__ import annotations
import json

from src.llm_client import OllamaClient, OLLAMA_URL

SYSTEM_PROMPT = (
    "Eres un asistente académico de análisis de datos. "
    "Responde únicamente con base en el contexto estadístico entregado. "
    "Distingue hechos observados de hipótesis. No inventes variables ni resultados. "
    "El dataset de triaje KTAS (Emergency Service - Triage Application) se usa solo con fines educativos; "
    "el nivel KTAS va de 1 (más urgente) a 5 (menos urgente). "
    "Las métricas se calcularon sobre el conjunto de prueba (pacientes que el modelo no vio al entrenar), no sobre el de entrenamiento. "
    "Traduce accuracy como exactitud (acierto global), no como precisión, que es otra métrica. "
    "Compara la exactitud con baseline_accuracy_most_common_level, que es el acierto de predecir siempre el nivel más frecuente. "
    "No emitas diagnósticos ni recomendaciones clínicas. Responde en español, breve y técnico."
)


class LocalDataAgent:
    """Agente simple que envía un contexto analítico resumido a Ollama."""

    def __init__(self, model: str = "llama3.2:3b", base_url: str = OLLAMA_URL):
        self.client = OllamaClient(model=model, base_url=base_url)

    @property
    def model(self) -> str:
        return self.client.model

    def status(self) -> tuple[bool, list[str], str]:
        return self.client.status()

    def ask(self, question: str, context: dict) -> str:
        user = f"CONTEXTO:\n{json.dumps(context, ensure_ascii=False)}\n\nPREGUNTA:\n{question}"
        return self.client.chat(SYSTEM_PROMPT, user)


def offline_summary(context: dict) -> str:
    """Respuesta de respaldo (sin LLM) construida solo con el contexto calculado."""
    d = context.get("dataset", {})
    t = context.get("target_summary", {})
    m = context.get("model", {})
    text = (
        f"El dataset contiene {d.get('rows', 0)} registros y {d.get('columns', 0)} variables, "
        f"con {d.get('missing_total', 0)} valores nulos."
    )
    if t:
        text += f" La variable objetivo presenta una media de {t.get('mean')} y una mediana de {t.get('median')}."
    if m.get("type") == "RandomForestClassifier":
        text += (
            f" El modelo RandomForest obtuvo accuracy={m.get('accuracy')}, "
            f"F1 macro={m.get('f1_macro')} y una proporción de {m.get('within_one_level')} "
            f"de predicciones con error máximo de un nivel KTAS."
        )
        top = context.get("top_features", [])
        if top:
            nombres = ", ".join(f["variable"] for f in top)
            text += f" Las variables más importantes fueron: {nombres}."
    elif m:
        text += (
            f" El modelo lineal de demostración obtuvo R²={m.get('r2')}, "
            f"MAE={m.get('mae')} y RMSE={m.get('rmse')}."
        )   
    return text + " Esta salida es descriptiva y académica."
