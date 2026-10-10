from __future__ import annotations
import pandas as pd
import plotly.express as px


def histogram(df: pd.DataFrame, variable: str):
    return px.histogram(df, x=variable, nbins=30, marginal="box", title=f"Distribución de {variable}")


def scatter(df: pd.DataFrame, x: str, y: str):
    return px.scatter(df, x=x, y=y, trendline="ols", title=f"{y} vs. {x}")


def correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True).round(2)
    return px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de correlación")


def predicted_vs_real(predictions: pd.DataFrame):
    return px.scatter(
        predictions,
        x="real",
        y="predicho",
        trendline="ols",
        title="Valores reales vs. predichos",
    )

def confusion_heatmap(cm: pd.DataFrame):
    return px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                     title="Matriz de confusión (filas = real, columnas = predicho)")


def feature_importance_bar(importances: pd.DataFrame):
    datos = importances.sort_values("importancia")
    return px.bar(datos, x="importancia", y="variable", orientation="h",
                  title="Importancia de variables (RandomForest)")