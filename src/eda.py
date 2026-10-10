from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

FEATURES_NUM = ["Age", "Patients number per hour", "NRS_pain",
                "SBP", "DBP", "HR", "RR", "BT", "Saturation"]
FEATURES_CAT = ["Group", "Sex", "Arrival mode", "Injury", "Mental", "Pain"]

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "variable": df.columns,
        "nulos": df.isna().sum().values,
        "porcentaje": (df.isna().mean().values * 100).round(2),
    })
    return report.sort_values(["nulos", "variable"], ascending=[False, True]).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    out = numeric.describe().T.reset_index().rename(columns={"index": "variable"})
    return out.round(3)


def target_correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        return pd.DataFrame(columns=["variable", "correlacion"])
    corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    return (
        corr.rename("correlacion")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
        .rename(columns={"index": "variable"})
    )


def train_regression_demo(df: pd.DataFrame, target: str) -> dict:
    numeric = df.select_dtypes(include="number").dropna().copy()
    if target not in numeric.columns:
        raise ValueError("La variable objetivo debe ser numérica.")

    X = numeric.drop(columns=[target])
    y = numeric[target]
    if X.shape[1] == 0 or len(numeric) < 20:
        raise ValueError("No hay suficientes variables o registros para entrenar el modelo.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    coefficients = pd.DataFrame({
        "variable": X.columns,
        "coeficiente": model.coef_,
    }).sort_values("coeficiente", key=lambda s: s.abs(), ascending=False)

    predictions = pd.DataFrame({
        "real": y_test.to_numpy(),
        "predicho": pred,
    }).reset_index(drop=True)

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": coefficients,
        "predictions": predictions,
        "features": list(X.columns),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def build_agent_context(df: pd.DataFrame, target: str, model_result: dict | None = None) -> dict:
    numeric = df.select_dtypes(include="number")
    context = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_total": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "target": target,
        },
        "target_summary": {},
        "top_correlations": [],
    }

    if target in numeric.columns:
        s = numeric[target].dropna()
        context["target_summary"] = {
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
        }
        context["top_correlations"] = target_correlations(df, target).head(5).round(3).to_dict("records")

    if model_result:
        context["model"] = {
            "type": "LinearRegression",
            "r2": round(model_result["metrics"]["r2"], 3),
            "mae": round(model_result["metrics"]["mae"], 3),
            "rmse": round(model_result["metrics"]["rmse"], 3),
            "n_train": model_result["n_train"],
            "n_test": model_result["n_test"],
        }
    return context

def prepare_classification_data(df: pd.DataFrame, target: str = "KTAS_expert", columnas=None):
    """Separa X (entradas) e y (objetivo). Si columnas=None usa todas menos el objetivo."""
    data = df.dropna(subset=[target]).copy()
    y = data[target].astype(int)
    X = data.drop(columns=[target]) if columnas is None else data[columnas]
    return X, y


def build_preprocessor(num_cols, cat_cols):
    """Prepara los datos para el modelo: rellena vacíos y convierte categorías en columnas 0/1."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

def train_classification_demo(df: pd.DataFrame, target: str = "KTAS_expert") -> dict:
    """Entrena un RandomForest para clasificar el nivel KTAS (1 a 5)."""
    columnas = FEATURES_NUM + FEATURES_CAT
    X, y = prepare_classification_data(df, target, columnas)

    # 80% de los pacientes para aprender, 20% para examinar al modelo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("prep", build_preprocessor(FEATURES_NUM, FEATURES_CAT)),
        ("rf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "pred": pred,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

def evaluate_classification(result: dict) -> dict:
    """Calcula las métricas del examen: acierto, matriz de confusión y reporte por nivel."""
    y_test = result["y_test"]
    pred = result["pred"]
    niveles = sorted(int(c) for c in result["model"].classes_)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        # En triaje, equivocarse por 1 nivel es menos grave que por 3
        "within_one_level": float((np.abs(y_test.to_numpy() - pred) <= 1).mean()),
    }

    confusion = pd.DataFrame(
        confusion_matrix(y_test, pred, labels=niveles),
        index=[f"real {n}" for n in niveles],
        columns=[f"pred {n}" for n in niveles],
    )

    report = pd.DataFrame(
        classification_report(y_test, pred, labels=niveles, output_dict=True, zero_division=0)
    ).T.round(3)

    return {"metrics": metrics, "confusion": confusion, "report": report}

def feature_importance_table(result: dict) -> pd.DataFrame:
    """Importancia de cada variable original (suma sus columnas 0/1 y su aviso de vacío)."""
    pipe = result["model"]
    nombres = pipe.named_steps["prep"].get_feature_names_out()
    valores = pipe.named_steps["rf"].feature_importances_

    filas = []
    for nombre, valor in zip(nombres, valores):
        sin_prefijo = nombre.split("__", 1)[1]          # quita "num__" o "cat__"
        if sin_prefijo.startswith("missingindicator_"):
            variable = sin_prefijo.replace("missingindicator_", "")
        elif nombre.startswith("cat__"):
            variable = sin_prefijo.rsplit("_", 1)[0]    # "Arrival mode_3" -> "Arrival mode"
        else:
            variable = sin_prefijo
        filas.append((variable, valor))

    tabla = pd.DataFrame(filas, columns=["variable", "importancia"])
    tabla = tabla.groupby("variable", as_index=False)["importancia"].sum()
    tabla["importancia"] = tabla["importancia"].round(3)
    return tabla.sort_values("importancia", ascending=False).reset_index(drop=True)

def build_classification_context(df, target, result, evaluation, importances) -> dict:
    """Resumen estadístico (sin el dataset completo) que se le entrega al asistente."""
    conteo = df[target].value_counts().sort_index()
    niveles = sorted(int(c) for c in result["model"].classes_)
    reporte = evaluation["report"]

    return {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_total": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "target": target,
        },
        "class_distribution": {str(k): int(v) for k, v in conteo.items()},
        "model": {
            "type": "RandomForestClassifier",
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "evaluated_on": "conjunto de prueba (pacientes que el modelo no vio al entrenar)",
            "baseline_accuracy_most_common_level": round(float(result["y_test"].value_counts(normalize=True).max()), 3),
            **{k: round(v, 3) for k, v in evaluation["metrics"].items()},
            "recall_per_level": {str(n): float(reporte.loc[str(n), "recall"]) for n in niveles},
        },
        "top_features": importances.head(5).to_dict("records"),
    }