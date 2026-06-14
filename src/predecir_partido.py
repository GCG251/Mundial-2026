"""
Predicción de partidos individuales del Mundial 2026.

Usa el modelo Poisson + Dixon-Coles (Fase 2) y el estado actual de Elo/forma
de cada selección (Fase 1) para estimar, para cualquier enfrentamiento:
- Goles esperados de cada equipo
- Probabilidad de victoria local / empate / victoria visitante
- Los marcadores más probables

También genera un reporte con las predicciones de los 72 partidos de la
fase de grupos (los cruces ya están definidos por `grupos_2026.csv`).

Salida:
- output/predicciones_fase_grupos.csv
"""

import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from model import matriz_resultado, probabilidades_resultado


# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

RUTA_ESTADO_ACTUAL = DATA_DIR / "estado_actual_selecciones.csv"
RUTA_ESTADO_MUNDIAL = DATA_DIR / "estado_actual_mundial2026.csv"
RUTA_MODELO = DATA_DIR / "modelo_goles.pickle"
RUTA_RHO = DATA_DIR / "dixon_coles_rho.json"
RUTA_GRUPOS = DATA_DIR / "grupos_2026.csv"
RUTA_SALIDA_GRUPOS = OUTPUT_DIR / "predicciones_fase_grupos.csv"

TOP_MARCADORES = 5  # cantidad de marcadores más probables a reportar


# ----------------------------------------------------------------------------
# Carga de recursos
# ----------------------------------------------------------------------------
def cargar_recursos():
    """Carga el estado actual de las selecciones, el modelo y el parámetro rho."""
    ruta_estado = RUTA_ESTADO_MUNDIAL if RUTA_ESTADO_MUNDIAL.exists() else RUTA_ESTADO_ACTUAL
    estado = pd.read_csv(ruta_estado).set_index("seleccion")

    with open(RUTA_MODELO, "rb") as f:
        modelo = pickle.load(f)
    with open(RUTA_RHO, "r", encoding="utf-8") as f:
        rho = json.load(f)["rho"]

    return estado, modelo, rho


# ----------------------------------------------------------------------------
# Predicción de un partido
# ----------------------------------------------------------------------------
def calcular_lambda(equipo_propio: str, equipo_rival: str, estado: pd.DataFrame, modelo, es_local: int = 0) -> float:
    """Calcula los goles esperados de `equipo_propio` frente a `equipo_rival`."""
    fp = estado.loc[equipo_propio]
    fr = estado.loc[equipo_rival]

    fila = pd.DataFrame([{
        "elo_diff_propio": fp["elo"] - fr["elo"],
        "forma_pts_propio": fp["forma_pts"],
        "forma_dif_gol_propio": fp["forma_dif_gol"],
        "forma_pts_rival": fr["forma_pts"],
        "forma_dif_gol_rival": fr["forma_dif_gol"],
        "gf_prom_propio": fp["gf_prom"],
        "gc_prom_rival": fr["gc_prom"],
        "es_local": es_local,
    }])

    return float(modelo.predict(fila).iloc[0])


def predecir_partido(equipo_local: str, equipo_visita: str, estado: pd.DataFrame, modelo, rho: float,
                      neutral: bool = True, top_marcadores: int = TOP_MARCADORES) -> dict:
    """
    Predice un partido entre `equipo_local` y `equipo_visita`.

    Si `neutral=False`, se le da al equipo local la ventaja de jugar en casa
    (tal como se modeló en la Fase 2 con la variable `es_local`).
    """
    es_local_flag = 0 if neutral else 1

    lam = calcular_lambda(equipo_local, equipo_visita, estado, modelo, es_local=es_local_flag)
    mu = calcular_lambda(equipo_visita, equipo_local, estado, modelo, es_local=0)

    matriz = matriz_resultado(lam, mu, rho)
    p_local, p_empate, p_visita = probabilidades_resultado(matriz)

    orden = np.dstack(np.unravel_index(np.argsort(-matriz.ravel()), matriz.shape))[0]
    marcadores_probables = [
        {"marcador": f"{i}-{j}", "probabilidad": float(matriz[i, j])}
        for i, j in orden[:top_marcadores]
    ]

    return {
        "equipo_local": equipo_local,
        "equipo_visita": equipo_visita,
        "goles_esperados_local": lam,
        "goles_esperados_visita": mu,
        "prob_victoria_local": p_local,
        "prob_empate": p_empate,
        "prob_victoria_visita": p_visita,
        "marcadores_probables": marcadores_probables,
    }


def imprimir_prediccion(pred: dict) -> None:
    """Imprime en consola, de forma legible, la predicción de un partido."""
    print(f"\n{pred['equipo_local']} vs {pred['equipo_visita']}")
    print(f"Goles esperados: {pred['goles_esperados_local']:.2f} - {pred['goles_esperados_visita']:.2f}")
    print(f"P(gana {pred['equipo_local']}):  {pred['prob_victoria_local'] * 100:5.1f}%")
    print(f"P(empate):              {pred['prob_empate'] * 100:5.1f}%")
    print(f"P(gana {pred['equipo_visita']}): {pred['prob_victoria_visita'] * 100:5.1f}%")
    print("Marcadores más probables:")
    for m in pred["marcadores_probables"]:
        print(f"   {m['marcador']}: {m['probabilidad'] * 100:4.1f}%")


# ----------------------------------------------------------------------------
# Reporte: predicciones de toda la fase de grupos
# ----------------------------------------------------------------------------
def generar_predicciones_fase_grupos(estado: pd.DataFrame, modelo, rho: float) -> pd.DataFrame:
    """Calcula las predicciones de los 72 partidos de la fase de grupos (6 por grupo)."""
    grupos_df = pd.read_csv(RUTA_GRUPOS)

    filas = []
    for grupo, equipos in grupos_df.groupby("grupo")["seleccion"]:
        for local, visita in combinations(equipos.tolist(), 2):
            pred = predecir_partido(local, visita, estado, modelo, rho, neutral=True)
            marcador_top = pred["marcadores_probables"][0]

            filas.append({
                "grupo": grupo,
                "equipo_local": local,
                "equipo_visita": visita,
                "goles_esperados_local": round(pred["goles_esperados_local"], 2),
                "goles_esperados_visita": round(pred["goles_esperados_visita"], 2),
                "prob_victoria_local": round(pred["prob_victoria_local"], 4),
                "prob_empate": round(pred["prob_empate"], 4),
                "prob_victoria_visita": round(pred["prob_victoria_visita"], 4),
                "marcador_mas_probable": marcador_top["marcador"],
                "prob_marcador_mas_probable": round(marcador_top["probabilidad"], 4),
            })

    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    print("Cargando modelo, rho y estado actual de selecciones...")
    estado, modelo, rho = cargar_recursos()

    print("Calculando predicciones de los 72 partidos de la fase de grupos...")
    predicciones = generar_predicciones_fase_grupos(estado, modelo, rho)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predicciones.to_csv(RUTA_SALIDA_GRUPOS, index=False)
    print(f"Guardado en '{RUTA_SALIDA_GRUPOS}' ({len(predicciones)} partidos)")

    # Ejemplo de uso interactivo
    print("\nEjemplo de predicción puntual:")
    pred = predecir_partido("Argentina", "Mexico", estado, modelo, rho, neutral=True)
    imprimir_prediccion(pred)


if __name__ == "__main__":
    main()
