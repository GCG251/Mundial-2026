"""
Calcula las predicciones (goles esperados, P(V/E/D), P(Over 0.5), marcador más
probable) para los partidos de la fase eliminatoria del Mundial 2026 cuyos dos
equipos ya estén definidos, según data/calendario_eliminatoria.csv (generado
por actualizar_resultados_fifa.py a partir del cuadro real de FIFA).

Se ejecuta después de actualizar_estado_mundial.py para usar el Elo/forma más
reciente (incorporando los resultados reales del Mundial jugados hasta ahora).

Salida:
- data/calendario_eliminatoria.csv (actualizado in-place con las predicciones)
"""

from pathlib import Path

import pandas as pd

from predecir_partido import cargar_recursos, predecir_partido

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RUTA_ELIMINATORIA = DATA_DIR / "calendario_eliminatoria.csv"

COLUMNAS_PREDICCION = [
    "goles_esperados_local", "goles_esperados_visita",
    "prob_victoria_local", "prob_empate", "prob_victoria_visita", "prob_over_0_5",
    "marcador_mas_probable", "prob_marcador_mas_probable",
]


def main():
    if not RUTA_ELIMINATORIA.exists():
        print(f"No existe '{RUTA_ELIMINATORIA.name}'; ejecutar primero actualizar_resultados_fifa.py")
        return

    df = pd.read_csv(RUTA_ELIMINATORIA)
    estado, modelo, rho = cargar_recursos()
    equipos_validos = set(estado.index)

    for columna in COLUMNAS_PREDICCION:
        if columna not in df.columns:
            df[columna] = pd.NA

    actualizados = 0
    for idx, fila in df.iterrows():
        local, visita = fila["equipo_local"], fila["equipo_visita"]
        if local not in equipos_validos or visita not in equipos_validos:
            continue

        pred = predecir_partido(local, visita, estado, modelo, rho, neutral=True)
        marcador_top = pred["marcadores_probables"][0]

        df.loc[idx, "goles_esperados_local"] = round(pred["goles_esperados_local"], 2)
        df.loc[idx, "goles_esperados_visita"] = round(pred["goles_esperados_visita"], 2)
        df.loc[idx, "prob_victoria_local"] = round(pred["prob_victoria_local"], 4)
        df.loc[idx, "prob_empate"] = round(pred["prob_empate"], 4)
        df.loc[idx, "prob_victoria_visita"] = round(pred["prob_victoria_visita"], 4)
        df.loc[idx, "prob_over_0_5"] = round(pred["prob_over_0_5"], 4)
        df.loc[idx, "marcador_mas_probable"] = marcador_top["marcador"]
        df.loc[idx, "prob_marcador_mas_probable"] = round(marcador_top["probabilidad"], 4)
        actualizados += 1

    df.to_csv(RUTA_ELIMINATORIA, index=False)
    print(f"Predicciones de fase eliminatoria actualizadas para {actualizados} partido(s).")


if __name__ == "__main__":
    main()
