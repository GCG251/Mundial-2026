"""
Preparación de datos para el modelo predictivo del Mundial 2026.

Este script:
1. Carga el histórico de partidos internacionales (data/results.csv)
2. Filtra los partidos desde 2018 en adelante
3. Calcula un rating Elo propio para cada selección, con distinto factor K
   según el tipo de competencia (mundial, eliminatoria, amistoso)
4. Genera features por partido: diferencia de Elo, forma reciente,
   promedio ponderado de goles a favor/en contra y condición de local/neutral

Salidas:
- data/dataset_features.csv : un registro por partido con las features y el resultado
- data/elo_ratings.csv       : rating Elo final de cada selección al cierre del histórico
"""

from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUTA_RESULTADOS = DATA_DIR / "results.csv"
RUTA_DATASET_SALIDA = DATA_DIR / "dataset_features.csv"
RUTA_ELO_SALIDA = DATA_DIR / "elo_ratings.csv"
RUTA_ESTADO_ACTUAL_SALIDA = DATA_DIR / "estado_actual_selecciones.csv"

FECHA_INICIO = "2018-01-01"

ELO_INICIAL = 1500.0
VENTAJA_LOCAL = 100.0  # puntos extra de Elo para el equipo local cuando el partido no es neutral

# Factores K del rating Elo según el tipo de competencia
K_MUNDIAL = 40
K_ELIMINATORIA = 30
K_AMISTOSO = 20

N_PARTIDOS_FORMA = 10  # tamaño de la ventana para "forma reciente"


# ----------------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------------
def clasificar_torneo(nombre_torneo: str) -> str:
    """
    Clasifica el torneo para asignar el factor K del Elo.

    - "mundial": Copa Mundial de la FIFA (fase final)
    - "eliminatoria": cualquier torneo de clasificación ("qualification")
    - "amistoso": el resto de competencias (amistosos, copas continentales, etc.)
    """
    nombre = str(nombre_torneo).lower()

    if "qualif" in nombre:
        return "eliminatoria"
    if "world cup" in nombre:
        return "mundial"
    return "amistoso"


def factor_k(tipo_torneo: str) -> int:
    """Devuelve el factor K del Elo según el tipo de torneo."""
    return {
        "mundial": K_MUNDIAL,
        "eliminatoria": K_ELIMINATORIA,
        "amistoso": K_AMISTOSO,
    }[tipo_torneo]


def elo_esperado(elo_a: float, elo_b: float) -> float:
    """Probabilidad esperada de que el equipo A gane (o empate cuente como 0.5) frente a B."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def resultado_partido(goles_local: int, goles_visita: int) -> tuple[float, float]:
    """Devuelve el puntaje Elo (1=victoria, 0.5=empate, 0=derrota) para local y visita."""
    if goles_local > goles_visita:
        return 1.0, 0.0
    if goles_local < goles_visita:
        return 0.0, 1.0
    return 0.5, 0.5


def puntos_futbol(goles_propios: int, goles_rival: int) -> int:
    """Puntos de torneo: 3 victoria, 1 empate, 0 derrota."""
    if goles_propios > goles_rival:
        return 3
    if goles_propios == goles_rival:
        return 1
    return 0


def promedio_ponderado(valores: list[float]) -> float:
    """
    Promedio ponderado donde los partidos más recientes pesan más.
    `valores` debe estar ordenado de más antiguo a más reciente.
    """
    if not valores:
        return np.nan
    pesos = np.arange(1, len(valores) + 1)  # 1, 2, 3, ... (el último pesa más)
    return float(np.dot(valores, pesos) / pesos.sum())


# ----------------------------------------------------------------------------
# Procesamiento principal
# ----------------------------------------------------------------------------
def cargar_resultados(ruta: Path) -> pd.DataFrame:
    """Carga el CSV de resultados históricos y aplica el filtro de fecha."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo '{ruta}'.\n"
            "Descarga el dataset de Kaggle 'International football results from 1872 to 2025' "
            "(martj42/international-football-results-from-1872-to-2017) y guarda el CSV como "
            f"'{ruta}'. Revisa el README.md para más detalles."
        )

    df = pd.read_csv(ruta, parse_dates=["date"])

    # Descartar partidos sin marcador registrado (datos incompletos)
    filas_antes = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    filas_descartadas = filas_antes - len(df)
    if filas_descartadas:
        print(f"Se descartaron {filas_descartadas} partidos sin marcador (NaN en home_score/away_score)")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def construir_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Recorre TODO el histórico de partidos en orden cronológico (incluyendo los
    anteriores a FECHA_INICIO) para "calentar" el Elo y la forma reciente de cada
    selección con datos reales previos. Solo los partidos desde FECHA_INICIO se
    incluyen en el dataset de features que se exporta.

    Para cada partido se calcula:
    - Elo pre-partido de ambos equipos (y luego se actualiza)
    - Forma reciente (puntos y diferencia de gol promedio en los últimos N partidos)
    - Promedio ponderado de goles a favor / en contra

    Devuelve el dataframe de features (solo desde FECHA_INICIO) y un diccionario
    con los Elo finales de todas las selecciones.
    """
    elo = defaultdict(lambda: ELO_INICIAL)

    # Historial reciente por selección: cada entrada es (puntos, diferencia_de_gol, gf, gc)
    historial = defaultdict(lambda: deque(maxlen=N_PARTIDOS_FORMA))

    registros = []

    for _, fila in df.iterrows():
        local = fila["home_team"]
        visita = fila["away_team"]
        goles_local = int(fila["home_score"])
        goles_visita = int(fila["away_score"])
        es_neutral = bool(fila.get("neutral", False))

        tipo_torneo = clasificar_torneo(fila["tournament"])
        k = factor_k(tipo_torneo)

        # --- Elo pre-partido ---
        elo_local = elo[local]
        elo_visita = elo[visita]
        ajuste_local = 0.0 if es_neutral else VENTAJA_LOCAL

        # --- Forma reciente pre-partido ---
        hist_local = list(historial[local])
        hist_visita = list(historial[visita])

        if hist_local:
            forma_pts_local = float(np.mean([h[0] for h in hist_local]))
            forma_dif_gol_local = float(np.mean([h[1] for h in hist_local]))
            gf_prom_local = promedio_ponderado([h[2] for h in hist_local])
            gc_prom_local = promedio_ponderado([h[3] for h in hist_local])
        else:
            forma_pts_local = np.nan
            forma_dif_gol_local = np.nan
            gf_prom_local = np.nan
            gc_prom_local = np.nan

        if hist_visita:
            forma_pts_visita = float(np.mean([h[0] for h in hist_visita]))
            forma_dif_gol_visita = float(np.mean([h[1] for h in hist_visita]))
            gf_prom_visita = promedio_ponderado([h[2] for h in hist_visita])
            gc_prom_visita = promedio_ponderado([h[3] for h in hist_visita])
        else:
            forma_pts_visita = np.nan
            forma_dif_gol_visita = np.nan
            gf_prom_visita = np.nan
            gc_prom_visita = np.nan

        # Solo exportamos como feature los partidos desde FECHA_INICIO; los anteriores
        # solo sirven para "calentar" el Elo y la forma reciente.
        if fila["date"] >= pd.Timestamp(FECHA_INICIO):
            registros.append({
                "date": fila["date"],
                "tournament": fila["tournament"],
                "tipo_torneo": tipo_torneo,
                "home_team": local,
                "away_team": visita,
                "home_score": goles_local,
                "away_score": goles_visita,
                "neutral": es_neutral,
                "elo_home": elo_local,
                "elo_away": elo_visita,
                "elo_diff": (elo_local + ajuste_local) - elo_visita,
                "home_forma_pts": forma_pts_local,
                "home_forma_dif_gol": forma_dif_gol_local,
                "away_forma_pts": forma_pts_visita,
                "away_forma_dif_gol": forma_dif_gol_visita,
                "home_gf_prom": gf_prom_local,
                "home_gc_prom": gc_prom_local,
                "away_gf_prom": gf_prom_visita,
                "away_gc_prom": gc_prom_visita,
            })

        # --- Actualizar Elo post-partido ---
        score_local, score_visita = resultado_partido(goles_local, goles_visita)
        esperado_local = elo_esperado(elo_local + ajuste_local, elo_visita)
        esperado_visita = 1.0 - esperado_local

        elo[local] = elo_local + k * (score_local - esperado_local)
        elo[visita] = elo_visita + k * (score_visita - esperado_visita)

        # --- Actualizar historial reciente ---
        pts_local = puntos_futbol(goles_local, goles_visita)
        pts_visita = puntos_futbol(goles_visita, goles_local)
        dif_gol_local = goles_local - goles_visita
        dif_gol_visita = goles_visita - goles_local

        historial[local].append((pts_local, dif_gol_local, goles_local, goles_visita))
        historial[visita].append((pts_visita, dif_gol_visita, goles_visita, goles_local))

    dataset = pd.DataFrame.from_records(registros)
    return dataset, dict(elo), dict(historial)


def construir_estado_actual(elo_final: dict, historial_final: dict) -> pd.DataFrame:
    """
    Construye, para cada selección, su "estado actual" (a la fecha del último partido
    disponible en el histórico): rating Elo y forma reciente (últimos N partidos).

    Esta tabla es la que se usará en la Fase 3 como punto de partida para simular
    los partidos del Mundial 2026.
    """
    filas = []
    for seleccion, elo_valor in elo_final.items():
        hist = list(historial_final.get(seleccion, []))
        if hist:
            forma_pts = float(np.mean([h[0] for h in hist]))
            forma_dif_gol = float(np.mean([h[1] for h in hist]))
            gf_prom = promedio_ponderado([h[2] for h in hist])
            gc_prom = promedio_ponderado([h[3] for h in hist])
        else:
            forma_pts = np.nan
            forma_dif_gol = np.nan
            gf_prom = np.nan
            gc_prom = np.nan

        filas.append({
            "seleccion": seleccion,
            "elo": elo_valor,
            "forma_pts": forma_pts,
            "forma_dif_gol": forma_dif_gol,
            "gf_prom": gf_prom,
            "gc_prom": gc_prom,
        })

    return pd.DataFrame.from_records(filas).sort_values("elo", ascending=False).reset_index(drop=True)


def main():
    print(f"Cargando resultados desde '{RUTA_RESULTADOS}'...")
    df = cargar_resultados(RUTA_RESULTADOS)
    print(f"Partidos totales (histórico completo, para calentar el Elo): {len(df)}")
    print(f"Partidos desde {FECHA_INICIO} (incluidos en el dataset de features): "
          f"{(df['date'] >= pd.Timestamp(FECHA_INICIO)).sum()}")

    print("Calculando Elo y features por partido...")
    dataset, elo_final, historial_final = construir_dataset(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(RUTA_DATASET_SALIDA, index=False)
    print(f"Dataset de features guardado en '{RUTA_DATASET_SALIDA}' ({len(dataset)} filas)")

    elo_df = (
        pd.DataFrame(elo_final.items(), columns=["seleccion", "elo"])
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    elo_df.to_csv(RUTA_ELO_SALIDA, index=False)
    print(f"Ratings Elo finales guardados en '{RUTA_ELO_SALIDA}'")
    print("\nTop 10 selecciones por Elo:")
    print(elo_df.head(10).to_string(index=False))

    estado_actual = construir_estado_actual(elo_final, historial_final)
    estado_actual.to_csv(RUTA_ESTADO_ACTUAL_SALIDA, index=False)
    print(f"\nEstado actual (Elo + forma reciente) guardado en '{RUTA_ESTADO_ACTUAL_SALIDA}'")


if __name__ == "__main__":
    main()
