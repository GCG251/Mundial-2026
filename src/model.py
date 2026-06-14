"""
Modelo de goles esperados (Poisson) con ajuste Dixon-Coles para partidos
de fútbol internacional, y validación contra un baseline basado en Elo.

Pasos:
1. Carga data/dataset_features.csv (generado por data_prep.py)
2. Transforma cada partido a formato "largo" (una fila por equipo por partido)
3. Entrena una regresión Poisson: goles_anotados ~ features de ataque/defensa propias y rivales
4. Estima el parámetro rho del ajuste Dixon-Coles (corrige la dependencia en marcadores bajos)
5. Valida en partidos de 2024-2025: log-loss y accuracy del resultado (V/E/D),
   comparado contra un baseline basado en Elo
6. Guarda el modelo entrenado y el parámetro rho para usarlos en la simulación (Fase 3)

Salidas:
- data/modelo_goles.pickle : modelo Poisson entrenado (statsmodels GLM)
- data/dixon_coles_rho.json : parámetro rho estimado
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUTA_DATASET = DATA_DIR / "dataset_features.csv"
RUTA_MODELO = DATA_DIR / "modelo_goles.pickle"
RUTA_RHO = DATA_DIR / "dixon_coles_rho.json"

FECHA_CORTE_TEST_INICIO = "2024-01-01"
FECHA_CORTE_TEST_FIN = "2026-01-01"  # partidos de 2024-2025 (exclusivo del 2026)

MAX_GOLES = 9  # tamaño de la matriz de marcadores posibles (0..MAX_GOLES)

# Fórmula de la regresión Poisson: goles anotados por "equipo" en función de
# su fortaleza ofensiva, la debilidad defensiva del rival y la condición de local
FORMULA = (
    "goles ~ elo_diff_propio + forma_pts_propio + forma_dif_gol_propio "
    "+ forma_pts_rival + forma_dif_gol_rival + gf_prom_propio + gc_prom_rival + es_local"
)

COLUMNAS_REQUERIDAS = [
    "home_forma_pts", "home_forma_dif_gol", "home_gf_prom", "home_gc_prom",
    "away_forma_pts", "away_forma_dif_gol", "away_gf_prom", "away_gc_prom",
]


# ----------------------------------------------------------------------------
# Construcción del formato "largo"
# ----------------------------------------------------------------------------
def construir_formato_largo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el dataset de partidos (una fila por partido, formato ancho) en un
    dataset "largo" con dos filas por partido: una desde la perspectiva del
    equipo local y otra desde la del visitante. Cada fila representa el número
    de goles anotados por "equipo" frente a "rival".
    """
    perspectiva_local = pd.DataFrame({
        "fecha": df["date"],
        "goles": df["home_score"],
        "elo_diff_propio": df["elo_home"] - df["elo_away"],
        "forma_pts_propio": df["home_forma_pts"],
        "forma_dif_gol_propio": df["home_forma_dif_gol"],
        "forma_pts_rival": df["away_forma_pts"],
        "forma_dif_gol_rival": df["away_forma_dif_gol"],
        "gf_prom_propio": df["home_gf_prom"],
        "gc_prom_rival": df["away_gc_prom"],
        "es_local": np.where(df["neutral"], 0, 1),
    })

    perspectiva_visita = pd.DataFrame({
        "fecha": df["date"],
        "goles": df["away_score"],
        "elo_diff_propio": df["elo_away"] - df["elo_home"],
        "forma_pts_propio": df["away_forma_pts"],
        "forma_dif_gol_propio": df["away_forma_dif_gol"],
        "forma_pts_rival": df["home_forma_pts"],
        "forma_dif_gol_rival": df["home_forma_dif_gol"],
        "gf_prom_propio": df["away_gf_prom"],
        "gc_prom_rival": df["home_gc_prom"],
        "es_local": 0,
    })

    largo = pd.concat([perspectiva_local, perspectiva_visita], ignore_index=True)
    return largo


# ----------------------------------------------------------------------------
# Ajuste Dixon-Coles
# ----------------------------------------------------------------------------
def tau_dixon_coles(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Factor de corrección de Dixon-Coles para la probabilidad conjunta P(goles_local=x, goles_visita=y).
    Corrige la sobre/sub-estimación de la independencia entre marcadores bajos (0-0, 1-0, 0-1, 1-1).
    """
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def log_verosimilitud_negativa(rho: float, lambdas: np.ndarray, mus: np.ndarray,
                                goles_local: np.ndarray, goles_visita: np.ndarray) -> float:
    """Negativo de la log-verosimilitud (solo el término tau, lo demás no depende de rho)."""
    total = 0.0
    for lam, mu, x, y in zip(lambdas, mus, goles_local, goles_visita):
        x, y = int(x), int(y)
        if x <= 1 and y <= 1:
            tau = tau_dixon_coles(x, y, lam, mu, rho)
            total += np.log(max(tau, 1e-10))
    return -total


def estimar_rho(lambdas: np.ndarray, mus: np.ndarray,
                 goles_local: np.ndarray, goles_visita: np.ndarray) -> float:
    """Estima por máxima verosimilitud el parámetro rho del ajuste Dixon-Coles."""
    resultado = minimize_scalar(
        log_verosimilitud_negativa,
        bounds=(-0.3, 0.3),
        method="bounded",
        args=(lambdas, mus, goles_local, goles_visita),
    )
    return float(resultado.x)


def matriz_resultado(lam: float, mu: float, rho: float, max_goles: int = MAX_GOLES) -> np.ndarray:
    """
    Construye la matriz de probabilidades P(goles_local=i, goles_visita=j) para
    i, j en 0..max_goles, aplicando el ajuste Dixon-Coles y normalizando para
    que la suma total sea 1.
    """
    goles = np.arange(0, max_goles + 1)
    p_local = poisson.pmf(goles, lam)
    p_visita = poisson.pmf(goles, mu)
    matriz = np.outer(p_local, p_visita)

    for x in (0, 1):
        for y in (0, 1):
            matriz[x, y] *= tau_dixon_coles(x, y, lam, mu, rho)

    matriz = np.clip(matriz, 0, None)
    matriz /= matriz.sum()
    return matriz


def probabilidades_resultado(matriz: np.ndarray) -> tuple[float, float, float]:
    """A partir de la matriz de marcadores, devuelve (P(local gana), P(empate), P(visita gana))."""
    n = matriz.shape[0]
    i, j = np.indices((n, n))
    p_local = matriz[i > j].sum()
    p_empate = matriz[i == j].sum()
    p_visita = matriz[i < j].sum()
    return float(p_local), float(p_empate), float(p_visita)


# ----------------------------------------------------------------------------
# Validación
# ----------------------------------------------------------------------------
def resultado_real(goles_local: int, goles_visita: int) -> str:
    """Etiqueta del resultado real: 'L' (gana local), 'E' (empate), 'V' (gana visita)."""
    if goles_local > goles_visita:
        return "L"
    if goles_local < goles_visita:
        return "V"
    return "E"


def evaluar_modelo(df_test: pd.DataFrame, modelo, rho: float) -> dict:
    """
    Evalúa el modelo Poisson + Dixon-Coles sobre el set de prueba:
    - log-loss del resultado (V/E/D)
    - accuracy del resultado (argmax de la probabilidad)
    También calcula el baseline basado en Elo (accuracy) y un baseline de
    frecuencias históricas (log-loss).
    """
    largo = construir_formato_largo(df_test)
    n = len(df_test)

    lambdas = modelo.predict(largo.iloc[:n])          # perspectiva local
    mus = modelo.predict(largo.iloc[n:].reset_index(drop=True))  # perspectiva visita

    log_losses, aciertos = [], []
    log_losses_baseline, aciertos_baseline = [], []

    # Frecuencias históricas de resultado (baseline de log-loss "naive")
    frecuencias = df_test.apply(
        lambda r: resultado_real(r["home_score"], r["away_score"]), axis=1
    ).value_counts(normalize=True)
    p_base = {
        "L": frecuencias.get("L", 1 / 3),
        "E": frecuencias.get("E", 1 / 3),
        "V": frecuencias.get("V", 1 / 3),
    }

    for idx, fila in enumerate(df_test.itertuples()):
        lam, mu = lambdas.iloc[idx], mus.iloc[idx]
        matriz = matriz_resultado(lam, mu, rho)
        p_l, p_e, p_v = probabilidades_resultado(matriz)
        probs = {"L": p_l, "E": p_e, "V": p_v}

        real = resultado_real(fila.home_score, fila.away_score)

        # --- Modelo Poisson + Dixon-Coles ---
        log_losses.append(-np.log(max(probs[real], 1e-10)))
        pred = max(probs, key=probs.get)
        aciertos.append(int(pred == real))

        # --- Baseline: gana el equipo con mayor Elo (ajustado por local) ---
        ajuste_local = 0 if fila.neutral else 100
        elo_local_ajustado = fila.elo_home + ajuste_local
        if elo_local_ajustado > fila.elo_away:
            pred_base = "L"
        elif elo_local_ajustado < fila.elo_away:
            pred_base = "V"
        else:
            pred_base = "E"
        aciertos_baseline.append(int(pred_base == real))

        # --- Baseline log-loss: frecuencias históricas constantes ---
        log_losses_baseline.append(-np.log(max(p_base[real], 1e-10)))

    return {
        "n_partidos": n,
        "log_loss_modelo": float(np.mean(log_losses)),
        "accuracy_modelo": float(np.mean(aciertos)),
        "log_loss_baseline_frecuencias": float(np.mean(log_losses_baseline)),
        "accuracy_baseline_elo": float(np.mean(aciertos_baseline)),
    }


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    print(f"Cargando dataset de features desde '{RUTA_DATASET}'...")
    df = pd.read_csv(RUTA_DATASET, parse_dates=["date"])

    filas_antes = len(df)
    df = df.dropna(subset=COLUMNAS_REQUERIDAS).reset_index(drop=True)
    print(f"Partidos con features completas: {len(df)} (se descartaron {filas_antes - len(df)} sin historial suficiente)")

    df_train = df[df["date"] < FECHA_CORTE_TEST_INICIO].reset_index(drop=True)
    df_test = df[
        (df["date"] >= FECHA_CORTE_TEST_INICIO) & (df["date"] < FECHA_CORTE_TEST_FIN)
    ].reset_index(drop=True)
    print(f"Entrenamiento: {len(df_train)} partidos (hasta {FECHA_CORTE_TEST_INICIO})")
    print(f"Prueba (2024-2025): {len(df_test)} partidos")

    # --- Entrenar regresión Poisson ---
    print("\nEntrenando regresión Poisson de goles esperados...")
    largo_train = construir_formato_largo(df_train)
    modelo = smf.glm(formula=FORMULA, data=largo_train, family=sm.families.Poisson()).fit()
    print(modelo.summary())

    # --- Estimar rho (ajuste Dixon-Coles) usando el set de entrenamiento ---
    print("\nEstimando parámetro rho del ajuste Dixon-Coles...")
    n_train = len(df_train)
    lambdas_train = modelo.predict(largo_train.iloc[:n_train]).to_numpy()
    mus_train = modelo.predict(largo_train.iloc[n_train:].reset_index(drop=True)).to_numpy()
    rho = estimar_rho(
        lambdas_train, mus_train,
        df_train["home_score"].to_numpy(), df_train["away_score"].to_numpy(),
    )
    print(f"rho estimado = {rho:.4f}")

    # --- Validación en 2024-2025 ---
    print("\nValidando en partidos de 2024-2025...")
    metricas = evaluar_modelo(df_test, modelo, rho)
    print(f"\nPartidos evaluados: {metricas['n_partidos']}")
    print(f"Log-loss modelo (Poisson + Dixon-Coles): {metricas['log_loss_modelo']:.4f}")
    print(f"Log-loss baseline (frecuencias históricas): {metricas['log_loss_baseline_frecuencias']:.4f}")
    print(f"Accuracy modelo (Poisson + Dixon-Coles): {metricas['accuracy_modelo']:.4f}")
    print(f"Accuracy baseline (gana el de mayor Elo): {metricas['accuracy_baseline_elo']:.4f}")

    # --- Guardar modelo y rho ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\nModelo guardado en '{RUTA_MODELO}'")

    with open(RUTA_RHO, "w", encoding="utf-8") as f:
        json.dump({"rho": rho, **metricas}, f, indent=2, ensure_ascii=False)
    print(f"Parámetro rho y métricas guardados en '{RUTA_RHO}'")


if __name__ == "__main__":
    main()
