"""
Simulación Monte Carlo del Mundial 2026.

Formato del torneo:
- 12 grupos de 4 selecciones (48 equipos en total), todos contra todos (6 partidos por grupo)
- Avanzan los 2 primeros de cada grupo (24 equipos) + los 8 mejores terceros = 32 equipos
- Eliminación directa desde dieciseisavos de final (32 -> 16 -> 8 -> 4 -> 2 -> 1)
- En la fase eliminatoria, los empates se resuelven con prórroga/penales: 50/50
  ajustado por la diferencia de Elo entre los equipos

El número de goles de cada partido se obtiene muestreando la matriz de probabilidades
de marcador (Poisson + ajuste Dixon-Coles) generada por el modelo de la Fase 2.

Salida:
- output/resultados.csv: probabilidad por selección de pasar de grupo, llegar a
  octavos (8vos), cuartos (4tos), semifinal, final y ser campeón.
"""

import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import elo_esperado
from model import matriz_resultado, MAX_GOLES


# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

RUTA_GRUPOS = DATA_DIR / "grupos_2026.csv"
RUTA_ESTADO_ACTUAL = DATA_DIR / "estado_actual_selecciones.csv"
RUTA_ESTADO_MUNDIAL = DATA_DIR / "estado_actual_mundial2026.csv"
RUTA_RESULTADOS_REALES = DATA_DIR / "resultados_reales_grupos.csv"
RUTA_ELIMINATORIA = DATA_DIR / "calendario_eliminatoria.csv"
RUTA_MODELO = DATA_DIR / "modelo_goles.pickle"
RUTA_RHO = DATA_DIR / "dixon_coles_rho.json"
RUTA_SALIDA = OUTPUT_DIR / "resultados.csv"

N_SIMULACIONES = 10_000
SEMILLA = 42

GRUPOS = list("ABCDEFGHIJKL")
RONDAS_ELIMINATORIA = ["llega_8vos", "llega_4tos", "llega_semis", "llega_final", "campeon"]

HOST_NATIONS = {"United States", "Canada", "Mexico"}


# ----------------------------------------------------------------------------
# Carga de datos y preparación
# ----------------------------------------------------------------------------
def cargar_datos():
    grupos_df = pd.read_csv(RUTA_GRUPOS)

    # Si existe un estado "en vivo" (actualizado con resultados reales del Mundial
    # vía actualizar_estado_mundial.py), se usa ese; si no, el estado base pre-torneo.
    if RUTA_ESTADO_MUNDIAL.exists():
        ruta_estado = RUTA_ESTADO_MUNDIAL
    else:
        ruta_estado = RUTA_ESTADO_ACTUAL
    print(f"Usando estado de selecciones desde '{ruta_estado.name}'")
    estado = pd.read_csv(ruta_estado).set_index("seleccion")

    with open(RUTA_MODELO, "rb") as f:
        modelo = pickle.load(f)
    with open(RUTA_RHO, "r", encoding="utf-8") as f:
        rho = json.load(f)["rho"]

    equipos_mundial = grupos_df["seleccion"].tolist()
    faltantes = [e for e in equipos_mundial if e not in estado.index]
    if faltantes:
        raise ValueError(f"Selecciones sin datos de Elo/forma actual: {faltantes}")

    grupo_de = dict(zip(grupos_df["seleccion"], grupos_df["grupo"]))
    equipos_por_grupo = {g: grupos_df.loc[grupos_df["grupo"] == g, "seleccion"].tolist() for g in GRUPOS}

    resultados_reales = cargar_resultados_reales()
    bracket_real = cargar_bracket_real()
    resultados_reales_eliminatoria = cargar_resultados_reales_eliminatoria()

    return (
        grupos_df, estado, modelo, rho, equipos_mundial, grupo_de, equipos_por_grupo,
        resultados_reales, bracket_real, resultados_reales_eliminatoria,
    )


def cargar_resultados_reales() -> dict:
    """
    Carga los partidos de fase de grupos ya jugados en el Mundial real
    (data/resultados_reales_grupos.csv, generado por
    actualizar_resultados_fifa.py o ingresado manualmente).

    Devuelve {(grupo, equipo_local, equipo_visita): (goles_local, goles_visita)}.
    Estos resultados se "fijan" en la simulación de cada grupo: no se vuelven
    a sortear, por lo que un equipo ya eliminado matemáticamente terminará con
    probabilidades cercanas a 0% en las rondas posteriores.
    """
    if not RUTA_RESULTADOS_REALES.exists():
        return {}

    reales = pd.read_csv(RUTA_RESULTADOS_REALES).dropna(subset=["goles_local_real", "goles_visita_real"])
    return {
        (fila["grupo"], fila["equipo_local"], fila["equipo_visita"]): (int(fila["goles_local_real"]), int(fila["goles_visita_real"]))
        for _, fila in reales.iterrows()
    }


def cargar_bracket_real() -> list[str] | None:
    """
    Carga el cuadro real de dieciseisavos de final (16 partidos, 32 equipos)
    tal como lo resuelve FIFA en cuanto termina la fase de grupos
    (data/calendario_eliminatoria.csv, generado por actualizar_resultados_fifa.py).

    Devuelve una lista de 32 equipos en orden de partido (pares consecutivos:
    (0,1), (2,3), ...) o None si todavía no están definidos los 16 cruces
    (en cuyo caso se usa el cuadro heurístico simplificado como aproximación).
    """
    if not RUTA_ELIMINATORIA.exists():
        return None

    df = pd.read_csv(RUTA_ELIMINATORIA)
    r32 = df[df["ronda"] == "Dieciseisavos de final"].sort_values("match_number")
    if len(r32) != 16 or not (r32["local_definido"] & r32["visita_definido"]).all():
        return None

    bracket = []
    for _, fila in r32.iterrows():
        bracket.append(fila["equipo_local"])
        bracket.append(fila["equipo_visita"])
    return bracket


def cargar_resultados_reales_eliminatoria() -> dict:
    """
    Carga los partidos de la fase eliminatoria ya jugados en el Mundial real
    (data/calendario_eliminatoria.csv). Se "fijan" en la simulación igual que
    los resultados reales de grupos: no se vuelven a sortear.

    Devuelve {(equipo_local, equipo_visita): (goles_local, goles_visita)}.
    """
    if not RUTA_ELIMINATORIA.exists():
        return {}

    df = pd.read_csv(RUTA_ELIMINATORIA).dropna(subset=["goles_local_real", "goles_visita_real"])
    return {
        (fila["equipo_local"], fila["equipo_visita"]): (int(fila["goles_local_real"]), int(fila["goles_visita_real"]))
        for _, fila in df.iterrows()
    }


def calcular_lambdas(equipos_mundial: list[str], estado: pd.DataFrame, modelo) -> dict:
    """
    Calcula, para cada par ordenado (equipo, rival) de las 48 selecciones,
    los goles esperados (lambda) del "equipo" frente al "rival" en cancha
    neutral, usando el modelo Poisson y el estado actual (Elo + forma) de cada
    selección. Se calcula una sola vez (no cambia entre simulaciones).
    """
    filas, pares = [], []
    for propio in equipos_mundial:
        for rival in equipos_mundial:
            if propio == rival:
                continue
            fp = estado.loc[propio]
            fr = estado.loc[rival]
            filas.append({
                "elo_diff_propio": fp["elo"] - fr["elo"],
                "forma_pts_propio": fp["forma_pts"],
                "forma_dif_gol_propio": fp["forma_dif_gol"],
                "forma_pts_rival": fr["forma_pts"],
                "forma_dif_gol_rival": fr["forma_dif_gol"],
                "gf_prom_propio": fp["gf_prom"],
                "gc_prom_rival": fr["gc_prom"],
                "es_local": 1 if propio in HOST_NATIONS else 0,
            })
            pares.append((propio, rival))

    df_pares = pd.DataFrame(filas)
    predicciones = modelo.predict(df_pares)
    return {par: float(lam) for par, lam in zip(pares, predicciones)}


def precalcular_matrices(equipos_mundial: list[str], lambdas: dict, rho: float) -> dict:
    """
    Precalcula, para cada pareja de selecciones (sin orden), la distribución
    acumulada de probabilidades de marcador (Poisson + Dixon-Coles), para
    poder muestrear resultados rápidamente en cada simulación.
    """
    indices_marcador = [(i, j) for i in range(MAX_GOLES + 1) for j in range(MAX_GOLES + 1)]
    cumsum_por_par = {}

    for a, b in combinations(equipos_mundial, 2):
        matriz = matriz_resultado(lambdas[(a, b)], lambdas[(b, a)], rho)
        probs = matriz.flatten()
        cumsum_por_par[(a, b)] = np.cumsum(probs)

    return cumsum_por_par, indices_marcador


# ----------------------------------------------------------------------------
# Simulación de partidos
# ----------------------------------------------------------------------------
def jugar_partido(a: str, b: str, cumsum_por_par: dict, indices_marcador: list, rng: np.random.Generator) -> tuple[int, int]:
    """Muestrea el marcador de un partido entre 'a' y 'b'. Devuelve (goles_a, goles_b)."""
    if (a, b) in cumsum_por_par:
        cum = cumsum_por_par[(a, b)]
        idx = int(np.searchsorted(cum, rng.random()))
        gol_a, gol_b = indices_marcador[idx]
    else:
        cum = cumsum_por_par[(b, a)]
        idx = int(np.searchsorted(cum, rng.random()))
        gol_b, gol_a = indices_marcador[idx]
    return gol_a, gol_b


def resolver_empate_eliminatoria(a: str, b: str, elo: pd.Series, rng: np.random.Generator) -> str:
    """
    En la fase eliminatoria, un empate se resuelve en prórroga/penales.
    Se asigna la victoria con una probabilidad 50/50 ajustada por la diferencia
    de Elo (el favorito tiene algo más de chances, pero sigue siendo muy parejo).
    """
    p_gana_a = elo_esperado(elo[a], elo[b])
    return a if rng.random() < p_gana_a else b


# ----------------------------------------------------------------------------
# Fase de grupos
# ----------------------------------------------------------------------------
def simular_grupo(equipos_grupo: list[str], partidos_jugados: dict, cumsum_por_par: dict,
                   indices_marcador: list, rng: np.random.Generator):
    """
    Simula el todos-contra-todos de un grupo de 4 equipos.

    Los partidos que ya se jugaron en el Mundial real (`partidos_jugados`,
    con sus resultados reales) no se vuelven a sortear: se usa el marcador
    real. Solo se simulan los partidos pendientes.

    Devuelve la lista de equipos ordenada (1° a 4°) y la tabla de resultados.
    """
    tabla = {e: {"pts": 0, "gf": 0, "gc": 0} for e in equipos_grupo}

    for a, b in combinations(equipos_grupo, 2):
        if (a, b) in partidos_jugados:
            gol_a, gol_b = partidos_jugados[(a, b)]
        elif (b, a) in partidos_jugados:
            gol_b, gol_a = partidos_jugados[(b, a)]
        else:
            gol_a, gol_b = jugar_partido(a, b, cumsum_por_par, indices_marcador, rng)

        tabla[a]["gf"] += gol_a
        tabla[a]["gc"] += gol_b
        tabla[b]["gf"] += gol_b
        tabla[b]["gc"] += gol_a

        if gol_a > gol_b:
            tabla[a]["pts"] += 3
        elif gol_a < gol_b:
            tabla[b]["pts"] += 3
        else:
            tabla[a]["pts"] += 1
            tabla[b]["pts"] += 1

    # Orden: puntos, diferencia de gol, goles a favor; empates restantes al azar
    orden = sorted(
        equipos_grupo,
        key=lambda e: (tabla[e]["pts"], tabla[e]["gf"] - tabla[e]["gc"], tabla[e]["gf"], rng.random()),
        reverse=True,
    )
    return orden, tabla


# ----------------------------------------------------------------------------
# Fase eliminatoria
# ----------------------------------------------------------------------------
def construir_bracket(primeros: dict, segundos: dict, mejores_terceros: list[str]) -> list[str]:
    """
    Construye el cuadro de 32 equipos para los dieciseisavos de final.

    Simplificación del bracket oficial de la FIFA: se enfrenta al 1° de cada
    grupo con el 2° del grupo siguiente (evitando que dos equipos del mismo
    grupo se cruquen en esta ronda), y los 8 mejores terceros se emparejan
    entre sí con un seeding tipo "serpiente" (mejor vs peor).

    Devuelve una lista de 32 equipos donde los partidos de dieciseisavos son
    las parejas consecutivas: (0,1), (2,3), ..., (30,31).
    """
    bracket = []
    for i, g in enumerate(GRUPOS):
        g_siguiente = GRUPOS[(i + 1) % len(GRUPOS)]
        bracket.append(primeros[g])
        bracket.append(segundos[g_siguiente])

    t = mejores_terceros
    parejas_terceros = [(t[0], t[7]), (t[1], t[6]), (t[2], t[5]), (t[3], t[4])]
    for a, b in parejas_terceros:
        bracket.append(a)
        bracket.append(b)

    return bracket


def simular_eliminatoria(bracket: list[str], cumsum_por_par: dict, indices_marcador: list,
                          elo: pd.Series, contadores: dict, rng: np.random.Generator,
                          resultados_reales_eliminatoria: dict | None = None) -> str:
    """
    Simula las 5 rondas de eliminación directa (dieciseisavos -> campeón).
    Los partidos ya jugados en el Mundial real (`resultados_reales_eliminatoria`)
    no se vuelven a sortear: se usa el marcador real.
    Incrementa, para cada ganador de ronda, el contador correspondiente en `contadores`.
    Devuelve el campeón.
    """
    resultados_reales_eliminatoria = resultados_reales_eliminatoria or {}
    ronda_actual = bracket
    for logro in RONDAS_ELIMINATORIA:
        siguiente_ronda = []
        for i in range(0, len(ronda_actual), 2):
            a, b = ronda_actual[i], ronda_actual[i + 1]
            if (a, b) in resultados_reales_eliminatoria:
                gol_a, gol_b = resultados_reales_eliminatoria[(a, b)]
            elif (b, a) in resultados_reales_eliminatoria:
                gol_b, gol_a = resultados_reales_eliminatoria[(b, a)]
            else:
                gol_a, gol_b = jugar_partido(a, b, cumsum_por_par, indices_marcador, rng)

            if gol_a > gol_b:
                ganador = a
            elif gol_a < gol_b:
                ganador = b
            else:
                ganador = resolver_empate_eliminatoria(a, b, elo, rng)

            contadores[ganador][logro] += 1
            siguiente_ronda.append(ganador)

        ronda_actual = siguiente_ronda

    return ronda_actual[0]


# ----------------------------------------------------------------------------
# Simulación completa del torneo
# ----------------------------------------------------------------------------
def simular_torneo(n_simulaciones: int, equipos_mundial: list[str], equipos_por_grupo: dict,
                    cumsum_por_par: dict, indices_marcador: list, elo: pd.Series,
                    resultados_reales: dict | None = None, bracket_real: list[str] | None = None,
                    resultados_reales_eliminatoria: dict | None = None, seed: int = SEMILLA) -> dict:
    rng = np.random.default_rng(seed)
    resultados_reales = resultados_reales or {}
    resultados_reales_eliminatoria = resultados_reales_eliminatoria or {}

    # Partidos ya jugados en el Mundial real, agrupados por grupo:
    # {grupo: {(equipo_local, equipo_visita): (goles_local, goles_visita)}}
    partidos_jugados_por_grupo = {g: {} for g in GRUPOS}
    for (grupo, local, visita), marcador in resultados_reales.items():
        partidos_jugados_por_grupo[grupo][(local, visita)] = marcador

    contadores = {
        e: {"pasa_grupo": 0, "llega_8vos": 0, "llega_4tos": 0, "llega_semis": 0, "llega_final": 0, "campeon": 0}
        for e in equipos_mundial
    }

    for _ in range(n_simulaciones):
        primeros, segundos, terceros = {}, {}, []

        for g in GRUPOS:
            orden, tabla = simular_grupo(
                equipos_por_grupo[g], partidos_jugados_por_grupo[g], cumsum_por_par, indices_marcador, rng
            )
            primeros[g] = orden[0]
            segundos[g] = orden[1]
            terceros.append((orden[2], tabla[orden[2]]))

            contadores[orden[0]]["pasa_grupo"] += 1
            contadores[orden[1]]["pasa_grupo"] += 1

        # Ranking de los 12 terceros lugares: los 8 mejores avanzan
        terceros_ordenados = sorted(
            terceros,
            key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["gc"], x[1]["gf"], rng.random()),
            reverse=True,
        )
        mejores_terceros = [equipo for equipo, _ in terceros_ordenados[:8]]
        for e in mejores_terceros:
            contadores[e]["pasa_grupo"] += 1

        # Si FIFA ya publicó el cuadro real de dieciseisavos (fase de grupos
        # terminada), se usa tal cual en vez del cruce heurístico simplificado.
        bracket = bracket_real if bracket_real is not None else construir_bracket(primeros, segundos, mejores_terceros)
        simular_eliminatoria(bracket, cumsum_por_par, indices_marcador, elo, contadores, rng, resultados_reales_eliminatoria)

    return contadores


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def main():
    print("Cargando datos (grupos, estado actual de selecciones, modelo y rho)...")
    (
        grupos_df, estado, modelo, rho, equipos_mundial, grupo_de, equipos_por_grupo,
        resultados_reales, bracket_real, resultados_reales_eliminatoria,
    ) = cargar_datos()
    print(f"Selecciones: {len(equipos_mundial)} | Grupos: {len(GRUPOS)} | rho = {rho:.4f}")
    print(f"Partidos de grupos ya jugados (fijados en la simulación): {len(resultados_reales)}")
    if bracket_real is not None:
        print("Cuadro real de dieciseisavos (FIFA) disponible: se usa en vez del cruce heurístico.")
    print(f"Partidos de eliminatoria ya jugados (fijados en la simulación): {len(resultados_reales_eliminatoria)}")

    print("Calculando goles esperados (lambda) para todos los enfrentamientos posibles...")
    lambdas = calcular_lambdas(equipos_mundial, estado, modelo)

    print("Precalculando matrices de probabilidad de marcador (Poisson + Dixon-Coles)...")
    cumsum_por_par, indices_marcador = precalcular_matrices(equipos_mundial, lambdas, rho)

    elo = estado["elo"]

    print(f"Simulando el torneo {N_SIMULACIONES:,} veces...")
    contadores = simular_torneo(
        N_SIMULACIONES, equipos_mundial, equipos_por_grupo, cumsum_por_par, indices_marcador, elo,
        resultados_reales, bracket_real, resultados_reales_eliminatoria,
    )

    filas = []
    for e in equipos_mundial:
        c = contadores[e]
        filas.append({
            "seleccion": e,
            "grupo": grupo_de[e],
            "elo": round(float(elo[e]), 1),
            "prob_pasa_grupo": c["pasa_grupo"] / N_SIMULACIONES,
            "prob_llega_8vos": c["llega_8vos"] / N_SIMULACIONES,
            "prob_llega_4tos": c["llega_4tos"] / N_SIMULACIONES,
            "prob_llega_semis": c["llega_semis"] / N_SIMULACIONES,
            "prob_llega_final": c["llega_final"] / N_SIMULACIONES,
            "prob_campeon": c["campeon"] / N_SIMULACIONES,
        })

    resultados = pd.DataFrame(filas).sort_values("prob_campeon", ascending=False).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(RUTA_SALIDA, index=False)
    print(f"\nResultados guardados en '{RUTA_SALIDA}'")

    print("\nTop 10 candidatos al título:")
    print(resultados.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
