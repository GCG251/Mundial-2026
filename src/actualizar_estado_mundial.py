"""
Actualiza el Elo y la forma reciente de las selecciones del Mundial 2026 a
partir de los resultados reales ingresados en el dashboard
(data/resultados_reales_grupos.csv).

Esto permite "reentrenar" el estado del modelo en tiempo real conforme avanza
el torneo:
- El rating Elo de cada selección se actualiza con K=40 (factor de Mundial),
  igual que en la Fase 1.
- La forma reciente (puntos, diferencia de gol, promedios de goles) se
  actualiza con una media móvil aproximada (ventana efectiva ~10 partidos).

Esto NO reentrena los coeficientes de la regresión Poisson (eso requiere
volver a ejecutar `data_prep.py` + `model.py` con los partidos reales
incorporados al histórico). Para torneos cortos como el Mundial, los
coeficientes del modelo cambian muy poco con unos pocos partidos nuevos; lo
que más impacta a las predicciones es el Elo y la forma actuales de cada
selección, que es justo lo que actualiza este script.

Salida:
- data/estado_actual_mundial2026.csv (usado por simulate.py / predecir_partido.py
  si existe; si no, se usa estado_actual_selecciones.csv)
"""

from pathlib import Path

import pandas as pd

from data_prep import elo_esperado, K_MUNDIAL


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RUTA_ESTADO_BASE = DATA_DIR / "estado_actual_selecciones.csv"
RUTA_RESULTADOS_REALES = DATA_DIR / "resultados_reales_grupos.csv"
RUTA_CALENDARIO = DATA_DIR / "calendario_grupos.csv"
RUTA_ESTADO_MUNDIAL = DATA_DIR / "estado_actual_mundial2026.csv"

PESO_RECIENTE = 0.10  # aproximación de media móvil con ventana efectiva ~10 partidos


def main():
    estado = pd.read_csv(RUTA_ESTADO_BASE).set_index("seleccion")

    if not RUTA_RESULTADOS_REALES.exists():
        print("No hay resultados reales registrados todavía (data/resultados_reales_grupos.csv no existe).")
        estado.reset_index().to_csv(RUTA_ESTADO_MUNDIAL, index=False)
        print(f"Se copió el estado base sin cambios a '{RUTA_ESTADO_MUNDIAL}'")
        return

    reales = pd.read_csv(RUTA_RESULTADOS_REALES)
    reales = reales.dropna(subset=["goles_local_real", "goles_visita_real"])

    calendario = pd.read_csv(RUTA_CALENDARIO, parse_dates=["fecha"])
    reales = reales.merge(
        calendario[["grupo", "equipo_local", "equipo_visita", "fecha"]],
        on=["grupo", "equipo_local", "equipo_visita"], how="left",
    ).sort_values("fecha")

    elo = estado["elo"].to_dict()
    forma_pts = estado["forma_pts"].to_dict()
    forma_dif_gol = estado["forma_dif_gol"].to_dict()
    gf_prom = estado["gf_prom"].to_dict()
    gc_prom = estado["gc_prom"].to_dict()

    for _, fila in reales.iterrows():
        local, visita = fila["equipo_local"], fila["equipo_visita"]
        gol_local, gol_visita = int(fila["goles_local_real"]), int(fila["goles_visita_real"])

        # --- Elo (K=40, cancha neutral) ---
        esperado_local = elo_esperado(elo[local], elo[visita])
        if gol_local > gol_visita:
            score_local, score_visita = 1.0, 0.0
        elif gol_local < gol_visita:
            score_local, score_visita = 0.0, 1.0
        else:
            score_local, score_visita = 0.5, 0.5

        elo[local] = elo[local] + K_MUNDIAL * (score_local - esperado_local)
        elo[visita] = elo[visita] + K_MUNDIAL * (score_visita - (1 - esperado_local))

        # --- Forma reciente (media móvil aproximada) ---
        pts_local = 3 if gol_local > gol_visita else (1 if gol_local == gol_visita else 0)
        pts_visita = 3 if gol_visita > gol_local else (1 if gol_local == gol_visita else 0)

        for equipo, pts, dif_gol, gf, gc in (
            (local, pts_local, gol_local - gol_visita, gol_local, gol_visita),
            (visita, pts_visita, gol_visita - gol_local, gol_visita, gol_local),
        ):
            forma_pts[equipo] = forma_pts[equipo] * (1 - PESO_RECIENTE) + pts * PESO_RECIENTE
            forma_dif_gol[equipo] = forma_dif_gol[equipo] * (1 - PESO_RECIENTE) + dif_gol * PESO_RECIENTE
            gf_prom[equipo] = gf_prom[equipo] * (1 - PESO_RECIENTE) + gf * PESO_RECIENTE
            gc_prom[equipo] = gc_prom[equipo] * (1 - PESO_RECIENTE) + gc * PESO_RECIENTE

    actualizado = estado.copy()
    actualizado["elo"] = pd.Series(elo)
    actualizado["forma_pts"] = pd.Series(forma_pts)
    actualizado["forma_dif_gol"] = pd.Series(forma_dif_gol)
    actualizado["gf_prom"] = pd.Series(gf_prom)
    actualizado["gc_prom"] = pd.Series(gc_prom)

    actualizado.reset_index().to_csv(RUTA_ESTADO_MUNDIAL, index=False)
    print(f"Estado actualizado con {len(reales)} partidos reales guardado en '{RUTA_ESTADO_MUNDIAL}'")


if __name__ == "__main__":
    main()
