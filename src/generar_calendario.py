"""
Genera el calendario de la fase de grupos del Mundial 2026 con las predicciones
del modelo para cada partido.

Reglas de emparejamiento por jornada (según el formato oficial FIFA, partidos
numerados 1-4 dentro de cada grupo según el orden de `grupos_2026.csv`):
- Jornada 1: 1 vs 2, 3 vs 4
- Jornada 2: 1 vs 3, 2 vs 4
- Jornada 3: 4 vs 1, 2 vs 3 (simultáneos)

Las fechas son una aproximación ilustrativa basada en las ventanas oficiales
de cada jornada (J1: 11-16 jun, J2: 17-22 jun, J3: 23-26 jun), repartiendo los
12 grupos entre los días de cada ventana. Si se cuenta con el calendario
oficial exacto (fecha/sede de cada partido), se puede reemplazar la columna
`fecha` de `data/calendario_grupos.csv` manualmente.

Salida:
- data/calendario_grupos.csv
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from predecir_partido import cargar_recursos, predecir_partido


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RUTA_GRUPOS = DATA_DIR / "grupos_2026.csv"
RUTA_SALIDA = DATA_DIR / "calendario_grupos.csv"

GRUPOS = list("ABCDEFGHIJKL")

# Ventanas de fecha por jornada (inicio) y cantidad de grupos por día
FECHA_INICIO_JORNADA = {1: date(2026, 6, 11), 2: date(2026, 6, 17), 3: date(2026, 6, 23)}
GRUPOS_POR_DIA = {1: 2, 2: 2, 3: 3}

# Emparejamientos por jornada, en términos de posición (0-indexado) dentro del grupo
PARTIDOS_POR_JORNADA = {
    1: [(0, 1), (2, 3)],
    2: [(0, 2), (1, 3)],
    3: [(3, 0), (1, 2)],
}


def fecha_partido(jornada: int, indice_grupo: int) -> date:
    """Calcula la fecha aproximada de los partidos del grupo `indice_grupo` (0-11) en la `jornada`."""
    grupos_por_dia = GRUPOS_POR_DIA[jornada]
    dia_offset = indice_grupo // grupos_por_dia
    return FECHA_INICIO_JORNADA[jornada] + timedelta(days=dia_offset)


def main():
    print("Cargando grupos, modelo y estado actual de selecciones...")
    grupos_df = pd.read_csv(RUTA_GRUPOS)
    estado, modelo, rho = cargar_recursos()

    equipos_por_grupo = {g: grupos_df.loc[grupos_df["grupo"] == g, "seleccion"].tolist() for g in GRUPOS}

    filas = []
    for indice_grupo, grupo in enumerate(GRUPOS):
        equipos = equipos_por_grupo[grupo]  # posiciones 0..3 según orden en grupos_2026.csv

        for jornada, partidos in PARTIDOS_POR_JORNADA.items():
            fecha = fecha_partido(jornada, indice_grupo)

            for pos_local, pos_visita in partidos:
                local = equipos[pos_local]
                visita = equipos[pos_visita]

                pred = predecir_partido(local, visita, estado, modelo, rho, neutral=True)
                marcador_top = pred["marcadores_probables"][0]

                filas.append({
                    "fecha": fecha,
                    "jornada": jornada,
                    "grupo": grupo,
                    "equipo_local": local,
                    "equipo_visita": visita,
                    "goles_esperados_local": round(pred["goles_esperados_local"], 2),
                    "goles_esperados_visita": round(pred["goles_esperados_visita"], 2),
                    "prob_victoria_local": round(pred["prob_victoria_local"], 4),
                    "prob_empate": round(pred["prob_empate"], 4),
                    "prob_victoria_visita": round(pred["prob_victoria_visita"], 4),
                    "prob_over_0_5": round(pred["prob_over_0_5"], 4),
                    "marcador_mas_probable": marcador_top["marcador"],
                })

    calendario = pd.DataFrame(filas).sort_values(["fecha", "grupo"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    calendario.to_csv(RUTA_SALIDA, index=False)
    print(f"Calendario con predicciones guardado en '{RUTA_SALIDA}' ({len(calendario)} partidos)")


if __name__ == "__main__":
    main()
