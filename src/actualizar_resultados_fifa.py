"""
Actualiza el calendario y los resultados reales de la fase de grupos del
Mundial 2026 consultando la API pública (no oficial) de FIFA.com:

    https://api.fifa.com/api/v3/calendar/matches?idCompetition=17&idSeason=285023

Para cada partido de la fase de grupos:
- Sincroniza la fecha real (LocalDate) en data/calendario_grupos.csv
- Si el partido ya se jugó (hay marcador), guarda el resultado en
  data/resultados_reales_grupos.csv (mismo formato que genera el dashboard)

Después de ejecutar este script conviene correr:
    python src/actualizar_estado_mundial.py
    python src/simulate.py

para que el Elo/forma y las probabilidades del torneo incorporen los
resultados reales más recientes.
"""

from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RUTA_EQUIPOS = DATA_DIR / "equipos_2026.csv"
RUTA_CALENDARIO = DATA_DIR / "calendario_grupos.csv"
RUTA_RESULTADOS_REALES = DATA_DIR / "resultados_reales_grupos.csv"

URL_API_FIFA = "https://api.fifa.com/api/v3/calendar/matches"
ID_COMPETICION = 17       # FIFA World Cup
ID_TEMPORADA = 285023     # FIFA World Cup 2026


def obtener_partidos_fifa() -> list[dict]:
    """Descarga el calendario completo del torneo desde la API de FIFA."""
    parametros = {
        "idCompetition": ID_COMPETICION,
        "idSeason": ID_TEMPORADA,
        "language": "en",
        "count": 500,
    }
    cabeceras = {"User-Agent": "Mozilla/5.0"}

    # Nota: verify=False porque, en esta máquina, la cadena de certificados
    # de api.fifa.com no valida (interferencia del antivirus/firewall local).
    respuesta = requests.get(URL_API_FIFA, params=parametros, headers=cabeceras, timeout=60, verify=False)
    respuesta.raise_for_status()
    return respuesta.json().get("Results", [])


def construir_mapa_equipos() -> dict[str, str]:
    """Devuelve un diccionario {nombre_equipo_en_FIFA: seleccion_en_nuestros_datos}."""
    equipos = pd.read_csv(RUTA_EQUIPOS)
    return dict(zip(equipos["nombre_fifa"], equipos["seleccion"]))


def main():
    print("Descargando calendario del Mundial 2026 desde api.fifa.com...")
    partidos = obtener_partidos_fifa()
    mapa_equipos = construir_mapa_equipos()

    calendario = pd.read_csv(RUTA_CALENDARIO, parse_dates=["fecha"])

    fechas_actualizadas = 0
    resultados_nuevos = []

    for partido in partidos:
        grupo_nombre = (partido.get("GroupName") or [{}])[0].get("Description", "")
        if not grupo_nombre.startswith("Group "):
            continue  # solo nos interesa la fase de grupos

        grupo = grupo_nombre.replace("Group ", "").strip()
        nombre_local = partido["Home"]["TeamName"][0]["Description"]
        nombre_visita = partido["Away"]["TeamName"][0]["Description"]

        equipo_local_fifa = mapa_equipos.get(nombre_local)
        equipo_visita_fifa = mapa_equipos.get(nombre_visita)
        if equipo_local_fifa is None or equipo_visita_fifa is None:
            continue

        # Ubicar la fila correspondiente en nuestro calendario (el orden
        # local/visita puede no coincidir con el de FIFA, así que se busca
        # por el par de equipos sin importar el orden).
        equipos_partido = {equipo_local_fifa, equipo_visita_fifa}
        coincidencias = calendario[
            (calendario["grupo"] == grupo)
            & (calendario.apply(lambda f: {f["equipo_local"], f["equipo_visita"]} == equipos_partido, axis=1))
        ]
        if coincidencias.empty:
            continue

        idx = coincidencias.index[0]
        fila = calendario.loc[idx]

        # --- Sincronizar fecha real ---
        fecha_real = pd.to_datetime(partido["LocalDate"]).date()
        if fila["fecha"].date() != fecha_real:
            calendario.loc[idx, "fecha"] = pd.Timestamp(fecha_real)
            fechas_actualizadas += 1

        # --- Resultado real (si ya se jugó) ---
        goles_local_fifa = partido["HomeTeamScore"]
        goles_visita_fifa = partido["AwayTeamScore"]
        if goles_local_fifa is None or goles_visita_fifa is None:
            continue

        if equipo_local_fifa == fila["equipo_local"]:
            goles_local, goles_visita = goles_local_fifa, goles_visita_fifa
        else:
            goles_local, goles_visita = goles_visita_fifa, goles_local_fifa

        resultados_nuevos.append({
            "grupo": grupo,
            "equipo_local": fila["equipo_local"],
            "equipo_visita": fila["equipo_visita"],
            "goles_local_real": goles_local,
            "goles_visita_real": goles_visita,
        })

    calendario.to_csv(RUTA_CALENDARIO, index=False)
    print(f"Fechas sincronizadas en '{RUTA_CALENDARIO.name}': {fechas_actualizadas} cambios")

    if resultados_nuevos:
        nuevos_df = pd.DataFrame(resultados_nuevos)

        if RUTA_RESULTADOS_REALES.exists():
            existentes = pd.read_csv(RUTA_RESULTADOS_REALES)
            combinados = pd.concat([existentes, nuevos_df], ignore_index=True)
        else:
            combinados = nuevos_df

        combinados = combinados.drop_duplicates(
            subset=["grupo", "equipo_local", "equipo_visita"], keep="last"
        )
        combinados.to_csv(RUTA_RESULTADOS_REALES, index=False)
        print(f"Resultados reales guardados en '{RUTA_RESULTADOS_REALES.name}' ({len(combinados)} partidos jugados)")
    else:
        print("Todavía no hay partidos con resultado.")


if __name__ == "__main__":
    main()
