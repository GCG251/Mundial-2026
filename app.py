"""
Dashboard del Mundial 2026: predicciones del modelo vs resultados reales.

- Fase de grupos: calendario con predicciones (goles esperados, prob. V/E/D,
  marcador más probable), ingreso de resultados reales y % de acierto del modelo.
- Fase final: tablas de posiciones de grupo (en vivo, según resultados ingresados)
  y cuadro de dieciseisavos de final con los cruces (placeholders hasta que se
  definan los clasificados).

Ejecutar con:
    streamlit run app.py
"""

import subprocess
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------------
# Configuración y rutas
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
SRC_DIR = BASE_DIR / "src"

RUTA_CALENDARIO = DATA_DIR / "calendario_grupos.csv"
RUTA_RESULTADOS_REALES = DATA_DIR / "resultados_reales_grupos.csv"
RUTA_SIMULACION = OUTPUT_DIR / "resultados.csv"
RUTA_EQUIPOS = DATA_DIR / "equipos_2026.csv"

GRUPOS = list("ABCDEFGHIJKL")

# Cada cuánto se refresca automáticamente con datos de FIFA.com al cargar/recargar
# el dashboard (en segundos).
INTERVALO_ACTUALIZACION_AUTOMATICA = 600  # 10 minutos

st.set_page_config(page_title="Mundial 2026 - Predicciones", layout="wide")


# ----------------------------------------------------------------------------
# Pipeline de actualización con datos reales de FIFA.com
# ----------------------------------------------------------------------------
def ejecutar_pipeline_fifa() -> tuple[bool, str | None]:
    """
    Ejecuta, en orden: descarga de resultados/fechas reales desde FIFA.com,
    recálculo de Elo/forma y nueva simulación Monte Carlo.
    Devuelve (ok, mensaje_error).
    """
    for script in ["actualizar_resultados_fifa.py", "actualizar_estado_mundial.py", "simulate.py"]:
        try:
            resultado = subprocess.run(
                [sys.executable, str(SRC_DIR / script)],
                capture_output=True, text=True, cwd=str(SRC_DIR), timeout=300,
            )
        except Exception as e:
            return False, f"{script}: {e}"
        if resultado.returncode != 0:
            return False, f"{script}:\n{resultado.stderr[-1000:]}"
    return True, None


# ----------------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------------
@st.cache_data
def cargar_calendario() -> pd.DataFrame:
    return pd.read_csv(RUTA_CALENDARIO, parse_dates=["fecha"])


def cargar_resultados_reales() -> pd.DataFrame:
    columnas = ["grupo", "equipo_local", "equipo_visita", "goles_local_real", "goles_visita_real"]
    if RUTA_RESULTADOS_REALES.exists():
        return pd.read_csv(RUTA_RESULTADOS_REALES)
    return pd.DataFrame(columns=columnas)


@st.cache_data
def cargar_simulacion() -> pd.DataFrame:
    return pd.read_csv(RUTA_SIMULACION)


@st.cache_data
def cargar_banderas() -> dict[str, str]:
    """Devuelve {seleccion: url_bandera} usando flagcdn.com (códigos ISO 3166-1)."""
    equipos = pd.read_csv(RUTA_EQUIPOS)
    return {
        fila["seleccion"]: f"https://flagcdn.com/h40/{fila['codigo_iso2']}.png"
        for _, fila in equipos.iterrows()
    }


BANDERAS = cargar_banderas()


def bandera(seleccion: str) -> str:
    return BANDERAS.get(seleccion, "")


# ----------------------------------------------------------------------------
# Lógica de predicción / acierto
# ----------------------------------------------------------------------------
def resultado_desde_goles(goles_local, goles_visita):
    if pd.isna(goles_local) or pd.isna(goles_visita):
        return None
    if goles_local > goles_visita:
        return "Gana local"
    if goles_local < goles_visita:
        return "Gana visita"
    return "Empate"


def prediccion_resultado(fila) -> str:
    """
    Predicción del modelo a partir del marcador más probable (no de las
    probabilidades agregadas de V/E/D, que casi nunca dan "Empate" como
    máximo porque la probabilidad de empate se reparte entre varios
    marcadores 0-0, 1-1, 2-2, etc.).
    """
    goles_local, goles_visita = fila["marcador_mas_probable"].split("-")
    return resultado_desde_goles(int(goles_local), int(goles_visita))


# ----------------------------------------------------------------------------
# Tabla de posiciones de un grupo a partir de resultados reales
# ----------------------------------------------------------------------------
def calcular_tabla_grupo(equipos: list[str], df_grupo: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Calcula la tabla de posiciones de un grupo con los resultados reales
    disponibles. Devuelve (tabla, grupo_completo).
    """
    tabla = {e: {"PJ": 0, "PTS": 0, "GF": 0, "GC": 0} for e in equipos}

    for _, fila in df_grupo.iterrows():
        gl, gv = fila["goles_local_real"], fila["goles_visita_real"]
        if pd.isna(gl) or pd.isna(gv):
            continue
        local, visita = fila["equipo_local"], fila["equipo_visita"]
        gl, gv = int(gl), int(gv)

        tabla[local]["PJ"] += 1
        tabla[visita]["PJ"] += 1
        tabla[local]["GF"] += gl
        tabla[local]["GC"] += gv
        tabla[visita]["GF"] += gv
        tabla[visita]["GC"] += gl

        if gl > gv:
            tabla[local]["PTS"] += 3
        elif gl < gv:
            tabla[visita]["PTS"] += 3
        else:
            tabla[local]["PTS"] += 1
            tabla[visita]["PTS"] += 1

    filas = []
    for e in equipos:
        t = tabla[e]
        filas.append({
            "seleccion": e, "PJ": t["PJ"], "PTS": t["PTS"],
            "GF": t["GF"], "GC": t["GC"], "DG": t["GF"] - t["GC"],
        })

    tabla_df = pd.DataFrame(filas).sort_values(
        ["PTS", "DG", "GF"], ascending=False
    ).reset_index(drop=True)

    grupo_completo = all(tabla[e]["PJ"] == 3 for e in equipos)
    return tabla_df, grupo_completo


# ----------------------------------------------------------------------------
# Actualización automática con datos de FIFA.com (al cargar o tras N minutos)
# ----------------------------------------------------------------------------
if "ultima_actualizacion_fifa" not in st.session_state:
    st.session_state["ultima_actualizacion_fifa"] = 0.0
    st.session_state["error_actualizacion_fifa"] = None

if time.time() - st.session_state["ultima_actualizacion_fifa"] > INTERVALO_ACTUALIZACION_AUTOMATICA:
    with st.spinner("Consultando api.fifa.com y recalculando Elo/forma..."):
        ok, error = ejecutar_pipeline_fifa()
    st.session_state["ultima_actualizacion_fifa"] = time.time()
    st.session_state["error_actualizacion_fifa"] = error
    if ok:
        st.cache_data.clear()


# ----------------------------------------------------------------------------
# Carga inicial
# ----------------------------------------------------------------------------
calendario = cargar_calendario()
reales = cargar_resultados_reales()
simulacion = cargar_simulacion()

df = calendario.merge(
    reales, on=["grupo", "equipo_local", "equipo_visita"], how="left"
)
if "goles_local_real" not in df.columns:
    df["goles_local_real"] = pd.NA
    df["goles_visita_real"] = pd.NA


# ----------------------------------------------------------------------------
# Encabezado: actualización en vivo + filtros (en la parte superior)
# ----------------------------------------------------------------------------
st.title("🏆 Mundial 2026 - Predicciones del modelo vs resultados reales")

col_actualizar, col_grupo, col_seleccion, col_fecha = st.columns([1.2, 1, 1, 1])

with col_actualizar:
    if st.button("🔄 Actualizar resultados desde FIFA.com"):
        with st.spinner("Consultando api.fifa.com y recalculando Elo/forma..."):
            ok, error = ejecutar_pipeline_fifa()
        st.session_state["ultima_actualizacion_fifa"] = time.time()
        st.session_state["error_actualizacion_fifa"] = error
        if ok:
            st.cache_data.clear()
            st.success("Resultados, calendario y simulación actualizados.")
            st.rerun()
        else:
            st.error(f"No se pudo actualizar:\n{error}")

    if st.session_state.get("error_actualizacion_fifa"):
        st.warning(f"Última actualización automática falló: {st.session_state['error_actualizacion_fifa']}")
    else:
        ultima = st.session_state.get("ultima_actualizacion_fifa", 0.0)
        if ultima:
            st.caption(
                f"Última actualización: {datetime.fromtimestamp(ultima).strftime('%Y-%m-%d %H:%M:%S')}"
            )

with col_grupo:
    grupos_sel = st.multiselect("Grupo", GRUPOS)

equipos_todos = sorted(set(df["equipo_local"]) | set(df["equipo_visita"]))
with col_seleccion:
    equipos_sel = st.multiselect("Selección", equipos_todos)

fechas_disponibles = sorted(df["fecha"].dt.date.unique())
with col_fecha:
    fechas_sel = st.multiselect("Fecha", fechas_disponibles)

df_filtrado = df.copy()
if grupos_sel:
    df_filtrado = df_filtrado[df_filtrado["grupo"].isin(grupos_sel)]
if equipos_sel:
    df_filtrado = df_filtrado[
        df_filtrado["equipo_local"].isin(equipos_sel) | df_filtrado["equipo_visita"].isin(equipos_sel)
    ]
if fechas_sel:
    df_filtrado = df_filtrado[df_filtrado["fecha"].dt.date.isin(fechas_sel)]


# ----------------------------------------------------------------------------
# Layout principal
# ----------------------------------------------------------------------------
tab_grupos, tab_final = st.tabs(["📋 Fase de grupos", "🏟️ Fase final"])


# --- Tab 1: Fase de grupos --------------------------------------------------
with tab_grupos:
    df_filtrado = df_filtrado.copy()
    df_filtrado["resultado_real"] = df_filtrado.apply(
        lambda r: resultado_desde_goles(r["goles_local_real"], r["goles_visita_real"]), axis=1
    )
    df_filtrado["prediccion_modelo"] = df_filtrado.apply(prediccion_resultado, axis=1)
    df_filtrado["acierto"] = df_filtrado.apply(
        lambda r: (r["prediccion_modelo"] == r["resultado_real"]) if r["resultado_real"] is not None else None,
        axis=1,
    )
    df_filtrado["marcador_real"] = df_filtrado.apply(
        lambda r: f"{int(r['goles_local_real'])}-{int(r['goles_visita_real'])}"
        if pd.notna(r["goles_local_real"]) and pd.notna(r["goles_visita_real"])
        else None,
        axis=1,
    )
    df_filtrado["acierto_marcador"] = df_filtrado.apply(
        lambda r: (r["marcador_mas_probable"] == r["marcador_real"]) if r["resultado_real"] is not None else None,
        axis=1,
    )

    jugados = df_filtrado["resultado_real"].notna().sum()
    aciertos = df_filtrado["acierto"].sum()
    aciertos_marcador = df_filtrado["acierto_marcador"].sum()

    # --- Over/Under -----------------------------------------------------------
    df_filtrado["goles_esperados_total"] = (
        df_filtrado["goles_esperados_local"] + df_filtrado["goles_esperados_visita"]
    )
    df_filtrado["goles_total_real"] = df_filtrado.apply(
        lambda r: int(r["goles_local_real"]) + int(r["goles_visita_real"])
        if pd.notna(r["goles_local_real"]) and pd.notna(r["goles_visita_real"])
        else None,
        axis=1,
    )

    umbral_ou = st.radio(
        "Línea Over/Under",
        [1.5, 2.5, 3.5],
        index=1,
        horizontal=True,
        format_func=lambda x: f"O/U {x}",
    )

    df_filtrado["pred_ou"] = df_filtrado["goles_esperados_total"].apply(
        lambda x: f"Over {umbral_ou}" if x > umbral_ou else f"Under {umbral_ou}"
    )
    df_filtrado["real_ou"] = df_filtrado["goles_total_real"].apply(
        lambda x: (f"Over {umbral_ou}" if x > umbral_ou else f"Under {umbral_ou}")
        if x is not None else None
    )
    df_filtrado["acierto_ou"] = df_filtrado.apply(
        lambda r: (r["pred_ou"] == r["real_ou"]) if r["real_ou"] is not None else None,
        axis=1,
    )
    aciertos_ou = df_filtrado["acierto_ou"].sum()

    # --- Métricas -------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Partidos con resultado", f"{jugados} / {len(df_filtrado)}")
    c2.metric("Aciertos del modelo (V/E/D)", int(aciertos) if jugados else 0)
    c3.metric("% de acierto (V/E/D)", f"{aciertos / jugados * 100:.1f}%" if jugados else "—")
    c4.metric("% de acierto (marcador exacto)", f"{aciertos_marcador / jugados * 100:.1f}%" if jugados else "—")
    c5.metric(f"% acierto O/U {umbral_ou}", f"{aciertos_ou / jugados * 100:.1f}%" if jugados else "—")

    st.subheader("Calendario y predicciones")

    df_filtrado["bandera_local"] = df_filtrado["equipo_local"].map(bandera)
    df_filtrado["bandera_visita"] = df_filtrado["equipo_visita"].map(bandera)

    columnas_mostrar = [
        "fecha", "jornada", "grupo",
        "bandera_local", "equipo_local", "bandera_visita", "equipo_visita",
        "goles_esperados_local", "goles_esperados_visita", "goles_esperados_total",
        "prob_victoria_local", "prob_empate", "prob_victoria_visita",
        "marcador_mas_probable", "prediccion_modelo",
        "resultado_real", "acierto",
        "marcador_real", "acierto_marcador",
        "pred_ou", "goles_total_real", "acierto_ou",
    ]

    tabla_mostrar = df_filtrado[columnas_mostrar].rename(columns={
        "fecha": "Fecha", "jornada": "Jornada", "grupo": "Grupo",
        "bandera_local": "🏳️ L", "equipo_local": "Local",
        "bandera_visita": "🏳️ V", "equipo_visita": "Visita",
        "goles_esperados_local": "xG Local", "goles_esperados_visita": "xG Visita",
        "goles_esperados_total": "xG Total",
        "prob_victoria_local": "P(Local)", "prob_empate": "P(Empate)", "prob_victoria_visita": "P(Visita)",
        "marcador_mas_probable": "Marcador más probable", "prediccion_modelo": "Predicción modelo",
        "resultado_real": "Resultado real", "acierto": "¿Acierto V/E/D?",
        "marcador_real": "Marcador real", "acierto_marcador": "¿Acierto marcador?",
        "pred_ou": f"Pred O/U {umbral_ou}", "goles_total_real": "Total goles real",
        "acierto_ou": f"¿Acierto O/U {umbral_ou}?",
    })

    st.dataframe(
        tabla_mostrar.style.format({
            "xG Local": "{:.2f}", "xG Visita": "{:.2f}", "xG Total": "{:.2f}",
            "P(Local)": "{:.1%}", "P(Empate)": "{:.1%}", "P(Visita)": "{:.1%}",
        }, na_rep=""),
        column_config={
            "🏳️ L": st.column_config.ImageColumn("🏳️ L", width="small"),
            "🏳️ V": st.column_config.ImageColumn("🏳️ V", width="small"),
        },
        use_container_width=True, hide_index=True,
    )


# --- Tab 2: Fase final -------------------------------------------------------
with tab_final:
    grupos_df = pd.read_csv(DATA_DIR / "grupos_2026.csv")
    equipos_por_grupo = {g: grupos_df.loc[grupos_df["grupo"] == g, "seleccion"].tolist() for g in GRUPOS}

    st.subheader("Tablas de posiciones por grupo (en vivo)")

    primeros, segundos, terceros, grupos_completos = {}, {}, {}, {}
    cols = st.columns(4)

    for i, g in enumerate(GRUPOS):
        df_grupo = df[df["grupo"] == g]
        tabla_g, completo = calcular_tabla_grupo(equipos_por_grupo[g], df_grupo)
        grupos_completos[g] = completo

        primeros[g] = tabla_g.iloc[0]["seleccion"] if completo else f"1° Grupo {g}"
        segundos[g] = tabla_g.iloc[1]["seleccion"] if completo else f"2° Grupo {g}"
        terceros[g] = tabla_g.iloc[2] if completo else None

        with cols[i % 4]:
            st.markdown(f"**Grupo {g}**")
            tabla_g_mostrar = tabla_g.copy()
            tabla_g_mostrar["bandera"] = tabla_g_mostrar["seleccion"].map(bandera)
            tabla_g_mostrar = tabla_g_mostrar[["bandera", "seleccion", "PJ", "PTS", "GF", "GC", "DG"]]
            st.dataframe(
                tabla_g_mostrar.rename(columns={"bandera": "🏳️", "seleccion": "Selección"}),
                column_config={"🏳️": st.column_config.ImageColumn("🏳️", width="small")},
                hide_index=True, use_container_width=True,
            )

    st.divider()
    st.subheader("Cuadro de dieciseisavos de final")

    todos_completos = all(grupos_completos.values())

    if todos_completos:
        ranking_terceros = sorted(
            [(g, terceros[g]) for g in GRUPOS],
            key=lambda x: (x[1]["PTS"], x[1]["DG"], x[1]["GF"]),
            reverse=True,
        )
        mejores_terceros = [g for g, _ in ranking_terceros[:8]]
        terceros_clasificados = [terceros[g]["seleccion"] for g in mejores_terceros]
    else:
        terceros_clasificados = [f"3° (Grupo {g})" for g in GRUPOS[:8]]

    st.caption(
        "Cruce simplificado (ver README): 1° de cada grupo vs 2° del grupo siguiente, "
        "y los 8 mejores terceros emparejados tipo 'serpiente'. "
        "Los nombres reales aparecen una vez que el grupo correspondiente esté completo."
    )

    bracket = []
    for i, g in enumerate(GRUPOS):
        g_siguiente = GRUPOS[(i + 1) % len(GRUPOS)]
        bracket.append((primeros[g], segundos[g_siguiente]))

    t = terceros_clasificados
    for a, b in [(t[0], t[7]), (t[1], t[6]), (t[2], t[5]), (t[3], t[4])]:
        bracket.append((a, b))

    def linea_equipo(nombre: str) -> str:
        url = bandera(nombre)
        return f"![]({url}) {nombre}" if url else nombre

    cols_bracket = st.columns(4)
    for idx, (a, b) in enumerate(bracket):
        with cols_bracket[idx % 4]:
            st.info(f"**Partido {idx + 1}**\n\n{linea_equipo(a)}\n\nvs\n\n{linea_equipo(b)}")

    st.divider()
    st.subheader("Probabilidades de la simulación Monte Carlo (10,000 torneos)")
    st.caption(
        "Heatmap de probabilidad de avance por ronda. Se recalcula al presionar "
        "'🔄 Actualizar resultados desde FIFA.com' en la barra lateral, que actualiza "
        "el Elo/forma de cada selección con los resultados reales y vuelve a simular "
        "el torneo completo."
    )

    sim_mostrar = simulacion.copy()
    sim_mostrar["bandera"] = sim_mostrar["seleccion"].map(bandera)
    sim_mostrar = sim_mostrar[[
        "bandera", "seleccion", "grupo", "elo",
        "prob_pasa_grupo", "prob_llega_8vos", "prob_llega_4tos",
        "prob_llega_semis", "prob_llega_final", "prob_campeon",
    ]].rename(columns={
        "bandera": "🏳️", "seleccion": "Selección", "grupo": "Grupo", "elo": "Elo",
        "prob_pasa_grupo": "Pasa grupo", "prob_llega_8vos": "8vos",
        "prob_llega_4tos": "4tos", "prob_llega_semis": "Semis",
        "prob_llega_final": "Final", "prob_campeon": "Campeón",
    })
    cols_pct = ["Pasa grupo", "8vos", "4tos", "Semis", "Final", "Campeón"]
    st.dataframe(
        sim_mostrar.style.format({c: "{:.1%}" for c in cols_pct}).background_gradient(
            subset=cols_pct, cmap="YlOrRd"
        ),
        column_config={"🏳️": st.column_config.ImageColumn("🏳️", width="small")},
        hide_index=True, use_container_width=True,
    )
