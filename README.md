# Predicción y Simulación del Mundial 2026

Proyecto de portafolio que combina un modelo de goles esperados (Poisson estilo Dixon-Coles) con una
simulación Monte Carlo para estimar las probabilidades de cada selección en el Mundial 2026.

## Estructura del proyecto

```
Proyecto Mundial 2026/
├── data/                   # Datos crudos y procesados
│   ├── results.csv         # Histórico de partidos internacionales (descargar de Kaggle, ver abajo)
│   ├── dataset_features.csv # Generado por src/data_prep.py
│   ├── elo_ratings.csv      # Ratings Elo finales por selección
│   ├── estado_actual_selecciones.csv # Elo + forma actual por selección (base, pre-torneo)
│   ├── estado_actual_mundial2026.csv # Estado "en vivo", actualizado con resultados reales (opcional)
│   ├── grupos_2026.csv     # Grupos oficiales del Mundial 2026 (lo provee el usuario)
│   ├── calendario_grupos.csv # Calendario de fase de grupos + predicciones (generado)
│   ├── resultados_reales_grupos.csv # Resultados reales (ingresados o desde FIFA.com)
│   ├── calendario_eliminatoria.csv # Cuadro real de eliminatoria + predicciones (desde FIFA.com)
│   ├── equipos_2026.csv     # Mapeo selección <-> nombre/código FIFA y bandera (ISO)
│   ├── modelo_goles.pickle  # Modelo Poisson entrenado
│   └── dixon_coles_rho.json # Parámetro rho del ajuste Dixon-Coles
├── src/
│   ├── data_prep.py        # Limpieza, cálculo de Elo y features por partido
│   ├── model.py             # Modelo Poisson / Dixon-Coles de goles esperados
│   ├── simulate.py          # Simulación Monte Carlo del torneo
│   ├── predecir_partido.py  # Predicción puntual de cualquier partido
│   ├── generar_calendario.py # Genera el calendario de grupos con predicciones
│   ├── predecir_eliminatoria.py # Predice los partidos de eliminatoria ya definidos
│   ├── actualizar_estado_mundial.py # Actualiza Elo/forma con resultados reales del Mundial
│   └── actualizar_resultados_fifa.py # Descarga fechas/resultados/cuadro real desde la API de FIFA.com
├── notebooks/
│   └── analisis.ipynb       # Visualización de resultados
├── output/
│   ├── resultados.csv       # Probabilidades por selección (grupo, 8vos, 4tos, semis, final, campeón)
│   ├── predicciones_fase_grupos.csv # Predicciones de los 72 partidos de grupos
│   └── resumen_whatsapp.txt # Resumen top 10 favoritos para compartir
├── app.py                   # Dashboard interactivo (Streamlit)
├── requirements.txt
└── README.md
```

## Dataset

Este proyecto usa el dataset de Kaggle **"International football results from 1872 to 2025"**
(`martj42/international-football-results-from-1872-to-2017`).

### Descarga manual (recomendado)

1. Ingresa a: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
2. Descarga el archivo `results.csv`
3. Colócalo en la carpeta `data/` de este proyecto, con el nombre `data/results.csv`

### Descarga vía API de Kaggle (opcional)

1. Instala el paquete `kaggle` (incluido en `requirements.txt`)
2. Genera un token API en tu cuenta de Kaggle (Account > Create New API Token) y guarda
   `kaggle.json` en `C:\Users\<usuario>\.kaggle\kaggle.json`
3. Ejecuta:
   ```powershell
   kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 -p data --unzip
   ```

## Flujo de trabajo

1. **Fase 1 - Datos**: `python src/data_prep.py` genera `data/dataset_features.csv`, `data/elo_ratings.csv`
   y `data/estado_actual_selecciones.csv`
2. **Fase 2 - Modelo**: `python src/model.py` entrena y valida el modelo Poisson/Dixon-Coles
3. **Fase 3 - Simulación**: `python src/simulate.py` simula el Mundial 2026 10,000 veces y genera `output/resultados.csv`
4. **Fase 4 - Visualización**: abrir `notebooks/analisis.ipynb`
5. **Predicciones puntuales**: `python src/predecir_partido.py` (genera `output/predicciones_fase_grupos.csv`
   y permite consultar cualquier partido vía `predecir_partido(...)`)
6. **Calendario**: `python src/generar_calendario.py` genera `data/calendario_grupos.csv`
7. **Dashboard**: `streamlit run app.py`

## Dashboard (Streamlit)

`app.py` muestra, para cada partido de la fase de grupos: la fecha, la predicción del modelo
(goles esperados, P(Local)/P(Empate)/P(Visita), marcador más probable) y permite ingresar el
resultado real para calcular el **% de acierto** del modelo en tiempo real. Incluye filtros por
grupo, selección y fecha en la barra lateral.

La pestaña "Fase final" muestra las tablas de posiciones de cada grupo calculadas en vivo a
partir de los resultados ingresados, y el cuadro real de la fase eliminatoria (dieciseisavos
en adelante) tal como lo publica FIFA: los cruces de dieciseisavos muestran los equipos reales
en cuanto termina la fase de grupos, y las rondas siguientes muestran "Ganador Partido N" hasta
que ese partido se dispute. Cada partido con ambos equipos definidos incluye su predicción
(goles esperados, P(V/E/D), P(Over 0.5), marcador más probable).

### Banderas

El dashboard muestra la bandera de cada selección usando [flagcdn.com](https://flagcdn.com),
a partir de los códigos ISO 3166-1 definidos en `data/equipos_2026.csv`.

### Actualizar resultados y fechas desde FIFA.com

El botón **"🔄 Actualizar resultados desde FIFA.com"** en la barra lateral del dashboard
ejecuta automáticamente, en orden:

```powershell
python src/actualizar_resultados_fifa.py   # descarga fechas, marcadores y el cuadro de eliminatoria
python src/actualizar_estado_mundial.py    # recalcula Elo/forma con esos resultados
python src/predecir_eliminatoria.py        # predice los partidos de eliminatoria ya definidos
python src/simulate.py                     # vuelve a simular el torneo
```

`actualizar_resultados_fifa.py` consulta la API pública (no oficial) de FIFA.com
(`api.fifa.com/api/v3/calendar/matches`) para el Mundial 2026, sincroniza las fechas
reales en `data/calendario_grupos.csv` y guarda los marcadores de los partidos ya
jugados en `data/resultados_reales_grupos.csv`, con el mismo formato que si se
ingresaran manualmente desde el dashboard. También guarda en
`data/calendario_eliminatoria.csv` el cuadro real de dieciseisavos en adelante: FIFA
resuelve los nombres reales de los equipos de dieciseisavos en cuanto termina la fase
de grupos (usando el campo `PlaceHolderA`/`PlaceHolderB` de su API), por lo que no es
necesario adivinar el cruce con un cuadro simplificado.

Nota: en esta máquina la verificación del certificado TLS de `api.fifa.com` falla
(interferencia de antivirus/firewall local), por lo que el script usa `verify=False`
únicamente para esa llamada de solo lectura a una API pública.

### Actualizar el modelo con resultados reales del Mundial

También puedes ejecutar manualmente, después de ingresar resultados en el dashboard
(se guardan en `data/resultados_reales_grupos.csv`):

```powershell
python src/actualizar_estado_mundial.py
```

Esto recalcula el **Elo** (K=40, factor de Mundial) y la **forma reciente** de cada selección
incorporando los partidos reales jugados, y guarda el resultado en
`data/estado_actual_mundial2026.csv`. Si este archivo existe, `simulate.py` y
`predecir_partido.py` lo usan automáticamente en lugar del estado pre-torneo — por lo que puedes
volver a correr `python src/simulate.py` para obtener probabilidades actualizadas del torneo a
medida que avanza.

Nota: este proceso no reentrana los coeficientes de la regresión Poisson (eso requeriría volver
a ejecutar `data_prep.py` + `model.py` con los partidos del Mundial incorporados al histórico).
Para un torneo de ~7 partidos por equipo, el Elo y la forma reciente son los componentes que más
cambian de un partido a otro; los coeficientes de la regresión son estables y no es necesario
reentrenarlos partido a partido — se podría hacer una vez al cerrar la fase de grupos si se
quiere máxima precisión.

## Metodología

- **Rating Elo propio**: K=40 para partidos de Mundial, K=30 para eliminatorias, K=20 para amistosos.
  Ventaja de local de 100 puntos cuando el partido no es en sede neutral.
- **Features por partido**: diferencia de Elo, forma reciente (puntos y diferencia de gol en los
  últimos 10 partidos), promedio ponderado de goles a favor/en contra, condición de local/neutral.
- **Modelo de goles esperados**: regresión Poisson sobre las features anteriores, con ajuste
  Dixon-Coles para la dependencia entre marcadores bajos.
- **Simulación**: Monte Carlo con 10,000 iteraciones del torneo completo (fase de grupos +
  eliminación directa desde 16avos, con prórroga/penales en empates).
