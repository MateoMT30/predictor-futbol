"""
app.py
======

App web (Flask) sobre el mismo pipeline que usa main.py (CLI). Reutiliza
data_loader, ratings, models, simulation y report_html sin duplicar nada
del cálculo estadístico.

Flujo principal (lo que se usa desde el celular):
  1. GET /                  -> elegir competición
  2. GET /partidos?competition=X -> lista de próximos partidos (fecha/hora),
     clickeable, traída de football-data.org y cacheada en el servidor
     (ver src/connectors/football_data_connector.py) para no gastar la
     cuota gratuita de la API en cada visita.
  3. GET /predecir?...      -> el reporte del partido elegido

No hay cuotas ni value bets en este flujo: son opcionales y le agregaban
fricción a un formulario que la mayoría de usuarios no necesita. La
capacidad sigue existiendo en el CLI (`--cuotas`) para quien la quiera.

Modo avanzado/pruebas: si no tienes configurada FOOTBALL_DATA_API_KEY (por
ejemplo, probando en tu computador sin API key todavía), /avanzado permite
escribir los nombres de los equipos a mano usando el CSV de ejemplo — el
mismo comportamiento que main.py, sin depender de la API.
"""

import json
import os

# IMPORTANTE: esto debe ejecutarse ANTES de importar numpy/scipy/pandas
# (más abajo, vía src/*). En hosting con recursos limitados y compartidos
# (ej. el plan gratuito de Render: 512 MB RAM / 0.1 CPU), las librerías de
# álgebra lineal que usa numpy/scipy (OpenBLAS) suelen detectar el número
# de núcleos del servidor físico completo, no la fracción real asignada al
# contenedor, e intentan lanzar un hilo de cálculo por núcleo detectado.
# Cada hilo reserva su propia memoria de trabajo, lo que puede disparar el
# consumo de RAM muy por encima de lo que el proceso necesitaría en
# realidad — esto causó un "Worker was sent SIGKILL! Perhaps out of
# memory?" real en producción. Forzar un solo hilo por librería resuelve
# ese problema clásico de numpy/scipy en contenedores compartidos.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request, redirect, url_for

from src.connectors.csv_connector import CSVConnector
from src.connectors.football_data_connector import FootballDataConnector, COMPETITIONS
from src.connectors.fifa_reports_connector import (
    clear_request_budget,
    enrich_with_fifa_reports,
    get_team_summary_stats,
    start_request_budget,
)
from src.connectors.football_couk_connector import enrich_with_couk_stats
from src.connectors.international_results_connector import (
    align_team_names,
    fetch_international_results,
)
from src.data_loader import load_from_connector
from src.ratings import EloRatingSystem, RatingsConfig
from src.models.goles import DixonColesModel, GoalsModelConfig
from src.models.corners import CornersModel, CornersModelConfig
from src.models.tiros import ShotsOnTargetModel, ShotsModelConfig
from src.models.tarjetas import CardsModel, CardsModelConfig
from src.simulation import MatchSimulator, SimulationConfig
from src.report_html import render_html_report
from src.main import load_config, build_report
from src.i18n import team_name_es, to_colombia_time, day_label_es
from src.web_style import wrap_page

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "examples" / "historico_ejemplo.csv"

# De qué competición traer historial de respaldo cuando un equipo no tiene
# partidos en la elegida (recién ascendidos/descendidos). Solo Inglaterra por
# ahora: es el único par primera/segunda división que cubre tanto el plan
# gratuito de football-data.org como football-data.co.uk.
FALLBACK_HISTORY = {
    "PL": "ELC",
    "ELC": "PL",
}

# --- Ventana de historial adaptativa (muestra chica) -----------------------
# Con 365 días, una selección puede llegar a un Mundial con 3 partidos en el
# histórico ("muestra muy chica, tómalo con pinzas"). Ampliar la ventana da
# más muestra REAL (clasificatorias/amistosos), y el sesgo de "mirar
# históricos viejos" que preocupa (cambio de plantilla, decadencia) lo
# controla el decaimiento temporal que ya tiene el modelo: con xi=0.0018,
# un partido de hace 1 año pesa ~50% y uno de hace 2 años ~27% — lo
# reciente sigue mandando. La ventana solo se amplía si hace falta.
# Torneos (vs ligas): agenda con torneo completo hacia atrás, sección de
# clasificatorias, y tarjeta "¿Quién clasifica?" en el reporte.
TOURNAMENTS = {"WC", "EC", "CL", "CLI"}
NATIONAL_TEAM_COMPETITIONS = {"WC", "EC"}  # torneos de selecciones
MIN_MUESTRA_EQUIPO = 10          # partidos mínimos deseados por equipo
VENTANAS_AMPLIADAS = (730, 1095)  # 2 años, luego 3, hasta lograr la muestra
MAX_EQUIPOS_MODELO = 150          # tope de equipos en el ajuste (memoria: ver
                                  # bug #2 de NOTAS — OOM real en Render free)


def _team_match_count(df, team):
    return int(((df["equipo_local"] == team) | (df["equipo_visitante"] == team)).sum())


def _prune_to_neighborhood(df, seeds, hops=2):
    """Recorta el histórico al "vecindario" de los equipos del partido:
    ellos, sus rivales, y los rivales de sus rivales (hops=2). Dixon-Coles
    solo necesita una red de comparación conectada alrededor de los dos
    equipos — un partido de una confederación lejana no aporta nada para
    este pronóstico y sí infla el número de parámetros a optimizar (cada
    equipo agrega 2), que fue la causa del OOM en Render free."""
    keep = set(seeds)
    for _ in range(hops):
        mask = df["equipo_local"].isin(keep) | df["equipo_visitante"].isin(keep)
        keep = keep | set(df.loc[mask, "equipo_local"]) | set(df.loc[mask, "equipo_visitante"])
    return df[df["equipo_local"].isin(keep) & df["equipo_visitante"].isin(keep)]


def _ensure_min_sample(connector, competition, matches_df, local, visitante, hasta, avisos):
    """Si alguno de los dos equipos tiene menos de MIN_MUESTRA_EQUIPO
    partidos en la ventana de 365 días, reintenta con ventanas más largas
    (2 y 3 años) y recorta al vecindario relevante para no exceder el
    presupuesto de memoria. Modifica avisos in-place para que el reporte
    explique qué se hizo."""
    faltantes = [t for t in (local, visitante)
                 if _team_match_count(matches_df, t) < MIN_MUESTRA_EQUIPO]
    if not faltantes:
        return matches_df

    hoy = datetime.now(timezone.utc)

    # SELECCIONES (Mundial/Eurocopa): la API .org solo trae los partidos del
    # torneo en sí — las clasificatorias y amistosos NO están en el plan
    # gratis, así que ampliar la ventana contra la misma API no consigue
    # nada (verificado: España seguía con 3 partidos). La muestra real está
    # en el dataset público de resultados internacionales.
    if competition in NATIONAL_TEAM_COMPETITIONS:
        try:
            intl = fetch_international_results(desde=hoy - timedelta(days=1095))
        except Exception:
            intl = None
        if intl is not None and not intl.empty:
            equipos_org = (set(matches_df["equipo_local"]) | set(matches_df["equipo_visitante"])
                           | {local, visitante})
            intl = align_team_names(intl, equipos_org)
            # Vecindario de 1 salto: los 2 equipos, sus rivales, y los
            # partidos entre esos rivales (red de comparación conectada sin
            # traerse todas las confederaciones del planeta).
            intl = _prune_to_neighborhood(intl, {local, visitante}, hops=1)
            # Dedupe contra lo que ya vino de la API (el torneo en curso
            # aparece en ambas fuentes): fecha + par de equipos.
            ya = set(zip(pd.to_datetime(matches_df["fecha"]).dt.date,
                         matches_df["equipo_local"], matches_df["equipo_visitante"]))
            intl = intl[[
                (f.date(), h, a) not in ya
                for f, h, a in zip(intl["fecha"], intl["equipo_local"], intl["equipo_visitante"])
            ]]
            if not intl.empty and any(
                _team_match_count(intl, t) > 0 for t in faltantes
            ):
                matches_df = pd.concat([matches_df, intl], ignore_index=True)
                # Las dos fuentes pueden traer la fecha con tipos distintos
                # (string vs datetime); se unifica antes de ordenar.
                matches_df["fecha"] = pd.to_datetime(matches_df["fecha"])
                matches_df = matches_df.sort_values("fecha").reset_index(drop=True)
                equipos_es = " y ".join(team_name_es(t) for t in faltantes)
                avisos.append(
                    f"Muestra ampliada: la API del torneo solo trae los partidos del torneo en sí, "
                    f"así que el historial de {equipos_es} se completó con sus clasificatorias y "
                    f"amistosos de los últimos 3 años (dataset público de resultados "
                    f"internacionales). Los partidos antiguos pesan menos en el modelo "
                    f"(decaimiento temporal): la forma reciente sigue dominando."
                )
        if all(_team_match_count(matches_df, t) >= MIN_MUESTRA_EQUIPO for t in (local, visitante)):
            return matches_df
        faltantes = [t for t in (local, visitante)
                     if _team_match_count(matches_df, t) < MIN_MUESTRA_EQUIPO]

    dias_adoptados = None
    for dias in VENTANAS_AMPLIADAS:
        try:
            ampliado, _ = load_from_connector(
                connector, liga=competition,
                desde=(hoy - timedelta(days=dias)).strftime("%Y-%m-%d"), hasta=hasta,
            )
        except Exception:
            break
        if ampliado.empty or len(ampliado) <= len(matches_df):
            continue
        recortado = _prune_to_neighborhood(ampliado, {local, visitante}, hops=2)
        n_equipos = len(set(recortado["equipo_local"]) | set(recortado["equipo_visitante"]))
        if n_equipos > MAX_EQUIPOS_MODELO:
            recortado = _prune_to_neighborhood(ampliado, {local, visitante}, hops=1)
        # Solo se adopta la ventana ampliada si de verdad mejora la muestra
        # de los equipos del partido (recortada al vecindario).
        if all(_team_match_count(recortado, t) > _team_match_count(matches_df, t) for t in faltantes):
            matches_df = recortado.sort_values("fecha").reset_index(drop=True)
            dias_adoptados = dias
        if all(_team_match_count(matches_df, t) >= MIN_MUESTRA_EQUIPO for t in (local, visitante)):
            break

    if dias_adoptados:
        equipos_es = " y ".join(team_name_es(t) for t in faltantes)
        avisos.append(
            f"Muestra ampliada: {equipos_es} tenía muy pocos partidos en los últimos 365 días, "
            f"así que se usó el historial de los últimos {dias_adoptados // 365} años "
            f"(clasificatorias y amistosos). Los partidos antiguos pesan menos en el modelo "
            f"(decaimiento temporal), así que la forma y la plantilla recientes siguen "
            f"dominando el pronóstico."
        )
    return matches_df

INDEX_BODY = """
<h1>⚽ Predictor Fútbol</h1>
<div class="subtitle">Elige una competición para ver los próximos partidos.</div>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
<div class="card">
  <form method="GET" action="/partidos">
    <label>Competición</label>
    <select name="competition">
      {% for code, name in competitions.items() %}
      <option value="{{ code }}">{{ name }}</option>
      {% endfor %}
    </select>
    <button type="submit">Ver próximos partidos →</button>
  </form>
</div>
<details>
  <summary>Modo avanzado (sin API, para pruebas con tus propios datos)</summary>
  <div class="card">
    <form method="POST" action="/predecir_manual">
      <label>Equipo local</label>
      <input type="text" name="local" required placeholder="ej. Colombia">
      <label>Equipo visitante</label>
      <input type="text" name="visitante" required placeholder="ej. Argentina">
      <label>Liga (opcional, filtra el histórico)</label>
      <input type="text" name="liga" placeholder="ej. Mundial">
      <label>Ruta al histórico (CSV en el servidor)</label>
      <input type="text" name="datos" value="examples/historico_ejemplo.csv">
      <label>Ajuste manual por bajas/lesiones (opcional, % de goles esperados)</label>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <input type="number" step="1" name="ajuste_local" placeholder="Local %">
        <input type="number" step="1" name="ajuste_visitante" placeholder="Visitante %">
      </div>
      <button type="submit">Generar pronóstico</button>
    </form>
  </div>
</details>
<div class="disclaimer">
  <b>Disclaimer:</b> modelo probabilístico basado en datos históricos. No garantiza
  resultados. Las apuestas deportivas implican riesgo real de pérdida de dinero.
</div>
"""

MATCHES_BODY = """
<h1>Partidos</h1>
<div class="subtitle">{{ competition_name }} — toca un partido próximo para ver el pronóstico. Sube para ver días anteriores.</div>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
{% if teams %}
<form class="picker-card" method="get" action="{{ url_for('predecir') }}">
  <div class="picker-title">Predecir cualquier enfrentamiento</div>
  <input type="hidden" name="competition" value="{{ competition }}">
  <div class="picker-row">
    <select name="local" required aria-label="Equipo local">
      <option value="" disabled selected>Local…</option>
      {% for t in teams %}<option value="{{ t.name }}">{{ t.name_es }}</option>{% endfor %}
    </select>
    <span class="picker-vs">vs</span>
    <select name="visitante" required aria-label="Equipo visitante">
      <option value="" disabled selected>Visitante…</option>
      {% for t in teams %}<option value="{{ t.name }}">{{ t.name_es }}</option>{% endfor %}
    </select>
  </div>
  <button type="submit">Ver pronóstico →</button>
</form>
{% endif %}
{% if hay_marcas_modelo %}
<div class="hit-legend">
  <span><span class="mr-hitmark">✓✓</span> clavó el marcador exacto</span>
  <span><span class="mr-hitmark result">✓</span> acertó el ganador</span>
  <span><span class="mr-hitmark miss">✗</span> falló el ganador</span>
  <span class="muted">— el marcador exacto es la vara más dura (10-15% hasta en modelos pro; en un Mundial goleador cae más). El ✓ azul es lo que de verdad acierta seguido. Toca la fila para ver qué dijo el modelo.</span>
</div>
{% endif %}
{% if grouped_matches %}
  {% for grupo in grouped_matches %}
  <div class="day-header"{% if grupo.ancla_hoy %} id="hoy"{% endif %}>{{ grupo.dia }}</div>
  {% for m in grupo.partidos %}
  {% if m.finalizado %}
  {# Partido jugado: clic -> predicción RETROACTIVA (qué habría dicho el
     modelo antes del partido, sin conocer el resultado), para comparar. #}
  <a class="match-row-v2 played{% if m.estado_modelo == 'exacto' %} hit{% elif m.estado_modelo == 'resultado' %} result{% elif m.estado_modelo == 'fallo' %} miss{% endif %}"
     href="{{ url_for('predecir', competition=competition, local=m.equipo_local, visitante=m.equipo_visitante, antes_de=m.fecha_iso) }}"
     {% if m.detalle_modelo %}title="{{ m.detalle_modelo }}"{% endif %}>
    <div class="mr-teams">
      <div class="mr-team">
        {% if m.escudo_local %}<img class="crest" loading="lazy" src="{{ m.escudo_local }}" onerror="this.style.visibility='hidden'">{% endif %}
        <span>{{ m.equipo_local_es }}</span>
      </div>
      <div class="mr-team">
        {% if m.escudo_visitante %}<img class="crest" loading="lazy" src="{{ m.escudo_visitante }}" onerror="this.style.visibility='hidden'">{% endif %}
        <span>{{ m.equipo_visitante_es }}</span>
      </div>
    </div>
    {% if m.estado_modelo == 'exacto' %}<span class="mr-hitmark" aria-label="El modelo clavó el marcador">✓✓</span>
    {% elif m.estado_modelo == 'resultado' %}<span class="mr-hitmark result" aria-label="El modelo acertó el ganador">✓</span>
    {% elif m.estado_modelo == 'fallo' %}<span class="mr-hitmark miss" aria-label="El modelo falló">✗</span>{% endif %}
    <div class="mr-time mr-score">{{ m.marcador }}</div>
  </a>
  {% else %}
  <a class="match-row-v2" href="{{ url_for('predecir', competition=competition, local=m.equipo_local, visitante=m.equipo_visitante) }}">
    <div class="mr-teams">
      <div class="mr-team">
        {% if m.escudo_local %}<img class="crest" loading="lazy" src="{{ m.escudo_local }}" onerror="this.style.visibility='hidden'">{% endif %}
        <span>{{ m.equipo_local_es }}</span>
      </div>
      <div class="mr-team">
        {% if m.escudo_visitante %}<img class="crest" loading="lazy" src="{{ m.escudo_visitante }}" onerror="this.style.visibility='hidden'">{% endif %}
        <span>{{ m.equipo_visitante_es }}</span>
      </div>
    </div>
    <div class="mr-time">{{ m.hora_str }}</div>
  </a>
  {% endif %}
  {% endfor %}
  {% endfor %}
  <a href="#hoy" class="today-fab" title="Ir a hoy">Hoy</a>
  <script>
    (function () {
      var hoy = document.getElementById('hoy');
      if (hoy) { hoy.scrollIntoView({block: 'start'}); }
    })();
  </script>
{% elif not error %}
  <p class="subtitle">No hay partidos para esta competición en estos días —
  puede estar en receso (entre temporadas o entre rondas). Prueba otra
  competición o vuelve más cerca de la próxima fecha.</p>
{% endif %}

{% if prev_groups %}
<details>
  <summary>Clasificatorias y partidos previos al torneo</summary>
  {% if prev_note %}<p class="subtitle" style="margin-top:8px;">{{ prev_note }}</p>{% endif %}
  {% for g in prev_groups %}
  <div class="day-header">{{ g.mes }}</div>
  {% for m in g.partidos %}
  <div class="match-row-v2 played">
    <div class="mr-teams">
      <div class="mr-team">
        {% if m.escudo_local %}<img class="crest" loading="lazy" src="{{ m.escudo_local }}" onerror="this.style.visibility='hidden'">{% endif %}
        <span>{{ m.equipo_local_es }}</span>
      </div>
      <div class="mr-team">
        {% if m.escudo_visitante %}<img class="crest" loading="lazy" src="{{ m.escudo_visitante }}" onerror="this.style.visibility='hidden'">{% endif %}
        <span>{{ m.equipo_visitante_es }}</span>
      </div>
    </div>
    <div class="mr-time mr-score">{{ m.marcador }}</div>
  </div>
  {% endfor %}
  {% endfor %}
</details>
{% endif %}

{% if standings %}
<details>
  <summary>Tabla de posiciones</summary>
  {% for grupo in standings %}
  <div class="card">
    <h2>{{ grupo.group or "Tabla general" }}</h2>
    <table>
      <thead><tr><th>#</th><th>Equipo</th><th>PJ</th><th>DG</th><th>Pts</th></tr></thead>
      <tbody>
        {% for row in grupo.table %}
        <tr>
          <td><span class="pos-badge">{{ row.position }}</span></td>
          <td style="display:flex;align-items:center;gap:8px;">
            {% if row.team.crest %}<img class="crest" loading="lazy" src="{{ row.team.crest }}" onerror="this.style.visibility='hidden'">{% endif %}
            {{ row.team_es }}
          </td>
          <td>{{ row.playedGames }}</td>
          <td>{{ row.goalDifference }}</td>
          <td><b>{{ row.points }}</b></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endfor %}
</details>
{% endif %}

{% if scorers %}
<details>
  <summary>Goleadores del torneo</summary>
  <div class="card">
    <table>
      <thead><tr><th>#</th><th>Jugador</th><th>Equipo</th><th>Goles</th></tr></thead>
      <tbody>
        {% for s in scorers %}
        <tr><td>{{ loop.index }}</td><td>{{ s.jugador }}</td><td>{{ s.equipo_es }}</td><td><b>{{ s.goles }}</b></td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</details>
{% endif %}

<a class="match-row" href="{{ url_for('rendimiento', competition=competition) }}" style="justify-content:center;">
  📊 Rendimiento del modelo — aciertos y margen de error en partidos ya jugados
</a>
<a class="match-row" href="/" style="justify-content:center;color:var(--muted);">← Elegir otra competición</a>
"""


RENDIMIENTO_BODY = """
<h1>Rendimiento del modelo</h1>
<div class="subtitle">{{ competition_name }} — cómo le fue al modelo prediciendo los últimos
{{ bt.n }} partidos ya jugados. Cada predicción se hizo usando SOLO los partidos anteriores a esa
fecha (sin hacer trampa mirando el futuro), igual que si hubieras consultado la app antes del partido.</div>

<div class="card">
  <h2>Resumen</h2>
  <div class="goals-summary">
    <div>Aciertos del pick<span>{{ bt.aciertos }}/{{ bt.n }}</span></div>
    <div>Tasa de acierto<span>{{ bt.acierto_pct }}%</span></div>
    <div>Brier score<span>{{ bt.brier }}</span></div>
  </div>
  <p class="muted" style="margin-top:12px;">
    <b>Cómo leerlo:</b> elegir al azar acierta ~33%. El <b>Brier score</b> mide la calidad de las
    probabilidades (0 = perfecto, 0.667 = tirar la moneda de tres caras): castiga estar muy seguro
    y equivocarse. Un Brier por debajo de 0.667 significa que las probabilidades del modelo
    aportan información real. El fútbol es de bajo marcador y alto azar: ni el mejor modelo del
    mundo pasa de ~55-60% de acierto en 1X2 — desconfía de quien prometa más.
  </p>
</div>

{% if mercados_rows %}
<div class="card">
  <h2>Acierto por tipo de apuesta</h2>
  <table>
    <thead><tr><th>Apuesta</th><th>Acierto</th><th>Brier</th><th>Azar</th></tr></thead>
    <tbody>
      {% for m in mercados_rows %}
      <tr>
        <td>{{ m.nombre }}</td>
        <td><b>{{ m.aciertos }}/{{ m.n }}</b> ({{ m.acierto_pct }}%)</td>
        <td {% if m.brier < m.brier_azar %}style="color:var(--green);font-weight:700;"{% endif %}>{{ m.brier }}</td>
        <td class="muted">{{ m.brier_azar }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  <p class="muted" style="margin-top:10px;">
    "Acierto" = el lado que el modelo veía más probable fue el que ocurrió. Ojo con la doble
    oportunidad: acierta mucho más seguido, pero las casas también la pagan mucho menos — el
    número que de verdad compara mercados es el <b>Brier contra su azar</b> (dos salidas = 0.25,
    tres salidas = 0.667; en verde cuando el modelo le gana al azar). Donde el Brier del modelo
    más se aleja del azar, más confiable es su probabilidad para ese tipo de apuesta.
  </p>
</div>
{% endif %}

{% for p in bt.partidos %}
<div class="card" style="padding:12px 16px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <div style="flex:1; min-width:0;">
      <div style="font-weight:600; font-size:0.9rem;">{{ p.local_es }} vs {{ p.visitante_es }}</div>
      <div class="muted">{{ p.fecha }} · terminó <b>{{ p.marcador }}</b></div>
      <div class="muted">Modelo: {{ p.pick_es }} ({{ (p.prob_pick * 100) | round(1) }}%)
        — L {{ (p.prob_local * 100) | round(0) | int }}% / E {{ (p.prob_empate * 100) | round(0) | int }}% / V {{ (p.prob_visitante * 100) | round(0) | int }}%</div>
    </div>
    <div style="flex-shrink:0; font-size:1.3rem;">{{ "✅" if p.acierto else "❌" }}</div>
  </div>
</div>
{% endfor %}

<a class="match-row" href="{{ url_for('partidos', competition=competition) }}"
   style="justify-content:center;color:var(--muted);">← Volver a {{ competition_name }}</a>
"""


def _clean_nan(value):
    """
    pandas convierte None a NaN (float) en columnas de tipo objeto cuando
    TODAS las filas de esa columna están vacías (ej. ningún partido trae
    escudo). Un NaN es "truthy" en Python (bool(float('nan')) == True), así
    que un {% if %} de Jinja no lo filtra y termina renderizando
    src="nan" — el navegador intenta cargar la URL literal "/nan". Esta
    función normaliza cualquier NaN a None antes de que llegue a la plantilla.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _get_api_connector():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        return None
    return FootballDataConnector(api_key=api_key)


@app.route("/", methods=["GET"])
def index():
    error = None
    if not os.environ.get("FOOTBALL_DATA_API_KEY"):
        error = ("No hay FOOTBALL_DATA_API_KEY configurada — la lista de próximos partidos "
                 "no va a funcionar hasta que la configures (ver README > Desplegar en Render). "
                 "Mientras tanto puedes usar el modo avanzado de abajo.")
    body = render_template_string(INDEX_BODY, competitions=COMPETITIONS, error=error)
    return wrap_page("Predictor Fútbol", body)


@app.route("/partidos", methods=["GET"])
def partidos():
    competition = request.args.get("competition", "")
    competition_name = COMPETITIONS.get(competition, competition)
    connector = _get_api_connector()
    if not connector:
        body = render_template_string(
            MATCHES_BODY, competition=competition, competition_name=competition_name,
            grouped_matches=[], standings=None, scorers=None, error="Falta configurar FOOTBALL_DATA_API_KEY en el servidor.",
        )
        return wrap_page(competition_name, body)
    # En torneos (Mundial, Euro, Champions, Libertadores) se muestra el
    # torneo completo hacia atrás (45 días cubren desde la fase de grupos
    # hasta la final), no solo la última semana: el usuario quiere poder
    # repasar los marcadores de grupos aunque ya se esté en eliminatorias.
    # En ligas se mantienen 7 días — mostrar mes y medio de jornadas viejas
    # sería puro ruido para llegar a "Hoy".
    dias_pasados = 45 if competition in TOURNAMENTS else 7
    try:
        agenda = connector.fetch_agenda(liga=competition, dias_pasados=dias_pasados)
    except Exception as e:
        body = render_template_string(
            MATCHES_BODY, competition=competition, competition_name=competition_name,
            grouped_matches=[], standings=None, scorers=None, error=f"No se pudo consultar football-data.org: {e}",
        )
        return wrap_page(competition_name, body)

    # Cruce con el backtest nocturno: para cada partido jugado, si el
    # backtest ya lo evaluó, se sabe si el pick del modelo (1X2, calculado
    # SOLO con datos anteriores al partido) acertó — la lista lo pinta en
    # verde (acierto) o rojo suave (fallo). Partidos aún no evaluados no
    # llevan marca. Cruce por fecha + nombres de la API (ambos vienen de
    # las mismas fuentes, así que coinciden exactos).
    bt = _load_backtest().get(competition) or {}
    bt_lookup = {
        (p["fecha"], p["local"], p["visitante"]): p
        for p in bt.get("partidos", [])
    }
    _PICK_ES = {"local": "gana local", "empate": "empate", "visitante": "gana visitante"}

    def _resultado_modelo(row):
        """Semáforo de 3 estados para el partido jugado, cruzado con el
        backtest walk-forward (pick calculado SOLO con datos anteriores):

          'exacto'    -> el modelo clavó el MARCADOR exacto (lo raro, ~10-15%)
          'resultado' -> acertó el GANADOR/1X2 aunque no el marcador (~48%)
          'fallo'     -> ni el ganador

        Antes se coloreaba solo por marcador exacto y, en torneos atípicos y
        goleadores (este Mundial), daba casi todo ✗ — hacía ver inútil un
        modelo que igual acierta ~la mitad de los ganadores. Separar los dos
        niveles muestra el valor real sin ocultar lo difícil que es el pleno.
        """
        if not getattr(row, "finalizado", False):
            return None, None
        p = bt_lookup.get((row.fecha_hora.strftime("%Y-%m-%d"), row.equipo_local, row.equipo_visitante))
        if not p or "marcador_pred" not in p:
            # Partido no evaluado aún, o backtest viejo sin marcador guardado
            # (se completa en el próximo refresco nocturno).
            return None, None
        if p.get("acierto_marcador"):
            estado = "exacto"
        elif p.get("acierto"):
            estado = "resultado"
        else:
            estado = "fallo"
        pick_es = _PICK_ES.get(p.get("pick"), p.get("pick", "?"))
        detalle = (f"Marcador que dijo el modelo: {p['marcador_pred']} "
                   f"({p['prob_marcador_pred']*100:.0f}%) · Pick 1X2: {pick_es}")
        return estado, detalle

    # Los nombres se traducen solo para mostrar; el valor que va en el link
    # (equipo_local/equipo_visitante) se deja en el idioma original de la
    # API, porque es el identificador que usa el modelo para cruzar contra
    # el histórico — traducirlo ahí rompería la búsqueda.
    matches = []
    for row in agenda.itertuples():
        estado_modelo, detalle_modelo = _resultado_modelo(row)
        matches.append({
            "estado_modelo": estado_modelo,
            "detalle_modelo": detalle_modelo,
            "equipo_local": row.equipo_local,
            "equipo_visitante": row.equipo_visitante,
            "equipo_local_es": team_name_es(row.equipo_local),
            "equipo_visitante_es": team_name_es(row.equipo_visitante),
            "escudo_local": _clean_nan(getattr(row, "escudo_local", None)),
            "escudo_visitante": _clean_nan(getattr(row, "escudo_visitante", None)),
            "hora_str": to_colombia_time(row.fecha_hora).strftime("%I:%M %p"),
            "dia": day_label_es(to_colombia_time(row.fecha_hora)),
            # Partidos ya jugados: se muestra el marcador y no son clicables
            # para predecir (predecir un partido terminado no tiene sentido).
            "finalizado": bool(getattr(row, "finalizado", False)),
            # Fecha (solo día) del partido, para la predicción retroactiva:
            # /predecir?antes_de=YYYY-MM-DD recorta el histórico a lo anterior.
            "fecha_iso": row.fecha_hora.strftime("%Y-%m-%d"),
            "marcador": (
                f"{int(row.goles_local)} - {int(row.goles_visitante)}"
                if getattr(row, "finalizado", False) and row.goles_local is not None
                else None
            ),
        })

    # La leyenda de ✓/✗ solo se muestra si al menos un partido tiene marca
    # (si el backtest no ha corrido para esta competición, no hay que explicar nada).
    hay_marcas_modelo = any(m["estado_modelo"] is not None for m in matches)

    # Agrupación por día (estilo apps deportivas: "Ayer", "Hoy", "Mañana",
    # etc.), preservando el orden cronológico — como ya viene ordenado
    # ascendente, agrupar consecutivamente basta.
    grouped_matches = []
    for m in matches:
        if grouped_matches and grouped_matches[-1]["dia"] == m["dia"]:
            grouped_matches[-1]["partidos"].append(m)
        else:
            grouped_matches.append({"dia": m["dia"], "partidos": [m], "ancla_hoy": False})

    # Se marca dónde va el ancla del botón "Hoy": el grupo cuyo día es "Hoy",
    # o si hoy no hay partidos, el primer grupo aún no jugado (el próximo).
    ancla = next((g for g in grouped_matches if g["dia"] == "Hoy"), None)
    if ancla is None:
        ancla = next((g for g in grouped_matches
                      if any(not p["finalizado"] for p in g["partidos"])), None)
    if ancla is not None:
        ancla["ancla_hoy"] = True

    # Tabla de posiciones y goleadores: puramente informativos (ver
    # discusión en el README sobre por qué no se usan para calcular
    # tiros al arco ni ningún otro mercado). Si la competición no tiene
    # standings/scorers disponibles (ej. torneos que no llevan tabla), se
    # muestra igual la lista de partidos sin esas secciones.
    standings = None
    scorers = None
    try:
        raw_standings = connector.fetch_standings(competition)
        standings = [
            {
                "group": s.get("group"),
                "table": [
                    {**row, "team_es": team_name_es(row["team"]["name"])}
                    for row in s.get("table", [])
                ],
            }
            for s in raw_standings
        ]
    except Exception:
        pass
    try:
        raw_scorers = connector.fetch_scorers(competition, limit=10)
        scorers = [{**s, "equipo_es": team_name_es(s["equipo"])} for s in raw_scorers]
    except Exception:
        pass

    # Clasificatorias y partidos previos al torneo (solo torneos): la agenda
    # de arriba cubre el torneo en sí (45 días), pero la clasificación se jugó
    # meses/años antes. Se traen los partidos FINISHED del último año que
    # queden ANTES de la ventana de la agenda (la API agrupa torneo y su
    # clasificación bajo el mismo código, ej. "WC"), agrupados por mes, en una
    # sección plegable para no enterrar la agenda del torneo. Son los mismos
    # datos que el modelo ya usa para predecir — esto solo los hace visibles.
    prev_groups = None
    prev_note = None
    if competition in TOURNAMENTS:
        try:
            now = datetime.now(timezone.utc)
            hist = connector.fetch_matches(
                liga=competition,
                desde=(now - timedelta(days=365)).strftime("%Y-%m-%d"),
                hasta=(now - timedelta(days=dias_pasados + 1)).strftime("%Y-%m-%d"),
            )
        except Exception:
            hist = None
        # SELECCIONES: la API .org no tiene clasificatorias ni amistosos (solo
        # el torneo), así que esta sección salía vacía en el Mundial. Se llena
        # con el MISMO dataset internacional que alimenta al modelo cuando la
        # muestra es chica — coherencia pedida por el usuario: lo que se
        # muestra como "partidos previos" es lo que de verdad afecta el
        # pronóstico, con los mismos nombres de selección (no "otra selección").
        if (hist is None or hist.empty) and competition in NATIONAL_TEAM_COMPETITIONS and matches:
            try:
                intl = fetch_international_results(desde=now - timedelta(days=365))
            except Exception:
                intl = None
            if intl is not None and not intl.empty:
                equipos_torneo = ({m["equipo_local"] for m in matches}
                                  | {m["equipo_visitante"] for m in matches})
                intl = align_team_names(intl, equipos_torneo)
                corte = (now - timedelta(days=dias_pasados)).date()
                previos = intl[
                    (intl["equipo_local"].isin(equipos_torneo)
                     | intl["equipo_visitante"].isin(equipos_torneo))
                    & (intl["fecha"].dt.date < corte)
                ].copy()
                if not previos.empty:
                    previos["fecha"] = previos["fecha"].dt.strftime("%Y-%m-%d")
                    previos["escudo_local"] = None
                    previos["escudo_visitante"] = None
                    hist = previos
                    prev_note = ("Clasificatorias y amistosos del último año de las selecciones del "
                                 "torneo. Estos partidos son parte de la muestra con la que se "
                                 "calculan los pronósticos (los más antiguos pesan menos).")
        if hist is not None and not hist.empty:
            meses_es = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                        "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
            # Descendente: lo más reciente (el repechaje/última fecha) primero.
            hist = hist.sort_values("fecha", ascending=False)
            prev_groups = []
            for row in hist.itertuples():
                f = datetime.strptime(row.fecha, "%Y-%m-%d")
                mes = f"{meses_es[f.month - 1].capitalize()} {f.year}"
                m = {
                    "equipo_local_es": team_name_es(row.equipo_local),
                    "equipo_visitante_es": team_name_es(row.equipo_visitante),
                    "escudo_local": _clean_nan(row.escudo_local),
                    "escudo_visitante": _clean_nan(row.escudo_visitante),
                    "marcador": f"{int(row.goles_local)} - {int(row.goles_visitante)}",
                }
                if prev_groups and prev_groups[-1]["mes"] == mes:
                    prev_groups[-1]["partidos"].append(m)
                else:
                    prev_groups.append({"mes": mes, "partidos": [m]})

    # Lista de equipos (de la tabla de posiciones) para el selector "predecir
    # cualquier enfrentamiento": permite pedir un pronóstico aunque no haya
    # partido programado (útil en receso o entre rondas), usando el historial
    # de la temporada. El value es el nombre original de la API (el que cruza
    # contra el histórico); se muestra el nombre en español.
    teams_picker = []
    if standings:
        seen = set()
        for grupo in standings:
            for row in grupo["table"]:
                name = row.get("team", {}).get("name")
                if name and name not in seen:
                    seen.add(name)
                    teams_picker.append({"name": name, "name_es": row.get("team_es", name)})
        teams_picker.sort(key=lambda t: t["name_es"])

    body = render_template_string(
        MATCHES_BODY, competition=competition, competition_name=competition_name,
        grouped_matches=grouped_matches, standings=standings, scorers=scorers,
        teams=teams_picker, prev_groups=prev_groups, prev_note=prev_note,
        hay_marcas_modelo=hay_marcas_modelo, error=None,
    )
    return wrap_page(competition_name, body)


def _run_prediction(matches_df, local, visitante, home_adjustment=0.0, away_adjustment=0.0,
                    fifa_context=None, avisos=None, es_eliminatoria=False):
    """Lógica compartida por el flujo de API y el modo manual: ajusta los
    modelos sobre el histórico ya cargado y arma el reporte HTML.

    es_eliminatoria: True para torneos (Mundial/Euro/Champions/Libertadores):
    muestra la tarjeta "¿Quién clasifica?" que traduce el 1X2 de 90 minutos
    a probabilidad de pasar de ronda (en muerte súbita no existe el empate)."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))

    ratings_cfg = RatingsConfig(
        k_factor=config["ratings"]["elo_k_factor"], initial_rating=config["ratings"]["elo_initial"],
        home_advantage=config["ratings"]["home_advantage_elo"], use_goal_diff_multiplier=config["ratings"]["goal_diff_multiplier"],
    )
    elo_system = EloRatingSystem(ratings_cfg)
    elo_system.replay_history(matches_df)

    goals_cfg = GoalsModelConfig(
        xi=config["goals_model"]["dixon_coles_xi"], max_goals=config["goals_model"]["max_goals"],
        low_score_correction=config["goals_model"]["low_score_correction"],
        regularization=config["goals_model"].get("regularization", 0.7),
        max_expected_goals=config["goals_model"].get("max_expected_goals", 4.5),
    )
    goals_model = DixonColesModel(goals_cfg).fit(matches_df)

    corners_model = CornersModel(CornersModelConfig(
        half_life_days=config["recency"]["half_life_days"], opponent_strength_weight=config["opponent_strength"]["strength_weight"],
    ))
    shots_model = ShotsOnTargetModel(ShotsModelConfig(
        half_life_days=config["recency"]["half_life_days"], opponent_strength_weight=config["opponent_strength"]["strength_weight"],
    ))
    cards_model = CardsModel(CardsModelConfig(
        half_life_days=config["recency"]["half_life_days"],
        opponent_strength_weight=min(0.3, config["opponent_strength"]["strength_weight"]),
    ))

    sim_config = SimulationConfig(
        n_iterations=config["simulation"]["n_iterations"], random_seed=config["simulation"]["random_seed"],
    )
    simulator = MatchSimulator(goals_model, corners_model, shots_model, cards_model, sim_config)
    sim_result = simulator.simulate(
        matches_df, local, visitante, elo_system.ratings,
        home_adjustment=home_adjustment, away_adjustment=away_adjustment,
    )

    report = build_report(
        local, visitante, goals_model, sim_result, config,
        home_adjustment=home_adjustment, away_adjustment=away_adjustment,
        matches_df=matches_df, avisos=avisos,
    )
    # Traducción solo de presentación: local/visitante ya se usaron para
    # todo el cálculo (Elo, Dixon-Coles, simulación) con el nombre
    # original — acá se reemplaza únicamente lo que se va a mostrar.
    report["partido"]["local"] = team_name_es(local)
    report["partido"]["visitante"] = team_name_es(visitante)
    report["escudo_local"] = _find_crest(matches_df, local)
    report["escudo_visitante"] = _find_crest(matches_df, visitante)
    report["fifa_context"] = fifa_context
    report["es_eliminatoria"] = es_eliminatoria
    return render_html_report(report, value_bets=None)


def _find_crest(matches_df, team_name):
    """Busca el escudo de un equipo en el histórico ya cargado (puede
    aparecer como local o visitante en distintas filas). Devuelve None si
    la fuente de datos no trae escudos (ej. CSV propio en modo avanzado)."""
    if "escudo_local" not in matches_df.columns:
        return None
    as_home = matches_df[matches_df["equipo_local"] == team_name]["escudo_local"].dropna()
    if len(as_home):
        return as_home.iloc[0]
    as_away = matches_df[matches_df["equipo_visitante"] == team_name]["escudo_visitante"].dropna()
    if len(as_away):
        return as_away.iloc[0]
    return None


@app.route("/predecir", methods=["GET"])
def predecir():
    competition = request.args.get("competition", "")
    local = request.args.get("local", "").strip()
    visitante = request.args.get("visitante", "").strip()

    connector = _get_api_connector()
    if not connector:
        return "Falta configurar FOOTBALL_DATA_API_KEY en el servidor.", 500

    # 365 días de historial recientes bastan para que Dixon-Coles y Elo
    # tengan suficiente muestra sin traer la competición completa. Esto no
    # es solo una optimización de velocidad: sin este límite, una
    # competición con muchos años de historial (ej. "WC" trae Mundiales y
    # clasificatorias de decenas de años) mete cientos de equipos distintos
    # en el ajuste de Dixon-Coles, disparando el número de parámetros a
    # optimizar y agotando la memoria del plan gratuito de Render (512 MB) —
    # esto causó un "Worker was sent SIGKILL! Perhaps out of memory?" real
    # en producción antes de este fix.
    hoy = datetime.now(timezone.utc)
    desde = (hoy - timedelta(days=365)).strftime("%Y-%m-%d")
    hasta = hoy.strftime("%Y-%m-%d")

    try:
        matches_df, cleaning_report = load_from_connector(connector, liga=competition, desde=desde, hasta=hasta)
    except Exception as e:
        return f"Error consultando el histórico: {e}", 502

    if matches_df.empty:
        return "No hay histórico suficiente para esta competición todavía.", 404

    # Predicción RETROACTIVA (clic en un partido ya jugado): se recorta el
    # histórico a lo estrictamente anterior a la fecha del partido, para
    # reproducir exactamente lo que el modelo habría dicho ANTES del pitazo
    # — sin fuga de información del resultado (misma disciplina que el
    # backtest de src/backtest.py, pero para un partido puntual a demanda).
    avisos = []
    antes_de = request.args.get("antes_de", "").strip()
    if antes_de:
        try:
            corte = pd.Timestamp(antes_de)
        except Exception:
            return "Parámetro antes_de inválido (formato esperado: YYYY-MM-DD).", 400
        fechas = pd.to_datetime(matches_df["fecha"])
        # El resultado real se rescata ANTES de recortar, solo para mostrarlo
        # al final como comparación (no participa en ningún cálculo).
        jugado = matches_df[
            (fechas.dt.date == corte.date())
            & (matches_df["equipo_local"] == local)
            & (matches_df["equipo_visitante"] == visitante)
        ]
        real = None
        if not jugado.empty:
            r = jugado.iloc[0]
            real = f"{int(r['goles_local'])} - {int(r['goles_visitante'])}"
        matches_df = matches_df[fechas < corte]
        if matches_df.empty:
            return ("No hay histórico anterior a ese partido para hacer la "
                    "predicción retroactiva."), 404
        aviso_retro = (f"Predicción retroactiva: calculada solo con partidos anteriores al "
                       f"{antes_de}, como si se hubiera consultado antes del pitazo.")
        if real:
            aviso_retro += f" Resultado real: {team_name_es(local)} {real} {team_name_es(visitante)}."
        avisos.append(aviso_retro)

    # Ventana adaptativa: si alguno de los dos equipos quedó con muestra
    # muy chica en 365 días (típico: selecciones que juegan poco), se
    # amplía a 2-3 años recortando al vecindario relevante. En modo
    # retroactivo NO se amplía: cambiaría la base contra la que se comparó
    # históricamente y el corte antes_de ya define su propio universo.
    if not antes_de:
        matches_df = _ensure_min_sample(
            connector, competition, matches_df, local, visitante, hasta, avisos,
        )

    # Historial de respaldo para equipos sin partidos en esta competición
    # (caso típico: recién ascendido — ej. un equipo nuevo en Premier League
    # no jugó ni un partido de PL el último año, así que el histórico de la
    # competición no sabe NADA de él y córners/tiros/tarjetas salían como
    # falsos ceros). Si el equipo no aparece, se trae su historial de la
    # competición "hermana" (PL<->Championship, las dos cubiertas por el plan
    # gratis de la API y por football-data.co.uk) y se agrega al histórico,
    # avisando en el reporte que se usó otra liga.
    fallback_comp = FALLBACK_HISTORY.get(competition)
    if fallback_comp:
        missing = [
            t for t in (local, visitante)
            if not ((matches_df["equipo_local"] == t) | (matches_df["equipo_visitante"] == t)).any()
        ]
        if missing:
            try:
                fb_df, _ = load_from_connector(connector, liga=fallback_comp, desde=desde, hasta=hasta)
            except Exception:
                fb_df = None
            for t in missing:
                fb_rows = None
                if fb_df is not None and not fb_df.empty:
                    fb_rows = fb_df[(fb_df["equipo_local"] == t) | (fb_df["equipo_visitante"] == t)]
                if fb_rows is not None and not fb_rows.empty:
                    try:
                        fb_rows = enrich_with_couk_stats(fb_rows, fallback_comp)
                    except Exception:
                        pass
                    matches_df = pd.concat([matches_df, fb_rows], ignore_index=True)
                    avisos.append(
                        f"{team_name_es(t)} no tiene partidos en {COMPETITIONS.get(competition, competition)} "
                        f"el último año; su pronóstico usa su historial de "
                        f"{COMPETITIONS.get(fallback_comp, fallback_comp)} ({len(fb_rows)} partidos)."
                    )
            # El pipeline (Elo, ponderación por recencia) asume orden cronológico.
            matches_df = matches_df.sort_values("fecha").reset_index(drop=True)

    # Enriquecimiento opcional con córners/tiros al arco oficiales de FIFA
    # (ver src/connectors/fifa_reports_connector.py) — football-data.org
    # no los trae. Solo se intenta para Mundial (los reportes PMSR son
    # específicos de esa competición) y nunca puede romper la predicción:
    # si la fuente falla, matches_df simplemente se queda como estaba.
    fifa_context = None
    if competition == "WC":
        # Presupuesto TOTAL de parseo de PDFs para todo este request (cubre el
        # enrich y los summaries de ambos equipos), para no pasarse del timeout
        # de gunicorn. Best-effort: lo que no alcance a parsearse se omite.
        start_request_budget()
        try:
            try:
                matches_df = enrich_with_fifa_reports(matches_df, {local, visitante})
            except Exception:
                pass
            # En predicción retroactiva NO se muestra el contexto FIFA: son
            # promedios de TODO el torneo (incluye partidos posteriores al
            # que se está "prediciendo") y contaminaría la comparación.
            if not antes_de:
                try:
                    fifa_context = {
                        "local": get_team_summary_stats(local),
                        "visitante": get_team_summary_stats(visitante),
                    }
                    if not fifa_context["local"] and not fifa_context["visitante"]:
                        fifa_context = None
                except Exception:
                    fifa_context = None
        finally:
            clear_request_budget()
    else:
        # Ligas de clubes: córners, tiros al arco y tarjetas desde los CSV
        # gratuitos de football-data.co.uk (la API .org no los trae). Solo
        # aplica a las ligas que esa fuente cubre; para el resto no hace nada.
        # Es liviano (CSV, no PDF) y nunca rompe la predicción.
        try:
            matches_df = enrich_with_couk_stats(matches_df, competition)
        except Exception:
            pass

    return _run_prediction(matches_df, local, visitante, fifa_context=fifa_context, avisos=avisos,
                           es_eliminatoria=competition in TOURNAMENTS)


@app.route("/rendimiento", methods=["GET"])
def rendimiento():
    """Backtest del modelo (ver src/backtest.py): aciertos y calibración
    sobre los últimos partidos jugados. Lee el JSON precomputado por
    scripts/run_backtest.py — no calcula nada pesado en el request."""
    competition = request.args.get("competition", "")
    competition_name = COMPETITIONS.get(competition, competition)
    bt = _load_backtest().get(competition)
    if not bt:
        body = render_template_string(
            MATCHES_BODY, competition=competition, competition_name=competition_name,
            grouped_matches=[], standings=None, scorers=None, teams=None,
            error="Todavía no hay backtest generado para esta competición "
                  "(se genera automáticamente cada noche; también puedes correr "
                  "scripts/run_backtest.py).",
        )
        return wrap_page(competition_name, body)

    bt = dict(bt)
    pick_es = {"local": "gana local", "empate": "empate", "visitante": "gana visitante"}
    bt["partidos"] = [
        {**p,
         "local_es": team_name_es(p["local"]),
         "visitante_es": team_name_es(p["visitante"]),
         "pick_es": pick_es.get(p["pick"], p["pick"])}
        for p in bt["partidos"]
    ]
    # Tabla "acierto por tipo de apuesta": el 1X2 (tres salidas) primero y
    # luego los mercados binarios. JSONs viejos sin "mercados" simplemente
    # no muestran la tabla (el nightly la agrega al siguiente refresco).
    nombres = {
        "doble_1x": "Doble oportunidad 1X (local o empate)",
        "doble_x2": "Doble oportunidad X2 (empate o visitante)",
        "doble_12": "Doble oportunidad 12 (no empate)",
        "over25": "Más de 2.5 goles",
        "btts": "Ambos anotan",
    }
    mercados_rows = [{
        "nombre": "1X2 (gana/empata/pierde)",
        "n": bt["n"], "aciertos": bt["aciertos"], "acierto_pct": bt["acierto_pct"],
        "brier": bt["brier"], "brier_azar": bt["brier_azar"],
    }]
    for key, nombre in nombres.items():
        m = (bt.get("mercados") or {}).get(key)
        if m:
            mercados_rows.append({"nombre": nombre, **m})
    if len(mercados_rows) == 1:
        mercados_rows = None  # JSON viejo: solo 1X2, la tabla no aporta nada

    body = render_template_string(
        RENDIMIENTO_BODY, competition=competition, competition_name=competition_name, bt=bt,
        mercados_rows=mercados_rows,
    )
    return wrap_page(f"Rendimiento — {competition_name}", body)


def _load_backtest() -> dict:
    """Lee data/backtest.json (precomputado offline). Dict vacío si no existe."""
    path = PROJECT_ROOT / "data" / "backtest.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@app.route("/predecir_manual", methods=["POST"])
def predecir_manual():
    form = request.form
    local = form.get("local", "").strip()
    visitante = form.get("visitante", "").strip()
    liga = form.get("liga", "").strip() or None
    datos_path = form.get("datos", "").strip() or str(DEFAULT_DATA_PATH)
    if not Path(datos_path).is_absolute():
        datos_path = str(PROJECT_ROOT / datos_path)

    def pct_to_fraction(name):
        raw = form.get(name, "").strip()
        return float(raw) / 100.0 if raw else 0.0

    if not local or not visitante:
        body = render_template_string(INDEX_BODY, competitions=COMPETITIONS, error="Debes indicar equipo local y visitante.")
        return wrap_page("Predictor Fútbol", body)

    try:
        connector = CSVConnector(matches_path=datos_path)
        matches_df, cleaning_report = load_from_connector(connector, liga=liga)
    except FileNotFoundError as e:
        body = render_template_string(INDEX_BODY, competitions=COMPETITIONS, error=str(e))
        return wrap_page("Predictor Fútbol", body)

    if matches_df.empty:
        body = render_template_string(
            INDEX_BODY, competitions=COMPETITIONS,
            error="No hay partidos históricos disponibles tras la limpieza (revisa la liga o el archivo).",
        )
        return wrap_page("Predictor Fútbol", body)

    return _run_prediction(
        matches_df, local, visitante,
        home_adjustment=pct_to_fraction("ajuste_local"), away_adjustment=pct_to_fraction("ajuste_visitante"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
