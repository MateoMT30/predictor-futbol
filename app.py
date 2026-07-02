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
{% if grouped_matches %}
  {% for grupo in grouped_matches %}
  <div class="day-header"{% if grupo.ancla_hoy %} id="hoy"{% endif %}>{{ grupo.dia }}</div>
  {% for m in grupo.partidos %}
  {% if m.finalizado %}
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

<a class="match-row" href="/" style="justify-content:center;color:var(--muted);">← Elegir otra competición</a>
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
    TOURNAMENTS = {"WC", "EC", "CL", "CLI"}
    dias_pasados = 45 if competition in TOURNAMENTS else 7
    try:
        agenda = connector.fetch_agenda(liga=competition, dias_pasados=dias_pasados)
    except Exception as e:
        body = render_template_string(
            MATCHES_BODY, competition=competition, competition_name=competition_name,
            grouped_matches=[], standings=None, scorers=None, error=f"No se pudo consultar football-data.org: {e}",
        )
        return wrap_page(competition_name, body)

    # Los nombres se traducen solo para mostrar; el valor que va en el link
    # (equipo_local/equipo_visitante) se deja en el idioma original de la
    # API, porque es el identificador que usa el modelo para cruzar contra
    # el histórico — traducirlo ahí rompería la búsqueda.
    matches = [
        {
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
            "marcador": (
                f"{int(row.goles_local)} - {int(row.goles_visitante)}"
                if getattr(row, "finalizado", False) and row.goles_local is not None
                else None
            ),
        }
        for row in agenda.itertuples()
    ]

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
        teams=teams_picker, prev_groups=prev_groups, error=None,
    )
    return wrap_page(competition_name, body)


def _run_prediction(matches_df, local, visitante, home_adjustment=0.0, away_adjustment=0.0,
                    fifa_context=None, avisos=None):
    """Lógica compartida por el flujo de API y el modo manual: ajusta los
    modelos sobre el histórico ya cargado y arma el reporte HTML."""
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

    # Historial de respaldo para equipos sin partidos en esta competición
    # (caso típico: recién ascendido — ej. un equipo nuevo en Premier League
    # no jugó ni un partido de PL el último año, así que el histórico de la
    # competición no sabe NADA de él y córners/tiros/tarjetas salían como
    # falsos ceros). Si el equipo no aparece, se trae su historial de la
    # competición "hermana" (PL<->Championship, las dos cubiertas por el plan
    # gratis de la API y por football-data.co.uk) y se agrega al histórico,
    # avisando en el reporte que se usó otra liga.
    avisos = []
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

    return _run_prediction(matches_df, local, visitante, fifa_context=fifa_context, avisos=avisos)


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
