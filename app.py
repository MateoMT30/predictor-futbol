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

from flask import Flask, render_template_string, request, redirect, url_for

from src.connectors.csv_connector import CSVConnector
from src.connectors.football_data_connector import FootballDataConnector, COMPETITIONS
from src.data_loader import load_from_connector
from src.ratings import EloRatingSystem, RatingsConfig
from src.models.goles import DixonColesModel, GoalsModelConfig
from src.models.corners import CornersModel, CornersModelConfig
from src.models.tiros import ShotsOnTargetModel, ShotsModelConfig
from src.models.tarjetas import CardsModel, CardsModelConfig
from src.simulation import MatchSimulator, SimulationConfig
from src.report_html import render_html_report
from src.main import load_config, build_report

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "examples" / "historico_ejemplo.csv"

BASE_STYLE = """
<style>
  :root { --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --accent:#3b82f6; }
  * { box-sizing: border-box; }
  body { margin:0; padding:16px; background:var(--bg); color:var(--text);
    font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width:720px; margin-inline:auto; }
  h1 { font-size: 1.3rem; }
  .subtitle { color: var(--muted); font-size: 0.8rem; margin-bottom: 16px; }
  .card { background: var(--card); border-radius: 12px; padding: 16px; margin-bottom: 14px; }
  label { display:block; font-size:0.85rem; color:var(--muted); margin:10px 0 4px; }
  input, select, button { width:100%; padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.15);
    background:#0f172a; color:var(--text); font-size:1rem; }
  button { background: var(--accent); border:none; font-weight:700; cursor:pointer; margin-top:16px; padding:12px; }
  .error { background: rgba(239,68,68,0.15); color:#fca5a5; padding:10px; border-radius:8px; margin-bottom:12px; }
  a.match-row { display:block; text-decoration:none; color:var(--text); background:#0f172a;
    border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:12px; margin-bottom:8px; }
  a.match-row:hover { border-color: var(--accent); }
  .match-date { color: var(--muted); font-size: 0.8rem; }
  .match-teams { font-weight: 700; margin-top: 2px; }
  details summary { color: var(--accent); cursor: pointer; font-size: 0.85rem; margin-top: 12px; }
</style>
"""

INDEX_TEMPLATE = BASE_STYLE + """
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
    <button type="submit">Ver próximos partidos</button>
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
<div class="subtitle" style="margin-top:20px;">
  Disclaimer: modelo probabilístico basado en datos históricos. No garantiza
  resultados. Las apuestas deportivas implican riesgo real de pérdida de dinero.
</div>
"""

MATCHES_TEMPLATE = BASE_STYLE + """
<h1>Próximos partidos</h1>
<div class="subtitle">{{ competition_name }} — toca un partido para ver el pronóstico.</div>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
{% if matches %}
  {% for m in matches %}
  <a class="match-row" href="{{ url_for('predecir', competition=competition, local=m.equipo_local, visitante=m.equipo_visitante) }}">
    <div class="match-date">{{ m.fecha_str }}</div>
    <div class="match-teams">{{ m.equipo_local }} vs {{ m.equipo_visitante }}</div>
  </a>
  {% endfor %}
{% elif not error %}
  <p class="subtitle">No hay partidos programados próximamente para esta competición.</p>
{% endif %}
<a class="match-row" href="/" style="text-align:center;color:var(--muted);">← Elegir otra competición</a>
"""


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
    return render_template_string(INDEX_TEMPLATE, competitions=COMPETITIONS, error=error)


@app.route("/partidos", methods=["GET"])
def partidos():
    competition = request.args.get("competition", "")
    connector = _get_api_connector()
    if not connector:
        return render_template_string(
            MATCHES_TEMPLATE, competition=competition, competition_name=COMPETITIONS.get(competition, competition),
            matches=[], error="Falta configurar FOOTBALL_DATA_API_KEY en el servidor.",
        )
    try:
        upcoming = connector.fetch_upcoming(liga=competition)
    except Exception as e:
        return render_template_string(
            MATCHES_TEMPLATE, competition=competition, competition_name=COMPETITIONS.get(competition, competition),
            matches=[], error=f"No se pudo consultar football-data.org: {e}",
        )

    matches = [
        {
            "equipo_local": row.equipo_local,
            "equipo_visitante": row.equipo_visitante,
            "fecha_str": row.fecha_hora.strftime("%a %d %b, %H:%M UTC"),
        }
        for row in upcoming.itertuples()
    ]
    return render_template_string(
        MATCHES_TEMPLATE, competition=competition, competition_name=COMPETITIONS.get(competition, competition),
        matches=matches, error=None,
    )


def _run_prediction(matches_df, local, visitante, home_adjustment=0.0, away_adjustment=0.0):
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
        matches_df=matches_df,
    )
    return render_html_report(report, value_bets=None)


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

    return _run_prediction(matches_df, local, visitante)


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
        return render_template_string(INDEX_TEMPLATE, competitions=COMPETITIONS, error="Debes indicar equipo local y visitante.")

    try:
        connector = CSVConnector(matches_path=datos_path)
        matches_df, cleaning_report = load_from_connector(connector, liga=liga)
    except FileNotFoundError as e:
        return render_template_string(INDEX_TEMPLATE, competitions=COMPETITIONS, error=str(e))

    if matches_df.empty:
        return render_template_string(
            INDEX_TEMPLATE, competitions=COMPETITIONS,
            error="No hay partidos históricos disponibles tras la limpieza (revisa la liga o el archivo).",
        )

    return _run_prediction(
        matches_df, local, visitante,
        home_adjustment=pct_to_fraction("ajuste_local"), away_adjustment=pct_to_fraction("ajuste_visitante"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
