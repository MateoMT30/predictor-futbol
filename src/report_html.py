"""
report_html.py
===============

Genera un reporte HTML autocontenido (un solo archivo, sin dependencias
externas ni servidor) a partir del mismo dict que produce main.build_report().

Por qué HTML y no solo JSON: el JSON es ideal para integrarse con otro
sistema, pero no es legible a simple vista, y mucho menos cómodo desde un
celular. Un único .html con CSS embebido se puede abrir directamente en
cualquier navegador (incluso enviándolo por WhatsApp/correo/Drive) sin
necesitar Python, un servidor, ni conexión a internet.

Todo el CSS y el layout son mobile-first (una sola columna, texto grande,
tarjetas apilables) porque el caso de uso principal es abrir esto desde el
teléfono.
"""

import html
from datetime import datetime, timezone
from typing import Optional

from .i18n import to_colombia_time


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _bar(label: str, value: float, color: str = "#3b82f6") -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return f"""
    <div class="bar-row">
      <div class="bar-label">{html.escape(label)}<span class="bar-value">{_pct(value)}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div></div>
    </div>"""


def _stat_card(title: str, summary_local: dict, summary_away: dict, summary_total: Optional[dict] = None) -> str:
    rows = []
    rows.append(f"""
      <tr><td>Local</td><td>{summary_local['media']:.1f}</td>
          <td>{summary_local['rango_esperado_p10_p90'][0]:.0f} - {summary_local['rango_esperado_p10_p90'][1]:.0f}</td></tr>""")
    rows.append(f"""
      <tr><td>Visitante</td><td>{summary_away['media']:.1f}</td>
          <td>{summary_away['rango_esperado_p10_p90'][0]:.0f} - {summary_away['rango_esperado_p10_p90'][1]:.0f}</td></tr>""")
    if summary_total:
        rows.append(f"""
      <tr class="total-row"><td>Total</td><td>{summary_total['media']:.1f}</td>
          <td>{summary_total['rango_esperado_p10_p90'][0]:.0f} - {summary_total['rango_esperado_p10_p90'][1]:.0f}</td></tr>""")

    return f"""
    <div class="card">
      <h2>{html.escape(title)}</h2>
      <table>
        <thead><tr><th></th><th>Media</th><th>Rango esperado (P10-P90)</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>"""


def _no_data_card(title: str) -> str:
    return f"""
    <div class="card">
      <h2>{html.escape(title)}</h2>
      <p class="muted">Sin datos suficientes: la fuente de histórico usada no incluye
      esta estadística (por ejemplo, la API gratuita de football-data.org solo trae
      goles). Carga tu propio histórico con estas columnas para habilitar este mercado.</p>
    </div>"""


def _over_under_card(title: str, lines: dict, metric_label: str) -> str:
    """
    metric_label: nombre en español de lo que se está contando (ej.
    "goles", "córners"), usado en la explicación de qué significa cada
    línea — este mercado es el que más confunde a alguien que no viene
    del mundo de las apuestas.
    """
    rows = "".join(
        f"<tr><td>{line}</td><td>{_pct(probs['over'])}</td><td>{_pct(probs['under'])}</td></tr>"
        for line, probs in lines.items()
    )
    return f"""
    <div class="card">
      <h2>{html.escape(title)}</h2>
      <p class="muted">"Over X" = probabilidad de que el total de {html.escape(metric_label)}
      del partido termine por ENCIMA de X. "Under X" = por DEBAJO de X. Ej.: si la línea es 2.5,
      "Over 2.5" significa 3 {html.escape(metric_label)} o más.</p>
      <table>
        <thead><tr><th>Línea</th><th>Over</th><th>Under</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def render_html_report(report: dict, value_bets: Optional[list] = None) -> str:
    """
    value_bets es opcional: la app web (app.py) ya no pide cuotas por
    defecto, así que la mayoría de las veces esto viene vacío y la
    sección de value bets simplemente no se muestra. Se mantiene la
    capacidad (útil para quien sí quiera comparar contra una cuota
    puntual desde el CLI con --cuotas) sin forzarla en el flujo principal.
    """
    p = report["partido"]
    home, away = p["local"], p["visitante"]

    x1x2 = report["1x2"]
    btts = report["ambos_anotan"]
    ge = report["goles_esperados"]

    value_bets_card = ""
    if value_bets:
        marked = [vb for vb in value_bets if vb["value_bet"]]
        if marked:
            rows = "".join(f"""
              <div class="value-bet">
                <div class="vb-market">{html.escape(vb['mercado'])} — {html.escape(str(vb['resultado']))}</div>
                <div class="vb-detail">Modelo: <b>{_pct(vb['probabilidad_modelo'])}</b> vs Implícita: {_pct(vb['probabilidad_implicita'])}
                  (cuota {vb['cuota']}, edge +{vb['edge']*100:.1f}pp)</div>
              </div>""" for vb in marked)
        else:
            rows = '<p class="muted">Ninguno por encima del umbral configurado.</p>'
        value_bets_card = f'<div class="card"><h2>Value bets</h2>{rows}</div>'

    if report.get("corners"):
        corners_section = (
            _stat_card("Córners", report["corners"]["local"], report["corners"]["visitante"], report["corners"]["total"])
            + _over_under_card("Over/Under córners", report["over_under_corners"], "córners")
        )
    else:
        corners_section = _no_data_card("Córners")

    if report.get("tiros_al_arco"):
        shots_section = (
            _stat_card("Tiros al arco", report["tiros_al_arco"]["local"], report["tiros_al_arco"]["visitante"], report["tiros_al_arco"]["total"])
            + _over_under_card("Over/Under tiros al arco", report["over_under_tiros"], "tiros al arco")
        )
    else:
        shots_section = _no_data_card("Tiros al arco")

    if report.get("tarjetas"):
        cards_section = (
            _stat_card("Tarjetas amarillas", report["tarjetas"]["amarillas_local"], report["tarjetas"]["amarillas_visitante"], report["tarjetas"]["amarillas_total"])
            + _over_under_card("Over/Under tarjetas amarillas", report["over_under_tarjetas"], "tarjetas amarillas")
            + f"""
    <div class="card">
      <h2>Tarjetas rojas (media esperada)</h2>
      <div class="goals-summary">
        <div>Local<span>{report['tarjetas']['rojas_local']['media']:.2f}</span></div>
        <div>Visitante<span>{report['tarjetas']['rojas_visitante']['media']:.2f}</span></div>
      </div>
    </div>"""
        )
    else:
        cards_section = _no_data_card("Tarjetas")

    generated_at = to_colombia_time(datetime.now(timezone.utc)).strftime("%Y-%m-%d %I:%M %p") + " (hora Colombia)"

    ajuste = report.get("ajuste_manual_aplicado", {"local": 0.0, "visitante": 0.0})
    ajuste_banner = ""
    if ajuste.get("local", 0) != 0 or ajuste.get("visitante", 0) != 0:
        ajuste_banner = f"""
    <div class="card" style="border-left:4px solid var(--amber);">
      <h2 style="color:var(--amber);">⚠ Ajuste manual aplicado (no estadístico)</h2>
      <p class="muted">Local: {ajuste['local']*100:+.0f}% &nbsp; Visitante: {ajuste['visitante']*100:+.0f}%</p>
      <p class="muted">Esto refleja criterio humano (bajas, lesiones, rotación), no algo que
      el modelo haya inferido de los resultados históricos.</p>
    </div>"""

    html_doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pronóstico: {html.escape(home)} vs {html.escape(away)}</title>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --muted: #94a3b8;
    --accent: #3b82f6; --green: #22c55e; --amber: #f59e0b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 16px; background: var(--bg); color: var(--text);
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    max-width: 720px; margin-inline: auto;
  }}
  h1 {{ font-size: 1.4rem; margin: 8px 0 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 16px; }}
  .card {{
    background: var(--card); border-radius: 12px; padding: 16px;
    margin-bottom: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  }}
  .card h2 {{ font-size: 1rem; margin: 0 0 12px; color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 6px 4px; border-bottom: 1px solid rgba(255,255,255,0.08); }}
  th {{ color: var(--muted); font-weight: 500; }}
  .total-row {{ font-weight: 700; }}
  .bar-row {{ margin-bottom: 12px; }}
  .bar-label {{ display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px; }}
  .bar-value {{ font-weight: 700; }}
  .bar-track {{ background: rgba(255,255,255,0.08); border-radius: 6px; height: 10px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 6px; }}
  .value-bet {{
    background: rgba(34,197,94,0.12); border-left: 4px solid var(--green);
    border-radius: 6px; padding: 10px 12px; margin-bottom: 10px;
  }}
  .vb-market {{ font-weight: 700; margin-bottom: 2px; }}
  .vb-detail {{ font-size: 0.85rem; color: var(--muted); }}
  .muted {{ color: var(--muted); font-size: 0.9rem; }}
  .disclaimer {{
    font-size: 0.78rem; color: var(--muted); border-top: 1px solid rgba(255,255,255,0.1);
    margin-top: 24px; padding-top: 12px; line-height: 1.4;
  }}
  .goals-summary {{ display: flex; justify-content: space-around; text-align: center; margin-top: 8px; }}
  .goals-summary div span {{ display: block; font-size: 1.3rem; font-weight: 700; color: var(--accent); }}
</style>
</head>
<body>
  <h1>{html.escape(home)} vs {html.escape(away)}</h1>
  <div class="subtitle">Generado el {generated_at} · predictor-futbol</div>
  {ajuste_banner}
  <div class="card">
    <h2>1X2</h2>
    {_bar(f"Local ({html.escape(home)})", x1x2["local"], "#3b82f6")}
    {_bar("Empate", x1x2["empate"], "#94a3b8")}
    {_bar(f"Visitante ({html.escape(away)})", x1x2["visitante"], "#ef4444")}
  </div>

  <div class="card">
    <h2>Ambos anotan</h2>
    {_bar("Sí", btts["si"], "#22c55e")}
    {_bar("No", btts["no"], "#ef4444")}
    <div class="goals-summary">
      <div>Local<span>{ge['local']:.2f}</span></div>
      <div>Visitante<span>{ge['visitante']:.2f}</span></div>
      <div>Total<span>{ge['total']:.2f}</span></div>
    </div>
  </div>

  {_over_under_card("Over/Under goles", report["over_under_goles"], "goles")}
  {corners_section}
  {shots_section}
  {cards_section}

  {value_bets_card}

  <div class="disclaimer">
    <b>Disclaimer:</b> este es un modelo probabilístico basado en datos históricos.
    No garantiza resultados. Las apuestas deportivas implican riesgo real de
    pérdida de dinero. Ver README.md para las limitaciones completas del modelo.
  </div>
</body>
</html>"""
    return html_doc
