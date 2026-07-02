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
from .web_style import wrap_page


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _bar(label: str, value: float, color: str = "#3b82f6") -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return f"""
    <div class="bar-row">
      <div class="bar-label">{html.escape(label)}<span class="bar-value">{_pct(value)}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div></div>
    </div>"""


def _stat_row(label: str, summary: Optional[dict], total: bool = False) -> str:
    """Fila de la tabla de un mercado. summary=None significa que ESE equipo
    no tiene datos de esa estadística en el histórico (distinto de que la
    fuente no la traiga): se muestra explícito en vez de un falso 0.0."""
    cls = ' class="total-row"' if total else ""
    if summary is None:
        return f"""
      <tr{cls}><td>{html.escape(label)}</td><td colspan="2" class="muted">Sin datos en el histórico</td></tr>"""
    return f"""
      <tr{cls}><td>{html.escape(label)}</td><td>{summary['media']:.1f}</td>
          <td>{summary['rango_esperado_p10_p90'][0]:.0f} - {summary['rango_esperado_p10_p90'][1]:.0f}</td></tr>"""


def _sample_footer(muestras: Optional[dict], home: str, away: str) -> str:
    """Pie con el tamaño de muestra por equipo — barato de mostrar y evita
    que un promedio calculado sobre 2 partidos se lea igual de confiable
    que uno sobre 19."""
    if not muestras or muestras.get("local") is None:
        return ""
    return f"""
      <p class="muted" style="margin-top:8px;">Muestra: {html.escape(home)} {muestras['local']} partido(s),
      {html.escape(away)} {muestras['visitante']} partido(s) con este dato.</p>"""


def _stat_card(title: str, summary_local: Optional[dict], summary_away: Optional[dict],
               summary_total: Optional[dict] = None, home: str = "Local", away: str = "Visitante",
               muestras: Optional[dict] = None) -> str:
    rows = [
        _stat_row(home, summary_local),
        _stat_row(away, summary_away),
    ]
    if summary_local is not None or summary_away is not None:
        rows.append(_stat_row("Total", summary_total, total=True))

    return f"""
    <div class="card">
      <h2>{html.escape(title)}</h2>
      <table>
        <thead><tr><th></th><th>Media</th><th>Rango esperado (P10-P90)</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>{_sample_footer(muestras, home, away)}
    </div>"""


def _incomplete_ou_card(title: str) -> str:
    """Se usa cuando el mercado existe pero uno de los dos equipos no tiene
    datos: un over/under del TOTAL calculado con la mitad del partido saldría
    artificialmente bajo y engañoso, así que se omite con explicación."""
    return f"""
    <div class="card">
      <h2>{html.escape(title)}</h2>
      <p class="muted">No se calcula: uno de los equipos no tiene datos de esta
      estadística en el histórico, y un total calculado solo con el otro equipo
      saldría engañosamente bajo.</p>
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
    "goles", "córners").

    Solo se muestra la probabilidad de "más de X" (">X"). No se repite el
    "menos de X": es matemáticamente el complemento (100% - "más de X"),
    mostrarlo en una segunda columna era información redundante.
    """
    rows = "".join(
        f"<tr><td>&gt;{line} {html.escape(metric_label)}</td><td>{_pct(probs['over'])}</td></tr>"
        for line, probs in lines.items()
    )
    return f"""
    <div class="card">
      <h2>{html.escape(title)}</h2>
      <table>
        <thead><tr><th>Línea</th><th>Probabilidad</th></tr></thead>
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
    mp = report.get("marcador_mas_probable")

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

    fifa_context = report.get("fifa_context")
    fifa_context_card = ""
    if fifa_context and (fifa_context.get("local") or fifa_context.get("visitante")):
        def _row(label, unit, key):
            l = fifa_context.get("local") or {}
            v = fifa_context.get("visitante") or {}
            lv, vv = l.get(key), v.get(key)
            if lv is None and vv is None:
                return ""
            lv_str = f"{lv}{unit}" if lv is not None else "—"
            vv_str = f"{vv}{unit}" if vv is not None else "—"
            return f"<tr><td>{html.escape(label)}</td><td>{lv_str}</td><td>{vv_str}</td></tr>"

        rows = "".join([
            _row("xG (goles esperados reales)", "", "xg"),
            _row("Posesión", "%", "posesion"),
            _row("Precisión de pase", "%", "precision_pase"),
            _row("Córners", "", "corners"),
            _row("Tiros libres", "", "tiros_libres"),
            _row("Penales", "", "penales"),
        ])
        n_local = (fifa_context.get("local") or {}).get("partidos_con_dato", 0)
        n_away = (fifa_context.get("visitante") or {}).get("partidos_con_dato", 0)
        fifa_context_card = f"""
    <div class="card">
      <h2>Datos oficiales FIFA (promedio del torneo)</h2>
      <p class="muted">Extraído de los reportes oficiales de partido de la FIFA
      ({n_local} partido(s) de {html.escape(home)}, {n_away} de {html.escape(away)}).
      Es contexto informativo real — no se usa para calcular ninguna probabilidad
      de las de arriba, a diferencia de córners/tiros al arco que sí alimentan el modelo.</p>
      <table>
        <thead><tr><th>Estadística</th><th>{html.escape(home)}</th><th>{html.escape(away)}</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    if report.get("corners"):
        c = report["corners"]
        corners_section = (
            _stat_card("Córners", c["local"], c["visitante"], c["total"],
                       home=home, away=away, muestras=c.get("muestras"))
            + (_over_under_card("Over/Under córners", report["over_under_corners"], "córners")
               if report.get("over_under_corners") else _incomplete_ou_card("Over/Under córners"))
        )
    else:
        corners_section = _no_data_card("Córners")

    if report.get("tiros_al_arco"):
        t = report["tiros_al_arco"]
        shots_section = (
            _stat_card("Tiros al arco", t["local"], t["visitante"], t["total"],
                       home=home, away=away, muestras=t.get("muestras"))
            + (_over_under_card("Over/Under tiros al arco", report["over_under_tiros"], "tiros al arco")
               if report.get("over_under_tiros") else _incomplete_ou_card("Over/Under tiros al arco"))
        )
    else:
        shots_section = _no_data_card("Tiros al arco")

    if report.get("tarjetas"):
        ta = report["tarjetas"]
        rl = ta["rojas_local"]
        rv = ta["rojas_visitante"]
        rl_str = f"{rl['media']:.2f}" if rl else "—"
        rv_str = f"{rv['media']:.2f}" if rv else "—"
        cards_section = (
            _stat_card("Tarjetas amarillas", ta["amarillas_local"], ta["amarillas_visitante"], ta["amarillas_total"],
                       home=home, away=away, muestras=ta.get("muestras"))
            + (_over_under_card("Over/Under tarjetas amarillas", report["over_under_tarjetas"], "tarjetas amarillas")
               if report.get("over_under_tarjetas") else _incomplete_ou_card("Over/Under tarjetas amarillas"))
            + f"""
    <div class="card">
      <h2>Tarjetas rojas (media esperada)</h2>
      <div class="goals-summary">
        <div>{html.escape(home)}<span>{rl_str}</span></div>
        <div>{html.escape(away)}<span>{rv_str}</span></div>
      </div>
    </div>"""
        )
    else:
        cards_section = _no_data_card("Tarjetas")

    generated_at = to_colombia_time(datetime.now(timezone.utc)).strftime("%Y-%m-%d %I:%M %p") + " (hora Colombia)"

    avisos = report.get("avisos") or []
    avisos_banner = ""
    if avisos:
        items = "".join(f"<li>{html.escape(a)}</li>" for a in avisos)
        avisos_banner = f"""
    <div class="card" style="border-left:4px solid var(--amber);">
      <h2 style="color:var(--amber);">⚠ Avisos sobre los datos</h2>
      <ul class="muted" style="margin:0;padding-left:18px;">{items}</ul>
    </div>"""

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

    def _crest_img(url):
        if not url:
            return ""
        return f'<img class="crest-lg" loading="lazy" src="{html.escape(url)}" onerror="this.style.visibility=\'hidden\'">'

    crest_home = report.get("escudo_local")
    crest_away = report.get("escudo_visitante")
    crests_html = ""
    if crest_home or crest_away:
        crests_html = f"""<div class="match-crests" style="margin-bottom:8px;">
      {_crest_img(crest_home)}
      {_crest_img(crest_away)}
    </div>"""

    body = f"""
  {crests_html}
  <h1>{html.escape(home)} vs {html.escape(away)}</h1>
  <div class="subtitle">Generado el {generated_at} · predictor-futbol</div>
  {avisos_banner}
  {ajuste_banner}
  <div class="card">
    <h2>1X2</h2>
    {_bar(home, x1x2["local"], "#6366f1")}
    {_bar("Empate", x1x2["empate"], "#94a3b8")}
    {_bar(away, x1x2["visitante"], "#ef4444")}
  </div>

  <div class="card">
    <h2>Ambos anotan</h2>
    {_bar("Sí", btts["si"], "#22c55e")}
    {_bar("No", btts["no"], "#ef4444")}
  </div>

  {f'''<div class="card">
    <h2>Marcador exacto más probable</h2>
    <div class="goals-summary">
      <div>Resultado<span>{mp['local']}-{mp['visitante']}</span></div>
      <div>Probabilidad<span>{_pct(mp['probabilidad'])}</span></div>
    </div>
    <p class="muted" style="margin-top:10px;">De los miles de marcadores posibles que simula el
    modelo, este es el único marcador entero específico con mayor probabilidad individual — su
    porcentaje suele ser bajo (a veces menos de 15%) porque compite contra muchos otros
    resultados posibles, no porque el modelo esté poco seguro.</p>
  </div>''' if mp else ''}

  <div class="card">
    <h2>Promedio de goles (referencia)</h2>
    <p class="muted">No es una predicción de marcador — es el promedio si el partido se jugara
    muchas veces en las mismas condiciones. Alimenta el cálculo del resto de mercados (1X2,
    over/under, marcador exacto de arriba).</p>
    <div class="goals-summary">
      <div>{html.escape(home)}<span>{ge['local']:.2f}</span></div>
      <div>{html.escape(away)}<span>{ge['visitante']:.2f}</span></div>
      <div>Total<span>{ge['total']:.2f}</span></div>
    </div>
  </div>

  {_over_under_card("Over/Under goles", report["over_under_goles"], "goles")}
  {fifa_context_card}
  {corners_section}
  {shots_section}
  {cards_section}

  {value_bets_card}

  <div class="disclaimer">
    <b>Disclaimer:</b> este es un modelo probabilístico basado en datos históricos.
    No garantiza resultados. Las apuestas deportivas implican riesgo real de
    pérdida de dinero. Ver README.md para las limitaciones completas del modelo.
  </div>
"""
    return wrap_page(f"Pronóstico: {home} vs {away}", body, redirect_home_on_reload=True)
