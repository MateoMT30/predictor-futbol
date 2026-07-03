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


def _cmp_row(label: str, val_local, val_away, fmt: str = "{:.1f}") -> str:
    """Fila de comparación local/visitante con barras que crecen desde el
    centro (estilo SofaScore): valor local | etiqueta+barras | valor visitante.
    Un valor None significa que ESE equipo no tiene datos de esa estadística
    en el histórico: se muestra explícito en vez de un falso 0.0."""
    def side(v):
        return fmt.format(v) if v is not None else '<span class="nd">Sin datos en el histórico</span>'

    bars = ""
    numeric = all(isinstance(v, (int, float)) for v in (val_local, val_away))
    if numeric and (val_local + val_away) > 0:
        wl = val_local / (val_local + val_away) * 100
        bars = f"""
        <div class="cmp-bars">
          <div class="half h"><div class="fill-h" style="width:{wl:.0f}%"></div></div>
          <div class="half"><div class="fill-a" style="width:{100 - wl:.0f}%"></div></div>
        </div>"""

    return f"""
      <div class="cmp-row">
        <div class="cmp-val">{side(val_local)}</div>
        <div class="cmp-mid"><div class="cmp-label">{html.escape(label)}</div>{bars}</div>
        <div class="cmp-val away">{side(val_away)}</div>
      </div>"""


def _sample_footer(muestras: Optional[dict], home: str, away: str) -> str:
    """Pie con el tamaño de muestra por equipo — barato de mostrar y evita
    que un promedio calculado sobre 2 partidos se lea igual de confiable
    que uno sobre 19."""
    if not muestras or muestras.get("local") is None:
        return ""
    return f"""
      <p class="muted" style="margin-top:8px;">Muestra: {html.escape(home)} {muestras['local']} partido(s),
      {html.escape(away)} {muestras['visitante']} partido(s) con este dato.</p>"""


def _rng(summary: Optional[dict]) -> Optional[str]:
    if summary is None:
        return None
    lo, hi = summary["rango_esperado_p10_p90"]
    return f"{lo:.0f} - {hi:.0f}"


def _stat_card(title: str, summary_local: Optional[dict], summary_away: Optional[dict],
               summary_total: Optional[dict] = None, home: str = "Local", away: str = "Visitante",
               muestras: Optional[dict] = None) -> str:
    rows = [
        _cmp_row("Media", summary_local["media"] if summary_local else None,
                 summary_away["media"] if summary_away else None),
        _cmp_row("Rango esperado (P10-P90)", _rng(summary_local), _rng(summary_away), fmt="{}"),
    ]
    total_line = ""
    if summary_total:
        total_line = f"""
      <div class="cmp-total">Total: media {summary_total['media']:.1f} · rango {_rng(summary_total)}</div>"""

    return f"""
    <div class="card">
      <h2>{html.escape(title)}</h2>{''.join(rows)}{total_line}{_sample_footer(muestras, home, away)}
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

    Chips con semáforo de probabilidad (verde = probable, ámbar = parejo,
    gris = improbable) en vez de tabla: se lee de un vistazo. Solo se
    muestra la probabilidad de "más de X" (">X"); el "menos de X" es su
    complemento y mostrarlo era redundante.
    """
    def chip_class(p: float) -> str:
        if p >= 0.55:
            return "chip-hi"
        if p >= 0.35:
            return "chip-mid"
        return "chip-lo"

    chips = "".join(
        f'<span class="chip {chip_class(probs["over"])}">&gt;{line} {html.escape(metric_label)}'
        f"<b>{_pct(probs['over'])}</b></span>"
        for line, probs in lines.items()
    )
    return f"""
    <div class="card">
      <h2>{html.escape(title)}</h2>
      <div class="chips">{chips}</div>
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

    # --- Header de partido (estilo SofaScore): escudos + nombres + barra de
    # probabilidad 1X2 segmentada. Reemplaza al antiguo h1 + tarjeta "1X2" de
    # barras apiladas: la misma información, de un solo vistazo.
    def _crest_or_placeholder(url):
        if not url:
            return '<div class="crest-ph">⚽</div>'
        return f'<img class="crest-xl" loading="lazy" src="{html.escape(url)}" onerror="this.style.visibility=\'hidden\'">'

    p_home, p_draw, p_away = x1x2["local"], x1x2["empate"], x1x2["visitante"]
    match_header = f"""
  <div class="card match-header">
    <div class="mh-teams">
      <div class="mh-team">{_crest_or_placeholder(report.get("escudo_local"))}
        <div class="mh-name">{html.escape(home)}</div></div>
      <div class="mh-vs">VS</div>
      <div class="mh-team">{_crest_or_placeholder(report.get("escudo_visitante"))}
        <div class="mh-name">{html.escape(away)}</div></div>
    </div>
    <div class="prob-strip">
      <div class="prob-home" style="flex:{p_home:.3f}"></div>
      <div class="prob-draw" style="flex:{p_draw:.3f}"></div>
      <div class="prob-away" style="flex:{p_away:.3f}"></div>
    </div>
    <div class="prob-legend">
      <span><b>{_pct(p_home)}</b> gana</span>
      <span><b>{_pct(p_draw)}</b> empate</span>
      <span><b>{_pct(p_away)}</b> gana</span>
    </div>
  </div>"""

    # --- Veredicto destacado (estilo 365Scores "nuestro pronóstico"): el
    # resultado más probable, el marcador exacto y los goles esperados en
    # UNA tarjeta arriba — antes eran tres tarjetas separadas que pesaban
    # visualmente igual que cualquier mercado secundario.
    pick_label, pick_side, pick_prob = max(
        ((f"Gana {home}", "local", p_home), ("Empate", "empate", p_draw), (f"Gana {away}", "visitante", p_away)),
        key=lambda t: t[2],
    )

    # Marcador destacado = el MÁS PROBABLE de la matriz (argmax global). Se
    # midió en backtest walk-forward que este acierta más que el marcador
    # "coherente con el veredicto" (18.2% vs 13.6% de plenos): en partidos
    # parejos el resultado real SÍ suele ser 1-1/0-0, aunque visualmente
    # choque con un "Gana X" por décimas. La coherencia se preserva con la
    # nota condicional de abajo ("si gana X, típico 2-1"), sin sacrificar
    # precisión en el número destacado.
    mc = report.get("marcador_condicional")
    score_pill = ""
    if mp:
        score_pill = f"""
      <div class="score-pill"><small>Marcador exacto más probable</small>
        <span>{mp['local']}-{mp['visitante']}</span>
        <small>{_pct(mp['probabilidad'])}</small></div>"""

    # Top 3 de marcadores: evita la lectura errónea "el modelo dice empate"
    # cuando el 1-1 apenas le gana por decimales a los marcadores de victoria
    # del favorito (la victoria reparte su probabilidad entre muchos
    # marcadores; el empate la concentra casi toda en uno).
    tops = report.get("marcadores_probables") or []
    top_scores_line = ""
    if len(tops) > 1:
        items = " · ".join(
            f"<b>{t['local']}-{t['visitante']}</b> ({_pct(t['probabilidad'])})" for t in tops
        )
        top_scores_line = f"""
      <p class="muted" style="margin-top:10px;">Marcadores más probables: {items}</p>"""

    # Nota condicional: el marcador más típico DENTRO del escenario que el
    # modelo ve más probable. Da la lectura "realista" del ganador (ej. 2-1)
    # como CONTEXTO, sin robarle precisión al marcador destacado de arriba
    # (que es el argmax, más certero). Cubre tanto victorias como el matiz
    # de "si el empate es lo más probable pero igual hay un favorito leve".
    conditional_line = ""
    if mc and mc.get("resultado") != "empate":
        quien = home if mc["resultado"] == "local" else away
        if pick_side == mc["resultado"]:
            conditional_line = f"""
      <p class="muted">Si se impone {html.escape(quien)}, su marcador más típico es
      <b>{mc['local']}-{mc['visitante']}</b> ({_pct(mc['prob_dentro_escenario'])} de sus
      victorias simuladas).</p>"""
        else:
            conditional_line = f"""
      <p class="muted">Aunque lo más probable es el empate, si se impone alguien sería
      {html.escape(quien)}, típicamente <b>{mc['local']}-{mc['visitante']}</b>
      ({_pct(mc['prob_dentro_escenario'])} de sus victorias simuladas).</p>"""
    # Nivel de confianza del pick, derivado de su propia probabilidad. Nace
    # del análisis de errores: el fallo más común es un favorito con 45-60%
    # que termina en empate — no es un error del modelo, es que ese pick era
    # de por sí flojo. Hacerlo explícito evita leer un "Gana X" al 47% como
    # una promesa. Umbrales: >=60% alta, 45-60% media, <45% "muy parejo"
    # (el favorito lo es por poco; empate o sorpresa son muy posibles).
    if pick_prob >= 0.60:
        conf_txt, conf_cls = "Confianza alta", "conf-alta"
    elif pick_prob >= 0.45:
        conf_txt, conf_cls = "Confianza media", "conf-media"
    else:
        conf_txt, conf_cls = "Muy parejo", "conf-baja"
    conf_badge = f'<span class="conf-badge {conf_cls}">{conf_txt}</span>'

    verdict_card = f"""
  <div class="card">
    <h2>Pronóstico del modelo</h2>
    <div class="verdict">
      <div>
        <div class="verdict-pick">{html.escape(pick_label)} {conf_badge}</div>
        <div class="verdict-sub">{_pct(pick_prob)} de probabilidad ·
        goles esperados {ge['local']:.2f} - {ge['visitante']:.2f} (total {ge['total']:.2f})</div>
      </div>{score_pill}
    </div>{top_scores_line}{conditional_line}
    <p class="muted" style="margin-top:12px;">Ojo: el marcador más probable puede ser un empate
    aunque el pronóstico favorezca a un equipo — la victoria reparte su probabilidad entre muchos
    marcadores (1-0, 2-0, 2-1...) mientras el empate la concentra casi toda en uno solo; por eso
    se muestran los tres primeros. Los goles esperados son el promedio si el partido se jugara
    muchas veces en las mismas condiciones, no una predicción de marcador.</p>
  </div>"""

    # Tarjeta de forma reciente ("tendencia"): racha de los últimos partidos
    # de cada equipo con goles a favor/en contra — el rendimiento sobre el
    # que se apoya el pronóstico, visible en vez de implícito.
    forma = report.get("forma") or {}
    forma_card = ""
    if forma.get("local") or forma.get("visitante"):
        def _forma_row(team, f):
            if not f:
                return f"""
        <div class="cmp-row"><div class="cmp-val">{html.escape(team)}</div>
          <div class="cmp-mid muted">Sin partidos recientes en el histórico</div></div>"""
            badges = "".join(f'<span class="fb fb-{r.lower()}">{r}</span>' for r in f["racha"])
            return f"""
        <div class="cmp-row">
          <div class="cmp-val" style="width:auto;min-width:70px;">{html.escape(team)}</div>
          <div class="cmp-mid" style="text-align:left;">{badges}</div>
          <div class="cmp-val away" style="width:auto;">GF {f['gf']} · GC {f['gc']}</div>
        </div>"""

        forma_card = f"""
    <div class="card">
      <h2>Forma reciente (tendencia)</h2>{_forma_row(home, forma.get("local"))}{_forma_row(away, forma.get("visitante"))}
      <p class="muted" style="margin-top:8px;">Últimos {max((forma.get("local") or {}).get("n", 0),
      (forma.get("visitante") or {}).get("n", 0))} partidos, el más reciente primero
      (G = ganó, E = empató, P = perdió; GF/GC = goles a favor/en contra en esa racha).
      El modelo ya pondera más lo reciente — esta tarjeta lo hace visible.</p>
    </div>"""

    # ¿Quién clasifica? Solo en torneos (app.py marca es_eliminatoria): el
    # 1X2 modela los 90 minutos; en muerte súbita el empate = alargue/penales
    # y lo que importa es quién pasa de ronda.
    clasif = report.get("clasificacion_eliminatoria")
    clasif_card = ""
    if clasif and report.get("es_eliminatoria"):
        clasif_card = f"""
    <div class="card">
      <h2>¿Quién clasifica? (eliminación directa)</h2>
      {_bar(home, clasif["local"], "#6366f1")}
      {_bar(away, clasif["visitante"], "#ef4444")}
      <p class="muted">El pronóstico de arriba es a 90 minutos (por eso existe el "empate"):
      si el partido es de eliminación directa, ese empate significa alargue y penales. Aquí esa
      probabilidad se reparte según la fuerza relativa de los equipos para responder la pregunta
      que importa: quién pasa de ronda.</p>
    </div>"""

    body = f"""
  {match_header}
  <div class="subtitle" style="text-align:center;">Generado el {generated_at} · predictor-futbol</div>
  {avisos_banner}
  {ajuste_banner}
  {verdict_card}
  {clasif_card}
  {forma_card}

  <div class="card">
    <h2>Ambos anotan</h2>
    {_bar("Sí", btts["si"], "#22c55e")}
    {_bar("No", btts["no"], "#ef4444")}
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
