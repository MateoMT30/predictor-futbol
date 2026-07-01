"""
web_style.py
============

Sistema visual compartido entre app.py (Flask) y report_html.py (reporte
autocontenido), para que ambos se vean consistentes sin duplicar el CSS.

Referencias de diseño: paleta oscura con acento en gradiente (estilo
dashboards modernos tipo Linear/Vercel/Stripe), tarjetas con blur sutil,
transiciones suaves en hover, tipografía Inter (Google Fonts) en vez de la
fuente del sistema. Todo sigue siendo mobile-first: una sola columna,
textos grandes, nada que dependa de mouse hover para ser usable.
"""

import html as _html

MODERN_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  :root {
    --bg: #0a0e1a;
    --card: rgba(30, 41, 59, 0.55);
    --card-border: rgba(148, 163, 184, 0.12);
    --text: #e8ecf3;
    --muted: #8b96ab;
    --accent: #6366f1;
    --accent2: #22d3ee;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0; padding: 20px 16px 40px; color: var(--text);
    font-family: 'Inter', -apple-system, Segoe UI, Roboto, sans-serif;
    max-width: 720px; margin-inline: auto;
    background:
      radial-gradient(ellipse 800px 500px at 20% -10%, rgba(99,102,241,0.25), transparent),
      radial-gradient(ellipse 600px 400px at 100% 0%, rgba(34,211,238,0.15), transparent),
      var(--bg);
    background-attachment: fixed;
  }
  h1 {
    font-size: 1.6rem; font-weight: 800; margin: 4px 0 4px; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #fff, #c7d2fe);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  }
  h2 { font-size: 1rem; font-weight: 700; margin: 0 0 12px; color: var(--accent2); letter-spacing: -0.01em; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 18px; line-height: 1.4; }
  .card {
    background: var(--card); border: 1px solid var(--card-border);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-radius: 16px; padding: 18px; margin-bottom: 14px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    animation: fadeUp 0.35s ease both;
  }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  label { display:block; font-size:0.82rem; color:var(--muted); margin:12px 0 6px; font-weight: 500; }
  input, select, button {
    width:100%; padding:12px 14px; border-radius:10px; border:1px solid var(--card-border);
    background: rgba(10,14,26,0.6); color:var(--text); font-size:1rem; font-family: inherit;
    transition: border-color 0.15s ease, background 0.15s ease;
  }
  input:focus, select:focus { outline: none; border-color: var(--accent); background: rgba(10,14,26,0.9); }
  button {
    background: linear-gradient(135deg, var(--accent), #818cf8); border:none; color: white;
    font-weight:700; cursor:pointer; margin-top:16px; padding:13px; font-size: 0.95rem;
    box-shadow: 0 4px 16px rgba(99,102,241,0.35);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
  }
  button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99,102,241,0.45); }
  button:active { transform: translateY(0); }
  .error {
    background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3); color:#fca5a5;
    padding:12px 14px; border-radius:12px; margin-bottom:14px; font-size: 0.88rem; line-height: 1.4;
  }
  a.match-row {
    display:flex; align-items:center; gap: 12px; text-decoration:none; color:var(--text);
    background: var(--card); border: 1px solid var(--card-border);
    backdrop-filter: blur(12px); border-radius: 14px; padding:14px; margin-bottom:10px;
    transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
  }
  a.match-row:hover { border-color: var(--accent); transform: translateY(-2px) scale(1.01); background: rgba(99,102,241,0.08); }
  a.match-row:active { transform: translateY(0) scale(0.99); }
  .match-date { color: var(--muted); font-size: 0.75rem; margin-bottom: 3px; }
  .match-teams { font-weight: 600; font-size: 0.95rem; }
  .crest { width: 22px; height: 22px; object-fit: contain; flex-shrink: 0; }
  .crest-lg { width: 40px; height: 40px; object-fit: contain; }
  .match-crests { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
  .muted { color: var(--muted); font-weight: 400; font-size: 0.8rem; }

  /* Lista de partidos estilo apps deportivas (agrupada por día) */
  .day-header {
    display: flex; align-items: center; gap: 10px; margin: 22px 0 10px;
    color: var(--muted); font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .day-header:first-of-type { margin-top: 4px; }
  .day-header::before, .day-header::after { content: ""; flex: 1; height: 1px; background: var(--card-border); }
  .match-row-v2 {
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
    text-decoration: none; color: var(--text); background: var(--card); border: 1px solid var(--card-border);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-radius: 14px; padding: 12px 14px; margin-bottom: 8px;
    transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
  }
  .match-row-v2:hover { border-color: var(--accent); transform: translateY(-2px); background: rgba(99,102,241,0.08); }
  .match-row-v2:active { transform: translateY(0) scale(0.99); }
  .mr-teams { display: flex; flex-direction: column; gap: 7px; flex: 1; min-width: 0; }
  .mr-team { display: flex; align-items: center; gap: 8px; font-size: 0.88rem; font-weight: 500; }
  .mr-team span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .mr-time { font-size: 0.82rem; color: var(--accent2); font-weight: 700; white-space: nowrap; flex-shrink: 0; }
  details summary {
    color: var(--accent2); cursor: pointer; font-size: 0.88rem; margin-top: 14px;
    font-weight: 600; list-style: none; display: flex; align-items: center; gap: 6px;
  }
  details summary::-webkit-details-marker { display: none; }
  details summary::before { content: "▸"; transition: transform 0.15s ease; }
  details[open] summary::before { transform: rotate(90deg); }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--card-border); }
  th { color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em; }
  .total-row { font-weight: 700; color: var(--accent2); }
  .bar-row { margin-bottom: 14px; }
  .bar-label { display: flex; justify-content: space-between; font-size: 0.92rem; margin-bottom: 6px; font-weight: 500; }
  .bar-value { font-weight: 800; }
  .bar-track { background: rgba(148,163,184,0.12); border-radius: 8px; height: 12px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 8px; transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1); }
  .value-bet {
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.25);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
  }
  .vb-market { font-weight: 700; margin-bottom: 2px; }
  .vb-detail { font-size: 0.85rem; color: var(--muted); }
  .goals-summary { display: flex; justify-content: space-around; text-align: center; margin-top: 10px; }
  .goals-summary div span { display: block; font-size: 1.4rem; font-weight: 800; color: var(--accent2); margin-top: 2px; }
  .pos-badge {
    display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px;
    border-radius: 6px; background: rgba(148,163,184,0.12); font-size: 0.75rem; font-weight: 700; color: var(--muted);
  }
  .disclaimer {
    font-size: 0.76rem; color: var(--muted); border-top: 1px solid var(--card-border);
    margin-top: 24px; padding-top: 14px; line-height: 1.5;
  }
</style>
"""


# Script que, SOLO cuando la página se recarga (F5 / botón refrescar), lleva
# al inicio en vez de re-ejecutar la predicción (que es lenta y no tiene
# sentido repetir). Usa el tipo de navegación del navegador: 'reload' vs
# 'navigate'/'back_forward', así que abrir el reporte normalmente NO redirige;
# solo el refresco explícito lo hace.
_RELOAD_TO_HOME_SCRIPT = """
<script>
(function () {
  try {
    var navs = performance.getEntriesByType ? performance.getEntriesByType('navigation') : [];
    var type = navs.length ? navs[0].type
             : (performance.navigation && performance.navigation.type === 1 ? 'reload' : '');
    if (type === 'reload' && location.pathname !== '/') {
      location.replace('/');
    }
  } catch (e) {}
})();
</script>"""


def wrap_page(title: str, body_html: str, subtitle_meta: str = "",
              redirect_home_on_reload: bool = False) -> str:
    """Envuelve un fragmento de contenido en un documento HTML completo
    (con <head>, viewport, favicon emoji), consistente para todas las
    páginas de la app y del reporte.

    redirect_home_on_reload: si True, al recargar la página (F5) se redirige
    al inicio en lugar de recalcular. Se usa en la página del reporte."""
    reload_script = _RELOAD_TO_HOME_SCRIPT if redirect_home_on_reload else ""
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>⚽</text></svg>">
<title>{_html.escape(title)}</title>
{MODERN_CSS}
</head>
<body>
{body_html}
{reload_script}
</body>
</html>"""
