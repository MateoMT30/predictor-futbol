"""
connectors/fifa_reports_connector.py
======================================

Conector para los "Post Match Summary Report" (PMSR) que la propia FIFA
publica públicamente en formato PDF en su FIFA Training Centre, para cada
partido del Mundial 2026 ya jugado:

    https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub.php

Por qué esta fuente y no otra API: es la ÚNICA fuente gratuita que
encontramos con córners y tiros al arco reales para el Mundial (después de
investigar football-data.org, API-Football, football-data.co.uk, FBref,
WC2026API, Statorium y TheStatsAPI — ver discusión en el README). Al ser
la propia FIFA publicando sus reportes oficiales para consulta pública
(pensados para prensa/cuerpos técnicos, sin API key ni paywall), no aplica
el mismo problema de "scraping no autorizado" que sí aplica a apps de
terceros como 365Scores: aquí no hay Términos de Servicio que prohíban
descargar un documento que la propia organización publica para ese fin.

Limitación importante y honesta: estos reportes NO incluyen tarjetas
amarillas/rojas ni faltas — son reportes tácticos/analíticos (posesión,
xG, líneas de pase, distancia recorrida), no reportes disciplinarios. Por
eso este conector solo enriquece córners y tiros al arco; tarjetas sigue
sin dato automático (ver models/tarjetas.py y el aviso "sin datos
suficientes" en el reporte cuando no hay fuente).

--- Cómo funciona ---
1. Se descarga la página "hub" (HTML) y se extraen con regex los enlaces
   a PDF — cada nombre de archivo trae los códigos de 3 letras de FIFA de
   ambos equipos (ej. "PMSR-M01 MEX V RSA.pdf").
2. Para no descargar los ~104 PDFs del torneo (pesan varios MB cada uno),
   solo se bajan los que involucran a los dos equipos del partido que se
   está prediciendo — un puñado de partidos por equipo, no el torneo
   completo.
3. Cada PDF se parsea con pypdf (extracción de texto) y se leen los
   números con expresiones regulares sobre las secciones "Key Statistics"
   (tiros/tiros al arco) y "Set Plays" (córners totales) del reporte.
4. Todo se cachea en memoria (igual que football_data_connector.py):
   la lista de links por 1 hora, cada PDF ya parseado indefinidamente
   (un reporte de un partido ya jugado no cambia).
"""

import contextvars
import json
import multiprocessing
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from . import _pdf_extract_worker

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - se valida en requirements.txt
    PdfReader = None

HUB_URL = "https://www.fifatrainingcentre.com/en/fifa-world-cup-2026/match-report-hub.php"
BASE_URL = "https://www.fifatrainingcentre.com"

# Códigos FIFA de 3 letras para las selecciones más comunes. No pretende
# ser exhaustivo — un equipo no listado simplemente no se enriquece con
# esta fuente (el pipeline sigue funcionando solo sin córners/tiros para
# ese equipo, nunca falla por esto).
FIFA_CODES = {
    "Mexico": "MEX", "South Africa": "RSA", "South Korea": "KOR", "Czech Republic": "CZE",
    "Canada": "CAN", "Bosnia and Herzegovina": "BIH", "Qatar": "QAT", "Switzerland": "SUI",
    "Belgium": "BEL", "Senegal": "SEN", "United States": "USA", "Spain": "ESP",
    "Austria": "AUT", "Portugal": "POR", "Croatia": "CRO", "Algeria": "ALG",
    "Australia": "AUS", "Egypt": "EGY", "Argentina": "ARG", "Cape Verde Islands": "CPV",
    "Colombia": "COL", "Ghana": "GHA", "Brazil": "BRA", "Germany": "GER",
    "France": "FRA", "England": "ENG", "Italy": "ITA", "Netherlands": "NED",
    "Uruguay": "URU", "Japan": "JPN", "Morocco": "MAR", "Ecuador": "ECU",
    "Peru": "PER", "Chile": "CHI", "Paraguay": "PAR", "Wales": "WAL",
    "Scotland": "SCO", "Poland": "POL", "Sweden": "SWE", "Norway": "NOR",
    "Denmark": "DEN", "Serbia": "SRB", "Turkey": "TUR", "Greece": "GRE",
    "Ukraine": "UKR", "Saudi Arabia": "KSA", "Iran": "IRN", "Nigeria": "NGA",
    "Cameroon": "CMR", "Tunisia": "TUN", "Ivory Coast": "CIV", "New Zealand": "NZL",
    "Costa Rica": "CRC", "Panama": "PAN", "Jamaica": "JAM", "Honduras": "HON",
    "Iceland": "ISL", "Uzbekistan": "UZB", "Jordan": "JOR", "Iraq": "IRQ",
    "Curaçao": "CUW", "Haiti": "HAI", "DR Congo": "COD",
}

_link_cache: dict = {}  # {"links": (timestamp, [urls])}
_pdf_cache: dict = {}   # {url: parsed_stats_dict}

# --- Cache en disco (JSON) ---------------------------------------------------
# Parsear los PDFs de FIFA cuesta ~8-15s cada uno y es demasiado lento/pesado
# para hacerlo DENTRO de un request en Render free (se veían timeouts y OOM).
# Solución: precomputar todos los reportes offline con
# `scripts/refresh_fifa_cache.py` y guardarlos en este JSON, que se sube al
# repo. En producción la web solo LEE el JSON (instantáneo); no parsea PDFs.
# Como un reporte de un partido ya jugado no cambia, basta re-correr el script
# cuando se jueguen partidos nuevos.
_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "fifa_cache.json"

# Por defecto NO se parsean PDFs en vivo (solo se lee el JSON). El script de
# refresco pone FIFA_LIVE_PARSE=1 para sí bajar y parsear los PDFs y
# regenerar el cache. Así la web nunca se arriesga a un timeout/OOM.
_LIVE_PARSE = os.environ.get("FIFA_LIVE_PARSE") == "1"

_disk_cache_loaded = False


def _load_disk_cache() -> None:
    """Carga el JSON precomputado en `_pdf_cache` (una sola vez). Si el
    archivo no existe o está corrupto, se sigue sin cache — el pipeline nunca
    falla por esto."""
    global _disk_cache_loaded
    if _disk_cache_loaded:
        return
    _disk_cache_loaded = True
    try:
        with open(_CACHE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for url, stats in data.items():
                _pdf_cache.setdefault(url, stats)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def save_disk_cache() -> int:
    """Vuelca `_pdf_cache` al JSON en disco. La usa el script de refresco.
    Devuelve cuántas entradas con datos (no None) quedaron guardadas."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {url: stats for url, stats in _pdf_cache.items()}
    with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return sum(1 for v in serializable.values() if v)

# Límites de memoria — Render free tiene solo 512 MB y numpy/pandas ya
# cargados ocupan una parte. Parsear los PMSR de FIFA con pypdf puede
# disparar la memoria de forma impredecible (content streams pesados, Form
# XObjects con gráficos, o CMaps patológicas de una fuente — se vieron OOM
# reales en `_parse_content_stream` y en `_cmap._parse_encoding`). Un OOM
# manda SIGKILL: NO se puede atrapar con try/except (no lanza excepción,
# mata el proceso). Por eso el parseo del PDF corre en un **subproceso
# aislado con tope de memoria** (ver `_pdf_extract_worker.py`): si revienta,
# muere solo el subproceso y este worker web sobrevive.
_MAX_PDF_BYTES = 15 * 1024 * 1024          # no descargar PDFs gigantes
_PDF_PARSE_TIMEOUT = 8                      # segundos máx. de parseo por PDF
_TEAM_TIME_BUDGET = 18                      # presupuesto de PDFs por equipo (s)
_MAX_PDFS_PER_TEAM = 4                      # tope de PDFs a parsear por equipo
_REQUEST_TIME_BUDGET = 15                   # presupuesto TOTAL de parseo por request (s)

# Presupuesto global por request, compartido entre las varias llamadas a
# get_match_stats_for_team que hace un mismo request (enrich + summaries de
# los 2 equipos). Sin esto, cada equipo tenía su propio presupuesto y el
# total (2 equipos) se pasaba del timeout de gunicorn. Es un ContextVar (no
# una global normal) porque con el worker gthread hay varios hilos atendiendo
# requests en paralelo, y cada hilo tiene su propio contexto: así el deadline
# de un request no pisa el de otro.
_request_deadline: "contextvars.ContextVar[Optional[float]]" = contextvars.ContextVar(
    "_fifa_request_deadline", default=None
)


def start_request_budget(seconds: float = _REQUEST_TIME_BUDGET) -> None:
    """Arranca (para el hilo/request actual) el presupuesto total de tiempo
    de parseo de PDFs de FIFA. Se llama una vez al inicio del bloque de
    enriquecimiento en app.py."""
    _request_deadline.set(time.monotonic() + seconds)


def clear_request_budget() -> None:
    _request_deadline.set(None)

# Se usa 'fork' (Linux/Render): NO re-importa el __main__ del padre (a
# diferencia de 'spawn', que bajo gunicorn re-ejecutaría el arranque), y es
# barato. Si un PDF revienta la RAM, el kernel mata al hijo (que se marca
# como blanco del OOM killer) y el worker web sobrevive. En Windows (solo
# dev local, con RAM de sobra) 'fork' no existe: ahí se parsea inline.
_USE_SUBPROCESS = "fork" in multiprocessing.get_all_start_methods()
_mp_ctx = multiprocessing.get_context("fork") if _USE_SUBPROCESS else None


def _extract_pdf_text_isolated(content: bytes) -> Optional[str]:
    """Extrae el texto del PDF sin arriesgar al worker web. En Linux/Render
    corre el parseo en un subproceso 'fork' aislado; si ese subproceso muere
    por OOM (o excede el timeout, o falla), devuelve None y el padre sigue
    vivo. En plataformas sin fork (Windows dev) parsea inline."""
    if not _USE_SUBPROCESS:
        try:
            return _pdf_extract_worker.extract_light_text(content)
        except Exception:
            return None

    parent_conn, child_conn = _mp_ctx.Pipe(duplex=False)
    proc = _mp_ctx.Process(
        target=_pdf_extract_worker._child_main,
        args=(content, child_conn),
    )
    proc.start()
    child_conn.close()  # el extremo de escritura queda solo en el hijo

    result: Optional[str] = None
    try:
        if parent_conn.poll(_PDF_PARSE_TIMEOUT):
            result = parent_conn.recv()
    except EOFError:
        # El hijo murió (OOM/SIGKILL) sin enviar nada.
        result = None
    finally:
        parent_conn.close()
        # SIGKILL (kill), no SIGTERM: un hijo atascado dentro de código C de
        # pypdf (allocando) puede ignorar SIGTERM y dejar al padre colgado en
        # join() hasta que gunicorn mate al worker por timeout. kill() lo
        # remata sin margen.
        if proc.is_alive():
            proc.kill()
        proc.join()

    if not isinstance(result, str):
        return None
    return result


def _fetch_report_links() -> list:
    now = time.time()
    if "links" in _link_cache:
        cached_at, links = _link_cache["links"]
        if now - cached_at < 3600:
            return links
    response = requests.get(HUB_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    hrefs = re.findall(r'href="([^"]+\.pdf)"', response.text, re.IGNORECASE)
    links = [BASE_URL + h if not h.startswith("http") else h for h in hrefs]
    _link_cache["links"] = (now, links)
    return links


def _links_for_code(code: str, links: list) -> list:
    """Filtra los links de PDF cuyo nombre de archivo menciona el código
    de 3 letras dado, tolerando que el separador sea espacio o guion."""
    pattern = re.compile(rf'(^|[ \-]){re.escape(code)}([ \-]|\.pdf)', re.IGNORECASE)
    return [l for l in links if pattern.search(l)]


def _parse_pdf(url: str) -> Optional[dict]:
    _load_disk_cache()
    if url in _pdf_cache:
        return _pdf_cache[url]
    if PdfReader is None:
        return None
    # En producción (web) no se parsea en vivo: solo se sirve lo que quedó
    # precomputado en el JSON. Parsear un PDF acá costaría ~10s y arriesga el
    # timeout/OOM que tanto costó eliminar. Solo el script de refresco
    # (FIFA_LIVE_PARSE=1) llega a descargar y parsear.
    if not _LIVE_PARSE:
        return None

    try:
        response = requests.get(
            url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, stream=True
        )
        response.raise_for_status()
        # Cortar la descarga si el PDF excede el límite: evita traer a RAM un
        # archivo enorme antes siquiera de parsearlo.
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            total += len(chunk)
            if total > _MAX_PDF_BYTES:
                _pdf_cache[url] = None
                return None
            chunks.append(chunk)
        content = b"".join(chunks)
    except Exception:
        _pdf_cache[url] = None
        return None

    # El parseo real (que es lo que puede reventar la memoria) se hace en un
    # subproceso aislado con tope de RAM, para que un OOM no mate al worker.
    full_text = _extract_pdf_text_isolated(content)
    if not full_text:
        _pdf_cache[url] = None
        return None

    stats = _extract_stats(full_text)
    _pdf_cache[url] = stats
    return stats


def _extract_stats(text: str) -> Optional[dict]:
    """
    Extrae equipos, tiros/tiros al arco (página "Key Statistics") y
    córners totales (páginas "Set Plays", una por equipo) del texto plano
    del PDF. Se usa regex tolerante a espacios/saltos de línea porque la
    extracción de texto de un PDF no preserva el layout visual exacto.

    Devuelve None si el formato no calza con lo esperado (reporte con
    layout distinto, versión futura del formato, etc.) — nunca lanza
    excepción hacia el llamador, para que un PDF con formato inesperado
    no tumbe el resto del pipeline.
    """
    # Página 1: "Mexico2 - 0\nSouth Africa" -> equipo local, marcador, equipo visitante
    header = re.search(r'^([A-Za-z .\-]+?)(\d+)\s*-\s*(\d+)\s*\n([A-Za-z .\-]+)', text)
    if not header:
        return None
    home_team = header.group(1).strip()
    away_team = header.group(4).strip().split("\n")[0].strip()

    # Fecha del partido (ej. "11 June 2026"), necesaria para cruzar este
    # reporte contra la fila correspondiente en el histórico de
    # football-data.org. No se cruza por nombre de equipo porque cada
    # fuente puede deletrearlo distinto (ej. "South Korea" vs "Korea
    # Republic") — la fecha + saber qué equipo jugó ese día es un cruce
    # más confiable que comparar strings de nombres.
    fecha = None
    date_match = re.search(r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})', text)
    if date_match:
        try:
            fecha = datetime.strptime(date_match.group(1), "%d %B %Y").strftime("%Y-%m-%d")
        except ValueError:
            fecha = None

    shots = re.search(
        r'(\d+)\s*\((\d+)\)\s*Attempts at Goal \(On Target\)\s*(\d+)\s*\((\d+)\)',
        text,
    )

    # --- Página "Key Statistics": posesión, xG, pases, % de acierto ---
    # Estos números son contexto informativo (se muestran en el reporte,
    # ver report_html.py), no alimentan ningún modelo estadístico — igual
    # criterio que la tabla de posiciones y goleadores: dato real de FIFA,
    # mostrado tal cual, sin usarlo para inferir otras estadísticas.
    possession = re.search(r'Total\s*([\d.]+)%.*?([\d.]+)%\s*Total', text, re.DOTALL)
    xg = re.search(r'([\d.]+)\s*xG \(Expected Goals\)\s*([\d.]+)', text)
    passes = re.search(
        r'(\d+)\s*\((\d+)\)\s*Total Passes \(Complete\)\s*(\d+)\s*\((\d+)\)', text,
    )
    pass_pct = re.search(r'(\d+)\s*%\s*Pass Completion %\s*(\d+)\s*%', text)

    def _set_play_stat_for_team(team_section_title: str, label: str) -> Optional[int]:
        # El límite de fin de sección NO puede ser "Free Kicks": ese texto
        # ya aparece antes, dentro de "Total Free Kicks", y cortaría la
        # búsqueda antes de llegar a las estadísticas siguientes. "Total
        # Throw Ins" sí aparece de forma fiable al final de este bloque
        # resumen, antes de las tablas de detalle.
        section = re.search(rf'Set Plays {re.escape(team_section_title)}(.*?)Total Throw Ins', text, re.DOTALL)
        if not section:
            return None
        m = re.search(rf'(\d+)\s*\n?\s*{re.escape(label)}', section.group(1))
        return int(m.group(1)) if m else None

    result = {
        "fecha": fecha,
        "equipo_local": home_team,
        "equipo_visitante": away_team,
        "tiros_local": int(shots.group(1)) if shots else None,
        "tiros_arco_local": int(shots.group(2)) if shots else None,
        "tiros_visitante": int(shots.group(3)) if shots else None,
        "tiros_arco_visitante": int(shots.group(4)) if shots else None,
        "corners_local": _set_play_stat_for_team(home_team, "Total Corners"),
        "corners_visitante": _set_play_stat_for_team(away_team, "Total Corners"),
        "tiros_libres_local": _set_play_stat_for_team(home_team, "Total Free Kicks"),
        "tiros_libres_visitante": _set_play_stat_for_team(away_team, "Total Free Kicks"),
        "penales_local": _set_play_stat_for_team(home_team, "Total Penalties"),
        "penales_visitante": _set_play_stat_for_team(away_team, "Total Penalties"),
        "posesion_local": float(possession.group(1)) if possession else None,
        "posesion_visitante": float(possession.group(2)) if possession else None,
        "xg_local": float(xg.group(1)) if xg else None,
        "xg_visitante": float(xg.group(2)) if xg else None,
        "pases_local": int(passes.group(1)) if passes else None,
        "pases_completos_local": int(passes.group(2)) if passes else None,
        "pases_visitante": int(passes.group(3)) if passes else None,
        "pases_completos_visitante": int(passes.group(4)) if passes else None,
        "precision_pase_local": int(pass_pct.group(1)) if pass_pct else None,
        "precision_pase_visitante": int(pass_pct.group(2)) if pass_pct else None,
    }
    return result


def enrich_with_fifa_reports(matches_df, teams: set):
    """
    Rellena corners_local/visitante y tiros_arco_local/visitante en
    matches_df (in-place sobre una copia) para las filas que involucren a
    alguno de los equipos en `teams`, usando los reportes oficiales de
    FIFA. Se cruza por FECHA, no por nombre de equipo — dos fuentes
    distintas pueden deletrear el mismo país distinto (ej. "South Korea"
    vs "Korea Republic" en el reporte de FIFA), pero es prácticamente
    imposible que dos partidos del mismo equipo caigan el mismo día, así
    que la fecha es un cruce confiable sin depender de coincidencia exacta
    de texto.

    No modifica filas que no correspondan a los equipos pedidos, y nunca
    lanza excepción — si la fuente falla o un partido no se pudo parsear,
    esas filas simplemente se quedan con los valores que ya tenían (None
    si no había otra fuente), consistente con el resto del proyecto: nunca
    se inventa un número donde no hay dato.
    """
    df = matches_df.copy()
    for team in teams:
        try:
            reports = get_match_stats_for_team(team)
        except Exception:
            continue
        for report in reports:
            if not report.get("fecha"):
                continue
            # df["fecha"] es datetime64 (ya normalizado por data_loader.py);
            # el string "YYYY-MM-DD" del reporte se convierte para comparar
            # tipos compatibles en vez de comparar Timestamp contra str.
            report_fecha = pd.to_datetime(report["fecha"])
            mask = (
                (df["fecha"] == report_fecha)
                & ((df["equipo_local"] == team) | (df["equipo_visitante"] == team))
            )
            if not mask.any():
                continue
            # Se asume que el orden local/visitante coincide entre ambas
            # fuentes para la misma fecha (el equipo que jugó de local en
            # la vida real es el mismo en cualquier fuente que describa
            # ese partido) — no depende de que los nombres se escriban
            # igual, solo de que la fecha identifique un único partido.
            if report.get("corners_local") is not None:
                df.loc[mask, "corners_local"] = report["corners_local"]
            if report.get("corners_visitante") is not None:
                df.loc[mask, "corners_visitante"] = report["corners_visitante"]
            if report.get("tiros_arco_local") is not None:
                df.loc[mask, "tiros_arco_local"] = report["tiros_arco_local"]
            if report.get("tiros_arco_visitante") is not None:
                df.loc[mask, "tiros_arco_visitante"] = report["tiros_arco_visitante"]
    return df


def get_match_stats_for_team(team_name: str) -> list:
    """
    Devuelve una lista de dicts (uno por partido ya jugado de ese
    equipo) con córners/tiros/tiros al arco extraídos de los reportes
    oficiales de FIFA. Lista vacía si el equipo no está en FIFA_CODES,
    si la fuente no responde, o si no se pudo parsear ningún reporte —
    nunca lanza excepción (uso defensivo: esto es un enriquecimiento
    opcional, no debe poder romper el flujo principal de predicción).
    """
    code = FIFA_CODES.get(team_name)
    if not code:
        return []
    try:
        links = _fetch_report_links()
    except Exception:
        return []

    # Presupuesto de tiempo y de cantidad: en Render free (1 worker, timeout
    # de gunicorn) no se puede pasar minutos parseando PDFs dentro de un
    # request. Se parsean como mucho _MAX_PDFS_PER_TEAM y solo mientras quede
    # presupuesto; los que ya estén en caché no cuentan contra el reloj (son
    # instantáneos). Es best-effort: si no alcanza el tiempo, se devuelve lo
    # que se haya podido, consistente con "nunca inventar, mostrar lo que hay".
    deadline = time.monotonic() + _TEAM_TIME_BUDGET
    request_deadline = _request_deadline.get()
    if request_deadline is not None:
        deadline = min(deadline, request_deadline)
    matches = []
    parsed_count = 0
    for url in _links_for_code(code, links):
        cached = url in _pdf_cache
        if not cached:
            if parsed_count >= _MAX_PDFS_PER_TEAM or time.monotonic() >= deadline:
                break
            parsed_count += 1
        stats = _parse_pdf(url)
        if stats:
            # Copia (no se muta el dict cacheado, que puede reutilizarse para
            # el rival) con la marca de si ESTE equipo jugó de local en ese
            # partido — se deduce del orden del nombre de archivo (HOME-V-AWAY,
            # ver _own_is_local), no de comparar nombres, porque el reporte de
            # FIFA puede escribir el país distinto que football-data.
            matches.append({**stats, "_own_is_local": _own_is_local(url, code)})
    return matches


def _own_is_local(url: str, code: str) -> Optional[bool]:
    """Deduce si el equipo con este código FIFA jugó de local en el partido,
    a partir del nombre del archivo del reporte, que lista los códigos en
    orden LOCAL-V-VISITANTE (ej. 'PMSR-M04-USA-V-PAR' -> USA local;
    'PMSR-M59-TUR-v-USA' -> USA visitante). None si no se puede determinar."""
    name = url.rsplit("/", 1)[-1]
    parts = re.split(r'[\s\-][Vv][\s\-]', name, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = parts[0].upper(), parts[1].upper()
    code_u = code.upper()
    left_has = re.search(rf'(^|[\s\-]){re.escape(code_u)}([\s\-]|$)', left) is not None
    right_has = re.search(rf'(^|[\s\-]){re.escape(code_u)}([\s\-]|\.)', right) is not None
    if left_has and not right_has:
        return True
    if right_has and not left_has:
        return False
    return None


# Campos "propios" de cada equipo (no del rival) que sí tiene sentido
# promediar como contexto informativo. Deliberadamente separado de
# enrich_with_fifa_reports: xG/posesión/pases no alimentan ningún modelo
# (a diferencia de córners/tiros al arco, que sí forman parte del cálculo
# de simulation.py) — mostrarlos mezclados con "datos que sí calculan
# probabilidades" sería confuso. Este resumen es puramente para que el
# usuario tenga más contexto oficial en pantalla.
_OWN_FIELD_SUFFIXES = ["xg", "posesion", "pases", "precision_pase", "tiros_libres", "penales", "corners", "tiros_arco"]


def get_team_summary_stats(team_name: str) -> Optional[dict]:
    """
    Promedios oficiales de FIFA para un equipo a lo largo de sus partidos
    ya jugados en el torneo (xG, posesión, pases, precisión de pase,
    córners, tiros libres, penales). None si no hay reportes disponibles
    para ese equipo — nunca inventa un promedio con muestra vacía.
    """
    try:
        reports = get_match_stats_for_team(team_name)
    except Exception:
        return None
    if not reports:
        return None

    def _own_value(report: dict, suffix: str):
        # Un reporte guarda el dato en la columna "_local" o "_visitante"
        # según si el equipo jugó de local o visitante ESE partido. Se usa la
        # marca `_own_is_local` (deducida del nombre de archivo en
        # get_match_stats_for_team) porque comparar nombres no sirve: el
        # reporte de FIFA puede escribir el país distinto que football-data
        # (ej. "USA" vs "United States"). Fallback a comparar nombres si la
        # marca no se pudo determinar.
        own_local = report.get("_own_is_local")
        if own_local is None:
            own_local = report.get("equipo_local") == team_name
        suffix_side = "local" if own_local else "visitante"
        return report.get(f"{suffix}_{suffix_side}")

    averages = {"partidos_con_dato": len(reports)}
    for suffix in _OWN_FIELD_SUFFIXES:
        values = [v for r in reports if (v := _own_value(r, suffix)) is not None]
        averages[suffix] = round(sum(values) / len(values), 1) if values else None
    return averages
