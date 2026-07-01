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

import io
import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

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
    if url in _pdf_cache:
        return _pdf_cache[url]
    if PdfReader is None:
        return None

    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        reader = PdfReader(io.BytesIO(response.content))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
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

    def _corners_for_team(team_section_title: str) -> Optional[int]:
        # El límite de fin de sección NO puede ser "Free Kicks": ese texto
        # ya aparece antes, dentro de "Total Free Kicks", y cortaría la
        # búsqueda antes de llegar a "Total Corners". "Total Throw Ins"
        # sí aparece de forma fiable después de "Total Corners" en el
        # layout de este reporte.
        section = re.search(rf'Set Plays {re.escape(team_section_title)}(.*?)Total Throw Ins', text, re.DOTALL)
        if not section:
            return None
        m = re.search(r'(\d+)\s*\n?\s*Total Corners', section.group(1))
        return int(m.group(1)) if m else None

    result = {
        "fecha": fecha,
        "equipo_local": home_team,
        "equipo_visitante": away_team,
        "tiros_local": int(shots.group(1)) if shots else None,
        "tiros_arco_local": int(shots.group(2)) if shots else None,
        "tiros_visitante": int(shots.group(3)) if shots else None,
        "tiros_arco_visitante": int(shots.group(4)) if shots else None,
        "corners_local": _corners_for_team(home_team),
        "corners_visitante": _corners_for_team(away_team),
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

    matches = []
    for url in _links_for_code(code, links):
        stats = _parse_pdf(url)
        if stats:
            matches.append(stats)
    return matches
