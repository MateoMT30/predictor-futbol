"""
connectors/football_couk_connector.py
=====================================

Enriquecimiento de córners, tiros al arco y tarjetas para LIGAS DE CLUBES,
usando los CSV históricos gratuitos de **football-data.co.uk** (distinto de
football-data.ORG, que es la API de resultados/goles que usamos aparte).

Por qué esta fuente: football-data.co.uk publica, gratis y sin API key, un
CSV por temporada y liga con estadísticas que la API .org no trae —
córners (HC/AC), tiros al arco (HST/AST) y tarjetas (HY/AY amarillas,
HR/AR rojas). Es la única fuente gratuita encontrada para estos datos en
ligas de clubes (todas las APIs que los tienen son de pago). No aplica al
Mundial (para eso están los PDFs oficiales de FIFA, ver
fifa_reports_connector.py); esta cubre las grandes ligas europeas.

Cómo cruza los datos con el histórico de football-data.org: **por fecha +
marcador exacto**, NO por nombre de equipo. Las dos fuentes escriben los
nombres distinto ("Man City" vs "Manchester City FC"), pero ambas tienen la
fecha y los goles de cada partido; fecha + goles local + goles visitante
identifica el partido sin depender de que los nombres coincidan (y el orden
local/visitante queda alineado solo, porque el marcador local va con el
equipo local en las dos fuentes). Si dos partidos de la misma liga caen el
mismo día con el mismo marcador exacto (rarísimo), se omite ese cruce en vez
de arriesgar asignarlo mal — consistente con "nunca inventar".

Es liviano (un CSV pesa cientos de KB y se parsea en milisegundos, nada que
ver con los PDFs de FIFA), así que corre directo en el request sin riesgo de
timeout/OOM. Nunca lanza excepción hacia arriba: si la fuente falla, el
pipeline sigue con lo que tenía.
"""

import io
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Mapa de código de competición (football-data.ORG, el que usa la app) al
# código de archivo de football-data.co.uk. Solo las ligas que co.uk cubre;
# las demás (Mundial, Champions, Libertadores, Brasileirão...) no están aquí
# y simplemente no se enriquecen por esta vía.
COUK_LEAGUE_CODES = {
    "PL": "E0",    # Premier League
    "ELC": "E1",   # Championship
    "PD": "SP1",   # La Liga
    "SA": "I1",    # Serie A
    "BL1": "D1",   # Bundesliga
    "FL1": "F1",   # Ligue 1
    "DED": "N1",   # Eredivisie
    "PPL": "P1",   # Primeira Liga
}

# Cache en memoria del DataFrame ya parseado por (código_couk, temporada).
_csv_cache: dict = {}   # {(code, season): (timestamp, dataframe_or_None)}
_CACHE_TTL = 3600       # 1h: los CSV se actualizan a lo sumo un par de veces al día


def _season_codes(today: Optional[datetime] = None) -> list:
    """Códigos de temporada de co.uk (ej. '2526' para 2025-26) a descargar:
    la temporada según la fecha actual y la anterior, para cubrir el historial
    de ~365 días que usa el modelo aunque cruce el cambio de temporada."""
    if today is None:
        today = datetime.now(timezone.utc)
    y = today.year
    # La temporada europea arranca a mitad de año; de julio en adelante ya
    # cuenta como la temporada "año/año+1".
    if today.month >= 7:
        cur = (y, y + 1)
    else:
        cur = (y - 1, y)
    prev = (cur[0] - 1, cur[1] - 1)
    return [f"{a % 100:02d}{b % 100:02d}" for a, b in (cur, prev)]


def _fetch_season_csv(code: str, season: str) -> Optional[pd.DataFrame]:
    key = (code, season)
    now = time.time()
    if key in _csv_cache:
        cached_at, df = _csv_cache[key]
        if now - cached_at < _CACHE_TTL:
            return df

    url = f"{BASE_URL}/{season}/{code}.csv"
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 404:
            _csv_cache[key] = (now, None)
            return None
        response.raise_for_status()
        # encoding tolerante: estos CSV a veces traen bytes latin-1.
        df = pd.read_csv(io.StringIO(response.content.decode("latin-1")), on_bad_lines="skip")
    except Exception:
        _csv_cache[key] = (now, None)
        return None

    _csv_cache[key] = (now, df)
    return df


def _load_stats(competition: str) -> Optional[pd.DataFrame]:
    """Concatena las temporadas disponibles para la liga en un solo DataFrame
    con columnas normalizadas: fecha (datetime), goles, córners, tiros al arco
    y tarjetas de local y visitante. None si la liga no aplica o no hay datos."""
    code = COUK_LEAGUE_CODES.get(competition)
    if not code:
        return None

    frames = []
    for season in _season_codes():
        df = _fetch_season_csv(code, season)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    if "Date" not in raw or "FTHG" not in raw:
        return None

    # Columnas esperadas (algunas pueden faltar en ligas/temporadas viejas).
    def col(name):
        return raw[name] if name in raw.columns else pd.Series([None] * len(raw))

    out = pd.DataFrame({
        # dayfirst: el formato de co.uk es DD/MM/YYYY (o DD/MM/YY).
        "fecha": pd.to_datetime(col("Date"), dayfirst=True, errors="coerce"),
        "goles_local": pd.to_numeric(col("FTHG"), errors="coerce"),
        "goles_visitante": pd.to_numeric(col("FTAG"), errors="coerce"),
        "corners_local": pd.to_numeric(col("HC"), errors="coerce"),
        "corners_visitante": pd.to_numeric(col("AC"), errors="coerce"),
        "tiros_arco_local": pd.to_numeric(col("HST"), errors="coerce"),
        "tiros_arco_visitante": pd.to_numeric(col("AST"), errors="coerce"),
        "tarjetas_amarillas_local": pd.to_numeric(col("HY"), errors="coerce"),
        "tarjetas_amarillas_visitante": pd.to_numeric(col("AY"), errors="coerce"),
        "tarjetas_rojas_local": pd.to_numeric(col("HR"), errors="coerce"),
        "tarjetas_rojas_visitante": pd.to_numeric(col("AR"), errors="coerce"),
    })
    return out.dropna(subset=["fecha", "goles_local", "goles_visitante"])


_ENRICH_COLUMNS = [
    "corners_local", "corners_visitante",
    "tiros_arco_local", "tiros_arco_visitante",
    "tarjetas_amarillas_local", "tarjetas_amarillas_visitante",
    "tarjetas_rojas_local", "tarjetas_rojas_visitante",
]


def enrich_with_couk_stats(matches_df, competition: str):
    """Rellena córners, tiros al arco y tarjetas en `matches_df` (sobre una
    copia) para las ligas de clubes que cubre football-data.co.uk, cruzando
    por FECHA + MARCADOR EXACTO. Devuelve el df (posiblemente sin cambios si la
    liga no aplica o la fuente falla). Nunca lanza excepción."""
    try:
        stats = _load_stats(competition)
    except Exception:
        return matches_df
    if stats is None or stats.empty:
        return matches_df

    df = matches_df.copy()
    if "fecha" not in df.columns:
        return matches_df
    df_fecha = pd.to_datetime(df["fecha"], errors="coerce")

    for _, s in stats.iterrows():
        # Cruce por fecha (mismo día) + goles local + goles visitante. El
        # orden local/visitante queda alineado solo porque el marcador local
        # va con el equipo local en ambas fuentes.
        mask = (
            (df_fecha.dt.date == s["fecha"].date())
            & (pd.to_numeric(df["goles_local"], errors="coerce") == s["goles_local"])
            & (pd.to_numeric(df["goles_visitante"], errors="coerce") == s["goles_visitante"])
        )
        if mask.sum() != 1:
            # 0 = ese partido no está en el histórico; >1 = ambiguo (mismo día
            # y marcador exacto repetido). En ambos casos se omite, no se adivina.
            continue
        for c in _ENRICH_COLUMNS:
            val = s[c]
            if pd.notna(val):
                df.loc[mask, c] = val
    return df
