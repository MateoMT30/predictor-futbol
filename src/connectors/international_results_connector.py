"""
connectors/international_results_connector.py
==============================================

Historial de partidos de SELECCIONES desde el dataset público
github.com/martj42/international_results (CSV, dominio público/CC0,
mantenido por la comunidad y actualizado tras cada fecha FIFA — verificado:
incluye hasta los partidos del Mundial en curso).

Por qué existe: football-data.org (plan gratis) solo trae los partidos del
TORNEO en sí — ni clasificatorias ni amistosos. Resultado real observado:
España llegaba al pronóstico con 3 partidos de muestra ("tómalo con
pinzas"). Este dataset le da ~40 partidos de los últimos 3 años a una
selección europea típica. Solo trae goles (sin córners/tiros/tarjetas),
que es exactamente lo que consumen Dixon-Coles y Elo.

Cruce de nombres entre fuentes: ambas escriben en inglés y casi igual
("Spain", "Austria", "South Korea"). Para los casos que difieren en orden
o conectores ("DR Congo" vs "Congo DR", "Bosnia and Herzegovina" vs
"Bosnia-Herzegovina") se normaliza: minúsculas, sin acentos, sin
"and"/"the", tokens ordenados alfabéticamente. Lo que aún así no cruce
simplemente queda con su nombre original (no ayuda a ese equipo, pero
tampoco daña nada).
"""

import io
import time
import unicodedata
from typing import Optional

import pandas as pd
import requests

DATASET_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Cache en memoria del CSV parseado (pesa ~4 MB, se descarga en ~1s). Los
# resultados de partidos jugados no cambian; 12h de TTL es de sobra.
_cache = {"at": 0.0, "df": None}
_CACHE_TTL = 12 * 3600


def normalize_team_name(name: str) -> str:
    """Clave de cruce tolerante entre fuentes: 'DR Congo' y 'Congo DR'
    producen la misma clave; 'Bosnia and Herzegovina' y
    'Bosnia-Herzegovina' también."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    tokens = [t for t in s.lower().replace("-", " ").split() if t not in ("and", "the")]
    return " ".join(sorted(tokens))


def _download() -> Optional[pd.DataFrame]:
    now = time.time()
    if _cache["df"] is not None and now - _cache["at"] < _CACHE_TTL:
        return _cache["df"]
    try:
        resp = requests.get(DATASET_URL, timeout=20)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text))
    except Exception:
        return _cache["df"]  # si falla la red, se sirve lo último que hubo
    _cache["at"] = now
    _cache["df"] = raw
    return raw


def fetch_international_results(desde) -> Optional[pd.DataFrame]:
    """Partidos de selecciones ya jugados desde la fecha `desde` (datetime o
    string YYYY-MM-DD), en el esquema estándar del proyecto (solo goles; las
    columnas de córners/tiros/tarjetas no existen y quedarán NaN al
    concatenar — el reporte ya sabe decir "sin datos" en esos mercados).
    None si el dataset no está disponible."""
    raw = _download()
    if raw is None:
        return None
    df = pd.DataFrame({
        "fecha": pd.to_datetime(raw["date"], errors="coerce"),
        "liga": raw["tournament"],
        "equipo_local": raw["home_team"],
        "equipo_visitante": raw["away_team"],
        "goles_local": pd.to_numeric(raw["home_score"], errors="coerce"),
        "goles_visitante": pd.to_numeric(raw["away_score"], errors="coerce"),
    })
    df = df.dropna(subset=["fecha", "goles_local", "goles_visitante"])
    return df[df["fecha"] >= pd.Timestamp(desde)].reset_index(drop=True)


def align_team_names(df: pd.DataFrame, reference_names) -> pd.DataFrame:
    """Renombra los equipos del dataset a la convención de football-data.org
    cuando la clave normalizada coincide con un nombre ya visto en la otra
    fuente — así el modelo ve UN solo 'Congo DR' y no dos equipos distintos."""
    key_to_org = {normalize_team_name(n): n for n in reference_names}

    def rename(name):
        return key_to_org.get(normalize_team_name(name), name)

    out = df.copy()
    out["equipo_local"] = out["equipo_local"].map(rename)
    out["equipo_visitante"] = out["equipo_visitante"].map(rename)
    return out
