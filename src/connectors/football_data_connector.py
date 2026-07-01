"""
connectors/football_data_connector.py
=======================================

Conector para football-data.org (plan gratuito). Trae tanto el histórico
de resultados (para ajustar los modelos) como los próximos partidos
programados (para que la app muestre una lista clickeable en vez de pedir
que el usuario escriba los nombres de los equipos a mano).

Por qué esto corre del lado del servidor (Flask) y no en el navegador:
football-data.org no permite llamadas directas desde JavaScript de
navegador (CORS) y, aunque lo permitiera, exponer la API key en el código
del cliente sería un problema de seguridad. Al vivir en el backend Python,
ninguno de esos dos problemas existe — es una llamada HTTP normal de
servidor a servidor.

--- Cache para minimizar consumo de la cuota gratuita ---
El plan gratuito de football-data.org limita a ~10 peticiones/minuto. Sin
cache, cada persona que abre la página dispararía una llamada nueva. Este
conector guarda en memoria (por proceso) la última respuesta de cada
competición durante `cache_ttl_seconds` (por defecto 1 hora) y la reutiliza
mientras no haya vencido — así 100 visitas en una hora cuestan 1 sola
llamada real a la API, no 100.

Limitación de esta cache: vive en memoria del proceso, así que se reinicia
si el servidor se reinicia o "despierta" tras dormir (plan free de Render).
Eso es aceptable: en el peor caso, la primera visita tras un reinicio paga
el costo de una llamada real, y las siguientes vuelven a estar cubiertas.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from .base import DataSourceConnector

BASE_URL = "https://api.football-data.org/v4"

# Competiciones cubiertas por el plan gratuito (ver football-data.org/documentation).
COMPETITIONS = {
    "WC": "Copa Mundial de la FIFA",
    "CL": "UEFA Champions League",
    "PL": "Premier League (Inglaterra)",
    "PD": "La Liga (España)",
    "SA": "Serie A (Italia)",
    "BL1": "Bundesliga (Alemania)",
    "FL1": "Ligue 1 (Francia)",
    "DED": "Eredivisie (Países Bajos)",
    "PPL": "Primeira Liga (Portugal)",
    "EC": "Eurocopa",
}

_cache: dict = {}  # {cache_key: (timestamp, dataframe)}


def _get_cached_or_fetch(cache_key: str, ttl_seconds: int, fetch_fn):
    now = time.time()
    if cache_key in _cache:
        cached_at, data = _cache[cache_key]
        if now - cached_at < ttl_seconds:
            return data
    data = fetch_fn()
    _cache[cache_key] = (now, data)
    return data


class FootballDataConnector(DataSourceConnector):
    def __init__(self, api_key: Optional[str] = None, cache_ttl_seconds: int = 3600):
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
        self.cache_ttl_seconds = cache_ttl_seconds
        if not self.api_key:
            raise ValueError(
                "Falta FOOTBALL_DATA_API_KEY. Configúrala como variable de entorno "
                "(ver README > Desplegar en Render, o exporta la variable en tu shell local)."
            )

    def _get(self, path: str, params: dict) -> dict:
        response = requests.get(
            f"{BASE_URL}{path}", params=params,
            headers={"X-Auth-Token": self.api_key}, timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def fetch_matches(
        self,
        liga: Optional[str] = None,
        desde: Optional[str] = None,
        hasta: Optional[str] = None,
    ) -> pd.DataFrame:
        if not liga:
            raise ValueError("FootballDataConnector requiere el código de competición en 'liga' (ej. 'PL', 'WC').")

        # Red de seguridad: si nadie especifica un rango, se limita a los
        # últimos 365 días por defecto (nunca "todo el historial"). Sin este
        # límite, una competición con muchos años acumulados (ej. el
        # Mundial, con clasificatorias de décadas) mete cientos de equipos
        # distintos en el ajuste de Dixon-Coles, disparando el número de
        # parámetros a optimizar y agotando la memoria en hosting gratuito
        # (esto causó un OOM real en producción con el plan free de Render).
        if not desde and not hasta:
            now = datetime.now(timezone.utc)
            hasta = now.strftime("%Y-%m-%d")
            desde = (now - timedelta(days=365)).strftime("%Y-%m-%d")

        def fetch():
            params = {"status": "FINISHED"}
            if desde:
                params["dateFrom"] = desde
            if hasta:
                params["dateTo"] = hasta
            data = self._get(f"/competitions/{liga}/matches", params)
            rows = []
            for m in data.get("matches", []):
                score = m.get("score", {}).get("fullTime", {})
                if score.get("home") is None:
                    continue
                rows.append({
                    "fecha": m["utcDate"][:10],
                    "liga": data.get("competition", {}).get("name", liga),
                    "equipo_local": m["homeTeam"]["name"],
                    "equipo_visitante": m["awayTeam"]["name"],
                    "goles_local": score["home"],
                    "goles_visitante": score["away"],
                    # El plan gratuito no incluye córners/tiros/tarjetas — se
                    # dejan explícitamente en None, nunca se inventan.
                    "corners_local": None, "corners_visitante": None,
                    "tiros_arco_local": None, "tiros_arco_visitante": None,
                    "tarjetas_amarillas_local": None, "tarjetas_amarillas_visitante": None,
                    "tarjetas_rojas_local": None, "tarjetas_rojas_visitante": None,
                })
            return pd.DataFrame(rows)

        cache_key = f"matches:{liga}:{desde}:{hasta}"
        return _get_cached_or_fetch(cache_key, self.cache_ttl_seconds, fetch)

    def fetch_upcoming(self, liga: Optional[str] = None, dias: int = 14) -> pd.DataFrame:
        if not liga:
            raise ValueError("FootballDataConnector requiere el código de competición (ej. 'PL', 'WC').")

        def fetch():
            data = self._get(f"/competitions/{liga}/matches", {"status": "SCHEDULED"})
            rows = []
            for m in data.get("matches", [])[: 50]:  # tope defensivo, no por cuota sino por tamaño de respuesta
                rows.append({
                    "fecha_hora": pd.to_datetime(m["utcDate"]),
                    "liga": data.get("competition", {}).get("name", liga),
                    "equipo_local": m["homeTeam"]["name"],
                    "equipo_visitante": m["awayTeam"]["name"],
                })
            return pd.DataFrame(rows).sort_values("fecha_hora").reset_index(drop=True)

        cache_key = f"upcoming:{liga}"
        return _get_cached_or_fetch(cache_key, self.cache_ttl_seconds, fetch)
