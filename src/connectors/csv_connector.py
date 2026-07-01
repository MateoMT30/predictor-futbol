"""
connectors/csv_connector.py
===========================

Implementación concreta de DataSourceConnector para archivos locales
CSV o JSON. Es el conector "de arranque": no depende de internet ni de
llaves de API, así que el sistema es utilizable desde el día uno con
datos históricos que el usuario ya tenga.

Cuando se consiga acceso a una API real, basta con escribir un nuevo
archivo (p. ej. api_football_connector.py) que implemente la misma
interfaz y pasárselo a data_loader en vez de este.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import DataSourceConnector


class CSVConnector(DataSourceConnector):
    def __init__(self, matches_path: str, odds_path: Optional[str] = None):
        self.matches_path = Path(matches_path)
        self.odds_path = Path(odds_path) if odds_path else None

    def fetch_matches(
        self,
        liga: Optional[str] = None,
        desde: Optional[str] = None,
        hasta: Optional[str] = None,
    ) -> pd.DataFrame:
        if not self.matches_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de histórico: {self.matches_path}")

        if self.matches_path.suffix.lower() == ".json":
            df = pd.read_json(self.matches_path)
        else:
            df = pd.read_csv(self.matches_path)

        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

        # Los filtros se aplican en memoria porque un CSV no tiene forma de
        # "consultar" — a diferencia de una API real, donde estos parámetros
        # normalmente van en el query string de la petición HTTP.
        if liga:
            df = df[df["liga"].str.casefold() == liga.casefold()]
        if desde:
            df = df[df["fecha"] >= pd.to_datetime(desde)]
        if hasta:
            df = df[df["fecha"] <= pd.to_datetime(hasta)]

        return df.reset_index(drop=True)

    def fetch_odds(self, equipo_local: str, equipo_visitante: str) -> Optional[dict]:
        if not self.odds_path or not self.odds_path.exists():
            return None
        with open(self.odds_path, "r", encoding="utf-8") as f:
            return json.load(f)
