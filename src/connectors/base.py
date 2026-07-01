"""
connectors/base.py
===================

Interfaz abstracta para cualquier fuente de datos de fútbol.

Por qué existe esto: el proyecto empieza alimentado por CSV/JSON locales,
pero la idea es poder "enchufar" mañana una API real (API-Football,
Understat, Sportmonks, etc.) sin tocar el resto del sistema (ratings,
modelos, simulación). Para eso, todo el pipeline habla contra esta interfaz,
nunca contra un conector concreto.

Cualquier conector nuevo solo necesita heredar de DataSourceConnector e
implementar fetch_matches() y fetch_odds() devolviendo los DataFrames/dicts
con el esquema esperado (documentado en data_loader.py).
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class DataSourceConnector(ABC):
    """Contrato que debe cumplir cualquier fuente de datos de partidos."""

    @abstractmethod
    def fetch_matches(
        self,
        liga: Optional[str] = None,
        desde: Optional[str] = None,
        hasta: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Debe devolver un DataFrame con (al menos) estas columnas:
        fecha, liga, equipo_local, equipo_visitante,
        goles_local, goles_visitante,
        corners_local, corners_visitante,
        tiros_arco_local, tiros_arco_visitante,
        tarjetas_amarillas_local, tarjetas_amarillas_visitante,
        tarjetas_rojas_local, tarjetas_rojas_visitante

        Los filtros (liga, rango de fechas) son opcionales: una API real
        normalmente los soporta de forma nativa; un CSV los aplica en memoria.
        """
        raise NotImplementedError

    def fetch_odds(self, equipo_local: str, equipo_visitante: str) -> Optional[dict]:
        """
        Opcional: un conector puede devolver cuotas si las tiene (mismo
        esquema que examples/cuotas_ejemplo.json). No es obligatorio
        implementarlo — por defecto no hay cuotas disponibles.
        """
        return None

    def fetch_upcoming(self, liga: Optional[str] = None, dias: int = 14) -> pd.DataFrame:
        """
        Opcional: partidos programados/futuros (no jugados aún), con fecha
        y hora, para que la app pueda mostrar "qué partidos hay" en vez de
        pedirle al usuario que escriba los nombres de los equipos a mano.

        Debe devolver un DataFrame con columnas: fecha_hora (datetime),
        liga, equipo_local, equipo_visitante.

        Por defecto no soportado (DataFrame vacío) — un CSV de resultados
        históricos, por ejemplo, no tiene forma de saber qué se jugará
        mañana; eso solo lo puede dar una fuente en vivo como una API.
        """
        return pd.DataFrame(columns=["fecha_hora", "liga", "equipo_local", "equipo_visitante"])
