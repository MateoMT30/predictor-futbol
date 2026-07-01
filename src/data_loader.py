"""
data_loader.py
==============

Capa de ingestión y limpieza de datos históricos de partidos.

Responsabilidades:
  1. Recibir datos crudos desde cualquier DataSourceConnector.
  2. Normalizar nombres de equipo (evita que "Man United" y "Manchester
     United" se traten como equipos distintos, lo cual rompería silenciosamente
     el sistema de ratings y los promedios por equipo).
  3. Validar que cada fila tenga los campos mínimos para ser usable, y
     descartar (con aviso) las que no los tengan, en vez de fallar toda
     la corrida por un solo registro corrupto.
  4. Ordenar cronológicamente, porque tanto el Elo como la ponderación
     por recencia son procesos secuenciales (dependen del orden de los
     partidos, no solo de su contenido).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "fecha", "liga", "equipo_local", "equipo_visitante",
    "goles_local", "goles_visitante",
]

OPTIONAL_STAT_COLUMNS = [
    "corners_local", "corners_visitante",
    "tiros_arco_local", "tiros_arco_visitante",
    "tarjetas_amarillas_local", "tarjetas_amarillas_visitante",
    "tarjetas_rojas_local", "tarjetas_rojas_visitante",
]


@dataclass
class CleaningReport:
    """Resumen de qué se hizo durante la limpieza, para que el usuario
    pueda auditar qué datos se descartaron y por qué."""
    filas_originales: int = 0
    filas_descartadas_incompletas: int = 0
    filas_finales: int = 0
    equipos_normalizados: Dict[str, str] = field(default_factory=dict)


def _build_alias_map(team_names: pd.Series) -> Dict[str, str]:
    """
    Construye un mapa alias -> nombre canónico.

    Estrategia simple pero robusta: se normaliza cada nombre (minúsculas,
    sin espacios extra, sin tildes) y se agrupan los equipos que comparten
    esa forma normalizada. El nombre "canónico" elegido es el más frecuente
    en los datos (asumimos que la grafía más común es la correcta).

    Esto resuelve casos como "Bogotá" vs "Bogota", o "  Colombia " vs
    "Colombia", sin necesitar un diccionario de alias mantenido a mano.
    """
    import unicodedata

    def normalize(name: str) -> str:
        name = str(name).strip()
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        return name.casefold()

    counts = team_names.value_counts()
    groups: Dict[str, list] = {}
    for name, count in counts.items():
        key = normalize(name)
        groups.setdefault(key, []).append((name, count))

    alias_map = {}
    for key, variants in groups.items():
        canonical = max(variants, key=lambda x: x[1])[0]
        for name, _ in variants:
            alias_map[name] = canonical
    return alias_map


def load_and_clean(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Punto de entrada principal de este módulo. Toma un DataFrame crudo
    (tal como lo entrega cualquier DataSourceConnector) y devuelve:
      - un DataFrame limpio, ordenado por fecha, con columnas de stats
        faltantes rellenadas con NaN (nunca inventamos números)
      - un CleaningReport para trazabilidad
    """
    report = CleaningReport(filas_originales=len(raw_df))
    df = raw_df.copy()

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Faltan columnas obligatorias en el histórico: {missing_required}. "
            f"Se requieren como mínimo: {REQUIRED_COLUMNS}"
        )

    for col in OPTIONAL_STAT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Un partido sin fecha, equipos o goles no sirve para nada del pipeline
    # (ni Elo, ni Poisson, ni ordenamiento cronológico) -> se descarta.
    core_cols = ["fecha", "equipo_local", "equipo_visitante", "goles_local", "goles_visitante"]
    before = len(df)
    df = df.dropna(subset=core_cols)
    report.filas_descartadas_incompletas = before - len(df)

    # Normalización de nombres de equipo: se construye el mapa de alias
    # sobre la unión de local + visitante para que sea consistente en
    # ambos roles.
    all_team_names = pd.concat([df["equipo_local"], df["equipo_visitante"]], ignore_index=True)
    alias_map = _build_alias_map(all_team_names)
    df["equipo_local"] = df["equipo_local"].map(alias_map)
    df["equipo_visitante"] = df["equipo_visitante"].map(alias_map)
    report.equipos_normalizados = {
        k: v for k, v in alias_map.items() if k != v
    }

    # Los goles deben ser enteros no negativos; cualquier otra cosa es un
    # dato corrupto (ej. un "-1" por un error de captura) y se descarta.
    for col in ["goles_local", "goles_visitante"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df[(df["goles_local"] >= 0) & (df["goles_visitante"] >= 0)]
    report.filas_descartadas_incompletas += before - len(df)

    df = df.sort_values("fecha").reset_index(drop=True)
    report.filas_finales = len(df)
    return df, report


def load_from_connector(connector, liga: Optional[str] = None,
                         desde: Optional[str] = None, hasta: Optional[str] = None):
    """Atajo: pide datos a cualquier DataSourceConnector y los limpia."""
    raw_df = connector.fetch_matches(liga=liga, desde=desde, hasta=hasta)
    return load_and_clean(raw_df)
