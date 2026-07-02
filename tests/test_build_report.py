"""
Tests de build_report enfocados en la honestidad del reporte cuando un
equipo no tiene datos (el caso real que motivó esto: un recién ascendido
elegido desde el selector de enfrentamientos salía con "media 0.0,
rango 0-0" en tiros/tarjetas como si fuera un pronóstico).
"""

import numpy as np
import pandas as pd

from src.main import build_report, load_config
from src.models.goles import DixonColesModel, GoalsModelConfig


class _FakeSim:
    """Resultado de simulación mínimo: arrays constantes bastan porque acá
    se prueba el armado del reporte, no la estadística."""

    def __init__(self, n=200):
        for name in ("goals_home", "goals_away", "corners_home", "corners_away",
                     "shots_home", "shots_away", "yellow_home", "yellow_away",
                     "red_home", "red_away"):
            setattr(self, name, np.ones(n))


def _history_with_stats(teams=("A", "B", "C", "D")):
    rows = []
    dates = pd.date_range("2026-01-01", periods=12, freq="7D")
    for i, d in enumerate(dates):
        home, away = teams[i % 2], teams[2 + i % 2]
        rows.append({
            "fecha": d, "equipo_local": home, "equipo_visitante": away,
            "goles_local": 1, "goles_visitante": 1,
            "corners_local": 5, "corners_visitante": 4,
            "tiros_arco_local": 4, "tiros_arco_visitante": 3,
            "tarjetas_amarillas_local": 2, "tarjetas_amarillas_visitante": 2,
            "tarjetas_rojas_local": 0, "tarjetas_rojas_visitante": 0,
        })
    return pd.DataFrame(rows)


def _report_for(home, away, matches_df):
    config = load_config()
    goals_model = DixonColesModel(GoalsModelConfig()).fit(matches_df)
    return build_report(home, away, goals_model, _FakeSim(), config, matches_df=matches_df)


def test_team_with_data_keeps_all_markets():
    df = _history_with_stats()
    report = _report_for("A", "C", df)
    assert report["corners"]["local"] is not None
    assert report["corners"]["visitante"] is not None
    assert report["over_under_corners"] is not None
    assert report["avisos"] == []


def test_team_absent_from_history_gets_none_and_aviso():
    df = _history_with_stats()
    report = _report_for("A", "Recien Ascendido FC", df)
    # El lado sin datos no debe reportar medias que parezcan pronóstico
    assert report["corners"]["visitante"] is None
    assert report["tiros_al_arco"]["visitante"] is None
    assert report["tarjetas"]["amarillas_visitante"] is None
    # Los mercados de total (mitad del partido faltante) se suprimen
    assert report["corners"]["total"] is None
    assert report["over_under_corners"] is None
    assert report["over_under_tiros"] is None
    assert report["over_under_tarjetas"] is None
    # El lado con datos se conserva
    assert report["corners"]["local"] is not None
    # Y el usuario queda avisado
    assert any("Recien Ascendido FC" in a for a in report["avisos"])
    assert report["corners"]["muestras"]["visitante"] == 0


def test_low_sample_team_generates_aviso():
    df = _history_with_stats()
    extra = df.iloc[:1].copy()
    extra["equipo_visitante"] = "Debutante FC"
    report = _report_for("A", "Debutante FC", pd.concat([df, extra], ignore_index=True))
    assert any("Debutante FC" in a and "muestra" in a.lower() for a in report["avisos"])
