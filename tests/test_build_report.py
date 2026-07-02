"""
Tests de build_report enfocados en la honestidad del reporte cuando un
equipo no tiene datos (el caso real que motivó esto: un recién ascendido
elegido desde el selector de enfrentamientos salía con "media 0.0,
rango 0-0" en tiros/tarjetas como si fuera un pronóstico).
"""

import numpy as np
import pytest
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


def test_build_report_incluye_top3_de_marcadores():
    df = _history_with_stats()
    report = _report_for("A", "C", df)
    tops = report["marcadores_probables"]
    assert len(tops) == 3
    # Ordenados de mayor a menor probabilidad y consistentes con el #1
    probs = [t["probabilidad"] for t in tops]
    assert probs == sorted(probs, reverse=True)
    mp = report["marcador_mas_probable"]
    assert (tops[0]["local"], tops[0]["visitante"]) == (mp["local"], mp["visitante"])


def test_build_report_incluye_marcador_condicional_y_forma():
    df = _history_with_stats()
    report = _report_for("A", "C", df)
    mc = report["marcador_condicional"]
    # El escenario condicional corresponde al pick del 1X2
    pick = max(report["1x2"], key=report["1x2"].get)
    assert mc["resultado"] == pick
    assert 0.0 < mc["prob_dentro_escenario"] <= 1.0
    # Coherencia con la región: si el pick es local, el marcador es de victoria local
    if pick == "local":
        assert mc["local"] > mc["visitante"]
    elif pick == "visitante":
        assert mc["local"] < mc["visitante"]
    else:
        assert mc["local"] == mc["visitante"]
    # Forma reciente: racha de hasta 5 con letras válidas y goles coherentes
    f = report["forma"]["local"]
    assert f is not None and 1 <= f["n"] <= 5
    assert set(f["racha"]) <= {"G", "E", "P"}
    assert f["gf"] >= 0 and f["gc"] >= 0


def test_forma_none_si_equipo_sin_partidos():
    df = _history_with_stats()
    report = _report_for("A", "Recien Ascendido FC", df)
    assert report["forma"]["visitante"] is None


def test_clasificacion_eliminatoria_reparte_el_empate_por_fuerza():
    df = _history_with_stats()
    report = _report_for("A", "C", df)
    cl = report["clasificacion_eliminatoria"]
    x = report["1x2"]
    # Suma 1 (el empate quedó repartido) y respeta el orden de fuerzas
    assert cl["local"] + cl["visitante"] == pytest.approx(1.0, abs=0.001)
    assert cl["local"] >= x["local"] and cl["visitante"] >= x["visitante"]
    if x["local"] > x["visitante"]:
        assert cl["local"] > cl["visitante"]
