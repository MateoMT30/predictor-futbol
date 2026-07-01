import numpy as np
import pytest

from src.models.goles import DixonColesModel, GoalsModelConfig


def test_fit_and_expected_goals_are_positive(sample_matches):
    model = DixonColesModel(GoalsModelConfig()).fit(sample_matches)
    lam_home, lam_away = model.expected_goals("Colombia", "Argentina")
    assert lam_home > 0
    assert lam_away > 0


def test_score_matrix_sums_to_one(sample_matches):
    model = DixonColesModel(GoalsModelConfig()).fit(sample_matches)
    matrix = model.score_matrix("Colombia", "Argentina")
    assert abs(matrix.sum() - 1.0) < 1e-6


def test_market_probabilities_consistent(sample_matches):
    model = DixonColesModel(GoalsModelConfig()).fit(sample_matches)
    report = model.market_probabilities("Colombia", "Argentina")
    total_1x2 = sum(report["1x2"].values())
    assert abs(total_1x2 - 1.0) < 1e-6
    assert abs(sum(report["ambos_anotan"].values()) - 1.0) < 1e-6


def test_stronger_team_has_higher_win_prob():
    import pandas as pd
    # Construimos un histórico sintético donde A domina sistemáticamente a B
    rows = []
    for i in range(20):
        rows.append({
            "fecha": f"2024-01-{(i % 28) + 1:02d}", "liga": "Test",
            "equipo_local": "A", "equipo_visitante": "B",
            "goles_local": 3, "goles_visitante": 0,
        })
    df = pd.DataFrame(rows)
    df["fecha"] = pd.to_datetime(df["fecha"])
    model = DixonColesModel(GoalsModelConfig()).fit(df)
    report = model.market_probabilities("A", "B")
    assert report["1x2"]["local"] > report["1x2"]["visitante"]


def test_unknown_team_does_not_crash(sample_matches):
    model = DixonColesModel(GoalsModelConfig()).fit(sample_matches)
    lam_home, lam_away = model.expected_goals("EquipoInexistente", "Colombia")
    assert lam_home > 0 and lam_away > 0


def test_manual_adjustment_reduces_expected_goals(sample_matches):
    model = DixonColesModel(GoalsModelConfig()).fit(sample_matches)
    lam_base, _ = model.expected_goals("Colombia", "Argentina")
    lam_adjusted, _ = model.expected_goals("Colombia", "Argentina", home_adjustment=-0.15)
    assert lam_adjusted == pytest.approx(lam_base * 0.85, rel=1e-6)


def test_manual_adjustment_reflected_in_market_probabilities(sample_matches):
    model = DixonColesModel(GoalsModelConfig()).fit(sample_matches)
    report = model.market_probabilities("Colombia", "Argentina", home_adjustment=-0.5)
    assert report["ajuste_manual_aplicado"]["local"] == -0.5
    # Con menos goles esperados del local, su probabilidad de ganar debe bajar
    base_report = model.market_probabilities("Colombia", "Argentina")
    assert report["1x2"]["local"] < base_report["1x2"]["local"]


def test_most_probable_score_is_integer_and_consistent_with_matrix(sample_matches):
    model = DixonColesModel(GoalsModelConfig()).fit(sample_matches)
    matrix = model.score_matrix("Colombia", "Argentina")
    report = model.market_probabilities("Colombia", "Argentina")
    mp = report["marcador_mas_probable"]

    assert isinstance(mp["local"], int)
    assert isinstance(mp["visitante"], int)
    # Debe ser exactamente la celda de mayor probabilidad de la matriz,
    # no un redondeo del promedio (goles_esperados).
    assert mp["probabilidad"] == pytest.approx(matrix.max(), rel=1e-9)
    assert matrix[mp["local"], mp["visitante"]] == matrix.max()
