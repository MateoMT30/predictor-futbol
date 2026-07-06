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


def test_blend_elo_mueve_1x2_hacia_el_elo_y_sigue_normalizado(sample_matches):
    """attach_elo debe mezclar el 1X2 del DC con el del Elo (acercarlo) y la
    matriz debe seguir sumando 1. Con dc_weight=1.0 no debe cambiar nada."""
    from src.ratings import EloRatingSystem
    model = DixonColesModel(GoalsModelConfig()).fit(sample_matches)
    base = model.market_probabilities("Colombia", "Argentina")["1x2"]

    elo = EloRatingSystem()
    elo.replay_history(sample_matches)
    elo_1x2 = elo.win_probabilities("Colombia", "Argentina")

    # dc_weight=1.0 -> sin efecto
    model.attach_elo(elo, dc_weight=1.0)
    assert model.score_matrix("Colombia", "Argentina").sum() == pytest.approx(1.0)
    igual = model.market_probabilities("Colombia", "Argentina")["1x2"]
    assert igual["local"] == pytest.approx(base["local"], abs=1e-9)

    # dc_weight=0.5 -> el 1X2 mezclado queda entre el DC y el Elo
    model.attach_elo(elo, dc_weight=0.5)
    blend = model.market_probabilities("Colombia", "Argentina")["1x2"]
    assert sum(blend.values()) == pytest.approx(1.0)
    for k in ("local", "empate", "visitante"):
        lo, hi = sorted((base[k], elo_1x2[k]))
        assert lo - 1e-9 <= blend[k] <= hi + 1e-9


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


def test_regularization_prevents_absurd_lambdas_with_sparse_data():
    """
    Reproduce el bug real: un torneo con muchos equipos y pocos partidos
    por equipo (ej. un Mundial recién empezado) puede llevar la MLE de
    Dixon-Coles a estimar goles esperados absurdos para un equipo con una
    sola goleada en su historial. La regularización debe evitarlo.
    """
    import pandas as pd
    rows = []
    # Equipo "Golstate" con una sola goleada 7-0 (poquísima evidencia)
    rows.append({
        "fecha": "2026-06-01", "liga": "Test",
        "equipo_local": "Golstate", "equipo_visitante": "Debil",
        "goles_local": 7, "goles_visitante": 0,
    })
    # Un grupo de equipos "normales" que se enfrentan entre sí varias
    # veces, para darle al modelo suficiente contexto de escala general.
    normales = ["A", "B", "C", "D", "E"]
    for i in range(15):
        h, a = normales[i % 5], normales[(i + 1) % 5]
        rows.append({
            "fecha": f"2026-05-{(i % 28) + 1:02d}", "liga": "Test",
            "equipo_local": h, "equipo_visitante": a,
            "goles_local": 1, "goles_visitante": 1,
        })
    df = pd.DataFrame(rows)
    df["fecha"] = pd.to_datetime(df["fecha"])

    model = DixonColesModel(GoalsModelConfig()).fit(df)
    lam_home, _ = model.expected_goals("Golstate", "A")
    # Sin regularización este valor puede dispararse a >15; con ella debe
    # quedarse en un rango futbolísticamente plausible.
    assert lam_home < 6.0


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
