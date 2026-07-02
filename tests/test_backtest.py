"""
Tests del backtest walk-forward (src/backtest.py). Lo crítico acá es la
ausencia de fuga de información: cada predicción debe usar solo partidos
anteriores — un backtest con leakage daría métricas infladas que el usuario
tomaría como reales.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import walk_forward_backtest, _actual_outcome


def _synthetic_history(n=80, seed=7):
    """Liga sintética de 6 equipos donde 'Fuerte' gana mucho y 'Debil'
    pierde mucho — así el backtest tiene señal real que capturar."""
    rng = np.random.default_rng(seed)
    teams = ["Fuerte", "Medio A", "Medio B", "Medio C", "Medio D", "Debil"]
    strength = {"Fuerte": 2.2, "Medio A": 1.3, "Medio B": 1.2, "Medio C": 1.1, "Medio D": 1.0, "Debil": 0.5}
    rows = []
    dates = pd.date_range("2025-08-01", periods=n, freq="3D")
    for i, d in enumerate(dates):
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append({
            "fecha": d,
            "equipo_local": home,
            "equipo_visitante": away,
            "goles_local": rng.poisson(strength[home] * 1.15),
            "goles_visitante": rng.poisson(strength[away]),
        })
    return pd.DataFrame(rows)


def test_actual_outcome():
    assert _actual_outcome(2, 0) == "local"
    assert _actual_outcome(0, 1) == "visitante"
    assert _actual_outcome(1, 1) == "empate"


def test_backtest_returns_metrics_and_details():
    result = walk_forward_backtest(_synthetic_history(), max_matches=8, min_history=40)
    assert result is not None
    assert result["n"] == len(result["partidos"]) > 0
    assert 0 <= result["aciertos"] <= result["n"]
    # Brier multiclase siempre está en [0, 2]
    assert 0.0 <= result["brier"] <= 2.0
    p = result["partidos"][0]
    for key in ("fecha", "local", "visitante", "marcador", "pick", "real", "acierto",
                "prob_local", "prob_empate", "prob_visitante", "prob_pick", "brier"):
        assert key in p
    # Las probabilidades de cada partido suman ~1
    assert p["prob_local"] + p["prob_empate"] + p["prob_visitante"] == pytest.approx(1.0, abs=0.01)


def test_backtest_without_enough_history_returns_none():
    assert walk_forward_backtest(_synthetic_history(n=20), min_history=40) is None


def test_backtest_only_evaluates_recent_max_matches():
    result = walk_forward_backtest(_synthetic_history(), max_matches=5, min_history=40)
    assert result["n"] <= 5


def test_backtest_evalua_mercados_adicionales():
    result = walk_forward_backtest(_synthetic_history(), max_matches=8, min_history=40)
    mercados = result["mercados"]
    for key in ("doble_1x", "doble_x2", "doble_12", "over25", "btts"):
        m = mercados[key]
        assert m["n"] == result["n"]
        assert 0 <= m["aciertos"] <= m["n"]
        assert 0.0 <= m["brier"] <= 1.0       # binario: rango [0, 1]
        assert m["brier_azar"] == 0.25
    # Detalle por partido trae las probabilidades de los mercados nuevos
    p = result["partidos"][0]
    assert 0.0 <= p["prob_over25"] <= 1.0
    assert 0.0 <= p["prob_btts"] <= 1.0
    assert isinstance(p["real_over25"], bool)
    assert isinstance(p["real_btts"], bool)
    # Consistencia doble oportunidad: 1X + prob_visitante = 1 (mismas probs)
    assert p["prob_local"] + p["prob_empate"] + p["prob_visitante"] == pytest.approx(1.0, abs=0.01)


def test_backtest_guarda_marcador_exacto_predicho():
    result = walk_forward_backtest(_synthetic_history(), max_matches=6, min_history=40)
    for p in result["partidos"]:
        assert " - " in p["marcador_pred"]
        assert 0.0 < p["prob_marcador_pred"] <= 1.0
        assert isinstance(p["acierto_marcador"], bool)
        # Coherencia: si acerto_marcador, el marcador real es el predicho
        if p["acierto_marcador"]:
            assert p["marcador_pred"] == p["marcador"]
