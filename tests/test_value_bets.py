from src.value_bets import implied_probabilities, find_value_bets, ValueBetsConfig


def test_implied_probabilities_removes_overround():
    odds = {"local": 2.0, "empate": 3.0, "visitante": 4.0}
    implied = implied_probabilities(odds, remove_overround=True)
    assert abs(sum(implied.values()) - 1.0) < 1e-9


def test_implied_probabilities_without_removing_overround():
    odds = {"local": 2.0, "empate": 3.0, "visitante": 4.0}
    implied = implied_probabilities(odds, remove_overround=False)
    # 1/2 + 1/3 + 1/4 = 1.0833... > 1 (overround presente)
    assert sum(implied.values()) > 1.0


def test_find_value_bets_detects_clear_edge():
    model_probs = {"local": 0.60, "empate": 0.25, "visitante": 0.15}
    odds = {"local": 3.0, "empate": 3.3, "visitante": 6.0}  # cuota de 3.0 implica ~33%, muy por debajo del modelo
    results = find_value_bets(model_probs, odds, "1X2", ValueBetsConfig(min_edge=0.03))
    local_result = next(r for r in results if r["resultado"] == "local")
    assert local_result["value_bet"] is True


def test_find_value_bets_no_edge_when_odds_match_model():
    model_probs = {"local": 0.5, "empate": 0.3, "visitante": 0.2}
    odds = {"local": 2.0, "empate": 3.333, "visitante": 5.0}  # cuotas ~ probabilidades del modelo
    results = find_value_bets(model_probs, odds, "1X2", ValueBetsConfig(min_edge=0.03))
    assert all(not r["value_bet"] for r in results)
