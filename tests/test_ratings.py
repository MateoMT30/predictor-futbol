from src.ratings import EloRatingSystem, RatingsConfig


def test_winner_gains_rating():
    elo = EloRatingSystem(RatingsConfig(initial_rating=1500))
    before = elo.get_rating("A")
    elo.update_match("A", "B", 2, 0)
    assert elo.get_rating("A") > before
    assert elo.get_rating("B") < before


def test_draw_barely_changes_equal_teams():
    elo = EloRatingSystem(RatingsConfig(initial_rating=1500, home_advantage=0))
    elo.update_match("A", "B", 1, 1)
    assert abs(elo.get_rating("A") - 1500) < 1e-6
    assert abs(elo.get_rating("B") - 1500) < 1e-6


def test_bigger_margin_moves_rating_more():
    elo1 = EloRatingSystem(RatingsConfig(initial_rating=1500, home_advantage=0))
    elo1.update_match("A", "B", 1, 0)
    delta_small = elo1.get_rating("A") - 1500

    elo2 = EloRatingSystem(RatingsConfig(initial_rating=1500, home_advantage=0))
    elo2.update_match("A", "B", 5, 0)
    delta_big = elo2.get_rating("A") - 1500

    assert delta_big > delta_small


def test_replay_history_adds_pre_match_columns(sample_matches):
    elo = EloRatingSystem()
    out = elo.replay_history(sample_matches)
    assert "elo_local_pre" in out.columns
    assert "elo_visitante_pre" in out.columns
    assert len(out) == len(sample_matches)


def test_win_probabilities_sum_to_one():
    elo = EloRatingSystem()
    elo.update_match("A", "B", 2, 0)
    probs = elo.win_probabilities("A", "B")
    assert abs(sum(probs.values()) - 1.0) < 1e-9
