from src.models.corners import CornersModel, CornersModelConfig


def test_team_distributions_positive_mean(sample_matches):
    model = CornersModel(CornersModelConfig())
    dists = model.team_distributions(sample_matches, "Colombia", "Argentina")
    assert dists["local"].mean >= 0
    assert dists["visitante"].mean >= 0


def test_opponent_strength_adjustment_changes_mean(sample_matches):
    model = CornersModel(CornersModelConfig(opponent_strength_weight=0.8))
    no_adjust = model.team_distributions(sample_matches, "Colombia", "Argentina")
    with_adjust = model.team_distributions(
        sample_matches, "Colombia", "Argentina", elo_ratings={"Colombia": 1700, "Argentina": 1300}
    )
    # Con un rival mucho más débil, se espera un ajuste al alza
    assert with_adjust["local"].mean != no_adjust["local"].mean
