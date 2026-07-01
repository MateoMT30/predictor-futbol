from src.models.tiros import ShotsOnTargetModel, ShotsModelConfig


def test_team_distributions_positive_mean(sample_matches):
    model = ShotsOnTargetModel(ShotsModelConfig())
    dists = model.team_distributions(sample_matches, "Colombia", "Argentina")
    assert dists["local"].mean >= 0
    assert dists["visitante"].mean >= 0


def test_missing_team_returns_neutral_distribution(sample_matches):
    model = ShotsOnTargetModel(ShotsModelConfig())
    dists = model.team_distributions(sample_matches, "EquipoInexistente", "OtroInexistente")
    assert dists["local"].mean == 0.0
    assert dists["local"].dispersion == 1.0
