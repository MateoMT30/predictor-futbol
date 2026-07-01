from src.models.tarjetas import CardsModel, CardsModelConfig


def test_team_distributions_structure(sample_matches):
    model = CardsModel(CardsModelConfig())
    dists = model.team_distributions(sample_matches, "Colombia", "Argentina")
    assert "amarillas" in dists["local"]
    assert "rojas" in dists["local"]
    assert dists["local"]["amarillas"].mean >= 0
    assert dists["local"]["rojas"].mean >= 0
