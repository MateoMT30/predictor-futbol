from src.models.goles import DixonColesModel, GoalsModelConfig
from src.models.corners import CornersModel, CornersModelConfig
from src.models.tiros import ShotsOnTargetModel, ShotsModelConfig
from src.models.tarjetas import CardsModel, CardsModelConfig
from src.ratings import EloRatingSystem
from src.simulation import MatchSimulator, SimulationConfig, summarize_distribution, over_under_probability


def _build_simulator(sample_matches):
    goals_model = DixonColesModel(GoalsModelConfig()).fit(sample_matches)
    corners_model = CornersModel(CornersModelConfig())
    shots_model = ShotsOnTargetModel(ShotsModelConfig())
    cards_model = CardsModel(CardsModelConfig())
    sim_config = SimulationConfig(n_iterations=2000, random_seed=1)
    return MatchSimulator(goals_model, corners_model, shots_model, cards_model, sim_config)


def test_simulation_produces_correct_shapes(sample_matches):
    simulator = _build_simulator(sample_matches)
    elo = EloRatingSystem()
    elo.replay_history(sample_matches)
    result = simulator.simulate(sample_matches, "Colombia", "Argentina", elo.ratings)
    assert len(result.goals_home) == 2000
    assert (result.goals_home >= 0).all()
    assert (result.corners_home >= 0).all()


def test_simulation_is_reproducible_with_same_seed(sample_matches):
    sim1 = _build_simulator(sample_matches)
    sim2 = _build_simulator(sample_matches)
    elo = EloRatingSystem()
    elo.replay_history(sample_matches)
    r1 = sim1.simulate(sample_matches, "Colombia", "Argentina", elo.ratings)
    r2 = sim2.simulate(sample_matches, "Colombia", "Argentina", elo.ratings)
    assert (r1.goals_home == r2.goals_home).all()
    assert (r1.corners_home == r2.corners_home).all()


def test_summarize_distribution_keys(sample_matches):
    simulator = _build_simulator(sample_matches)
    elo = EloRatingSystem()
    elo.replay_history(sample_matches)
    result = simulator.simulate(sample_matches, "Colombia", "Argentina", elo.ratings)
    summary = summarize_distribution(result.corners_home)
    assert "media" in summary and "rango_esperado_p10_p90" in summary


def test_over_under_probabilities_are_complementary(sample_matches):
    simulator = _build_simulator(sample_matches)
    elo = EloRatingSystem()
    elo.replay_history(sample_matches)
    result = simulator.simulate(sample_matches, "Colombia", "Argentina", elo.ratings)
    probs = over_under_probability(result.goals_home + result.goals_away, 2.5)
    assert abs(probs["over"] + probs["under"] - 1.0) < 1e-9
