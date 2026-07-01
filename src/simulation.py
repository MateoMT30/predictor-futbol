"""
simulation.py
=============

Motor de Montecarlo: combina los modelos individuales (goles, córners,
tiros, tarjetas) en una simulación conjunta del partido, repetida miles
de veces, para derivar distribuciones empíricas de cada estadística y
sus mercados asociados.

--- Por qué Montecarlo en vez de fórmulas cerradas para todo ---
Para goles, Dixon-Coles ya da una matriz de probabilidad exacta (no hace
falta simular). Pero para preguntas como "¿cuál es la probabilidad de
más de 9.5 córners Y ambos equipos anotan simultáneamente?" (mercados
combinados), o para reportar un rango esperado con percentiles
(ej. "córners totales: probablemente entre 7 y 13"), es mucho más simple
y transparente simular miles de partidos completos y contar frecuencias,
que derivar una fórmula cerrada para cada combinación posible.

--- Independencia entre mercados ---
Simplificación explícita y documentada: en esta versión, goles, córners,
tiros y tarjetas se muestrean de forma independiente entre sí dentro de
cada iteración (cada uno ya incorpora la fuerza del equipo y del rival,
pero no covarían entre ellos más allá de eso). En la realidad hay
correlación cruzada (un partido con muchos tiros suele tener más córners
y más goles), que el modelo actual no captura. Ver README > Limitaciones.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .models.goles import DixonColesModel
from .models.corners import CornersModel
from .models.tiros import ShotsOnTargetModel
from .models.tarjetas import CardsModel
from ._nbinom_utils import sample_negative_binomial


@dataclass
class SimulationConfig:
    n_iterations: int = 15000
    random_seed: int = 42


@dataclass
class MatchSimulationResult:
    goals_home: np.ndarray
    goals_away: np.ndarray
    corners_home: np.ndarray
    corners_away: np.ndarray
    shots_home: np.ndarray
    shots_away: np.ndarray
    yellow_home: np.ndarray
    yellow_away: np.ndarray
    red_home: np.ndarray
    red_away: np.ndarray


class MatchSimulator:
    def __init__(
        self,
        goals_model: DixonColesModel,
        corners_model: CornersModel,
        shots_model: ShotsOnTargetModel,
        cards_model: CardsModel,
        config: Optional[SimulationConfig] = None,
    ):
        self.goals_model = goals_model
        self.corners_model = corners_model
        self.shots_model = shots_model
        self.cards_model = cards_model
        self.config = config or SimulationConfig()

    def simulate(
        self,
        matches_history: pd.DataFrame,
        home: str,
        away: str,
        elo_ratings: Dict[str, float],
        home_adjustment: float = 0.0,
        away_adjustment: float = 0.0,
    ) -> MatchSimulationResult:
        rng = np.random.default_rng(self.config.random_seed)
        n = self.config.n_iterations

        # --- Goles: se muestrea directamente de la matriz conjunta de
        # Dixon-Coles, para preservar exactamente la corrección de
        # marcador bajo (no se re-deriva con Poisson independiente aquí).
        # home_adjustment/away_adjustment: ver docstring de
        # GoalsModel.expected_goals — ajuste manual por bajas/lesiones que
        # el modelo estadístico no puede inferir de resultados históricos.
        score_matrix = self.goals_model.score_matrix(home, away, home_adjustment, away_adjustment)
        max_g = score_matrix.shape[0]
        flat_probs = score_matrix.flatten()
        flat_probs = flat_probs / flat_probs.sum()
        sampled_flat_idx = rng.choice(len(flat_probs), size=n, p=flat_probs)
        goals_home = sampled_flat_idx // max_g
        goals_away = sampled_flat_idx % max_g

        # --- Córners / tiros / tarjetas: binomial negativa independiente
        # por equipo (media y dispersión ya ajustadas por fuerza del rival
        # en sus respectivos modelos).
        corners_dist = self.corners_model.team_distributions(matches_history, home, away, elo_ratings)
        shots_dist = self.shots_model.team_distributions(matches_history, home, away, elo_ratings)
        cards_dist = self.cards_model.team_distributions(matches_history, home, away, elo_ratings)

        corners_home = sample_negative_binomial(corners_dist["local"], n, rng)
        corners_away = sample_negative_binomial(corners_dist["visitante"], n, rng)
        shots_home = sample_negative_binomial(shots_dist["local"], n, rng)
        shots_away = sample_negative_binomial(shots_dist["visitante"], n, rng)
        yellow_home = sample_negative_binomial(cards_dist["local"]["amarillas"], n, rng)
        yellow_away = sample_negative_binomial(cards_dist["visitante"]["amarillas"], n, rng)
        red_home = sample_negative_binomial(cards_dist["local"]["rojas"], n, rng)
        red_away = sample_negative_binomial(cards_dist["visitante"]["rojas"], n, rng)

        return MatchSimulationResult(
            goals_home=goals_home, goals_away=goals_away,
            corners_home=corners_home, corners_away=corners_away,
            shots_home=shots_home, shots_away=shots_away,
            yellow_home=yellow_home, yellow_away=yellow_away,
            red_home=red_home, red_away=red_away,
        )


def summarize_distribution(samples: np.ndarray) -> dict:
    """
    Resumen estándar de cualquier distribución simulada: media, rango
    esperado (percentiles 10-90, más informativo para apuestas que un
    intervalo de confianza clásico porque no asume normalidad) y P(over)
    para las líneas más comunes se calculan aparte según el mercado.
    """
    return {
        "media": float(np.mean(samples)),
        "mediana": float(np.median(samples)),
        "rango_esperado_p10_p90": [float(np.percentile(samples, 10)), float(np.percentile(samples, 90))],
        "desviacion_estandar": float(np.std(samples)),
    }


def over_under_probability(samples: np.ndarray, line: float) -> dict:
    over = float(np.mean(samples > line))
    return {"over": over, "under": 1 - over}
