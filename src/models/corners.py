"""
models/corners.py
==================

Estima la distribución de córners por equipo para un partido dado.

Ver models/_stat_common.py para la justificación estadística completa
(binomial negativa por sobredispersión, ponderación por recencia, ajuste
por fuerza del rival). Este archivo solo define la interfaz específica
del mercado de córners: qué columnas usa y cómo arma el resultado por
equipo (local/visitante) en vez de un promedio agregado.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from ._stat_common import TeamStatDistribution, team_role_distribution, adjust_for_opponent_strength


@dataclass
class CornersModelConfig:
    half_life_days: float = 180.0
    opponent_strength_weight: float = 0.5


class CornersModel:
    def __init__(self, config: CornersModelConfig = None):
        self.config = config or CornersModelConfig()

    def team_distributions(
        self,
        matches: pd.DataFrame,
        home: str,
        away: str,
        elo_ratings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, TeamStatDistribution]:
        """
        Devuelve la distribución esperada de córners a favor de `home`
        (jugando de local) y de `away` (jugando de visitante), ajustada
        por la fuerza relativa del rival si se proveen ratings Elo.
        """
        home_dist = team_role_distribution(matches, home, "local", "corners_local", self.config.half_life_days)
        away_dist = team_role_distribution(matches, away, "visitante", "corners_visitante", self.config.half_life_days)

        if elo_ratings and self.config.opponent_strength_weight > 0:
            home_elo = elo_ratings.get(home, 1500.0)
            away_elo = elo_ratings.get(away, 1500.0)
            home_dist.mean = adjust_for_opponent_strength(
                home_dist.mean, home_elo, away_elo, self.config.opponent_strength_weight
            )
            away_dist.mean = adjust_for_opponent_strength(
                away_dist.mean, away_elo, home_elo, self.config.opponent_strength_weight
            )

        return {"local": home_dist, "visitante": away_dist}
