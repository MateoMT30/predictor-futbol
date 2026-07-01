"""
models/tiros.py
================

Estima la distribución de tiros al arco por equipo para un partido dado.

Misma metodología que corners.py (binomial negativa ponderada por
recencia + ajuste por fuerza del rival) — ver models/_stat_common.py para
el detalle estadístico. Se separa en su propio archivo, en vez de
reutilizar CornersModel con otro nombre de columna, porque cada mercado
puede evolucionar con reglas propias (p. ej. tiros al arco correlaciona
más directamente con goles que córners, algo que podría aprovecharse en
una versión futura para reducir varianza en la simulación conjunta).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from ._stat_common import TeamStatDistribution, team_role_distribution, adjust_for_opponent_strength


@dataclass
class ShotsModelConfig:
    half_life_days: float = 180.0
    opponent_strength_weight: float = 0.5


class ShotsOnTargetModel:
    def __init__(self, config: ShotsModelConfig = None):
        self.config = config or ShotsModelConfig()

    def team_distributions(
        self,
        matches: pd.DataFrame,
        home: str,
        away: str,
        elo_ratings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, TeamStatDistribution]:
        home_dist = team_role_distribution(matches, home, "local", "tiros_arco_local", self.config.half_life_days)
        away_dist = team_role_distribution(matches, away, "visitante", "tiros_arco_visitante", self.config.half_life_days)

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
