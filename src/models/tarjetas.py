"""
models/tarjetas.py
===================

Estima la distribución de tarjetas amarillas y rojas por equipo.

Diferencia respecto a corners.py / tiros.py: las tarjetas rojas son un
evento raro (muchos partidos con 0), por lo que su varianza histórica
suele ser pequeña en términos absolutos aunque relativamente alta frente
a la media — exactamente el escenario para el que la binomial negativa
está pensada (a diferencia de un Poisson que subestimaría la probabilidad
de "al menos una roja" en partidos de alta tensión). No se trata distinto
en código: el mismo ajuste de _stat_common.py maneja bien este caso porque
la dispersión estimada ya captura esa cola pesada relativa a la media baja.

Las tarjetas amarillas, en cambio, están más influenciadas por el árbitro
y el nivel de intensidad/rivalidad del partido que por la "fuerza" pura del
equipo — el ajuste por fuerza del rival aquí es más débil conceptualmente
que en córners/tiros, pero se deja disponible por consistencia y porque
un rival más fuerte sí suele inducir más faltas por parte del equipo que
defiende con desventaja.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from ._stat_common import TeamStatDistribution, team_role_distribution, adjust_for_opponent_strength


@dataclass
class CardsModelConfig:
    half_life_days: float = 180.0
    opponent_strength_weight: float = 0.3  # más débil que en córners/tiros: ver docstring del módulo


class CardsModel:
    def __init__(self, config: CardsModelConfig = None):
        self.config = config or CardsModelConfig()

    def team_distributions(
        self,
        matches: pd.DataFrame,
        home: str,
        away: str,
        elo_ratings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, TeamStatDistribution]]:
        """
        Devuelve, para local y visitante, las distribuciones de amarillas
        y rojas por separado (son eventos distintos con distinta base rate,
        así que se simulan independientemente en simulation.py).
        """
        result = {}
        for role, team, other in [("local", home, away), ("visitante", away, home)]:
            yellow_col = f"tarjetas_amarillas_{'local' if role == 'local' else 'visitante'}"
            red_col = f"tarjetas_rojas_{'local' if role == 'local' else 'visitante'}"

            yellow = team_role_distribution(matches, team, role, yellow_col, self.config.half_life_days)
            red = team_role_distribution(matches, team, role, red_col, self.config.half_life_days)

            if elo_ratings and self.config.opponent_strength_weight > 0:
                own_elo = elo_ratings.get(team, 1500.0)
                opp_elo = elo_ratings.get(other, 1500.0)
                # Un equipo en desventaja de fuerza tiende a cometer más
                # faltas defendiendo -> se invierte el signo del ajuste
                # respecto a córners/tiros (donde más fuerza = más stat).
                yellow.mean = adjust_for_opponent_strength(
                    yellow.mean, opp_elo, own_elo, self.config.opponent_strength_weight
                )
                red.mean = adjust_for_opponent_strength(
                    red.mean, opp_elo, own_elo, self.config.opponent_strength_weight
                )

            result[role] = {"amarillas": yellow, "rojas": red}
        return result
