"""
ratings.py
==========

Sistema de rating tipo Elo para medir la "fuerza" relativa de cada equipo,
actualizada partido a partido.

Por qué Elo y no solo promedios históricos: un promedio de goles a favor
no distingue si esos goles se metieron contra un rival débil o uno fuerte.
Elo resuelve esto con actualizaciones relativas: ganarle a un rival fuerte
sube más el rating que ganarle a uno débil, y el sistema converge con el
tiempo hacia una jerarquía consistente de fuerza de equipos.

Extensiones sobre el Elo clásico de ajedrez, pensadas para fútbol:
  - Ventaja de local: se le suma un bono de Elo al equipo local antes de
    calcular la probabilidad esperada de resultado (los locales ganan más
    seguido por factores no reflejados en el rating puro: clima, público,
    viaje del rival, etc.)
  - Margen de goles: un 4-0 dice más sobre la diferencia de fuerza real
    entre dos equipos que un 1-0, así que el ajuste se escala con un
    multiplicador basado en la diferencia de goles (similar al "Elo con
    margen de victoria" usado en FiveThirtyEight/clubelo.com).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class RatingsConfig:
    k_factor: float = 20.0
    initial_rating: float = 1500.0
    home_advantage: float = 60.0
    use_goal_diff_multiplier: bool = True


class EloRatingSystem:
    """
    Mantiene un diccionario {equipo: rating} y lo actualiza partido a
    partido, en orden cronológico. El estado interno es mutable a
    propósito: refleja el rating "vivo" del equipo en el momento del
    último partido procesado.
    """

    def __init__(self, config: Optional[RatingsConfig] = None):
        self.config = config or RatingsConfig()
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.initial_rating)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Fórmula logística estándar de Elo: probabilidad esperada de que
        el equipo A "gane" (en el sentido de acumular más puntos Elo)
        frente al equipo B, dado el diferencial de rating.
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _goal_diff_multiplier(self, goal_diff: int) -> float:
        """
        Multiplicador inspirado en el sistema usado por clubelo.com /
        FiveThirtyEight: goleadas mueven más el rating que victorias
        ajustadas, pero con retornos decrecientes (raíz, no lineal) para
        que un 6-0 no dispare el rating de forma desproporcionada frente
        a un 3-0.
        """
        if not self.config.use_goal_diff_multiplier or goal_diff <= 1:
            return 1.0
        return np.sqrt(goal_diff)

    def update_match(self, home: str, away: str, goals_home: int, goals_away: int) -> None:
        r_home = self.get_rating(home)
        r_away = self.get_rating(away)

        # Resultado real en escala [0, 1]: 1 = victoria local, 0.5 = empate,
        # 0 = victoria visitante. Es el "score" que compara Elo contra la
        # expectativa.
        if goals_home > goals_away:
            actual_home = 1.0
        elif goals_home < goals_away:
            actual_home = 0.0
        else:
            actual_home = 0.5

        expected_home = self._expected_score(r_home + self.config.home_advantage, r_away)
        multiplier = self._goal_diff_multiplier(abs(goals_home - goals_away))

        delta = self.config.k_factor * multiplier * (actual_home - expected_home)
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta

    def replay_history(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa todo el histórico en orden cronológico y devuelve el mismo
        DataFrame con dos columnas nuevas: el rating de cada equipo *antes*
        de disputarse ese partido (así se puede usar como feature para
        entrenar otros modelos sin fuga de información del resultado futuro).
        """
        elo_home_pre = []
        elo_away_pre = []
        for row in matches.itertuples():
            elo_home_pre.append(self.get_rating(row.equipo_local))
            elo_away_pre.append(self.get_rating(row.equipo_visitante))
            self.update_match(row.equipo_local, row.equipo_visitante,
                               int(row.goles_local), int(row.goles_visitante))

        out = matches.copy()
        out["elo_local_pre"] = elo_home_pre
        out["elo_visitante_pre"] = elo_away_pre
        return out

    def win_probabilities(self, home: str, away: str) -> Dict[str, float]:
        """
        Probabilidad 1X2 derivada puramente del Elo (sin goles). Se usa
        como referencia rápida / sanity check frente al modelo de Poisson
        de goles.py, que es el que realmente alimenta el reporte final.

        El empate se modela con una banda alrededor de expected_home=0.5,
        de ancho fijo — una aproximación estándar cuando no se quiere
        ajustar una tercera categoría vía regresión logística multinomial.
        """
        r_home = self.get_rating(home)
        r_away = self.get_rating(away)
        expected_home = self._expected_score(r_home + self.config.home_advantage, r_away)

        draw_band = 0.15
        p_draw = max(0.0, draw_band - abs(expected_home - 0.5) * 0.3)
        p_home = expected_home * (1 - p_draw)
        p_away = (1 - expected_home) * (1 - p_draw)

        total = p_home + p_draw + p_away
        return {
            "local": p_home / total,
            "empate": p_draw / total,
            "visitante": p_away / total,
        }
