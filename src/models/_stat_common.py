"""
models/_stat_common.py
=======================

Lógica compartida por corners.py, tiros.py y tarjetas.py.

Por qué estos tres mercados comparten un enfoque (y goles.py no):
a diferencia de los goles, córners/tiros/tarjetas no tienen un modelo
teórico tan establecido como Dixon-Coles. Pero comparten un problema
estadístico común: son conteos con *sobredispersión* (varianza mayor
que la media), lo cual viola el supuesto de Poisson (donde media =
varianza). Un equipo puede tener córners muy parejos partido a partido,
u otro con mucha variabilidad según el rival y el plan de partido — Poisson
no distingue esto, la binomial negativa sí, vía su parámetro de dispersión.

Enfoque:
  1. Promedio histórico por equipo, en su rol (local/visitante) y ponderado
     por recencia — igual criterio que en ratings.py, para que un equipo
     que cambió de forma recientemente no quede anclado a su pasado lejano.
  2. Ajuste por fuerza relativa del rival: un equipo que dominó posesión
     ante rivales fuertes probablemente generó incluso más córners/tiros
     de los que su promedio crudo sugiere si el próximo rival es débil.
     Se usa el rating Elo (o la fuerza de ataque/defensa) como proxy de esa
     fuerza relativa.
  3. Estimación de la dispersión vía regresión binomial negativa
     (statsmodels), que entrega tanto la media esperada como el parámetro
     de forma (alpha) necesario para muestrear en la simulación de Montecarlo.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class TeamStatDistribution:
    """Resultado del ajuste para un equipo: media esperada y dispersión
    (parámetro alpha de la binomial negativa) para simular con numpy."""
    mean: float
    dispersion: float  # alpha: mayor alpha => mayor varianza sobre la media


def recency_weights(dates: pd.Series, half_life_days: float, min_weight: float = 0.05) -> np.ndarray:
    """
    Peso exponencial: un partido de hace `half_life_days` pesa la mitad
    que uno de hoy. Se usa el mismo criterio en todo el proyecto (ratings,
    goles, y estos modelos de conteo) para que la "memoria" del sistema
    sea consistente entre mercados.
    """
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.to_numpy()
    weights = 0.5 ** (days_ago / half_life_days)
    return np.maximum(weights, min_weight)


def fit_negative_binomial(counts: np.ndarray, weights: np.ndarray) -> TeamStatDistribution:
    """
    Ajusta una binomial negativa ponderada a una serie de conteos
    (córners, tiros o tarjetas de un equipo a lo largo de su historial).

    Se usa GLM con familia NegativeBinomial de statsmodels. El parámetro
    alpha se fija con el estimador de momentos (Cameron & Trivedi) como
    punto de partida robusto: alpha = (var - media) / media^2, acotado en
    0 si la muestra no muestra sobredispersión (en cuyo caso el
    comportamiento colapsa al de Poisson, que es el caso límite alpha=0).
    """
    if len(counts) == 0:
        # Sin historial para este equipo: se devuelve una distribución
        # "neutra" ampliamente dispersa en vez de fallar, para que el
        # pipeline pueda seguir corriendo con un aviso explícito aguas arriba.
        return TeamStatDistribution(mean=0.0, dispersion=1.0)

    weighted_mean = np.average(counts, weights=weights)
    weighted_var = np.average((counts - weighted_mean) ** 2, weights=weights)

    if weighted_mean <= 0:
        return TeamStatDistribution(mean=max(weighted_mean, 0.01), dispersion=0.5)

    alpha_mom = max((weighted_var - weighted_mean) / (weighted_mean ** 2), 1e-4)

    # Con pocas observaciones, un GLM completo es inestable; el estimador
    # de momentos ya es suficientemente informativo para simular. Se
    # intenta el GLM solo cuando hay muestra decente, y si falla se cae
    # de vuelta al estimador de momentos (nunca se propaga la excepción
    # hacia el usuario final).
    if len(counts) >= 8:
        try:
            X = np.ones((len(counts), 1))
            model = sm.GLM(
                counts, X,
                family=sm.families.NegativeBinomial(alpha=alpha_mom),
                freq_weights=weights,
            )
            res = model.fit()
            fitted_mean = float(np.exp(res.params[0]))
            return TeamStatDistribution(mean=fitted_mean, dispersion=alpha_mom)
        except Exception:
            pass

    return TeamStatDistribution(mean=float(weighted_mean), dispersion=float(alpha_mom))


def adjust_for_opponent_strength(
    base_mean: float,
    own_elo: float,
    opponent_elo: float,
    strength_weight: float,
) -> float:
    """
    Ajusta la media cruda por la fuerza relativa del rival esperado
    respecto al promedio de rivales históricos implícito en base_mean.

    Se usa una función logística centrada en la diferencia de Elo: si el
    próximo rival es más débil que el promedio histórico de rivales, se
    espera *más* del stat (más córners/tiros a favor); si es más fuerte,
    menos. strength_weight (0 a 1) controla qué tan agresivo es este ajuste,
    dejando en 0 el comportamiento "promedio crudo sin ajustar" para quien
    prefiera no asumir este supuesto adicional.
    """
    elo_diff = own_elo - opponent_elo
    # Factor multiplicativo acotado en un rango razonable (0.6x a 1.4x)
    # para que un desbalance extremo de Elo no dispare el ajuste a
    # valores absurdos.
    raw_factor = 1.0 + (elo_diff / 800.0)
    factor = np.clip(raw_factor, 0.6, 1.4)
    blended_factor = 1.0 + strength_weight * (factor - 1.0)
    return base_mean * blended_factor


def team_role_distribution(
    matches: pd.DataFrame,
    team: str,
    role: str,
    stat_column: str,
    half_life_days: float,
) -> TeamStatDistribution:
    """
    Calcula la distribución (media, dispersión) de `stat_column` para
    `team` jugando de `role` ("local" o "visitante"), usando su historial
    disponible en `matches`.
    """
    col_team = "equipo_local" if role == "local" else "equipo_visitante"
    subset = matches[matches[col_team] == team].dropna(subset=[stat_column])
    if subset.empty:
        return TeamStatDistribution(mean=0.0, dispersion=1.0)

    weights = recency_weights(subset["fecha"], half_life_days)
    counts = subset[stat_column].to_numpy(dtype=float)
    return fit_negative_binomial(counts, weights)
