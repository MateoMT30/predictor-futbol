"""
_nbinom_utils.py
================

Conversión de (media, dispersión alpha) -> parámetros (n, p) de la
parametrización que usa numpy para la binomial negativa, y utilidad de
muestreo compartida por simulation.py.

Parametrización usada en todo el proyecto (media/varianza, la estándar en
econometría de conteos — Cameron & Trivedi):
    Var(X) = mean + alpha * mean^2

numpy.random.negative_binomial(n, p) usa en cambio la parametrización
(n, p) de "número de fallos antes de n éxitos". La conversión es:
    n = 1 / alpha
    p = n / (n + mean)
"""

import numpy as np

from .models._stat_common import TeamStatDistribution


def sample_negative_binomial(dist: TeamStatDistribution, size: int, rng: np.random.Generator) -> np.ndarray:
    mean = max(dist.mean, 1e-6)
    alpha = max(dist.dispersion, 1e-6)

    n = 1.0 / alpha
    p = n / (n + mean)
    # Clip de seguridad: valores extremos de alpha muy cercanos a 0 (equipo
    # con historial perfectamente constante) pueden producir p fuera de
    # [0, 1] por error de redondeo de punto flotante.
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return rng.negative_binomial(n, p, size=size)
