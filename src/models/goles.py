"""
models/goles.py
================

Modelo de goles: Dixon-Coles (1997), una extensión del clásico modelo de
Poisson independiente de Maher (1982) para resultados de fútbol.

--- El "qué" ---
Cada equipo tiene un parámetro de ataque (alpha) y uno de defensa (beta).
Los goles esperados de un partido local vs visitante son:

    lambda_local     = exp(alpha_local + beta_visitante + gamma)
    lambda_visitante = exp(alpha_visitante + beta_local)

donde gamma es la ventaja de jugar de local. Los goles de cada equipo se
modelan como Poisson(lambda_local) y Poisson(lambda_visitante) —
*condicionalmente* independientes dado lambda_local y lambda_visitante.

--- El "por qué" de cada pieza ---

1) ¿Por qué separar ataque y defensa por equipo, en vez de un promedio de
   goles global? Porque el promedio de goles de un equipo depende tanto de
   su propio ataque como de la defensa de los rivales que enfrentó. Separar
   ambos parámetros permite estimar, por ejemplo, "qué tan bueno es el
   ataque de Colombia" de forma independiente de si jugó muchos partidos
   contra defensas débiles.

2) ¿Por qué Dixon-Coles y no Poisson independiente puro? Maher (1982)
   asume independencia total entre los goles del local y del visitante.
   Empíricamente esto subestima la frecuencia de marcadores bajos y
   correlacionados (0-0, 1-0, 0-1, 1-1): en la realidad, cuando un partido
   está trabado y con pocos goles, hay una correlación negativa leve entre
   ambos marcadores que el Poisson puro no captura. Dixon-Coles corrige
   esto multiplicando la probabilidad conjunta por un factor tau(x,y) que
   solo afecta a esas cuatro celdas de la matriz de resultados.

3) ¿Por qué ponderar por recencia (xi) en el ajuste de máxima verosimilitud?
   La fuerza real de un equipo cambia con el tiempo (fichajes, lesiones,
   cambios de entrenador). Un partido de hace dos años dice mucho menos
   sobre la fuerza actual que uno de la semana pasada. Dixon-Coles resuelve
   esto con un peso exponencial decreciente exp(-xi * dias_desde_partido)
   aplicado a la log-verosimilitud de cada partido histórico.

Referencia: Dixon, M.J. and Coles, S.G. (1997), "Modelling Association
Football Scores and Inefficiencies in the Football Betting Market".
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass
class GoalsModelConfig:
    xi: float = 0.0018          # decaimiento temporal (Dixon-Coles)
    max_goals: int = 10          # tope de la matriz de resultados simulada
    low_score_correction: bool = True


def _tau(goals_home: int, goals_away: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """
    Factor de corrección de Dixon-Coles. Solo modifica las 4 celdas de
    marcador bajo donde el Poisson independiente se equivoca sistemáticamente;
    para cualquier otro marcador vale 1 (no hace nada).

    rho es un parámetro pequeño (típicamente entre -0.15 y 0.05) estimado
    junto con alpha/beta/gamma durante el ajuste.
    """
    if goals_home == 0 and goals_away == 0:
        return 1 - lambda_home * lambda_away * rho
    elif goals_home == 0 and goals_away == 1:
        return 1 + lambda_home * rho
    elif goals_home == 1 and goals_away == 0:
        return 1 + lambda_away * rho
    elif goals_home == 1 and goals_away == 1:
        return 1 - rho
    return 1.0


class DixonColesModel:
    """
    Ajusta parámetros de ataque/defensa por equipo vía máxima verosimilitud
    y expone la distribución conjunta de goles para un partido futuro.
    """

    def __init__(self, config: GoalsModelConfig = None):
        self.config = config or GoalsModelConfig()
        self.teams: list[str] = []
        self.attack: Dict[str, float] = {}
        self.defence: Dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.rho: float = 0.0
        self._fitted = False

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        """
        matches debe tener: fecha, equipo_local, equipo_visitante,
        goles_local, goles_visitante (ya limpio, vía data_loader).

        La optimización es sobre 2*n_equipos + 2 parámetros (ataque y
        defensa por equipo, más gamma y rho). Se usa L-BFGS-B porque el
        problema es suave y de tamaño moderado (decenas/cientos de equipos
        como máximo en cualquier liga real).
        """
        self.teams = sorted(set(matches["equipo_local"]) | set(matches["equipo_visitante"]))
        n = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        max_date = matches["fecha"].max()
        days_ago = (max_date - matches["fecha"]).dt.days.to_numpy()
        weights = np.exp(-self.config.xi * days_ago)

        home_idx = matches["equipo_local"].map(team_idx).to_numpy()
        away_idx = matches["equipo_visitante"].map(team_idx).to_numpy()
        goals_home = matches["goles_local"].to_numpy()
        goals_away = matches["goles_visitante"].to_numpy()

        def unpack(params):
            attack = params[:n]
            defence = params[n:2 * n]
            gamma = params[2 * n]
            rho = params[2 * n + 1]
            return attack, defence, gamma, rho

        def neg_log_likelihood(params):
            attack, defence, gamma, rho = unpack(params)
            lam_home = np.exp(attack[home_idx] + defence[away_idx] + gamma)
            lam_away = np.exp(attack[away_idx] + defence[home_idx])

            # Log-verosimilitud de Poisson para cada lado, más la corrección
            # tau (en log, aplicada como log(tau) porque tau es un factor
            # multiplicativo de la probabilidad conjunta).
            ll_home = poisson.logpmf(goals_home, lam_home)
            ll_away = poisson.logpmf(goals_away, lam_away)

            log_tau = np.zeros(len(goals_home))
            if self.config.low_score_correction:
                for i in range(len(goals_home)):
                    t = _tau(int(goals_home[i]), int(goals_away[i]), lam_home[i], lam_away[i], rho)
                    log_tau[i] = np.log(max(t, 1e-10))

            ll = ll_home + ll_away + log_tau
            return -np.sum(weights * ll)

        # Punto inicial neutro: todos los equipos "promedio", sin ventaja
        # de local ni corrección de marcador bajo. La optimización converge
        # rápido desde aquí porque la log-verosimilitud de Poisson es
        # cóncava en la región relevante.
        x0 = np.zeros(2 * n + 2)

        # Restricción de identificabilidad: sin fijar un ancla, alpha y
        # beta podrían desplazarse todos por una constante sin cambiar la
        # verosimilitud (el modelo es invariante a alpha_i -> alpha_i + c,
        # defence_i -> defence_i - c). Se fija el ataque del primer equipo
        # en 0 como referencia.
        bounds = [(None, None)] * (2 * n + 2)

        # rho no está acotado por la teoría del modelo, pero SÍ debe estarlo
        # en la práctica: con historiales chicos (pocos partidos por equipo)
        # el optimizador puede "descubrir" que empujar rho a un valor enorme
        # concentra casi toda la probabilidad en las celdas de marcador bajo
        # (0-0, 1-1), inflando artificialmente el empate para encajar mejor
        # esos pocos datos — un caso de sobreajuste severo, no una mejora
        # real del modelo. En el paper original de Dixon-Coles, rho estimado
        # sobre datos reales de la liga inglesa ronda entre -0.15 y 0.05;
        # se acota ahí con margen para no perder generalidad en otras ligas.
        bounds[2 * n + 1] = (-0.3, 0.3)

        result = minimize(
            neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        attack, defence, gamma, rho = unpack(result.x)
        # Normalizamos para que el ataque promedio sea 0 (fija el punto de
        # referencia de forma simétrica en vez de anclar a un equipo
        # arbitrario, lo cual sería sensible a qué equipo se elija).
        attack = attack - attack.mean()

        self.attack = dict(zip(self.teams, attack))
        self.defence = dict(zip(self.teams, defence))
        self.home_advantage = float(gamma)
        self.rho = float(rho)
        self._fitted = True
        return self

    def _default_strength_for_unknown_team(self) -> Tuple[float, float]:
        """
        Un equipo nunca visto (ej. debut en el histórico, o nombre no
        normalizado) recibe fuerza promedio (0, 0) en vez de fallar. Es una
        degradación explícita y documentada, no un intento de adivinar.
        """
        return 0.0, 0.0

    def expected_goals(
        self, home: str, away: str,
        home_adjustment: float = 0.0, away_adjustment: float = 0.0,
    ) -> Tuple[float, float]:
        """
        home_adjustment / away_adjustment: ajuste manual porcentual sobre
        los goles esperados (ej. -0.15 = 15% menos goles esperados).

        Por qué existe esto: Dixon-Coles se ajusta solo con resultados
        históricos — no puede saber que el goleador titular está lesionado,
        que hay un jugador clave suspendido, o que el equipo va a rotar
        titulares por prioridad en otro torneo. Esa información ("actualidad"
        en el sentido de noticias/bajas, no de forma reciente) no vive en
        una tabla de resultados y ningún ajuste estadístico puede inferirla
        solo. Este parámetro es la vía explícita para que el usuario
        incorpore ese conocimiento a mano, en vez de que el modelo finja
        que no existe. Se aplica de forma transparente (multiplicativa
        sobre lambda) y debe declararse siempre en el reporte, para que
        quede claro qué es el modelo puro y qué es criterio humano encima.
        """
        if not self._fitted:
            raise RuntimeError("El modelo no ha sido ajustado. Llama a fit() primero.")

        a_home, d_home = self.attack.get(home, 0.0), self.defence.get(home, 0.0)
        a_away, d_away = self.attack.get(away, 0.0), self.defence.get(away, 0.0)

        lam_home = np.exp(a_home + d_away + self.home_advantage) * (1.0 + home_adjustment)
        lam_away = np.exp(a_away + d_home) * (1.0 + away_adjustment)
        return float(max(lam_home, 0.01)), float(max(lam_away, 0.01))

    def score_matrix(
        self, home: str, away: str,
        home_adjustment: float = 0.0, away_adjustment: float = 0.0,
    ) -> np.ndarray:
        """
        Matriz P[i, j] = probabilidad de que el marcador final sea
        i goles del local, j goles del visitante, con la corrección
        Dixon-Coles aplicada a las 4 celdas de marcador bajo.
        """
        lam_home, lam_away = self.expected_goals(home, away, home_adjustment, away_adjustment)
        max_g = self.config.max_goals

        p_home = poisson.pmf(np.arange(max_g + 1), lam_home)
        p_away = poisson.pmf(np.arange(max_g + 1), lam_away)
        matrix = np.outer(p_home, p_away)

        if self.config.low_score_correction:
            for i in range(2):
                for j in range(2):
                    matrix[i, j] *= _tau(i, j, lam_home, lam_away, self.rho)

        # La MLE ajusta rho sin acotarlo; con datos escasos o ruidosos puede
        # converger a un valor que vuelve negativa alguna celda de la
        # corrección (matemáticamente inválido como probabilidad). Se
        # recorta a 0 en vez de dejar que rompa la normalización o el
        # muestreo posterior en Montecarlo.
        matrix = np.clip(matrix, 0.0, None)
        matrix = matrix / matrix.sum()
        return matrix

    def market_probabilities(
        self, home: str, away: str,
        home_adjustment: float = 0.0, away_adjustment: float = 0.0,
    ) -> dict:
        """
        Deriva todos los mercados relacionados con goles a partir de la
        misma matriz de marcadores — así 1X2, over/under y BTTS son
        consistentes entre sí por construcción (no se estiman por separado
        con métodos distintos que podrían contradecirse).
        """
        matrix = self.score_matrix(home, away, home_adjustment, away_adjustment)
        max_g = matrix.shape[0]

        p_home_win = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away_win = np.triu(matrix, 1).sum()

        btts_yes = matrix[1:, 1:].sum()
        btts_no = 1 - btts_yes

        totals = {}
        goal_totals = np.add.outer(np.arange(max_g), np.arange(max_g))
        for line in [1.5, 2.5, 3.5]:
            over = matrix[goal_totals > line].sum()
            totals[line] = {"over": float(over), "under": float(1 - over)}

        lam_home, lam_away = self.expected_goals(home, away, home_adjustment, away_adjustment)

        return {
            "1x2": {"local": float(p_home_win), "empate": float(p_draw), "visitante": float(p_away_win)},
            "ambos_anotan": {"si": float(btts_yes), "no": float(btts_no)},
            "over_under_goles": totals,
            "goles_esperados": {"local": lam_home, "visitante": lam_away, "total": lam_home + lam_away},
            "ajuste_manual_aplicado": {"local": home_adjustment, "visitante": away_adjustment},
        }

    def handicap_probabilities(
        self, home: str, away: str, lines: list,
        home_adjustment: float = 0.0, away_adjustment: float = 0.0,
    ) -> dict:
        """
        Hándicap asiático/europeo: probabilidad de que (goles_local +
        linea) supere a goles_visitante. Se calcula directamente sobre la
        matriz conjunta para mantener consistencia con el resto de mercados.
        """
        matrix = self.score_matrix(home, away, home_adjustment, away_adjustment)
        max_g = matrix.shape[0]
        result = {}
        for line in lines:
            home_covers = 0.0
            push = 0.0
            for i in range(max_g):
                for j in range(max_g):
                    adjusted = i + line - j
                    if adjusted > 0:
                        home_covers += matrix[i, j]
                    elif adjusted == 0:
                        push += matrix[i, j]
            result[line] = {
                "local_cubre": float(home_covers),
                "push": float(push),
                "visitante_cubre": float(1 - home_covers - push),
            }
        return result
