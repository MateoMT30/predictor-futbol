"""
backtest.py
===========

Evaluación honesta del modelo contra partidos ya jugados ("walk-forward
backtest"): para cada partido pasado se reajusta el modelo usando SOLO los
partidos estrictamente anteriores a su fecha, se predice el 1X2, y se compara
con lo que de verdad pasó. Así se responde la pregunta del usuario: "¿qué
probabilidad de error tiene esto?" — con datos, no con promesas.

Por qué walk-forward y no simplemente "predecir el pasado con el modelo de
hoy": el modelo de hoy ya vio esos resultados dentro de su histórico de
ajuste (fuga de información / data leakage) y sacaría números
artificialmente buenos. Entrenar solo con lo anterior a cada partido
reproduce exactamente la situación en la que estarías al apostar antes del
pitazo inicial.

Métricas reportadas:
  - Acierto del pick (¿el resultado más probable según el modelo fue el
    real?). Referencia: elegir al azar acierta ~33%; elegir "siempre gana
    el local" suele rondar 40-45% en ligas.
  - Brier score multiclase (0 = perfecto, 0.667 = probabilidades uniformes
    de 1/3): castiga estar seguro y equivocado, premia probabilidades bien
    calibradas. Es la métrica estándar para esto porque mide la CALIDAD de
    la probabilidad, no solo el acierto seco — un modelo que acierta 50%
    diciendo 51% está mejor calibrado que uno que acierta 50% diciendo 90%.

Esto corre OFFLINE (scripts/run_backtest.py + GitHub Actions), igual que el
cache de FIFA: reajustar Dixon-Coles decenas de veces es demasiado pesado
para un request del plan free de Render. La web solo lee el JSON resultante.
"""

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .models.goles import DixonColesModel, GoalsModelConfig


def _actual_outcome(goles_local, goles_visitante) -> str:
    if goles_local > goles_visitante:
        return "local"
    if goles_local < goles_visitante:
        return "visitante"
    return "empate"


def _binary_market_summary(pairs) -> Optional[dict]:
    """
    Resumen de un mercado a dos salidas (sí/no): doble oportunidad,
    over/under, ambos anotan. `pairs` = lista de (prob_sí, ocurrió).

    Regla de decisión evaluada: "apostar al lado que el modelo ve más
    probable" (prob >= 50% → sí, si no → no). El Brier binario usa una sola
    probabilidad: (p - resultado)². Referencia de azar: decir siempre 50%
    da Brier 0.25 — distinta de la del 1X2 (0.667, tres salidas), por eso
    cada mercado guarda su propia referencia.
    """
    if not pairs:
        return None
    n = len(pairs)
    aciertos = sum(1 for p, happened in pairs if (p >= 0.5) == happened)
    brier = sum((p - (1.0 if happened else 0.0)) ** 2 for p, happened in pairs) / n
    return {
        "n": n,
        "aciertos": aciertos,
        "acierto_pct": round(aciertos / n * 100, 1),
        "brier": round(brier, 4),
        "brier_azar": 0.25,
    }


def walk_forward_backtest(
    matches_df: pd.DataFrame,
    goals_config: Optional[GoalsModelConfig] = None,
    max_matches: int = 40,
    min_history: int = 40,
) -> Optional[dict]:
    """
    Evalúa el modelo sobre los últimos `max_matches` partidos jugados del
    histórico, reajustando con los partidos anteriores a cada uno.

    min_history: mínimo de partidos de entrenamiento para intentar una
    predicción — con menos, Dixon-Coles no tiene muestra seria y evaluar
    sobre eso diría más del tamaño de muestra que del modelo.

    Devuelve dict con resumen + detalle por partido, o None si no hay
    suficiente historial para evaluar nada.
    """
    goals_config = goals_config or GoalsModelConfig()
    df = matches_df.dropna(subset=["goles_local", "goles_visitante"]).copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    candidates = df.iloc[min_history:]
    if candidates.empty:
        return None
    candidates = candidates.tail(max_matches)

    partidos = []
    for idx, row in candidates.iterrows():
        # Estrictamente ANTES de la fecha del partido: los partidos del mismo
        # día se excluyen del entrenamiento (conservador — en un mismo día de
        # jornada no habrías conocido los otros resultados al predecir).
        train = df[df["fecha"] < row["fecha"]]
        if len(train) < min_history:
            continue
        try:
            model = DixonColesModel(goals_config).fit(train)
            markets = model.market_probabilities(row["equipo_local"], row["equipo_visitante"], 0.0, 0.0)
            matrix = model.score_matrix(row["equipo_local"], row["equipo_visitante"], 0.0, 0.0)
        except Exception:
            continue
        probs = markets["1x2"]

        # Probabilidades derivadas de la MISMA matriz de marcadores (así
        # todos los mercados evaluados son consistentes entre sí):
        # más de 2.5 goles = suma de las celdas con 3+ goles totales.
        max_g = matrix.shape[0]
        total_goals = np.add.outer(np.arange(max_g), np.arange(max_g))
        p_over25 = float(matrix[total_goals >= 3].sum())
        p_btts = float(markets["ambos_anotan"]["si"])

        gl, gv = int(row["goles_local"]), int(row["goles_visitante"])
        real = _actual_outcome(gl, gv)
        pick = max(probs, key=probs.get)
        brier = sum((probs[k] - (1.0 if k == real else 0.0)) ** 2 for k in ("local", "empate", "visitante"))

        partidos.append({
            "fecha": row["fecha"].strftime("%Y-%m-%d"),
            "local": row["equipo_local"],
            "visitante": row["equipo_visitante"],
            "marcador": f"{gl} - {gv}",
            "prob_local": round(probs["local"], 4),
            "prob_empate": round(probs["empate"], 4),
            "prob_visitante": round(probs["visitante"], 4),
            "pick": pick,
            "prob_pick": round(probs[pick], 4),
            "real": real,
            "acierto": pick == real,
            "brier": round(brier, 4),
            # Mercados adicionales (probabilidad del "sí" y qué pasó):
            "prob_over25": round(p_over25, 4),
            "real_over25": (gl + gv) > 2.5,
            "prob_btts": round(p_btts, 4),
            "real_btts": gl > 0 and gv > 0,
        })

    if not partidos:
        return None

    n = len(partidos)
    aciertos = sum(1 for p in partidos if p["acierto"])

    # Rendimiento POR TIPO DE APUESTA (pedido del usuario: "si quiero
    # apostar gana/empata eso cambia, ¿no?"). La doble oportunidad se
    # deriva de las mismas probabilidades 1X2 (1X = local + empate, etc.),
    # así que no requiere re-evaluar nada — solo re-agregar.
    mercados = {
        "doble_1x": _binary_market_summary([
            (p["prob_local"] + p["prob_empate"], p["real"] in ("local", "empate")) for p in partidos
        ]),
        "doble_x2": _binary_market_summary([
            (p["prob_empate"] + p["prob_visitante"], p["real"] in ("empate", "visitante")) for p in partidos
        ]),
        "doble_12": _binary_market_summary([
            (p["prob_local"] + p["prob_visitante"], p["real"] in ("local", "visitante")) for p in partidos
        ]),
        "over25": _binary_market_summary([
            (p["prob_over25"], p["real_over25"]) for p in partidos
        ]),
        "btts": _binary_market_summary([
            (p["prob_btts"], p["real_btts"]) for p in partidos
        ]),
    }

    # Más reciente primero para mostrar (se evaluó en orden cronológico).
    partidos.reverse()
    return {
        "mercados": mercados,
        "generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": n,
        "aciertos": aciertos,
        "acierto_pct": round(aciertos / n * 100, 1),
        "brier": round(sum(p["brier"] for p in partidos) / n, 4),
        # Referencias para leer el Brier sin ser estadístico: azar uniforme
        # (1/3 a cada resultado) da 0.6667 — el modelo debe estar por debajo.
        "brier_azar": 0.6667,
        "partidos": partidos,
    }
