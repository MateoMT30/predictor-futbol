"""
value_bets.py
=============

Compara las probabilidades estimadas por el modelo contra las cuotas de
casas de apuestas, para detectar "value bets": apuestas donde el modelo
cree que un resultado es más probable de lo que la cuota implica.

--- El "qué" ---
Una cuota decimal `o` implica una probabilidad `1/o` (sin comisión). Pero
las casas de apuestas fijan cuotas con un margen (overround): la suma de
las probabilidades implícitas de todos los resultados de un mercado es
mayor a 1 (ej. 1.05-1.10 típicamente). Si no se corrige ese margen, se
subestima sistemáticamente el valor real de cada opción.

--- El "por qué" de quitar el overround ---
Si el mercado 1X2 tiene cuotas 2.10 / 3.30 / 3.40, las probabilidades
implícitas crudas suman más de 100%. Repartir ese exceso proporcionalmente
entre las tres opciones (normalización simple) da una estimación más justa
de "cuánto cree la casa" en cada resultado, aislando el margen comercial.
Es el método estándar más simple (hay alternativas más sofisticadas, como
el método de Shin, pero la normalización proporcional es suficientemente
buena para este propósito y mucho más transparente).

--- Detección de value bet ---
Se marca value bet cuando:
    prob_modelo - prob_implícita_sin_margen  >=  min_edge

El umbral min_edge (config.yaml) existe porque un edge minúsculo puede ser
puro ruido de estimación del modelo, no una ventaja real.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ValueBetsConfig:
    min_edge: float = 0.03
    remove_overround: bool = True


def implied_probabilities(odds: Dict[str, float], remove_overround: bool = True) -> Dict[str, float]:
    """
    odds: dict {resultado: cuota_decimal}, ej. {"local": 2.10, "empate": 3.30, "visitante": 3.40}
    Devuelve las probabilidades implícitas, normalizadas para quitar el
    overround si remove_overround=True.
    """
    raw = {k: 1.0 / v for k, v in odds.items() if v and v > 0}
    if not remove_overround:
        return raw
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def find_value_bets(
    model_probs: Dict[str, float],
    odds: Dict[str, float],
    market_name: str,
    config: Optional[ValueBetsConfig] = None,
) -> list:
    """
    Compara un mercado (ej. 1X2, BTTS, over/under de una línea) entre el
    modelo y las cuotas, devolviendo una lista de hallazgos por opción.
    """
    config = config or ValueBetsConfig()
    implied = implied_probabilities(odds, config.remove_overround)

    results = []
    for outcome, model_p in model_probs.items():
        if outcome not in implied:
            continue
        implied_p = implied[outcome]
        edge = model_p - implied_p
        results.append({
            "mercado": market_name,
            "resultado": outcome,
            "probabilidad_modelo": round(model_p, 4),
            "probabilidad_implicita": round(implied_p, 4),
            "cuota": odds[outcome],
            "edge": round(edge, 4),
            "value_bet": bool(edge >= config.min_edge),
        })
    return results


def evaluate_all_markets(model_report: dict, odds_json: dict, config: Optional[ValueBetsConfig] = None) -> list:
    """
    Recorre los mercados disponibles tanto en el reporte del modelo como
    en el JSON de cuotas provisto por el usuario, y agrega todos los
    hallazgos de value bet en una sola lista para el reporte final.

    Se ignoran silenciosamente los mercados presentes en las cuotas pero
    no soportados por el modelo (o viceversa) — no todo proveedor de
    cuotas ofrece los mismos mercados que este sistema modela.
    """
    all_results = []

    if "1x2" in odds_json and "1x2" in model_report:
        all_results += find_value_bets(model_report["1x2"], odds_json["1x2"], "1X2", config)

    if "ambos_anotan" in odds_json and "ambos_anotan" in model_report:
        all_results += find_value_bets(model_report["ambos_anotan"], odds_json["ambos_anotan"], "Ambos anotan", config)

    if "over_under_goles" in odds_json and "over_under_goles" in model_report:
        for line_str, line_odds in odds_json["over_under_goles"].items():
            line = float(line_str)
            if line in model_report["over_under_goles"]:
                all_results += find_value_bets(
                    model_report["over_under_goles"][line], line_odds, f"Over/Under {line} goles", config
                )

    if "over_under_corners" in odds_json and "over_under_corners" in model_report:
        for line_str, line_odds in odds_json["over_under_corners"].items():
            line = float(line_str)
            if line in model_report["over_under_corners"]:
                all_results += find_value_bets(
                    model_report["over_under_corners"][line], line_odds, f"Over/Under {line} corners", config
                )

    return all_results
