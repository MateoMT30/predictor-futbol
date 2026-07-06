"""
scripts/run_backtest.py
=======================

Genera `data/backtest.json`: el rendimiento real del modelo sobre los
últimos partidos jugados de cada competición (ver src/backtest.py para la
metodología walk-forward y las métricas).

Corre OFFLINE (igual que refresh_fifa_cache.py) porque reajustar
Dixon-Coles decenas de veces por competición es demasiado pesado para un
request en Render free. La web solo lee el JSON.

Uso:
    FOOTBALL_DATA_API_KEY=tu_key python scripts/run_backtest.py
    git add data/backtest.json && git commit && git push

O automático vía GitHub Actions (.github/workflows/backtest.yml), que
necesita el secret FOOTBALL_DATA_API_KEY configurado en el repo.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.backtest import walk_forward_backtest  # noqa: E402
from src.connectors.football_data_connector import FootballDataConnector, COMPETITIONS  # noqa: E402
from src.connectors.international_results_connector import (  # noqa: E402
    align_team_names, fetch_international_results,
)
from src.data_loader import load_from_connector  # noqa: E402
from src.main import load_config  # noqa: E402
from src.models.goles import GoalsModelConfig  # noqa: E402

OUT_PATH = _REPO_ROOT / "data" / "backtest.json"

# Torneos de selecciones: football-data solo trae los partidos del torneo y su
# `fullTime` incluye la prórroga. Para estos se mezcla el dataset internacional
# (clasificatorias + amistosos, marcador a 90 MINUTOS) y se PREFIERE su versión
# en los partidos presentes en ambas fuentes — así el backtest evalúa a 90',
# igual que predice el modelo (evita calificar un 1-1 real a 90' como 4-5).
NATIONAL_TEAM_COMPETITIONS = {"WC", "EC"}


def _merge_90min(matches_df, api_desde):
    """Devuelve matches_df con los partidos de torneo reemplazados por su
    versión a 90' del dataset internacional (nombres alineados a la API).
    Degradación segura: si el dataset no está disponible, devuelve la entrada
    tal cual (mejor un backtest con prórroga que ninguno)."""
    try:
        intl = fetch_international_results(desde=api_desde)
    except Exception:
        intl = None
    if intl is None or intl.empty:
        return matches_df
    equipos_api = set(matches_df["equipo_local"]) | set(matches_df["equipo_visitante"])
    intl = align_team_names(intl, equipos_api)
    # Solo interesan equipos que también están en la API del torneo (no traer
    # el planeta entero); esto conecta el histórico sin inflarlo.
    intl = intl[intl["equipo_local"].isin(equipos_api) | intl["equipo_visitante"].isin(equipos_api)]
    if intl.empty:
        return matches_df
    # Dedup TOLERANTE A ±1 DÍA: un mismo partido puede figurar en football-data
    # y en el internacional con la fecha corrida un día (el saque cruza la
    # medianoche UTC). Con dedup por fecha exacta, la versión con prórroga
    # sobrevivía junto a la de 90' (partido duplicado + sesgo de temporalidad
    # de vuelta). Dos selecciones no juegan dos veces en 1 día, así que casar
    # por (local, visitante) dentro de ±1 día es seguro.
    from collections import defaultdict
    intl_pair_dates = defaultdict(list)
    for f, h, a in zip(intl["fecha"], intl["equipo_local"], intl["equipo_visitante"]):
        intl_pair_dates[(h, a)].append(pd.Timestamp(f).date())

    def _dup_en_intl(f, h, a):
        fd = pd.Timestamp(f).date()
        return any(abs((fd - d).days) <= 1 for d in intl_pair_dates.get((h, a), ()))

    base = matches_df[[
        not _dup_en_intl(f, h, a)
        for f, h, a in zip(matches_df["fecha"], matches_df["equipo_local"], matches_df["equipo_visitante"])
    ]]
    out = pd.concat([base, intl], ignore_index=True)
    out["fecha"] = pd.to_datetime(out["fecha"])
    return out.sort_values("fecha").reset_index(drop=True)


def main() -> int:
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("ERROR: falta FOOTBALL_DATA_API_KEY en el entorno.")
        return 1

    config = load_config(str(_REPO_ROOT / "config.yaml"))
    goals_cfg = GoalsModelConfig(
        xi=config["goals_model"]["dixon_coles_xi"],
        max_goals=config["goals_model"]["max_goals"],
        low_score_correction=config["goals_model"]["low_score_correction"],
        regularization=config["goals_model"].get("regularization", 0.7),
        max_expected_goals=config["goals_model"].get("max_expected_goals", 4.5),
    )

    connector = FootballDataConnector(api_key=api_key)
    hoy = datetime.now(timezone.utc)
    desde = (hoy - timedelta(days=365)).strftime("%Y-%m-%d")
    hasta = hoy.strftime("%Y-%m-%d")

    out = {}
    for code, name in COMPETITIONS.items():
        try:
            matches_df, _ = load_from_connector(connector, liga=code, desde=desde, hasta=hasta)
        except Exception as exc:
            print(f"  {code}: sin datos ({exc})")
            continue
        if matches_df.empty:
            print(f"  {code}: histórico vacío, se omite.")
            continue

        # Selecciones: reemplazar por marcadores a 90' (ver _merge_90min).
        if code in NATIONAL_TEAM_COMPETITIONS:
            antes = len(matches_df)
            matches_df = _merge_90min(matches_df, hoy - timedelta(days=1095))
            print(f"  {code}: histórico a 90' (internacional): {antes} -> {len(matches_df)} partidos")

        t0 = time.time()
        result = walk_forward_backtest(
            matches_df, goals_cfg,
            elo_dc_weight=config["goals_model"].get("elo_blend_dc_weight", 0.8),
        )
        if result is None:
            print(f"  {code}: historial insuficiente para evaluar, se omite.")
        else:
            out[code] = result
            print(f"  {code}: {result['aciertos']}/{result['n']} aciertos "
                  f"({result['acierto_pct']}%), Brier {result['brier']} "
                  f"[{time.time() - t0:.0f}s]")
        # El plan gratuito de football-data.org limita a 10 requests/minuto;
        # una pausa corta entre competiciones evita el 429.
        time.sleep(7)

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nGuardado: {OUT_PATH} ({len(out)} competiciones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
