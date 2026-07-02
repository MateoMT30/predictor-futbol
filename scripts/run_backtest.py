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

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.backtest import walk_forward_backtest  # noqa: E402
from src.connectors.football_data_connector import FootballDataConnector, COMPETITIONS  # noqa: E402
from src.data_loader import load_from_connector  # noqa: E402
from src.main import load_config  # noqa: E402
from src.models.goles import GoalsModelConfig  # noqa: E402

OUT_PATH = _REPO_ROOT / "data" / "backtest.json"


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

        t0 = time.time()
        result = walk_forward_backtest(matches_df, goals_cfg)
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
