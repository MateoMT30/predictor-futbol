"""
main.py
=======

CLI del predictor. Orquesta todo el pipeline:

  1. Carga y limpia el histórico (data_loader + connectors)
  2. Recalcula ratings Elo partido a partido (ratings.py)
  3. Ajusta el modelo de goles Dixon-Coles (models/goles.py)
  4. Corre la simulación de Montecarlo combinando todos los mercados
     (simulation.py)
  5. Si se proveen cuotas, calcula value bets (value_bets.py)
  6. Imprime un reporte legible en consola, y opcionalmente lo vuelca a JSON

Ejemplo de uso:
    python main.py --local "Colombia" --visitante "Argentina" --liga "Mundial" \
        --datos examples/historico_ejemplo.csv --cuotas examples/cuotas_ejemplo.json
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

# La consola de Windows por defecto usa cp1252, que no puede imprimir
# caracteres como "★" o tildes en algunos casos. Se fuerza UTF-8 en
# stdout para que el reporte se vea igual en Windows/macOS/Linux.
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.connectors.csv_connector import CSVConnector
from src.data_loader import load_from_connector
from src.ratings import EloRatingSystem, RatingsConfig
from src.models.goles import DixonColesModel, GoalsModelConfig
from src.models.corners import CornersModel, CornersModelConfig
from src.models.tiros import ShotsOnTargetModel, ShotsModelConfig
from src.models.tarjetas import CardsModel, CardsModelConfig
from src.simulation import MatchSimulator, SimulationConfig, summarize_distribution, over_under_probability
from src.value_bets import evaluate_all_markets, ValueBetsConfig
from src.report_html import render_html_report


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        # ruta relativa al proyecto si se corrió desde otro directorio
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_report(
    home: str,
    away: str,
    goals_model: DixonColesModel,
    sim_result,
    config: dict,
    home_adjustment: float = 0.0,
    away_adjustment: float = 0.0,
    matches_df=None,
) -> dict:
    """
    matches_df (opcional): el histórico usado para ajustar los modelos. Se
    usa únicamente para detectar si la fuente de datos realmente trae
    córners/tiros/tarjetas (ej. football-data.org gratuito NO los trae) —
    si una columna está enteramente vacía, esos mercados se marcan como
    no disponibles en vez de mostrar un falso "0.0" que parecería un
    pronóstico real cuando en realidad es ausencia de dato.
    """
    goals_report = goals_model.market_probabilities(home, away, home_adjustment, away_adjustment)
    handicap_lines = config["markets"]["handicap_lines"]
    goals_report["handicap"] = goals_model.handicap_probabilities(
        home, away, handicap_lines, home_adjustment, away_adjustment
    )

    def has_data(*columns) -> bool:
        if matches_df is None:
            return True
        return any(col in matches_df.columns and matches_df[col].notna().any() for col in columns)

    has_corners = has_data("corners_local", "corners_visitante")
    has_shots = has_data("tiros_arco_local", "tiros_arco_visitante")
    has_cards = has_data("tarjetas_amarillas_local", "tarjetas_amarillas_visitante",
                          "tarjetas_rojas_local", "tarjetas_rojas_visitante")

    corners_totals = sim_result.corners_home + sim_result.corners_away
    shots_totals = sim_result.shots_home + sim_result.shots_away
    yellow_totals = sim_result.yellow_home + sim_result.yellow_away
    red_totals = sim_result.red_home + sim_result.red_away

    report = {
        "partido": {"local": home, "visitante": away},
        "1x2": goals_report["1x2"],
        "handicap": goals_report["handicap"],
        "ambos_anotan": goals_report["ambos_anotan"],
        "goles_esperados": goals_report["goles_esperados"],
        "ajuste_manual_aplicado": goals_report["ajuste_manual_aplicado"],
        "over_under_goles": {
            line: over_under_probability(sim_result.goals_home + sim_result.goals_away, line)
            for line in config["markets"]["over_under_lines"]["goals"]
        },
        "corners": {
            "local": summarize_distribution(sim_result.corners_home),
            "visitante": summarize_distribution(sim_result.corners_away),
            "total": summarize_distribution(corners_totals),
        } if has_corners else None,
        "over_under_corners": {
            line: over_under_probability(corners_totals, line)
            for line in config["markets"]["over_under_lines"]["corners"]
        } if has_corners else None,
        "tiros_al_arco": {
            "local": summarize_distribution(sim_result.shots_home),
            "visitante": summarize_distribution(sim_result.shots_away),
            "total": summarize_distribution(shots_totals),
        } if has_shots else None,
        "over_under_tiros": {
            line: over_under_probability(shots_totals, line)
            for line in config["markets"]["over_under_lines"]["shots_on_target"]
        } if has_shots else None,
        "tarjetas": {
            "amarillas_local": summarize_distribution(sim_result.yellow_home),
            "amarillas_visitante": summarize_distribution(sim_result.yellow_away),
            "amarillas_total": summarize_distribution(yellow_totals),
            "rojas_local": summarize_distribution(sim_result.red_home),
            "rojas_visitante": summarize_distribution(sim_result.red_away),
        } if has_cards else None,
        "over_under_tarjetas": {
            line: over_under_probability(yellow_totals, line)
            for line in config["markets"]["over_under_lines"]["cards"]
        } if has_cards else None,
    }
    return report


def print_human_report(report: dict, value_bets: list) -> None:
    p = report["partido"]
    print("=" * 70)
    print(f"  PRONÓSTICO: {p['local']} vs {p['visitante']}")
    print("=" * 70)

    ajuste = report.get("ajuste_manual_aplicado", {})
    if ajuste.get("local", 0) != 0 or ajuste.get("visitante", 0) != 0:
        print(f"\n[AVISO] Se aplicó un ajuste manual (no estadístico) sobre los goles esperados:")
        print(f"  Local: {ajuste['local']*100:+.0f}%   Visitante: {ajuste['visitante']*100:+.0f}%")
        print(f"  Esto refleja criterio humano (bajas, lesiones, etc.), no algo que el modelo haya inferido solo.")

    print("\n[1X2]")
    for k, v in report["1x2"].items():
        print(f"  {k:12s}: {v*100:5.1f}%")

    print("\n[Ambos anotan]")
    for k, v in report["ambos_anotan"].items():
        print(f"  {k:12s}: {v*100:5.1f}%")

    print("\n[Goles esperados]")
    ge = report["goles_esperados"]
    print(f"  Local: {ge['local']:.2f}  Visitante: {ge['visitante']:.2f}  Total: {ge['total']:.2f}")

    print("\n[Over/Under goles]")
    for line, probs in report["over_under_goles"].items():
        print(f"  {line}: Over {probs['over']*100:5.1f}%  Under {probs['under']*100:5.1f}%")

    print("\n[Córners]")
    c = report["corners"]
    if c is None:
        print("  Sin datos suficientes (la fuente de histórico usada no trae córners).")
    else:
        print(f"  Local     -> media {c['local']['media']:.1f}, rango esperado {c['local']['rango_esperado_p10_p90']}")
        print(f"  Visitante -> media {c['visitante']['media']:.1f}, rango esperado {c['visitante']['rango_esperado_p10_p90']}")
        print(f"  Total     -> media {c['total']['media']:.1f}, rango esperado {c['total']['rango_esperado_p10_p90']}")
        for line, probs in report["over_under_corners"].items():
            print(f"  Over/Under {line}: Over {probs['over']*100:5.1f}%  Under {probs['under']*100:5.1f}%")

    print("\n[Tiros al arco]")
    t = report["tiros_al_arco"]
    if t is None:
        print("  Sin datos suficientes (la fuente de histórico usada no trae tiros al arco).")
    else:
        print(f"  Local     -> media {t['local']['media']:.1f}, rango esperado {t['local']['rango_esperado_p10_p90']}")
        print(f"  Visitante -> media {t['visitante']['media']:.1f}, rango esperado {t['visitante']['rango_esperado_p10_p90']}")

    print("\n[Tarjetas amarillas]")
    ta = report["tarjetas"]
    if ta is None:
        print("  Sin datos suficientes (la fuente de histórico usada no trae tarjetas).")
    else:
        print(f"  Local     -> media {ta['amarillas_local']['media']:.1f}")
        print(f"  Visitante -> media {ta['amarillas_visitante']['media']:.1f}")
        print(f"  Rojas (local/visitante) -> media {ta['rojas_local']['media']:.2f} / {ta['rojas_visitante']['media']:.2f}")

    if value_bets:
        print("\n[VALUE BETS DETECTADOS]")
        for vb in value_bets:
            if vb["value_bet"]:
                print(f"  ★ {vb['mercado']} -> {vb['resultado']}: modelo {vb['probabilidad_modelo']*100:.1f}% "
                      f"vs implícita {vb['probabilidad_implicita']*100:.1f}% (cuota {vb['cuota']}, edge {vb['edge']*100:.1f}pp)")
        if not any(vb["value_bet"] for vb in value_bets):
            print("  Ninguno por encima del umbral configurado.")

    print("\n" + "=" * 70)
    print("  DISCLAIMER: modelo probabilístico basado en datos históricos.")
    print("  No garantiza resultados. Las apuestas implican riesgo real de")
    print("  pérdida de dinero. Ver README.md para limitaciones del modelo.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Predictor probabilístico de fútbol")
    parser.add_argument("--local", required=True, help="Equipo local")
    parser.add_argument("--visitante", required=True, help="Equipo visitante")
    parser.add_argument("--liga", default=None, help="Liga o torneo (opcional, filtra el histórico)")
    parser.add_argument("--fecha", default=None, help="Fecha del partido (informativo, YYYY-MM-DD)")
    parser.add_argument("--datos", default="examples/historico_ejemplo.csv", help="Ruta al CSV/JSON de histórico")
    parser.add_argument("--cuotas", default=None, help="Ruta a JSON de cuotas (opcional)")
    parser.add_argument("--config", default="config.yaml", help="Ruta a config.yaml")
    parser.add_argument("--json-out", default=None, help="Ruta para volcar el reporte en JSON")
    parser.add_argument("--html-out", default=None, help="Ruta para generar un reporte HTML autocontenido (ideal para abrir desde el celular)")
    parser.add_argument("--ajuste-local", type=float, default=0.0,
                         help="Ajuste manual (%% como decimal, ej. -0.15) sobre los goles esperados del local. "
                              "Úsalo para reflejar bajas/lesiones/rotación que el modelo no puede inferir de datos históricos.")
    parser.add_argument("--ajuste-visitante", type=float, default=0.0,
                         help="Mismo ajuste manual pero para el visitante.")
    args = parser.parse_args()

    config = load_config(args.config)

    connector = CSVConnector(matches_path=args.datos, odds_path=args.cuotas)
    matches, cleaning_report = load_from_connector(connector, liga=args.liga)

    if cleaning_report.filas_descartadas_incompletas > 0:
        print(f"[aviso] Se descartaron {cleaning_report.filas_descartadas_incompletas} filas incompletas del histórico.")
    if matches.empty:
        print("[error] No hay partidos históricos disponibles tras la limpieza. No se puede predecir.")
        sys.exit(1)

    ratings_cfg = RatingsConfig(
        k_factor=config["ratings"]["elo_k_factor"],
        initial_rating=config["ratings"]["elo_initial"],
        home_advantage=config["ratings"]["home_advantage_elo"],
        use_goal_diff_multiplier=config["ratings"]["goal_diff_multiplier"],
    )
    elo_system = EloRatingSystem(ratings_cfg)
    elo_system.replay_history(matches)

    goals_cfg = GoalsModelConfig(
        xi=config["goals_model"]["dixon_coles_xi"],
        max_goals=config["goals_model"]["max_goals"],
        low_score_correction=config["goals_model"]["low_score_correction"],
    )
    goals_model = DixonColesModel(goals_cfg).fit(matches)

    corners_model = CornersModel(CornersModelConfig(
        half_life_days=config["recency"]["half_life_days"],
        opponent_strength_weight=config["opponent_strength"]["strength_weight"],
    ))
    shots_model = ShotsOnTargetModel(ShotsModelConfig(
        half_life_days=config["recency"]["half_life_days"],
        opponent_strength_weight=config["opponent_strength"]["strength_weight"],
    ))
    cards_model = CardsModel(CardsModelConfig(
        half_life_days=config["recency"]["half_life_days"],
        opponent_strength_weight=min(0.3, config["opponent_strength"]["strength_weight"]),
    ))

    sim_config = SimulationConfig(
        n_iterations=config["simulation"]["n_iterations"],
        random_seed=config["simulation"]["random_seed"],
    )
    simulator = MatchSimulator(goals_model, corners_model, shots_model, cards_model, sim_config)
    sim_result = simulator.simulate(
        matches, args.local, args.visitante, elo_system.ratings,
        home_adjustment=args.ajuste_local, away_adjustment=args.ajuste_visitante,
    )

    report = build_report(
        args.local, args.visitante, goals_model, sim_result, config,
        home_adjustment=args.ajuste_local, away_adjustment=args.ajuste_visitante,
        matches_df=matches,
    )

    value_bets = []
    if args.cuotas:
        odds_path = Path(args.cuotas)
        if odds_path.exists():
            with open(odds_path, "r", encoding="utf-8") as f:
                odds_json = json.load(f)
            vb_config = ValueBetsConfig(
                min_edge=config["value_bets"]["min_edge"],
                remove_overround=config["value_bets"]["remove_overround"],
            )
            value_bets = evaluate_all_markets(report, odds_json, vb_config)
        else:
            print(f"[aviso] No se encontró el archivo de cuotas: {args.cuotas}")

    print_human_report(report, value_bets)

    if args.json_out:
        output = {"reporte": report, "value_bets": value_bets}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n[info] Reporte JSON guardado en: {args.json_out}")

    if args.html_out:
        html_content = render_html_report(report, value_bets)
        with open(args.html_out, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[info] Reporte HTML guardado en: {args.html_out} (ábrelo en el navegador de tu celular)")


if __name__ == "__main__":
    main()
