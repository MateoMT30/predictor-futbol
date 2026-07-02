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

import numpy as np
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
    avisos=None,
) -> dict:
    """
    matches_df (opcional): el histórico usado para ajustar los modelos. Se
    usa para detectar ausencia de datos en dos niveles:

      1. Fuente completa: si una columna (córners/tiros/tarjetas) está
         enteramente vacía (ej. football-data.org gratuito no las trae),
         el mercado entero se marca como no disponible.
      2. POR EQUIPO: aunque la fuente sí traiga la columna, un equipo
         puntual puede no tener ni una fila con dato (típico: recién
         ascendido que no jugó esta competición el último año). Antes esto
         producía un falso "media 0.0, rango 0-0" que parecía pronóstico
         real; ahora ese lado se marca None ("sin datos") y los mercados
         que combinan a ambos equipos (total, over/under) se suprimen,
         porque un total calculado con la mitad del partido es basura.

    avisos (opcional): lista de mensajes generados aguas arriba (ej. "se
    usó historial de otra liga para X") que el reporte muestra en un
    banner. Acá se agregan además los avisos de muestra insuficiente.
    """
    goals_report = goals_model.market_probabilities(home, away, home_adjustment, away_adjustment)
    handicap_lines = config["markets"]["handicap_lines"]
    goals_report["handicap"] = goals_model.handicap_probabilities(
        home, away, handicap_lines, home_adjustment, away_adjustment
    )

    # Top 3 de marcadores exactos (no solo el #1): mostrar un único marcador
    # confunde cuando el más probable es un empate pero el 1X2 favorece a un
    # equipo — el 1-1 compite SOLO por el empate mientras la victoria reparte
    # su probabilidad entre muchos marcadores (1-0, 2-0, 2-1...). Con el top 3
    # se ve que la diferencia entre ellos suele ser de decimales.
    matrix = goals_model.score_matrix(home, away, home_adjustment, away_adjustment)
    # kind="stable" sobre el negativo: ante probabilidades empatadas gana la
    # celda de índice menor — el mismo desempate que el argmax de
    # marcador_mas_probable, para que el #1 del top coincida siempre con él.
    top_idx = np.argsort(-matrix, axis=None, kind="stable")[:3]
    marcadores_probables = [
        {"local": int(i // matrix.shape[1]), "visitante": int(i % matrix.shape[1]),
         "probabilidad": float(matrix.flat[i])}
        for i in top_idx
    ]

    # Marcador CONDICIONAL al resultado más probable ("si gana X, ¿qué
    # marcador logra?"). Pedido del usuario: el marcador más probable a
    # secas es conservador (gana casi siempre un 1-0/1-1 de probabilidad
    # bajita); esto muestra la tendencia del ganador: el marcador más
    # típico DENTRO de sus victorias (o de los empates, si el empate es
    # lo más probable), con su peso relativo dentro de ese escenario.
    x1x2 = goals_report["1x2"]
    pick_1x2 = max(x1x2, key=x1x2.get)
    rows_idx, cols_idx = np.indices(matrix.shape)
    region = {"local": rows_idx > cols_idx, "empate": rows_idx == cols_idx,
              "visitante": rows_idx < cols_idx}[pick_1x2]
    masked = np.where(region, matrix, -1.0)
    ci, cj = np.unravel_index(np.argmax(masked), matrix.shape)
    region_total = float(matrix[region].sum())
    marcador_condicional = {
        "resultado": pick_1x2,
        "local": int(ci), "visitante": int(cj),
        "probabilidad": float(matrix[ci, cj]),
        "prob_dentro_escenario": float(matrix[ci, cj] / region_total) if region_total > 0 else 0.0,
    }

    def _forma_reciente(team, n_partidos=5):
        """Racha de los últimos partidos del equipo (más reciente primero):
        'G'/'E'/'P' + goles a favor y en contra. Es la "tendencia" visible:
        el rendimiento reciente sobre el que se apoya el pronóstico (los
        modelos ya ponderan más lo reciente; esto lo hace transparente)."""
        if matches_df is None:
            return None
        sub = matches_df[(matches_df["equipo_local"] == team)
                         | (matches_df["equipo_visitante"] == team)]
        sub = sub.sort_values("fecha").tail(n_partidos)
        racha, gf, gc = [], 0, 0
        for row in sub.itertuples():
            mine, theirs = ((row.goles_local, row.goles_visitante)
                            if row.equipo_local == team
                            else (row.goles_visitante, row.goles_local))
            racha.append("G" if mine > theirs else ("E" if mine == theirs else "P"))
            gf += int(mine)
            gc += int(theirs)
        if not racha:
            return None
        racha.reverse()  # más reciente primero
        return {"racha": racha, "gf": gf, "gc": gc, "n": len(racha)}

    def has_data(*columns) -> bool:
        if matches_df is None:
            return True
        return any(col in matches_df.columns and matches_df[col].notna().any() for col in columns)

    def team_sample(team: str, role: str, *columns) -> int:
        """Cuántos partidos de `team` en su rol (local/visitante) traen dato
        en alguna de `columns`. None si no hay matches_df (CLI viejo/tests):
        en ese caso no se puede auditar y se asume que hay datos."""
        if matches_df is None:
            return None
        col_team = "equipo_local" if role == "local" else "equipo_visitante"
        cols = [c for c in columns if c in matches_df.columns]
        if not cols:
            return 0
        subset = matches_df[matches_df[col_team] == team]
        return int(subset[cols].notna().any(axis=1).sum())

    def side_ok(n) -> bool:
        return n is None or n > 0

    has_corners = has_data("corners_local", "corners_visitante")
    has_shots = has_data("tiros_arco_local", "tiros_arco_visitante")
    has_cards = has_data("tarjetas_amarillas_local", "tarjetas_amarillas_visitante",
                          "tarjetas_rojas_local", "tarjetas_rojas_visitante")

    n_corners = {"local": team_sample(home, "local", "corners_local"),
                 "visitante": team_sample(away, "visitante", "corners_visitante")}
    n_shots = {"local": team_sample(home, "local", "tiros_arco_local"),
               "visitante": team_sample(away, "visitante", "tiros_arco_visitante")}
    n_cards = {"local": team_sample(home, "local", "tarjetas_amarillas_local", "tarjetas_rojas_local"),
               "visitante": team_sample(away, "visitante", "tarjetas_amarillas_visitante", "tarjetas_rojas_visitante")}

    corners_totals = sim_result.corners_home + sim_result.corners_away
    shots_totals = sim_result.shots_home + sim_result.shots_away
    yellow_totals = sim_result.yellow_home + sim_result.yellow_away
    red_totals = sim_result.red_home + sim_result.red_away

    # Avisos de calidad de muestra: incluyen el caso extremo (equipo sin
    # NINGÚN partido en el histórico: goles/1X2 lo tratan como equipo
    # promedio de la liga — ver DixonColesModel) para que el usuario sepa
    # que ese pronóstico es más débil de lo normal.
    avisos = list(avisos or [])
    if matches_df is not None:
        for team in (home, away):
            n_total = int(((matches_df["equipo_local"] == team)
                           | (matches_df["equipo_visitante"] == team)).sum())
            if n_total == 0:
                avisos.append(
                    f"{team} no tiene ningún partido en el histórico usado: el 1X2 y los goles "
                    f"lo tratan como un equipo promedio de la competición, y los mercados de "
                    f"córners/tiros/tarjetas de ese lado quedan sin datos."
                )
            elif n_total < 5:
                avisos.append(
                    f"{team} tiene solo {n_total} partido(s) en el histórico usado — "
                    f"pronóstico con muestra muy chica, tómalo con pinzas."
                )

    # Probabilidad de CLASIFICAR en eliminación directa: el 1X2 modela los
    # 90 minutos (igual que las casas de apuestas), pero en muerte súbita el
    # "empate" significa alargue/penales. Se traduce repartiendo la
    # probabilidad del empate proporcional a la fuerza relativa sin empate
    # (P(local)/(P(local)+P(visitante))): el alargue lo sigue jugando el
    # más fuerte, aunque con más azar — un reparto 50/50 ignoraría la
    # diferencia de nivel y uno 100/0 la exageraría. La tarjeta solo se
    # muestra en torneos (report["es_eliminatoria"], lo marca app.py).
    p_l, p_e, p_v = (goals_report["1x2"]["local"], goals_report["1x2"]["empate"],
                     goals_report["1x2"]["visitante"])
    rel = p_l / (p_l + p_v) if (p_l + p_v) > 0 else 0.5
    clasificacion = {"local": p_l + p_e * rel, "visitante": p_v + p_e * (1 - rel)}

    report = {
        "avisos": avisos,
        "clasificacion_eliminatoria": clasificacion,
        "partido": {"local": home, "visitante": away},
        "1x2": goals_report["1x2"],
        "handicap": goals_report["handicap"],
        "ambos_anotan": goals_report["ambos_anotan"],
        "goles_esperados": goals_report["goles_esperados"],
        "marcador_mas_probable": goals_report["marcador_mas_probable"],
        "marcadores_probables": marcadores_probables,
        "marcador_condicional": marcador_condicional,
        "forma": {"local": _forma_reciente(home), "visitante": _forma_reciente(away)},
        "ajuste_manual_aplicado": goals_report["ajuste_manual_aplicado"],
        "over_under_goles": {
            line: over_under_probability(sim_result.goals_home + sim_result.goals_away, line)
            for line in config["markets"]["over_under_lines"]["goals"]
        },
        "corners": {
            "local": summarize_distribution(sim_result.corners_home) if side_ok(n_corners["local"]) else None,
            "visitante": summarize_distribution(sim_result.corners_away) if side_ok(n_corners["visitante"]) else None,
            "total": summarize_distribution(corners_totals)
                     if side_ok(n_corners["local"]) and side_ok(n_corners["visitante"]) else None,
            "muestras": n_corners,
        } if has_corners else None,
        "over_under_corners": {
            line: over_under_probability(corners_totals, line)
            for line in config["markets"]["over_under_lines"]["corners"]
        } if has_corners and side_ok(n_corners["local"]) and side_ok(n_corners["visitante"]) else None,
        "tiros_al_arco": {
            "local": summarize_distribution(sim_result.shots_home) if side_ok(n_shots["local"]) else None,
            "visitante": summarize_distribution(sim_result.shots_away) if side_ok(n_shots["visitante"]) else None,
            "total": summarize_distribution(shots_totals)
                     if side_ok(n_shots["local"]) and side_ok(n_shots["visitante"]) else None,
            "muestras": n_shots,
        } if has_shots else None,
        "over_under_tiros": {
            line: over_under_probability(shots_totals, line)
            for line in config["markets"]["over_under_lines"]["shots_on_target"]
        } if has_shots and side_ok(n_shots["local"]) and side_ok(n_shots["visitante"]) else None,
        "tarjetas": {
            "amarillas_local": summarize_distribution(sim_result.yellow_home) if side_ok(n_cards["local"]) else None,
            "amarillas_visitante": summarize_distribution(sim_result.yellow_away) if side_ok(n_cards["visitante"]) else None,
            "amarillas_total": summarize_distribution(yellow_totals)
                               if side_ok(n_cards["local"]) and side_ok(n_cards["visitante"]) else None,
            "rojas_local": summarize_distribution(sim_result.red_home) if side_ok(n_cards["local"]) else None,
            "rojas_visitante": summarize_distribution(sim_result.red_away) if side_ok(n_cards["visitante"]) else None,
            "muestras": n_cards,
        } if has_cards else None,
        "over_under_tarjetas": {
            line: over_under_probability(yellow_totals, line)
            for line in config["markets"]["over_under_lines"]["cards"]
        } if has_cards and side_ok(n_cards["local"]) and side_ok(n_cards["visitante"]) else None,
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

    def fmt_side(summary, decimals=1):
        # None = ese equipo no tiene datos de esta estadística en el histórico
        # (distinto de que la fuente entera no la traiga).
        if summary is None:
            return "sin datos de este equipo"
        return f"media {summary['media']:.{decimals}f}, rango esperado {summary['rango_esperado_p10_p90']}"

    for aviso in report.get("avisos", []):
        print(f"\n[AVISO] {aviso}")

    print("\n[Córners]")
    c = report["corners"]
    if c is None:
        print("  Sin datos suficientes (la fuente de histórico usada no trae córners).")
    else:
        print(f"  Local     -> {fmt_side(c['local'])}")
        print(f"  Visitante -> {fmt_side(c['visitante'])}")
        print(f"  Total     -> {fmt_side(c['total'])}")
        if report["over_under_corners"]:
            for line, probs in report["over_under_corners"].items():
                print(f"  Over/Under {line}: Over {probs['over']*100:5.1f}%  Under {probs['under']*100:5.1f}%")

    print("\n[Tiros al arco]")
    t = report["tiros_al_arco"]
    if t is None:
        print("  Sin datos suficientes (la fuente de histórico usada no trae tiros al arco).")
    else:
        print(f"  Local     -> {fmt_side(t['local'])}")
        print(f"  Visitante -> {fmt_side(t['visitante'])}")

    print("\n[Tarjetas amarillas]")
    ta = report["tarjetas"]
    if ta is None:
        print("  Sin datos suficientes (la fuente de histórico usada no trae tarjetas).")
    else:
        print(f"  Local     -> {fmt_side(ta['amarillas_local'])}")
        print(f"  Visitante -> {fmt_side(ta['amarillas_visitante'])}")
        rl = ta["rojas_local"]
        rv = ta["rojas_visitante"]
        rl_str = f"{rl['media']:.2f}" if rl else "sin datos"
        rv_str = f"{rv['media']:.2f}" if rv else "sin datos"
        print(f"  Rojas (local/visitante) -> media {rl_str} / {rv_str}")

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
