"""
Tests del conector de resultados internacionales (dataset martj42) y su
integración con la ventana adaptativa: es la fuente que le da muestra real
a las selecciones (la API .org solo trae el torneo en sí).
"""

from unittest.mock import MagicMock, patch

import pandas as pd

import app as app_module
from src.connectors import international_results_connector as intl_mod
from src.connectors.international_results_connector import (
    align_team_names,
    fetch_international_results,
    normalize_team_name,
)


def _fake_raw_csv_df():
    return pd.DataFrame({
        "date": ["2025-03-20", "2025-06-07", "2026-06-15", "1999-01-01", "2026-07-04"],
        "home_team": ["Spain", "DR Congo", "Spain", "Spain", "Canada"],
        "away_team": ["Netherlands", "Spain", "Austria", "France", "Morocco"],
        "home_score": [3, 0, 2, 1, None],   # None = partido futuro sin jugar
        "away_score": [3, 2, 0, 1, None],
        "tournament": ["UEFA Nations League", "Friendly", "FIFA World Cup", "Friendly", "FIFA World Cup"],
        "city": ["Valencia", "Kinshasa", "Boston", "Paris", "Houston"],
        "country": ["Spain", "DR Congo", "United States", "France", "United States"],
        "neutral": [False, False, True, False, True],
    })


def test_normalize_team_name_cruza_variantes_entre_fuentes():
    assert normalize_team_name("DR Congo") == normalize_team_name("Congo DR")
    assert normalize_team_name("Bosnia and Herzegovina") == normalize_team_name("Bosnia-Herzegovina")
    assert normalize_team_name("Spain") != normalize_team_name("Portugal")


def test_fetch_filtra_por_fecha_y_descarta_no_jugados(monkeypatch):
    monkeypatch.setattr(intl_mod, "_download", lambda: _fake_raw_csv_df())
    df = fetch_international_results(desde="2024-01-01")
    # El de 1999 queda fuera por fecha; el de julio 2026 por no tener marcador
    assert len(df) == 3
    assert set(df.columns) >= {"fecha", "equipo_local", "equipo_visitante", "goles_local", "goles_visitante"}
    assert (df["fecha"] >= pd.Timestamp("2024-01-01")).all()


def test_align_team_names_renombra_a_la_convencion_de_la_api():
    df = pd.DataFrame({
        "fecha": [pd.Timestamp("2025-06-07")],
        "equipo_local": ["DR Congo"],
        "equipo_visitante": ["Bosnia and Herzegovina"],
        "goles_local": [1], "goles_visitante": [0],
    })
    out = align_team_names(df, reference_names={"Congo DR", "Bosnia-Herzegovina", "Spain"})
    assert out.iloc[0]["equipo_local"] == "Congo DR"
    assert out.iloc[0]["equipo_visitante"] == "Bosnia-Herzegovina"


def test_ensure_min_sample_usa_dataset_internacional_para_selecciones(monkeypatch):
    # Histórico .org: solo 3 partidos del torneo por equipo (el caso real)
    org = pd.DataFrame([
        {"fecha": f"2026-06-{d}", "equipo_local": "Spain", "equipo_visitante": "Austria",
         "goles_local": 1, "goles_visitante": 0} for d in (11, 15, 19)
    ])
    # Dataset internacional: mucha más muestra para ambos
    intl_rows = []
    for i in range(16):
        intl_rows.append({
            "fecha": pd.Timestamp(f"2025-{(i % 12) + 1:02d}-{5 + i // 12:02d}"),
            "liga": "Friendly",
            "equipo_local": "Spain" if i % 2 == 0 else "Austria",
            "equipo_visitante": "France",
            "goles_local": 2, "goles_visitante": 1,
        })
    monkeypatch.setattr(app_module, "fetch_international_results",
                        lambda desde: pd.DataFrame(intl_rows))

    avisos = []
    with patch.object(app_module, "load_from_connector") as org_loader:
        result = app_module._ensure_min_sample(
            MagicMock(), "WC", org, "Spain", "Austria", "2026-07-02", avisos)

    assert app_module._team_match_count(result, "Spain") >= 10
    assert app_module._team_match_count(result, "Austria") >= 10
    assert any("clasificatorias y" in a and "amistosos" in a for a in avisos)
    # No hizo falta recargar la API .org con ventanas más largas
    org_loader.assert_not_called()


def test_partidos_wc_llena_clasificatorias_desde_dataset_internacional(monkeypatch):
    """Si la API .org no trae partidos previos (caso selecciones), la sección
    de clasificatorias se llena con el dataset internacional, con los nombres
    alineados a la convención de la API y una nota de coherencia con el modelo."""
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    agenda_df = pd.DataFrame([
        {"fecha_hora": pd.Timestamp("2026-06-11 18:00"), "liga": "Copa Mundial",
         "equipo_local": "Spain", "equipo_visitante": "Congo DR",
         "finalizado": True, "goles_local": 2, "goles_visitante": 0},
    ])
    intl = pd.DataFrame([
        {"fecha": pd.Timestamp("2026-03-28"), "liga": "FIFA World Cup qualification",
         "equipo_local": "Spain", "equipo_visitante": "Norway",
         "goles_local": 3, "goles_visitante": 0},
        # Nombre en la convención del dataset: debe renombrarse a "Congo DR"
        {"fecha": pd.Timestamp("2026-03-20"), "liga": "FIFA World Cup qualification",
         "equipo_local": "DR Congo", "equipo_visitante": "Zambia",
         "goles_local": 1, "goles_visitante": 0},
    ])
    mock_connector = MagicMock()
    mock_connector.fetch_agenda.return_value = agenda_df
    mock_connector.fetch_matches.return_value = pd.DataFrame()  # .org sin previos
    mock_connector.fetch_standings.side_effect = Exception("sin tabla")
    mock_connector.fetch_scorers.side_effect = Exception("sin goleadores")
    monkeypatch.setattr(app_module, "fetch_international_results", lambda desde: intl)

    from app import app
    with patch.object(app_module, "FootballDataConnector", return_value=mock_connector):
        res = app.test_client().get("/partidos?competition=WC")
    html = res.data.decode("utf-8")
    assert "Clasificatorias y partidos previos al torneo" in html
    assert "3 - 0" in html
    # Nombre renombrado a la convención .org y traducido para mostrar
    assert "República Democrática del Congo" in html
    # Nota de coherencia con el modelo
    assert "parte de la muestra" in html


def test_fetch_acepta_fecha_con_zona_horaria(monkeypatch):
    """Regresión del bug de producción: app.py pasa datetime.now(timezone.utc)
    (tz-aware) y el dataset trae fechas naive — la comparación lanzaba
    TypeError que un try/except aguas arriba silenciaba, desactivando la
    fuente entera sin que nadie lo viera."""
    from datetime import datetime, timedelta, timezone
    monkeypatch.setattr(intl_mod, "_download", lambda: _fake_raw_csv_df())
    df = fetch_international_results(desde=datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert len(df) == 3
