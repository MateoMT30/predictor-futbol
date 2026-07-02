from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import app as app_module
from app import app


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    yield


def test_index_loads():
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    assert b"Predictor F" in res.data


def test_index_warns_when_no_api_key():
    client = app.test_client()
    res = client.get("/")
    assert "FOOTBALL_DATA_API_KEY".encode() in res.data


def test_predecir_manual_returns_report():
    client = app.test_client()
    res = client.post("/predecir_manual", data={
        "local": "Colombia", "visitante": "Argentina", "liga": "Liga Ejemplo",
    })
    assert res.status_code == 200
    assert "Colombia vs Argentina".encode("utf-8") in res.data
    # No debe pedir ni mostrar cuotas/value bets en el flujo por defecto
    assert b"Value bets" not in res.data


def test_predecir_manual_missing_team_shows_error():
    client = app.test_client()
    res = client.post("/predecir_manual", data={"local": "", "visitante": "Argentina"})
    assert res.status_code == 200
    assert "Debes indicar".encode("utf-8") in res.data


def test_predecir_manual_applies_adjustment():
    client = app.test_client()
    res = client.post("/predecir_manual", data={
        "local": "Colombia", "visitante": "Argentina", "liga": "Liga Ejemplo",
        "ajuste_local": "-20",
    })
    assert res.status_code == 200
    assert "Ajuste manual aplicado".encode("utf-8") in res.data


def test_partidos_without_api_key_shows_error():
    client = app.test_client()
    res = client.get("/partidos?competition=PL")
    assert res.status_code == 200
    assert "Falta configurar".encode("utf-8") in res.data


def test_partidos_with_mocked_connector(monkeypatch):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "fake-key")
    fake_df = pd.DataFrame([
        # Partido ya jugado (dia anterior) -> debe mostrar marcador
        {"fecha_hora": pd.Timestamp("2026-08-08 18:00"), "liga": "Premier League",
         "equipo_local": "Team C", "equipo_visitante": "Team D",
         "finalizado": True, "goles_local": 2, "goles_visitante": 1},
        # Partido proximo -> clicable
        {"fecha_hora": pd.Timestamp("2026-08-10 18:00"), "liga": "Premier League",
         "equipo_local": "Team A", "equipo_visitante": "Team B",
         "finalizado": False, "goles_local": None, "goles_visitante": None},
    ])
    mock_connector = MagicMock()
    mock_connector.fetch_agenda.return_value = fake_df
    with patch.object(app_module, "FootballDataConnector", return_value=mock_connector):
        client = app.test_client()
        res = client.get("/partidos?competition=PL")
    assert res.status_code == 200
    assert b"Team A" in res.data
    assert b"Team B" in res.data
    # Agrupado por dia (estilo apps deportivas): debe mostrar la etiqueta
    # del dia relativo, ej. el nombre del dia de la semana si es lejano.
    assert b"mr-teams" in res.data
    # El partido jugado muestra el marcador y el boton "Hoy" esta presente
    assert b"2 - 1" in res.data
    assert b"today-fab" in res.data


def test_predecir_without_api_key_returns_500():
    client = app.test_client()
    res = client.get("/predecir?competition=PL&local=A&visitante=B")
    assert res.status_code == 500


def test_partidos_torneo_muestra_clasificatorias(monkeypatch):
    """En torneos (ej. WC), los partidos previos al torneo (clasificatorias)
    aparecen en una seccion plegable agrupada por mes."""
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    agenda_df = pd.DataFrame([
        {"fecha_hora": pd.Timestamp("2026-06-11 18:00"), "liga": "Copa Mundial",
         "equipo_local": "Mexico", "equipo_visitante": "South Africa",
         "finalizado": True, "goles_local": 2, "goles_visitante": 0},
    ])
    hist_df = pd.DataFrame([
        {"fecha": "2026-03-28", "liga": "Copa Mundial",
         "equipo_local": "Colombia", "equipo_visitante": "Bolivia",
         "escudo_local": None, "escudo_visitante": None,
         "goles_local": 3, "goles_visitante": 0},
        {"fecha": "2025-11-14", "liga": "Copa Mundial",
         "equipo_local": "Italy", "equipo_visitante": "Norway",
         "escudo_local": None, "escudo_visitante": None,
         "goles_local": 1, "goles_visitante": 1},
    ])
    mock_connector = MagicMock()
    mock_connector.fetch_agenda.return_value = agenda_df
    mock_connector.fetch_matches.return_value = hist_df
    mock_connector.fetch_standings.side_effect = Exception("sin tabla")
    mock_connector.fetch_scorers.side_effect = Exception("sin goleadores")
    with patch.object(app_module, "FootballDataConnector", return_value=mock_connector):
        client = app.test_client()
        res = client.get("/partidos?competition=WC")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Clasificatorias y partidos previos al torneo" in html
    assert "Marzo 2026" in html
    assert "Noviembre 2025" in html
    assert "3 - 0" in html
    # En ligas NO se pide el historico previo (seria ruido)
    mock_connector.fetch_matches.reset_mock()
    with patch.object(app_module, "FootballDataConnector", return_value=mock_connector):
        client.get("/partidos?competition=PL")
    mock_connector.fetch_matches.assert_not_called()


def test_rendimiento_sin_backtest_muestra_mensaje():
    with patch.object(app_module, "_load_backtest", return_value={}):
        client = app.test_client()
        res = client.get("/rendimiento?competition=PL")
    assert res.status_code == 200
    assert "Todavía no hay backtest".encode("utf-8") in res.data


def test_rendimiento_con_datos_renderiza_resumen_y_partidos():
    fake = {"PL": {
        "generado_en": "2026-07-01T08:00:00+00:00",
        "n": 2, "aciertos": 1, "acierto_pct": 50.0,
        "brier": 0.59, "brier_azar": 0.6667,
        "partidos": [
            {"fecha": "2026-06-28", "local": "Arsenal FC", "visitante": "Chelsea FC",
             "marcador": "2 - 0", "prob_local": 0.55, "prob_empate": 0.25,
             "prob_visitante": 0.20, "pick": "local", "prob_pick": 0.55,
             "real": "local", "acierto": True, "brier": 0.3},
            {"fecha": "2026-06-27", "local": "Everton FC", "visitante": "Fulham FC",
             "marcador": "0 - 1", "prob_local": 0.45, "prob_empate": 0.30,
             "prob_visitante": 0.25, "pick": "local", "prob_pick": 0.45,
             "real": "visitante", "acierto": False, "brier": 0.9},
        ],
    }}
    with patch.object(app_module, "_load_backtest", return_value=fake):
        client = app.test_client()
        res = client.get("/rendimiento?competition=PL")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "1/2" in html          # aciertos
    assert "50.0%" in html        # tasa de acierto
    assert "Arsenal" in html
    assert "✅" in html and "❌" in html
    assert "Brier" in html


def test_partido_jugado_linkea_prediccion_retroactiva(monkeypatch):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    fake_df = pd.DataFrame([
        {"fecha_hora": pd.Timestamp("2026-06-28 15:00"), "liga": "Copa Mundial",
         "equipo_local": "England", "equipo_visitante": "Congo DR",
         "finalizado": True, "goles_local": 2, "goles_visitante": 1},
    ])
    mock_connector = MagicMock()
    mock_connector.fetch_agenda.return_value = fake_df
    mock_connector.fetch_matches.return_value = pd.DataFrame()
    mock_connector.fetch_standings.side_effect = Exception("sin tabla")
    mock_connector.fetch_scorers.side_effect = Exception("sin goleadores")
    with patch.object(app_module, "FootballDataConnector", return_value=mock_connector):
        client = app.test_client()
        res = client.get("/partidos?competition=WC")
    html = res.data.decode("utf-8")
    # El partido jugado ahora es un link a la prediccion retroactiva
    assert "antes_de=2026-06-28" in html
    assert 'class="match-row-v2 played' in html


def _df_partidos(rows):
    base = {"escudo_local": None, "escudo_visitante": None}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_prune_to_neighborhood_conserva_vecindario_y_descarta_lejanos():
    df = _df_partidos([
        # España y su rival directo
        {"fecha": "2026-06-01", "equipo_local": "Spain", "equipo_visitante": "Austria",
         "goles_local": 1, "goles_visitante": 1},
        # Rival del rival (2 saltos: debe conservarse)
        {"fecha": "2026-05-01", "equipo_local": "Austria", "equipo_visitante": "France",
         "goles_local": 0, "goles_visitante": 2},
        # Cluster totalmente desconectado (otra confederación): debe salir
        {"fecha": "2026-05-02", "equipo_local": "Fiji", "equipo_visitante": "Tonga",
         "goles_local": 3, "goles_visitante": 0},
    ])
    result = app_module._prune_to_neighborhood(df, {"Spain", "Austria"}, hops=2)
    equipos = set(result["equipo_local"]) | set(result["equipo_visitante"])
    assert {"Spain", "Austria", "France"} <= equipos
    assert "Fiji" not in equipos


def test_ensure_min_sample_amplia_ventana_cuando_hay_pocos_partidos():
    chico = _df_partidos([
        {"fecha": "2026-06-20", "equipo_local": "Spain", "equipo_visitante": "Austria",
         "goles_local": 2, "goles_visitante": 0},
    ])
    # Historial ampliado (2 años): más partidos de ambos equipos
    filas = []
    for i in range(12):
        filas.append({"fecha": f"2025-{(i % 12) + 1:02d}-10",
                      "equipo_local": "Spain" if i % 2 == 0 else "Austria",
                      "equipo_visitante": "France",
                      "goles_local": 1, "goles_visitante": 0})
    grande = pd.concat([chico, _df_partidos(filas)], ignore_index=True)

    avisos = []
    with patch.object(app_module, "load_from_connector", return_value=(grande, None)) as loader:
        result = app_module._ensure_min_sample(
            MagicMock(), "WC", chico, "Spain", "Austria", "2026-07-01", avisos)
    assert len(result) > len(chico)
    assert app_module._team_match_count(result, "Spain") >= 6
    assert any("Muestra ampliada" in a for a in avisos)
    # El aviso explica el control del sesgo (decaimiento temporal)
    assert any("decaimiento temporal" in a for a in avisos)


def test_ensure_min_sample_no_hace_nada_si_la_muestra_alcanza():
    filas = [{"fecha": f"2026-0{(i % 6) + 1}-15",
              "equipo_local": "Spain" if i % 2 == 0 else "Austria",
              "equipo_visitante": "France", "goles_local": 1, "goles_visitante": 1}
             for i in range(24)]
    df = _df_partidos(filas)
    with patch.object(app_module, "load_from_connector") as loader:
        result = app_module._ensure_min_sample(
            MagicMock(), "WC", df, "Spain", "Austria", "2026-07-01", [])
    loader.assert_not_called()
    assert len(result) == len(df)


def test_partidos_jugados_se_pintan_segun_acierto_del_backtest(monkeypatch):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    agenda_df = pd.DataFrame([
        {"fecha_hora": pd.Timestamp("2026-07-01 15:00"), "liga": "Copa Mundial",
         "equipo_local": "England", "equipo_visitante": "Congo DR",
         "finalizado": True, "goles_local": 2, "goles_visitante": 1},
        {"fecha_hora": pd.Timestamp("2026-06-29 15:00"), "liga": "Copa Mundial",
         "equipo_local": "Germany", "equipo_visitante": "Paraguay",
         "finalizado": True, "goles_local": 4, "goles_visitante": 5},
    ])
    fake_bt = {"WC": {"partidos": [
        {"fecha": "2026-07-01", "local": "England", "visitante": "Congo DR",
         "pick": "local", "prob_pick": 0.56, "acierto": True},
        {"fecha": "2026-06-29", "local": "Germany", "visitante": "Paraguay",
         "pick": "local", "prob_pick": 0.68, "acierto": False},
    ]}}
    mock_connector = MagicMock()
    mock_connector.fetch_agenda.return_value = agenda_df
    mock_connector.fetch_matches.return_value = pd.DataFrame()
    mock_connector.fetch_standings.side_effect = Exception()
    mock_connector.fetch_scorers.side_effect = Exception()
    monkeypatch.setattr(app_module, "fetch_international_results", lambda desde: pd.DataFrame())
    with patch.object(app_module, "_load_backtest", return_value=fake_bt), \
         patch.object(app_module, "FootballDataConnector", return_value=mock_connector):
        res = app.test_client().get("/partidos?competition=WC")
    html = res.data.decode("utf-8")
    assert "played hit" in html      # Inglaterra: acierto -> verde
    assert "played miss" in html     # Alemania: fallo -> rojo suave
    assert "El modelo dijo: gana local (56%)" in html
