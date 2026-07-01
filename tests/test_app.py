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
        {"fecha_hora": pd.Timestamp("2026-08-10 18:00"), "liga": "Premier League",
         "equipo_local": "Team A", "equipo_visitante": "Team B"},
    ])
    mock_connector = MagicMock()
    mock_connector.fetch_upcoming.return_value = fake_df
    with patch.object(app_module, "FootballDataConnector", return_value=mock_connector):
        client = app.test_client()
        res = client.get("/partidos?competition=PL")
    assert res.status_code == 200
    assert b"Team A" in res.data
    assert b"Team B" in res.data
    assert "(local)".encode() in res.data
    assert "(visitante)".encode() in res.data


def test_predecir_without_api_key_returns_500():
    client = app.test_client()
    res = client.get("/predecir?competition=PL&local=A&visitante=B")
    assert res.status_code == 500
