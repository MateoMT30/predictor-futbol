from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.connectors.football_data_connector import FootballDataConnector, _cache


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


def _mock_response(payload):
    mock = MagicMock()
    mock.json.return_value = payload
    mock.raise_for_status.return_value = None
    return mock


def test_requires_api_key():
    with pytest.raises(ValueError):
        FootballDataConnector(api_key=None)


def test_fetch_matches_parses_finished_games():
    connector = FootballDataConnector(api_key="fake-key")
    payload = {
        "competition": {"name": "Premier League"},
        "matches": [
            {
                "utcDate": "2026-05-01T15:00:00Z",
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team B"},
                "score": {"fullTime": {"home": 2, "away": 1}},
            },
            {
                # partido sin marcador (no debería incluirse)
                "utcDate": "2026-05-02T15:00:00Z",
                "homeTeam": {"name": "Team C"},
                "awayTeam": {"name": "Team D"},
                "score": {"fullTime": {"home": None, "away": None}},
            },
        ],
    }
    with patch("requests.get", return_value=_mock_response(payload)) as mock_get:
        df = connector.fetch_matches(liga="PL")
    assert len(df) == 1
    assert df.iloc[0]["equipo_local"] == "Team A"
    assert df.iloc[0]["goles_local"] == 2
    assert df.iloc[0]["corners_local"] is None
    mock_get.assert_called_once()


def test_fetch_matches_uses_cache_on_second_call():
    connector = FootballDataConnector(api_key="fake-key")
    payload = {"competition": {"name": "PL"}, "matches": []}
    with patch("requests.get", return_value=_mock_response(payload)) as mock_get:
        connector.fetch_matches(liga="PL")
        connector.fetch_matches(liga="PL")
    assert mock_get.call_count == 1  # la segunda llamada debe salir de cache


def test_fetch_upcoming_parses_scheduled_games():
    connector = FootballDataConnector(api_key="fake-key")
    payload = {
        "competition": {"name": "Premier League"},
        "matches": [
            {
                "utcDate": "2026-08-10T18:00:00Z",
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team B"},
            }
        ],
    }
    with patch("requests.get", return_value=_mock_response(payload)):
        df = connector.fetch_upcoming(liga="PL")
    assert len(df) == 1
    assert df.iloc[0]["equipo_local"] == "Team A"
    assert isinstance(df.iloc[0]["fecha_hora"], pd.Timestamp)


def test_fetch_matches_requires_liga():
    connector = FootballDataConnector(api_key="fake-key")
    with pytest.raises(ValueError):
        connector.fetch_matches(liga=None)
