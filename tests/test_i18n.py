from datetime import datetime, timezone

from src.i18n import team_name_es, to_colombia_time


def test_translates_known_team():
    assert team_name_es("Belgium") == "Bélgica"
    assert team_name_es("Spain") == "España"


def test_returns_original_for_unknown_team():
    assert team_name_es("Wakanda") == "Wakanda"


def test_converts_utc_to_colombia_time():
    utc_dt = datetime(2026, 8, 10, 18, 0, tzinfo=timezone.utc)
    co_dt = to_colombia_time(utc_dt)
    # Colombia es UTC-5 todo el año (sin horario de verano)
    assert co_dt.hour == 13


def test_assumes_utc_when_no_tzinfo():
    naive_dt = datetime(2026, 8, 10, 18, 0)
    co_dt = to_colombia_time(naive_dt)
    assert co_dt.hour == 13
