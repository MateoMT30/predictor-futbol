from datetime import datetime, timezone

from src.i18n import team_name_es, to_colombia_time, day_label_es


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


def test_day_label_today_tomorrow_yesterday():
    ref = datetime(2026, 7, 1, 10, 0)
    assert day_label_es(datetime(2026, 7, 1, 20, 0), reference=ref) == "Hoy"
    assert day_label_es(datetime(2026, 7, 2, 9, 0), reference=ref) == "Mañana"
    assert day_label_es(datetime(2026, 6, 30, 23, 0), reference=ref) == "Ayer"


def test_day_label_far_date_shows_weekday():
    ref = datetime(2026, 7, 1, 10, 0)
    label = day_label_es(datetime(2026, 7, 8, 20, 0), reference=ref)  # miércoles siguiente
    assert "8 jul" in label
    assert any(dia in label for dia in ["Miércoles"])
