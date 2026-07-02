"""Tests del conector football-data.co.uk (córners/tiros/tarjetas por CSV)."""

from unittest.mock import patch

import pandas as pd

from src.connectors import football_couk_connector as C


SAMPLE_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HST,AST,HC,AC,HY,AY,HR,AR\n"
    "E0,10/05/2026,Man City,Arsenal,2,1,7,3,8,4,1,2,0,1\n"
    "E0,11/05/2026,Liverpool,Chelsea,0,0,5,5,6,6,0,0,0,0\n"
)


class _FakeResp:
    def __init__(self, text, status=200):
        self.content = text.encode("latin-1")
        self.status_code = status

    def raise_for_status(self):
        pass


def _fake_get(url, **kwargs):
    # Una temporada trae datos, la otra (previa) 404: se debe tolerar.
    if "/2526/" in url:
        return _FakeResp(SAMPLE_CSV)
    return _FakeResp("", status=404)


def _clear_cache():
    C._csv_cache.clear()


def test_enrich_matches_by_date_and_score():
    _clear_cache()
    # Histórico estilo football-data.org: nombres DISTINTOS a co.uk, pero misma
    # fecha y marcador -> debe cruzar igual.
    df = pd.DataFrame([
        {"fecha": pd.Timestamp("2026-05-10"), "equipo_local": "Manchester City FC",
         "equipo_visitante": "Arsenal FC", "goles_local": 2, "goles_visitante": 1,
         "corners_local": None, "corners_visitante": None,
         "tiros_arco_local": None, "tiros_arco_visitante": None,
         "tarjetas_amarillas_local": None, "tarjetas_amarillas_visitante": None,
         "tarjetas_rojas_local": None, "tarjetas_rojas_visitante": None},
    ])
    with patch.object(C.requests, "get", side_effect=_fake_get):
        out = C.enrich_with_couk_stats(df, "PL")
    row = out.iloc[0]
    assert row["corners_local"] == 8 and row["corners_visitante"] == 4
    assert row["tiros_arco_local"] == 7 and row["tiros_arco_visitante"] == 3
    assert row["tarjetas_amarillas_local"] == 1 and row["tarjetas_amarillas_visitante"] == 2
    assert row["tarjetas_rojas_visitante"] == 1


def test_non_covered_competition_is_noop():
    _clear_cache()
    df = pd.DataFrame([
        {"fecha": pd.Timestamp("2026-05-10"), "equipo_local": "A", "equipo_visitante": "B",
         "goles_local": 1, "goles_visitante": 0, "corners_local": None, "corners_visitante": None},
    ])
    # WC no está en COUK_LEAGUE_CODES -> no debe siquiera intentar descargar.
    with patch.object(C.requests, "get", side_effect=AssertionError("no debe descargar")):
        out = C.enrich_with_couk_stats(df, "WC")
    assert out.iloc[0]["corners_local"] is None


def test_ambiguous_same_day_same_score_is_skipped():
    _clear_cache()
    # Dos partidos el mismo día con el mismo marcador exacto -> ambiguo, no se
    # asigna a ninguno (no se adivina).
    df = pd.DataFrame([
        {"fecha": pd.Timestamp("2026-05-10"), "equipo_local": "X", "equipo_visitante": "Y",
         "goles_local": 2, "goles_visitante": 1, "corners_local": None, "corners_visitante": None,
         "tiros_arco_local": None, "tiros_arco_visitante": None,
         "tarjetas_amarillas_local": None, "tarjetas_amarillas_visitante": None,
         "tarjetas_rojas_local": None, "tarjetas_rojas_visitante": None},
        {"fecha": pd.Timestamp("2026-05-10"), "equipo_local": "Z", "equipo_visitante": "W",
         "goles_local": 2, "goles_visitante": 1, "corners_local": None, "corners_visitante": None,
         "tiros_arco_local": None, "tiros_arco_visitante": None,
         "tarjetas_amarillas_local": None, "tarjetas_amarillas_visitante": None,
         "tarjetas_rojas_local": None, "tarjetas_rojas_visitante": None},
    ])
    with patch.object(C.requests, "get", side_effect=_fake_get):
        out = C.enrich_with_couk_stats(df, "PL")
    assert out["corners_local"].isna().all()


def test_season_codes_format():
    from datetime import datetime, timezone
    codes = C._season_codes(datetime(2026, 7, 1, tzinfo=timezone.utc))
    assert codes == ["2627", "2526"]
    codes2 = C._season_codes(datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert codes2 == ["2526", "2425"]
