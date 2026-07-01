from unittest.mock import patch

import pandas as pd
import pytest

from src.connectors.fifa_reports_connector import _extract_stats, _links_for_code, enrich_with_fifa_reports


SAMPLE_PDF_TEXT = """Mexico2 - 0
South Africa
Group A - Match 1
11 June 2026
13:00 Kick Off
Mexico City Stadium
POST MATCH SUMMARY REPORT

Match Summary - Key Statistics
Mexico South Africa
Possession
Total 57.1% 6.8% 36.1% Total
2 Goals 0
1.78 xG (Expected Goals) 0.1
16 (4) Attempts at Goal (On Target) 3 (2)

Set Plays Mexico
36
Total Set Plays
12
Total Free Kicks
0
Total Penalties
3
Total Corners
21
Total Throw Ins
Free Kicks
Type Total

Set Plays South Africa
25
Total Set Plays
13
Total Free Kicks
0
Total Penalties
1
Total Corners
11
Total Throw Ins
Free Kicks
Type Total
"""


def test_extract_stats_parses_teams_shots_and_corners():
    stats = _extract_stats(SAMPLE_PDF_TEXT)
    assert stats is not None
    assert stats["equipo_local"] == "Mexico"
    assert stats["equipo_visitante"] == "South Africa"
    assert stats["tiros_local"] == 16
    assert stats["tiros_arco_local"] == 4
    assert stats["tiros_visitante"] == 3
    assert stats["tiros_arco_visitante"] == 2
    assert stats["corners_local"] == 3
    assert stats["corners_visitante"] == 1


def test_extract_stats_returns_none_for_unrecognized_format():
    assert _extract_stats("un texto cualquiera sin el formato esperado") is None


def test_links_for_code_matches_space_and_dash_separators():
    links = [
        "https://x.com/PMSR-M01 MEX V RSA.pdf",
        "https://x.com/PMSR-M03-CAN-V-BIH-V2.pdf",
        "https://x.com/PMSR-M99-ARG-V-BRA.pdf",
    ]
    assert len(_links_for_code("MEX", links)) == 1
    assert len(_links_for_code("BIH", links)) == 1
    assert len(_links_for_code("ARG", links)) == 1
    assert len(_links_for_code("ZZZ", links)) == 0


def test_enrich_with_fifa_reports_fills_by_date_not_by_name_spelling():
    """
    Simula el caso real que motivó cruzar por fecha en vez de por nombre:
    football-data.org llama a un equipo "South Korea" pero el reporte de
    FIFA lo llama "Korea Republic" — deben cruzar igual porque comparten
    fecha de partido.
    """
    matches_df = pd.DataFrame([
        {
            "fecha": pd.Timestamp("2026-06-11"), "liga": "WC",
            "equipo_local": "Mexico", "equipo_visitante": "South Korea",
            "goles_local": 2, "goles_visitante": 1,
            "corners_local": None, "corners_visitante": None,
            "tiros_arco_local": None, "tiros_arco_visitante": None,
        }
    ])

    fake_report = {
        "fecha": "2026-06-11", "equipo_local": "Mexico", "equipo_visitante": "Korea Republic",
        "corners_local": 8, "corners_visitante": 2,
        "tiros_arco_local": 4, "tiros_arco_visitante": 2,
    }

    with patch(
        "src.connectors.fifa_reports_connector.get_match_stats_for_team",
        return_value=[fake_report],
    ):
        enriched = enrich_with_fifa_reports(matches_df, {"Mexico"})

    assert enriched.iloc[0]["corners_local"] == 8
    assert enriched.iloc[0]["corners_visitante"] == 2
    assert enriched.iloc[0]["tiros_arco_local"] == 4


def test_enrich_with_fifa_reports_never_raises_when_source_fails():
    matches_df = pd.DataFrame([
        {
            "fecha": pd.Timestamp("2026-06-11"), "liga": "WC",
            "equipo_local": "Mexico", "equipo_visitante": "South Korea",
            "goles_local": 2, "goles_visitante": 1,
            "corners_local": None, "corners_visitante": None,
        }
    ])
    with patch(
        "src.connectors.fifa_reports_connector.get_match_stats_for_team",
        side_effect=Exception("la fuente esta caida"),
    ):
        enriched = enrich_with_fifa_reports(matches_df, {"Mexico"})
    assert enriched.iloc[0]["corners_local"] is None
