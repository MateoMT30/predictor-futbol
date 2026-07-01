import pandas as pd

from src.data_loader import load_and_clean


def test_normalizes_team_name_variants():
    raw = pd.DataFrame({
        "fecha": ["2024-01-01", "2024-01-08", "2024-01-15"],
        "liga": ["Liga X"] * 3,
        "equipo_local": ["Colombia", "Colombia", " colombia "],
        "equipo_visitante": ["Peru", "Peru", "Peru"],
        "goles_local": [1, 2, 0],
        "goles_visitante": [0, 1, 0],
    })
    df, report = load_and_clean(raw)
    assert df["equipo_local"].nunique() == 1
    assert df["equipo_local"].iloc[0] == "Colombia"


def test_discards_incomplete_rows():
    raw = pd.DataFrame({
        "fecha": ["2024-01-01", None],
        "liga": ["Liga X", "Liga X"],
        "equipo_local": ["Colombia", "Peru"],
        "equipo_visitante": ["Peru", "Colombia"],
        "goles_local": [1, 2],
        "goles_visitante": [0, 1],
    })
    df, report = load_and_clean(raw)
    assert len(df) == 1
    assert report.filas_descartadas_incompletas == 1


def test_missing_required_columns_raises():
    raw = pd.DataFrame({"equipo_local": ["A"], "equipo_visitante": ["B"]})
    try:
        load_and_clean(raw)
        assert False, "debería haber lanzado ValueError"
    except ValueError:
        pass


def test_sample_history_loads(sample_matches):
    assert len(sample_matches) > 0
    assert "elo_local_pre" not in sample_matches.columns  # aún no pasó por ratings
