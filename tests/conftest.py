import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_matches():
    from src.data_loader import load_and_clean

    raw = pd.read_csv(Path(__file__).resolve().parent.parent / "examples" / "historico_ejemplo.csv")
    df, _ = load_and_clean(raw)
    return df
