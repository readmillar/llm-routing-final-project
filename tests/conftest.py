import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest


@pytest.fixture
def synthetic_csv(tmp_path):
    rows = [
        ("AIME", "p1", "free-small", 0.0, 0.0),
        ("AIME", "p1", "cheap-solid", 1.0, 0.1),
        ("AIME", "p1", "balanced", 1.0, 0.5),
        ("AIME", "p1", "strong", 1.0, 2.0),
        ("AIME", "p2", "free-small", 0.0, 0.0),
        ("AIME", "p2", "cheap-solid", 1.0, 0.1),
        ("AIME", "p2", "balanced", 0.0, 0.5),
        ("AIME", "p2", "strong", 1.0, 2.0),
        ("LCB", "p3", "free-small", 0.0, 0.0),
        ("LCB", "p3", "cheap-solid", 0.0, 0.1),
        ("LCB", "p3", "balanced", 1.0, 0.5),
        ("LCB", "p3", "strong", 1.0, 2.0),
        ("GPQA", "p4", "free-small", 1.0, 0.0),
        ("GPQA", "p4", "cheap-solid", 0.0, 0.1),
        ("GPQA", "p4", "balanced", 1.0, 0.5),
        ("GPQA", "p4", "strong", 1.0, 2.0),
    ]
    df = pd.DataFrame(rows, columns=["dataset", "prompt_id", "model", "score", "cost"])
    path = tmp_path / "synthetic_routerbench.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_data(synthetic_csv, tmp_path):
    from src.load_data import load_dataset

    return load_dataset(str(synthetic_csv), output_dir=str(tmp_path / "outputs"))
