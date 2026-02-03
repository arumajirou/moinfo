# C:\moinfo\libs\timesfm\02_src\moinfo_timesfm_ext\moinfo_timesfm_ext\utils.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_dummy_train_csv(
    out_csv: str | Path,
    n_points: int = 30000,
    value_col: str = "y",
) -> Path:
    """
    単系列のダミーCSVを生成。動作確認用。
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, 200, n_points, dtype=np.float32)
    y = np.sin(t) + 0.1 * np.random.randn(n_points).astype(np.float32)

    df = pd.DataFrame({value_col: y})
    df.to_csv(out_csv, index=False)
    return out_csv
