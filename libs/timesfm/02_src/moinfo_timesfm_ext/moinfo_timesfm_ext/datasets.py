# C:\moinfo\libs\timesfm\02_src\moinfo_timesfm_ext\moinfo_timesfm_ext\datasets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_series_from_csv(
    csv_path: str | Path,
    value_col: str = "y",
    series_id_col: str | None = None,
) -> dict[str, np.ndarray]:
    """
    CSVから系列を読み込む。
    - 単系列: series_id_col=None のとき、全行を1系列として扱う
    - 多系列: series_id_col を指定すると、series_idごとに系列を分ける
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSVが見つかりません: {p}")

    df = pd.read_csv(p)

    if value_col not in df.columns:
        raise ValueError(f"value_col='{value_col}' がCSVにありません。columns={list(df.columns)}")

    if series_id_col is None:
        y = df[value_col].to_numpy(dtype=np.float32)
        return {"0": y}

    if series_id_col not in df.columns:
        raise ValueError(f"series_id_col='{series_id_col}' がCSVにありません。columns={list(df.columns)}")

    out: dict[str, np.ndarray] = {}
    for sid, g in df.groupby(series_id_col, sort=False):
        y = g[value_col].to_numpy(dtype=np.float32)
        out[str(sid)] = y
    return out


@dataclass(frozen=True)
class WindowSpec:
    context_len: int
    horizon_len: int


class RandomWindowDataset(Dataset):
    """
    多系列/単系列どちらも対応。
    メモリ節約のため「事前に全窓を列挙せず」、getitemのたびにランダムに窓を切る。
    """
    def __init__(
        self,
        series: dict[str, np.ndarray],
        window: WindowSpec,
        samples_per_epoch: int,
        freq_id: int = 0,
        seed: int = 42,
    ):
        self.series = {k: v.astype(np.float32) for k, v in series.items()}
        self.window = window
        self.samples_per_epoch = int(samples_per_epoch)
        self.freq_id = int(freq_id)

        self.rng = np.random.default_rng(seed)

        # 有効な系列だけ残す
        self.valid_keys: list[str] = []
        for k, y in self.series.items():
            if len(y) > (window.context_len + window.horizon_len):
                self.valid_keys.append(k)
        if not self.valid_keys:
            raise ValueError("有効な系列がありません。context_len + horizon_len より長い系列が必要です。")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        k = self.rng.choice(self.valid_keys)
        y = self.series[k]
        max_start = len(y) - (self.window.context_len + self.window.horizon_len)
        s = int(self.rng.integers(0, max_start))

        past = y[s : s + self.window.context_len]
        future = y[s + self.window.context_len : s + self.window.context_len + self.window.horizon_len]

        return {
            "past_values": torch.from_numpy(past),
            "future_values": torch.from_numpy(future),
            "freq": torch.tensor(self.freq_id, dtype=torch.long),
        }
