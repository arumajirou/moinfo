# C:\moinfo\libs\timesfm\02_src\moinfo_timesfm_ext\moinfo_timesfm_ext\config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch


@dataclass
class FineTuneConfig:
    # 入力
    model_path: str                       # 例: r"C:\moinfo\timesfm_v2.5_local" or HF id
    csv_path: str                         # 例: r"C:\moinfo\data\train.csv"
    value_col: str = "y"                  # 目的変数列
    series_id_col: str | None = None      # 多系列なら指定（例: "series_id"）
    time_col: str | None = None           # あってもなくてもOK（例: "ds"）
    freq_id: int = 0                      # 周期ID（まず固定でOK）

    # ウィンドウ設定
    context_len: int = 512                # 過去長
    horizon_len: int = 128                # 未来長（予測長）
    samples_per_epoch: int = 20000        # 1 epoch あたりのランダム窓サンプル数

    # 学習
    batch_size: int = 8
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0

    # 出力
    output_dir: str = r".\timesfm_finetuned"

    # 実行
    device: str = "auto"                  # "cuda" / "cpu" / "auto"
    dtype: str = "auto"                   # "auto" / "fp16" / "bf16" / "fp32"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
