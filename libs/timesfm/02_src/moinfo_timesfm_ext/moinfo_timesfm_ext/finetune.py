# C:\moinfo\libs\timesfm\02_src\moinfo_timesfm_ext\moinfo_timesfm_ext\finetune.py
from __future__ import annotations

from pathlib import Path
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import FineTuneConfig
from .datasets import load_series_from_csv, RandomWindowDataset, WindowSpec
from .hf_model import load_timesfm_for_prediction


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _sanity_forward(model, batch, device: str):
    model.eval()
    past = batch["past_values"].to(device)
    future = batch["future_values"].to(device)
    freq = batch["freq"].to(device)

    out = model(past_values=past, freq=freq, future_values=future, return_dict=True)
    if not hasattr(out, "loss") or out.loss is None:
        raise RuntimeError("モデル出力にlossがありません。TimesFmModelForPrediction + future_values が必要です。")
    model.train()


def finetune_from_csv(cfg: FineTuneConfig) -> Path:
    """
    TimesFM(Hugging Face版)の微調整。
    - CSVから系列を読み込む
    - ランダム窓で学習
    - save_pretrainedで保存
    戻り値: 保存先ディレクトリ(Path)
    """
    device = cfg.resolved_device()
    set_seed(cfg.seed)

    series = load_series_from_csv(
        csv_path=cfg.csv_path,
        value_col=cfg.value_col,
        series_id_col=cfg.series_id_col,
    )

    ds = RandomWindowDataset(
        series=series,
        window=WindowSpec(cfg.context_len, cfg.horizon_len),
        samples_per_epoch=cfg.samples_per_epoch,
        freq_id=cfg.freq_id,
        seed=cfg.seed,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )

    model = load_timesfm_for_prediction(cfg.model_path, device=device, dtype=cfg.dtype)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # まず1バッチだけ通して shape/仕様の破綻を早期検出
    first = next(iter(dl))
    _sanity_forward(model, first, device)

    for epoch in range(cfg.epochs):
        running = 0.0
        for step, batch in enumerate(dl, 1):
            past = batch["past_values"].to(device)
            future = batch["future_values"].to(device)
            freq = batch["freq"].to(device)

            out = model(past_values=past, freq=freq, future_values=future, return_dict=True)
            loss = out.loss

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            opt.step()

            running += float(loss.item())
            if step % 100 == 0:
                print(f"[epoch {epoch+1}/{cfg.epochs}] step {step} loss {running/100:.4f}")
                running = 0.0

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 設定も一緒に残す
    cfg.to_json(out_dir / "finetune_config.json")

    # HF形式で保存（config.json + model.safetensors）
    model.save_pretrained(str(out_dir))
    print("saved to:", out_dir)

    return out_dir
