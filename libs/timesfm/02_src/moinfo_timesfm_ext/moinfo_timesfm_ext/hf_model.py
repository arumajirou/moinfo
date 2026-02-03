# C:\moinfo\libs\timesfm\02_src\moinfo_timesfm_ext\moinfo_timesfm_ext\hf_model.py
from __future__ import annotations

from pathlib import Path
import torch
from transformers import TimesFmModelForPrediction


def resolve_torch_dtype(dtype: str, device: str) -> torch.dtype | None:
    """
    dtype指定を torch.dtype に変換。Noneならfrom_pretrainedに任せる。
    """
    dtype = (dtype or "auto").lower()
    if dtype == "auto":
        # GPUなら半精度寄り、CPUならfp32寄りにする
        if device == "cuda":
            return torch.float16
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"不明なdtype: {dtype} (auto/fp16/bf16/fp32)")


def load_timesfm_for_prediction(model_path: str, device: str, dtype: str = "auto") -> TimesFmModelForPrediction:
    """
    model_path:
      - ローカル: r"C:\moinfo\timesfm_v2.5_local" 等（config.jsonとmodel.safetensorsがある想定）
      - もしくはHF id: "google/timesfm-2.0-500m-pytorch" 等
    """
    torch_dtype = resolve_torch_dtype(dtype, device)

    # local path かどうかの見分け（存在すればローカル優先）
    p = Path(model_path)
    path_or_id = str(p) if p.exists() else model_path

    model = TimesFmModelForPrediction.from_pretrained(
        path_or_id,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    return model
