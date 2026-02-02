# C:\moinfo\libs\timesfm\02_src\moinfo_timesfm_ext\moinfo_timesfm_ext\client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

import timesfm


@dataclass
class ForecastResult:
    point: np.ndarray                 # shape: (batch, horizon)
    quantiles: Optional[np.ndarray]   # shape: (batch, horizon, q) or None


class TimesFMClient:
    """
    timesfmの推論APIを“あなた用に安定化”する薄いラッパー。
    上流が多少変わっても、ここで吸収してNotebook側のコードを固定化できる。
    """

    def __init__(self, model: Any):
        self.model = model
        self.is_compiled = False

    @classmethod
    def from_pretrained_2p5_200m_torch(cls, repo_id: str = "google/timesfm-2.5-200m-pytorch"):
        # 上流READMEの利用方法に合わせる（from_pretrained → compile → forecast）:contentReference[oaicite:5]{index=5}
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo_id)
        return cls(model)

    def compile(self, *, config: "timesfm.ForecastConfig") -> None:
        self.model.compile(config)
        self.is_compiled = True

    def forecast(
        self,
        *,
        horizon: int,
        inputs: List[np.ndarray],
    ) -> ForecastResult:
        if not self.is_compiled:
            raise RuntimeError("compile() を先に呼んでください。")
        point, quantiles = self.model.forecast(horizon=horizon, inputs=inputs)
        return ForecastResult(point=np.asarray(point), quantiles=None if quantiles is None else np.asarray(quantiles))
