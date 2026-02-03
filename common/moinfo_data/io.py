# C:\moinfo\common\moinfo_data\io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .schema import STANDARD_COLS, SchemaReport, validate_timeseries_df


def _coerce_datetime(series: pd.Series) -> pd.Series:
    """
    timestamp を datetime に寄せる。
    文字列/日時/数値(UNIX秒・ms)をある程度吸収する。
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # 数値なら UNIX time っぽさを推定
    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna()
        if len(s) == 0:
            return pd.to_datetime(series, errors="coerce")
        m = float(s.median())
        # ざっくり判定:
        # 秒: 1e9~1e10 (2001-2286)
        # ms: 1e12~1e13
        if 1e12 <= m <= 1e13:
            return pd.to_datetime(series, unit="ms", errors="coerce")
        if 1e9 <= m <= 1e10:
            return pd.to_datetime(series, unit="s", errors="coerce")
        return pd.to_datetime(series, errors="coerce")

    return pd.to_datetime(series, errors="coerce")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def normalize_timeseries_df(
    df: pd.DataFrame,
    *,
    timestamp_col: Optional[str] = None,
    target_col: Optional[str] = None,
    item_id_col: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
    default_item_id: str = "series_0",
    sort: bool = True,
) -> pd.DataFrame:
    """
    df を標準スキーマ（timestamp/target/item_id）へ正規化。

    column_map は「標準名 -> 元CSV列名」のマップを想定。
      例: {"timestamp": "date", "target": "y", "item_id": "store_id"}

    直接 timestamp_col 等で指定してもOK（column_mapより優先）。
    """
    out = df.copy()

    # 1) どの列をどれに合わせるか決める
    ts_src = timestamp_col or (column_map.get("timestamp") if column_map else None)
    y_src = target_col or (column_map.get("target") if column_map else None)
    id_src = item_id_col or (column_map.get("item_id") if column_map else None)

    # 2) 列名が既に標準名の場合はそのままでも動く
    if ts_src is None and STANDARD_COLS["timestamp"] in out.columns:
        ts_src = STANDARD_COLS["timestamp"]
    if y_src is None and STANDARD_COLS["target"] in out.columns:
        y_src = STANDARD_COLS["target"]
    if id_src is None and STANDARD_COLS["item_id"] in out.columns:
        id_src = STANDARD_COLS["item_id"]

    # 3) rename
    rename_map: Dict[str, str] = {}
    if ts_src and ts_src != STANDARD_COLS["timestamp"]:
        rename_map[ts_src] = STANDARD_COLS["timestamp"]
    if y_src and y_src != STANDARD_COLS["target"]:
        rename_map[y_src] = STANDARD_COLS["target"]
    if id_src and id_src != STANDARD_COLS["item_id"]:
        rename_map[id_src] = STANDARD_COLS["item_id"]

    if rename_map:
        out = out.rename(columns=rename_map)

    # 4) item_id が無いなら付与
    if STANDARD_COLS["item_id"] not in out.columns:
        out[STANDARD_COLS["item_id"]] = default_item_id

    # 5) 型変換
    if STANDARD_COLS["timestamp"] in out.columns:
        out[STANDARD_COLS["timestamp"]] = _coerce_datetime(out[STANDARD_COLS["timestamp"]])
    if STANDARD_COLS["target"] in out.columns:
        out[STANDARD_COLS["target"]] = _coerce_numeric(out[STANDARD_COLS["target"]])

    # 6) 必要列だけに絞る…は、後で特徴量を足す可能性があるので今は保持
    # ただし標準列が先頭に来るよう並び替える
    cols = list(out.columns)
    ordered = [c for c in (STANDARD_COLS["timestamp"], STANDARD_COLS["target"], STANDARD_COLS["item_id"]) if c in cols]
    rest = [c for c in cols if c not in ordered]
    out = out[ordered + rest]

    # 7) ソート（推奨）
    if sort and (STANDARD_COLS["timestamp"] in out.columns) and (STANDARD_COLS["item_id"] in out.columns):
        out = out.sort_values([STANDARD_COLS["item_id"], STANDARD_COLS["timestamp"]]).reset_index(drop=True)

    return out


@dataclass
class LoadResult:
    df: pd.DataFrame
    report: SchemaReport


def load_timeseries_csv(
    csv_path: str | Path,
    *,
    encoding: str = "utf-8",
    column_map: Optional[Dict[str, str]] = None,
    timestamp_col: Optional[str] = None,
    target_col: Optional[str] = None,
    item_id_col: Optional[str] = None,
    default_item_id: str = "series_0",
    sort: bool = True,
    strict: bool = True,
) -> LoadResult:
    """
    CSVをロード→正規化→スキーマ検証。
    strict=True ならエラー時に例外を投げる（Notebookで状況確認したいなら strict=False）
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    df_raw = pd.read_csv(csv_path, encoding=encoding)

    df_norm = normalize_timeseries_df(
        df_raw,
        timestamp_col=timestamp_col,
        target_col=target_col,
        item_id_col=item_id_col,
        column_map=column_map,
        default_item_id=default_item_id,
        sort=sort,
    )

    report = validate_timeseries_df(
        df_norm,
        timestamp_col=STANDARD_COLS["timestamp"],
        target_col=STANDARD_COLS["target"],
        item_id_col=STANDARD_COLS["item_id"],
        require_sorted=False,  # sort=Trueなら既に並ぶので、ここではWARNを出さない運用でもOK
    )

    if strict:
        report.raise_if_invalid()

    return LoadResult(df=df_norm, report=report)
