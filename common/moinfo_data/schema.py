# C:\moinfo\common\moinfo_data\schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


STANDARD_COLS = {
    "timestamp": "timestamp",
    "target": "target",
    "item_id": "item_id",
}


class SchemaError(ValueError):
    pass


@dataclass
class SchemaIssue:
    level: str  # "ERROR" | "WARN"
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class SchemaReport:
    ok: bool
    errors: List[SchemaIssue]
    warnings: List[SchemaIssue]
    summary: Dict[str, Any]

    def raise_if_invalid(self) -> None:
        if self.ok:
            return
        msgs = ["スキーマ検証に失敗しました。"]
        for e in self.errors:
            msgs.append(f"- [{e.code}] {e.message}")
            if e.details:
                msgs.append(f"  details={e.details}")
        raise SchemaError("\n".join(msgs))


def _issue(level: str, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> SchemaIssue:
    return SchemaIssue(level=level, code=code, message=message, details=details)


def _nan_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].isna().sum())


def validate_timeseries_df(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    target_col: str = "target",
    item_id_col: str = "item_id",
    require_sorted: bool = True,
    allow_duplicate_timestamps_per_item: bool = False,
    max_bad_examples: int = 5,
) -> SchemaReport:
    """
    標準スキーマ:
      - timestamp: datetime64[ns]（NaTなし）
      - target: 数値（NaNなし推奨：NaNなら明示）
      - item_id: 文字列/カテゴリ（欠損なし）

    返り値の report で欠損・型不一致を明示する。
    """
    errors: List[SchemaIssue] = []
    warns: List[SchemaIssue] = []

    # 必須列チェック
    for col in (timestamp_col, target_col, item_id_col):
        if col not in df.columns:
            errors.append(_issue("ERROR", "MISSING_COLUMN", f"必須列がありません: {col}", {"columns": list(df.columns)}))

    if errors:
        return SchemaReport(ok=False, errors=errors, warnings=warns, summary={"n_rows": int(len(df))})

    # 型チェック: timestamp
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        errors.append(
            _issue(
                "ERROR",
                "BAD_DTYPE_TIMESTAMP",
                f"{timestamp_col} が datetime 型ではありません（現在: {df[timestamp_col].dtype}）",
                {"dtype": str(df[timestamp_col].dtype)},
            )
        )

    # 欠損チェック
    ts_nan = _nan_count(df, timestamp_col)
    if ts_nan > 0:
        bad = df[df[timestamp_col].isna()].head(max_bad_examples)
        errors.append(
            _issue(
                "ERROR",
                "TIMESTAMP_NAN",
                f"{timestamp_col} に欠損（NaT）が {ts_nan} 件あります",
                {"examples": bad.to_dict(orient="records")},
            )
        )

    # target 型
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        errors.append(
            _issue(
                "ERROR",
                "BAD_DTYPE_TARGET",
                f"{target_col} が数値型ではありません（現在: {df[target_col].dtype}）",
                {"dtype": str(df[target_col].dtype)},
            )
        )

    tgt_nan = _nan_count(df, target_col)
    if tgt_nan > 0:
        bad = df[df[target_col].isna()].head(max_bad_examples)
        warns.append(
            _issue(
                "WARN",
                "TARGET_NAN",
                f"{target_col} に欠損（NaN）が {tgt_nan} 件あります（前処理で補完/除外が必要）",
                {"examples": bad.to_dict(orient="records")},
            )
        )

    # item_id 欠損
    id_nan = _nan_count(df, item_id_col)
    if id_nan > 0:
        bad = df[df[item_id_col].isna()].head(max_bad_examples)
        errors.append(
            _issue(
                "ERROR",
                "ITEM_ID_NAN",
                f"{item_id_col} に欠損が {id_nan} 件あります",
                {"examples": bad.to_dict(orient="records")},
            )
        )

    # 重複 timestamp（itemごと）
    if not allow_duplicate_timestamps_per_item:
        dup_mask = df.duplicated(subset=[item_id_col, timestamp_col], keep=False)
        dup_cnt = int(dup_mask.sum())
        if dup_cnt > 0:
            bad = df[dup_mask].sort_values([item_id_col, timestamp_col]).head(max_bad_examples)
            errors.append(
                _issue(
                    "ERROR",
                    "DUPLICATE_TIMESTAMP",
                    f"{item_id_col} ごとに {timestamp_col} が重複している行が {dup_cnt} 件あります（集約/ユニーク化が必要）",
                    {"examples": bad.to_dict(orient="records")},
                )
            )

    # ソート/単調増加チェック（itemごと）
    if require_sorted:
        # item単位で timestamp が昇順になっているかざっくり確認
        is_sorted = True
        # groupbyの全件比較は重いので、差分の負を検出
        tmp = df[[item_id_col, timestamp_col]].copy()
        tmp["_ts_diff"] = tmp.groupby(item_id_col)[timestamp_col].diff()
        bad_cnt = int((tmp["_ts_diff"] < pd.Timedelta(0)).sum())
        if bad_cnt > 0:
            is_sorted = False
            bad_rows = df.loc[(tmp["_ts_diff"] < pd.Timedelta(0)).fillna(False)].head(max_bad_examples)
            warns.append(
                _issue(
                    "WARN",
                    "NOT_SORTED",
                    f"{item_id_col} ごとに {timestamp_col} が昇順に並んでいない箇所が {bad_cnt} 件あります（ソート推奨）",
                    {"examples": bad_rows.to_dict(orient="records")},
                )
            )

    summary = {
        "n_rows": int(len(df)),
        "n_items": int(df[item_id_col].nunique()),
        "min_timestamp": None if len(df) == 0 else str(df[timestamp_col].min()),
        "max_timestamp": None if len(df) == 0 else str(df[timestamp_col].max()),
        "nan_counts": {
            timestamp_col: ts_nan,
            target_col: tgt_nan,
            item_id_col: id_nan,
        },
        "dtypes": {
            timestamp_col: str(df[timestamp_col].dtype),
            target_col: str(df[target_col].dtype),
            item_id_col: str(df[item_id_col].dtype),
        },
    }

    ok = (len(errors) == 0)
    return SchemaReport(ok=ok, errors=errors, warnings=warns, summary=summary)
