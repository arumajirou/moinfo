# C:\moinfo\common\moinfo_data\preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class PreprocessConfig:
    # 標準列名
    timestamp_col: str = "timestamp"
    target_col: str = "target"
    item_id_col: str = "item_id"

    sort: bool = True

    # 重複解消: "mean" | "sum" | "first" | "raise"
    deduplicate: str = "mean"

    # 周波数（freq）: Noneなら推定
    freq: Optional[str] = None
    freq_infer: str = "median"  # "infer" | "median"
    freq_fallback: str = "D"

    # 欠損時刻の穴埋め（時刻グリッド化）
    fill_missing_timestamps: bool = True

    # 外れ値
    outlier_method: str = "mad"  # "none" | "mad" | "iqr"
    outlier_action: str = "clip"  # "none" | "clip" | "nan"
    mad_z: float = 6.0
    iqr_k: float = 3.0

    # 欠損補完
    impute_method: str = "ffill_bfill"  # "none" | "ffill" | "bfill" | "ffill_bfill" | "interpolate" | "zero"
    interpolate_method: str = "time"
    impute_limit: Optional[int] = None

    # ★追加：timestamp が無いときに疑似生成する
    create_timestamp_if_missing: bool = False
    generated_timestamp_start: str = "2000-01-01"
    generated_timestamp_freq: Optional[str] = None  # Noneなら freq or freq_fallback を使う


@dataclass
class PreprocessReport:
    ok: bool
    inferred_freq: str
    before: Dict[str, Any]
    after: Dict[str, Any]
    actions: List[Dict[str, Any]]


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"必須列が不足しています: {missing} / columns={list(df.columns)}")


def infer_frequency(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    item_id_col: str,
    mode: str = "median",
    fallback: str = "D",
) -> Union[str, pd.Timedelta]:
    _require_cols(df, [timestamp_col, item_id_col])
    if len(df) < 3:
        return fallback

    modes: List[Union[str, pd.Timedelta]] = []
    for _, g in df[[item_id_col, timestamp_col]].dropna().groupby(item_id_col):
        ts = g[timestamp_col].sort_values()
        if ts.nunique() < 3:
            continue

        if mode == "infer":
            try:
                f = pd.infer_freq(ts)
                if f:
                    modes.append(f)
                    continue
            except Exception:
                pass

        diffs = ts.diff().dropna()
        if len(diffs) == 0:
            continue
        td = diffs.median()
        if pd.isna(td) or td <= pd.Timedelta(0):
            continue
        modes.append(td)

    if not modes:
        return fallback

    str_modes = [m for m in modes if isinstance(m, str)]
    if str_modes:
        return pd.Series(str_modes).mode().iloc[0]

    td_modes = [m for m in modes if isinstance(m, pd.Timedelta)]
    return pd.Series(td_modes).median()


def _freq_to_str(freq: Union[str, pd.Timedelta]) -> str:
    if isinstance(freq, str):
        return freq
    sec = int(freq.total_seconds())
    if sec % 86400 == 0:
        return f"{sec // 86400}D"
    if sec % 3600 == 0:
        return f"{sec // 3600}H"
    if sec % 60 == 0:
        return f"{sec // 60}min"
    return f"{sec}s"


def _summary(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    target_col: str,
    item_id_col: str,
    freq: Optional[Union[str, pd.Timedelta]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n_rows"] = int(len(df))
    out["n_items"] = int(df[item_id_col].nunique()) if item_id_col in df.columns else None

    if timestamp_col in df.columns and len(df) > 0:
        out["min_timestamp"] = None if df[timestamp_col].isna().all() else str(df[timestamp_col].min())
        out["max_timestamp"] = None if df[timestamp_col].isna().all() else str(df[timestamp_col].max())
    else:
        out["min_timestamp"] = None
        out["max_timestamp"] = None

    out["nan_counts"] = {
        timestamp_col: int(df[timestamp_col].isna().sum()) if timestamp_col in df.columns else None,
        target_col: int(df[target_col].isna().sum()) if target_col in df.columns else None,
        item_id_col: int(df[item_id_col].isna().sum()) if item_id_col in df.columns else None,
    }

    if all(c in df.columns for c in (item_id_col, timestamp_col)):
        out["duplicate_item_timestamp_rows"] = int(df.duplicated(subset=[item_id_col, timestamp_col]).sum())
    else:
        out["duplicate_item_timestamp_rows"] = None

    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        s = df[target_col]
        out["target_stats"] = {
            "min": None if s.dropna().empty else float(s.min()),
            "max": None if s.dropna().empty else float(s.max()),
            "mean": None if s.dropna().empty else float(s.mean()),
            "std": None if s.dropna().empty else float(s.std(ddof=1)) if s.dropna().shape[0] > 1 else 0.0,
            "q01": None if s.dropna().empty else float(s.quantile(0.01)),
            "q50": None if s.dropna().empty else float(s.quantile(0.50)),
            "q99": None if s.dropna().empty else float(s.quantile(0.99)),
        }
    else:
        out["target_stats"] = None

    if freq is not None and all(c in df.columns for c in (item_id_col, timestamp_col)):
        missing_total = 0
        expected_total = 0
        for _, g in df[[item_id_col, timestamp_col]].dropna().groupby(item_id_col):
            ts = g[timestamp_col].sort_values()
            if ts.empty:
                continue
            start, end = ts.iloc[0], ts.iloc[-1]
            full = pd.date_range(start=start, end=end, freq=freq)
            expected = len(full)
            actual = ts.nunique()
            expected_total += expected
            missing_total += max(0, expected - actual)
        out["expected_rows_if_full_grid"] = int(expected_total)
        out["missing_timestamps_estimate"] = int(missing_total)
        out["freq_used_for_gap_estimate"] = _freq_to_str(freq)
    else:
        out["expected_rows_if_full_grid"] = None
        out["missing_timestamps_estimate"] = None
        out["freq_used_for_gap_estimate"] = None

    return out


def _maybe_create_timestamp(df: pd.DataFrame, cfg: PreprocessConfig, actions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    timestamp列が無い場合に、itemごとの行順を時間とみなして疑似timestampを生成する。
    ※これは“仮説”であり、真の日時列が存在するならそれを使うべき。
    """
    t, i = cfg.timestamp_col, cfg.item_id_col
    if t in df.columns:
        return df

    if not cfg.create_timestamp_if_missing:
        raise ValueError(
            f"必須列 '{t}' がありません。columns={list(df.columns)}\n"
            f"対処:\n"
            f"  1) CSVに日時列があるなら PreprocessConfig(timestamp_col='その列名') か ID3のcolumn_mapで指定\n"
            f"  2) 日時列が本当に無いなら PreprocessConfig(create_timestamp_if_missing=True) で疑似timestampを生成\n"
        )

    _require_cols(df, [i])

    freq = cfg.generated_timestamp_freq or cfg.freq or cfg.freq_fallback
    start = pd.Timestamp(cfg.generated_timestamp_start)

    work = df.copy()
    work["_orig_order"] = np.arange(len(work), dtype=np.int64)

    parts: List[pd.DataFrame] = []
    for item, g in work.groupby(i, sort=False):
        g = g.sort_values("_orig_order")
        n = len(g)
        ts = pd.date_range(start=start, periods=n, freq=freq)
        gg = g.copy()
        gg[t] = ts.values
        parts.append(gg)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)

    actions.append({
        "action": "create_timestamp_if_missing",
        "timestamp_col": t,
        "strategy": "per_item_row_order",
        "start": str(start),
        "freq": str(freq),
        "note": "timestampが無いので item内の行順を等間隔と仮定して生成（仮の時間軸）",
    })
    return out


def _deduplicate(df: pd.DataFrame, cfg: PreprocessConfig, actions: List[Dict[str, Any]]) -> pd.DataFrame:
    t, y, i = cfg.timestamp_col, cfg.target_col, cfg.item_id_col
    dup = df.duplicated(subset=[i, t]).sum()
    if dup == 0:
        return df

    if cfg.deduplicate == "raise":
        raise ValueError(f"timestamp×item_id の重複が {int(dup)} 行あります（deduplicate='raise'）")

    if cfg.deduplicate not in ("mean", "sum", "first"):
        raise ValueError(f"deduplicate は mean/sum/first/raise のいずれか: {cfg.deduplicate}")

    actions.append({"action": "deduplicate", "duplicates_rows": int(dup), "mode": cfg.deduplicate})

    agg: Dict[str, Any] = {}
    for col in df.columns:
        if col in (i, t):
            continue
        if col == y:
            agg[col] = cfg.deduplicate
        else:
            agg[col] = "first"

    out = df.groupby([i, t], as_index=False).agg(agg)
    return out


def _outlier_bounds_mad(x: pd.Series, z: float) -> Tuple[float, float]:
    s = x.dropna()
    if s.empty:
        return (-np.inf, np.inf)
    med = float(s.median())
    mad = float((s - med).abs().median())
    if mad == 0.0:
        return (-np.inf, np.inf)
    scale = 1.4826 * mad
    lo = med - z * scale
    hi = med + z * scale
    return (lo, hi)


def _outlier_bounds_iqr(x: pd.Series, k: float) -> Tuple[float, float]:
    s = x.dropna()
    if s.empty:
        return (-np.inf, np.inf)
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0.0:
        return (-np.inf, np.inf)
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (lo, hi)


def _handle_outliers(df: pd.DataFrame, cfg: PreprocessConfig, actions: List[Dict[str, Any]]) -> pd.DataFrame:
    if cfg.outlier_method == "none" or cfg.outlier_action == "none":
        return df

    t, y, i = cfg.timestamp_col, cfg.target_col, cfg.item_id_col
    if y not in df.columns or not pd.api.types.is_numeric_dtype(df[y]):
        actions.append({"action": "outlier_skip", "reason": "target is not numeric"})
        return df

    method = cfg.outlier_method
    action = cfg.outlier_action

    out = df.copy()
    outlier_total = 0

    for item, idx in out.groupby(i).groups.items():
        s = out.loc[idx, y]
        if method == "mad":
            lo, hi = _outlier_bounds_mad(s, cfg.mad_z)
        elif method == "iqr":
            lo, hi = _outlier_bounds_iqr(s, cfg.iqr_k)
        else:
            raise ValueError(f"outlier_method は none/mad/iqr: {method}")

        mask = (out.loc[idx, y] < lo) | (out.loc[idx, y] > hi)
        cnt = int(mask.sum())
        outlier_total += cnt

        if cnt == 0:
            continue

        if action == "clip":
            out.loc[idx, y] = out.loc[idx, y].clip(lower=lo, upper=hi)
        elif action == "nan":
            out.loc[idx, y] = out.loc[idx, y].where(~mask, np.nan)
        else:
            raise ValueError(f"outlier_action は none/clip/nan: {action}")

    actions.append({
        "action": "outlier",
        "method": method,
        "outlier_action": action,
        "outlier_count": int(outlier_total),
        "mad_z": cfg.mad_z,
        "iqr_k": cfg.iqr_k,
    })
    return out


def _fill_missing_grid(df: pd.DataFrame, cfg: PreprocessConfig, freq: Union[str, pd.Timedelta], actions: List[Dict[str, Any]]) -> pd.DataFrame:
    if not cfg.fill_missing_timestamps:
        return df

    t, y, i = cfg.timestamp_col, cfg.target_col, cfg.item_id_col

    out_parts: List[pd.DataFrame] = []
    meta_cols = [c for c in df.columns if c not in (t, y, i)]

    for item, g in df.groupby(i, sort=False):
        g = g.sort_values(t)
        if g.empty:
            continue
        start, end = g[t].iloc[0], g[t].iloc[-1]
        full_index = pd.date_range(start=start, end=end, freq=freq)

        g2 = g.set_index(t).reindex(full_index)
        g2.index.name = t
        g2 = g2.reset_index()
        g2[i] = item

        for c in meta_cols:
            if c in g.columns:
                g2[c] = g[c].iloc[0]

        out_parts.append(g2)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else df.copy()
    actions.append({
        "action": "fill_missing_timestamps",
        "enabled": True,
        "freq": _freq_to_str(freq),
        "added_rows_estimate": int(max(0, len(out) - len(df))),
    })
    return out


def _impute_target(df: pd.DataFrame, cfg: PreprocessConfig, actions: List[Dict[str, Any]]) -> pd.DataFrame:
    method = cfg.impute_method
    if method == "none":
        return df

    t, y, i = cfg.timestamp_col, cfg.target_col, cfg.item_id_col
    out = df.copy()

    if y not in out.columns or not pd.api.types.is_numeric_dtype(out[y]):
        actions.append({"action": "impute_skip", "reason": "target is not numeric"})
        return out

    before_nan = int(out[y].isna().sum())

    if method == "zero":
        out[y] = out[y].fillna(0.0)
    else:
        parts: List[pd.DataFrame] = []
        for item, g in out.groupby(i, sort=False):
            g = g.sort_values(t)
            if method == "ffill":
                g[y] = g[y].ffill(limit=cfg.impute_limit)
            elif method == "bfill":
                g[y] = g[y].bfill(limit=cfg.impute_limit)
            elif method == "ffill_bfill":
                g[y] = g[y].ffill(limit=cfg.impute_limit).bfill(limit=cfg.impute_limit)
            elif method == "interpolate":
                gg = g.set_index(t)
                gg[y] = gg[y].interpolate(method=cfg.interpolate_method, limit=cfg.impute_limit)
                g = gg.reset_index()
            else:
                raise ValueError(f"impute_method は none/ffill/bfill/ffill_bfill/interpolate/zero: {method}")
            parts.append(g)
        out = pd.concat(parts, ignore_index=True)

    after_nan = int(out[y].isna().sum())
    actions.append({
        "action": "impute_target",
        "method": method,
        "filled_count": int(before_nan - after_nan),
        "nan_before": int(before_nan),
        "nan_after": int(after_nan),
        "limit": cfg.impute_limit,
    })
    return out


def preprocess_timeseries_df(df: pd.DataFrame, cfg: Optional[PreprocessConfig] = None) -> Tuple[pd.DataFrame, PreprocessReport]:
    cfg = cfg or PreprocessConfig()
    t, y, i = cfg.timestamp_col, cfg.target_col, cfg.item_id_col

    actions: List[Dict[str, Any]] = []
    work = df.copy()

    # ★timestampが無い場合の救済
    work = _maybe_create_timestamp(work, cfg, actions)

    # 必須列チェック（ここで確実に揃っているはず）
    _require_cols(work, [t, y, i])

    # sort
    if cfg.sort:
        work = work.sort_values([i, t]).reset_index(drop=True)
        actions.append({"action": "sort", "by": [i, t]})

    # before集計（freq推定はこの時点でも可能）
    before_freq_guess = infer_frequency(work, timestamp_col=t, item_id_col=i, mode=cfg.freq_infer, fallback=cfg.freq_fallback)
    before_summary = _summary(work, timestamp_col=t, target_col=y, item_id_col=i, freq=before_freq_guess)

    # 重複解消
    work = _deduplicate(work, cfg, actions)

    # freq
    if cfg.freq is not None:
        freq: Union[str, pd.Timedelta] = cfg.freq
        actions.append({"action": "freq", "mode": "fixed", "freq": str(cfg.freq)})
    else:
        freq = infer_frequency(work, timestamp_col=t, item_id_col=i, mode=cfg.freq_infer, fallback=cfg.freq_fallback)
        actions.append({"action": "freq", "mode": f"infer:{cfg.freq_infer}", "freq": _freq_to_str(freq)})

    # 欠損時刻の穴埋め
    work = _fill_missing_grid(work, cfg, freq, actions)

    # 外れ値
    work = _handle_outliers(work, cfg, actions)

    # 欠損補完
    work = _impute_target(work, cfg, actions)

    if cfg.sort:
        work = work.sort_values([i, t]).reset_index(drop=True)

    after_summary = _summary(work, timestamp_col=t, target_col=y, item_id_col=i, freq=freq)

    report = PreprocessReport(
        ok=True,
        inferred_freq=_freq_to_str(freq),
        before=before_summary,
        after=after_summary,
        actions=actions,
    )
    return work, report


def report_to_dataframe(report: PreprocessReport) -> pd.DataFrame:
    keys = set(report.before.keys()) | set(report.after.keys())
    rows = []
    for k in sorted(keys):
        rows.append({"metric": k, "before": report.before.get(k), "after": report.after.get(k)})
    return pd.DataFrame(rows)
