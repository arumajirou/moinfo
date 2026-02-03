# C:\moinfo\common\moinfo_core\config.py
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(text: str) -> Dict[str, Any]:
    """
    YAML対応。PyYAML が無い場合は明示的にエラーにする（静かに壊れるのを防ぐ）。
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "YAMLを読むには PyYAML が必要です: `pip install pyyaml`"
        ) from e
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("設定ファイルのトップレベルは dict（マップ）である必要があります。")
    return data


def _load_json(text: str) -> Dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("設定JSONのトップレベルは dict（オブジェクト）である必要があります。")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深いマージ（dict を再帰的に結合）。override が優先。
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce_scalar(value: str) -> Any:
    """
    環境変数オーバーライド等の文字列を、可能なら bool/int/float/null に変換。
    """
    s = value.strip()
    lower = s.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none"):
        return None
    # int
    try:
        if s.startswith("0") and len(s) > 1 and s.isdigit():
            # 先頭ゼロは文字列として扱う（ID系の事故防止）
            return s
        return int(s)
    except Exception:
        pass
    # float
    try:
        return float(s)
    except Exception:
        return s


def apply_env_overrides(cfg: Dict[str, Any], prefix: str = "MOINFO_") -> Dict[str, Any]:
    """
    環境変数で設定上書き。
    例: MOINFO_EXPERIMENT__SEED=123
        -> cfg["experiment"]["seed"] = 123
    セパレータ: "__"
    """
    out = copy.deepcopy(cfg)
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):]
        parts = [p.strip().lower() for p in path.split("__") if p.strip()]
        if not parts:
            continue

        cur: Any = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _coerce_scalar(val)
    return out


def _normalize_for_hash(obj: Any) -> Any:
    """
    設定のハッシュ用に正規化（順序安定・Path→strなど）。
    """
    if isinstance(obj, dict):
        # key順を安定化
        return {str(k): _normalize_for_hash(obj[k]) for k in sorted(obj.keys(), key=lambda x: str(x))}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(x) for x in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def config_fingerprint(cfg: Dict[str, Any], length: int = 8) -> str:
    """
    同じ設定なら同じハッシュになる（出力パス固定用）。
    """
    normalized = _normalize_for_hash(cfg)
    blob = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(blob.encode("utf-8")).hexdigest()
    return h[:length]


def _default_config() -> Dict[str, Any]:
    """
    最小デフォルト。必要に応じて増やす。
    """
    return {
        "experiment": {
            "name": "default",
            "seed": 42,
        },
        "paths": {
            # ここは paths.py 側で推定もできるが、固定したいならここを使う
            "root": "C:/moinfo",
            "output_root": "C:/moinfo/_artifacts/runs",
            "run_mode": "deterministic",  # deterministic | timestamped
            "allow_overwrite": True,      # deterministic時に同じdirへ上書きOKか
        },
    }


def _resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    設定内のパスを正規化（Windows/Posix混在を吸収）。
    """
    out = copy.deepcopy(cfg)

    root = out.get("paths", {}).get("root", None)
    if root is not None:
        out["paths"]["root"] = str(Path(root))

    out_root = out.get("paths", {}).get("output_root", None)
    if out_root is not None:
        out["paths"]["output_root"] = str(Path(out_root))

    # データ系のパスも将来ここで統一できる（例: data.train_path 等）
    # 今は最小限に留める

    return out


def load_config(
    config_path: str | Path,
    *,
    defaults: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    env_prefix: str = "MOINFO_",
) -> Dict[str, Any]:
    """
    設定ロード（YAML/JSON）→ デフォルト合成 → overrides 合成 → env 上書き → パス正規化
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    text = _read_text(config_path)
    suffix = config_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        raw = _load_yaml(text)
    elif suffix == ".json":
        raw = _load_json(text)
    else:
        raise ValueError(f"未対応の設定拡張子です: {suffix}（.yaml/.yml/.json）")

    base = defaults if defaults is not None else _default_config()
    merged = deep_merge(base, raw)
    if overrides:
        merged = deep_merge(merged, overrides)
    merged = apply_env_overrides(merged, prefix=env_prefix)
    merged = _resolve_paths(merged)

    # run_id 生成のために fingerprint を埋め込む（runner側で参照しやすく）
    merged.setdefault("experiment", {})
    merged["experiment"].setdefault("name", "default")
    merged["experiment"]["fingerprint"] = config_fingerprint(merged)

    return merged


def save_effective_config(cfg: Dict[str, Any], output_dir: str | Path, filename: str = "effective_config.json") -> Path:
    """
    実行に使った“最終設定”を出力先へ保存（再現性の要）。
    YAMLが欲しければ filename を .yaml にして、PyYAMLがあれば YAMLでも保存できる。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("YAML保存には PyYAML が必要です: `pip install pyyaml`") from e
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    # JSONデフォルト
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
