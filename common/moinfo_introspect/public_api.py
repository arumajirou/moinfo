# C:\moinfo\common\moinfo_introspect\public_api.py
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import sys
from dataclasses import dataclass, asdict
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd  # optional (for to_dataframe)
except Exception:  # pragma: no cover
    pd = None  # type: ignore


@dataclass
class ApiItem:
    qualname: str                 # fully-qualified name: timesfm.xxx.yyy
    kind: str                     # "class" | "function"
    module: str                   # defining module
    signature: Optional[str]      # signature (引数情報)
    doc_first_line: Optional[str] # docstringの先頭1行


def _first_line(doc: Optional[str]) -> Optional[str]:
    if not doc:
        return None
    s = doc.strip()
    return s.splitlines()[0].strip() if s else None


def _safe_signature(obj: Any) -> Optional[str]:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def _kind_of(obj: Any) -> Optional[str]:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    return None


def _iter_public_members(mod: ModuleType) -> List[Tuple[str, Any]]:
    """
    公開APIらしいものだけ列挙:
      - __all__ があればそれを優先
      - なければ "_" で始まらない属性
    """
    if hasattr(mod, "__all__") and isinstance(getattr(mod, "__all__"), (list, tuple)):
        names = [n for n in getattr(mod, "__all__") if isinstance(n, str)]
        out: List[Tuple[str, Any]] = []
        for n in names:
            if hasattr(mod, n):
                out.append((n, getattr(mod, n)))
        return out

    out: List[Tuple[str, Any]] = []
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        out.append((name, obj))
    return out


def collect_public_api(
    root_pkg_name: str,
    include_submodules: bool = True,
    kinds: Tuple[str, ...] = ("class", "function"),
) -> Dict[str, Any]:
    """
    パッケージの公開API(関数/クラス)を内省（inspect）で収集する。
    - optional dependency不足で import失敗するサブモジュールは記録してスキップ。
    """
    result: Dict[str, Any] = {
        "root": root_pkg_name,
        "python": sys.version,
        "modules_ok": [],
        "modules_failed": [],
        "api": [],  # list[dict]
    }

    root = importlib.import_module(root_pkg_name)

    def add_from_module(mod: ModuleType, mod_name_for_qual: str) -> None:
        for name, obj in _iter_public_members(mod):
            kind = _kind_of(obj)
            if kind is None or kind not in kinds:
                continue
            result["api"].append(
                asdict(
                    ApiItem(
                        qualname=f"{mod_name_for_qual}.{name}",
                        kind=kind,
                        module=getattr(obj, "__module__", mod_name_for_qual),
                        signature=_safe_signature(obj),
                        doc_first_line=_first_line(getattr(obj, "__doc__", None)),
                    )
                )
            )

    # Top-level
    add_from_module(root, root_pkg_name)

    # Submodules
    if include_submodules:
        pkg_paths = getattr(root, "__path__", None)
        if pkg_paths:
            for m in pkgutil.walk_packages(pkg_paths, prefix=f"{root_pkg_name}."):
                mod_name = m.name
                try:
                    mod = importlib.import_module(mod_name)
                    result["modules_ok"].append(mod_name)
                except Exception as e:
                    result["modules_failed"].append({"module": mod_name, "error": repr(e)})
                    continue
                add_from_module(mod, mod_name)

    # Dedup by qualname
    dedup: Dict[str, Dict[str, Any]] = {}
    for item in result["api"]:
        dedup[item["qualname"]] = item
    result["api"] = sorted(dedup.values(), key=lambda x: x["qualname"])

    return result


def write_json(data: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def to_dataframe(data: Dict[str, Any]):
    """
    pandas.DataFrameへ変換。pandas未導入なら例外。
    """
    if pd is None:
        raise RuntimeError("pandas が import できません。pip install pandas を実行してください。")
    return pd.DataFrame(data.get("api", []))
