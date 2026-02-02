# C:\moinfo\libs\timesfm\02_src\moinfo_timesfm_ext\moinfo_timesfm_ext\introspect_timesfm.py
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class ApiRow:
    package: str
    module: str
    qualname: str
    name: str
    kind: str  # function | class | method | attribute | import_error
    signature: str
    doc_summary: str
    is_public: bool
    source_file: str
    source_line: Optional[int]
    import_error: str


def _doc_summary(obj: Any, max_chars: int = 300) -> str:
    doc = inspect.getdoc(obj) or ""
    doc = re.sub(r"\s+", " ", doc).strip()
    return (doc[: max_chars - 1] + "…") if len(doc) >= max_chars else doc


def _safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return ""


def _safe_source(obj: Any) -> Tuple[str, Optional[int]]:
    try:
        f = inspect.getsourcefile(obj) or ""
        lines, start = inspect.getsourcelines(obj)
        return f, int(start)
    except Exception:
        return "", None


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def _iter_submodules(package_mod) -> Iterable[str]:
    """Walk submodules under a package without importing everything upfront."""
    if not hasattr(package_mod, "__path__"):
        return []
    prefix = package_mod.__name__ + "."
    for m in pkgutil.walk_packages(package_mod.__path__, prefix=prefix):
        yield m.name


def _collect_from_module(
    package_name: str,
    module_name: str,
    include_private: bool,
    include_class_members: bool,
    max_doc_chars: int,
) -> List[ApiRow]:
    rows: List[ApiRow] = []

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        rows.append(
            ApiRow(
                package=package_name,
                module=module_name,
                qualname=module_name,
                name=module_name.split(".")[-1],
                kind="import_error",
                signature="",
                doc_summary="",
                is_public=False,
                source_file="",
                source_line=None,
                import_error=repr(e),
            )
        )
        return rows

    # 1) モジュール直下のメンバー
    for name, obj in inspect.getmembers(mod):
        is_public = _is_public_name(name)
        if (not include_private) and (not is_public):
            continue

        kind = None
        if inspect.isfunction(obj):
            kind = "function"
        elif inspect.isclass(obj):
            kind = "class"
        else:
            # 変数/定数/その他
            kind = "attribute"

        qualname = f"{module_name}.{name}"
        sig = _safe_signature(obj) if kind in ("function", "class") else ""
        doc = _doc_summary(obj, max_chars=max_doc_chars) if kind != "attribute" else ""
        src_file, src_line = _safe_source(obj) if kind in ("function", "class") else ("", None)

        rows.append(
            ApiRow(
                package=package_name,
                module=module_name,
                qualname=qualname,
                name=name,
                kind=kind,
                signature=sig,
                doc_summary=doc,
                is_public=is_public,
                source_file=src_file,
                source_line=src_line,
                import_error="",
            )
        )

        # 2) クラスなら public メソッドも取る（任意）
        if include_class_members and inspect.isclass(obj):
            for mname, mobj in inspect.getmembers(obj):
                mis_public = _is_public_name(mname)
                if (not include_private) and (not mis_public):
                    continue

                if not (inspect.isfunction(mobj) or inspect.ismethod(mobj) or inspect.isroutine(mobj)):
                    continue

                mqual = f"{module_name}.{name}.{mname}"
                msig = _safe_signature(mobj)
                mdoc = _doc_summary(mobj, max_chars=max_doc_chars)
                mfile, mline = _safe_source(mobj)

                rows.append(
                    ApiRow(
                        package=package_name,
                        module=module_name,
                        qualname=mqual,
                        name=mname,
                        kind="method",
                        signature=msig,
                        doc_summary=mdoc,
                        is_public=mis_public,
                        source_file=mfile,
                        source_line=mline,
                        import_error="",
                    )
                )

    return rows


def collect_public_api(
    package_name: str = "timesfm",
    include_submodules: bool = True,
    include_private: bool = False,
    include_class_members: bool = True,
    max_doc_chars: int = 300,
) -> pd.DataFrame:
    """
    指定パッケージの公開APIを列挙してDataFrame化。
    - include_submodules=True でサブモジュールも走査（import副作用が出る可能性あり）
    """
    pkg = importlib.import_module(package_name)

    module_names = [package_name]
    if include_submodules:
        module_names.extend(list(_iter_submodules(pkg)))

    all_rows: List[ApiRow] = []
    for m in module_names:
        all_rows.extend(
            _collect_from_module(
                package_name=package_name,
                module_name=m,
                include_private=include_private,
                include_class_members=include_class_members,
                max_doc_chars=max_doc_chars,
            )
        )

    df = pd.DataFrame([asdict(r) for r in all_rows])

    # 便利列：最終セグメント（検索しやすい）
    if not df.empty:
        df["last"] = df["qualname"].str.split(".").str[-1]
    return df


def save_api_df(df: pd.DataFrame, out_dir: str | Path, stem: str = "timesfm_public_api") -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    return {"csv": str(csv_path), "json": str(json_path)}


def main() -> int:
    p = argparse.ArgumentParser(description="Introspect a Python package public API and export as CSV/JSON.")
    p.add_argument("--package", default="timesfm", help="target package name (default: timesfm)")
    p.add_argument("--out", default=r"C:\moinfo\libs\timesfm\04_outputs\api", help="output directory")
    p.add_argument("--no-submodules", action="store_true", help="do not walk submodules")
    p.add_argument("--include-private", action="store_true", help="include private members (leading underscore)")
    p.add_argument("--no-class-members", action="store_true", help="do not include class methods")
    p.add_argument("--max-doc-chars", type=int, default=300, help="truncate docstring summary length")
    args = p.parse_args()

    df = collect_public_api(
        package_name=args.package,
        include_submodules=not args.no_submodules,
        include_private=args.include_private,
        include_class_members=not args.no_class_members,
        max_doc_chars=args.max_doc_chars,
    )
    paths = save_api_df(df, args.out, stem=f"{args.package}_public_api")
    print(f"[OK] rows={len(df)} csv={paths['csv']} json={paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
