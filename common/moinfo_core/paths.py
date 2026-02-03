# C:\moinfo\common\moinfo_core\paths.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _norm(p: str | Path) -> Path:
    # Windows/Posix混在の吸収（C:/moinfo と C:\moinfo をどちらも許す）
    return Path(str(p)).expanduser().resolve()


def infer_repo_root(start: Optional[str | Path] = None) -> Path:
    """
    ルート推定：
    1) 環境変数 MOINFO_ROOT があればそれ
    2) start（未指定ならこのファイル位置）から上へ辿って README.md または .gitignore を探す
    """
    env = os.environ.get("MOINFO_ROOT")
    if env:
        return _norm(env)

    cur = _norm(start) if start else Path(__file__).resolve()
    if cur.is_file():
        cur = cur.parent

    markers = ("README.md", ".gitignore")
    for _ in range(20):
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    # 最後の手段：CWD
    return Path.cwd().resolve()


@dataclass(frozen=True)
class Paths:
    root: Path
    output_root: Path

    run_id: str
    run_dir: Path

    tables_dir: Path
    figs_dir: Path
    logs_dir: Path
    api_dir: Path
    config_dir: Path

    def ensure_dirs(self) -> None:
        for d in (self.run_dir, self.tables_dir, self.figs_dir, self.logs_dir, self.api_dir, self.config_dir):
            d.mkdir(parents=True, exist_ok=True)


def _build_run_id(cfg: Dict[str, Any]) -> str:
    exp = cfg.get("experiment", {}) or {}
    name = str(exp.get("name", "default"))
    fp = str(exp.get("fingerprint", "nohash"))
    # 同設定なら同じ run_id（出力構造が固定される）
    return f"{name}-{fp}"


def build_paths(cfg: Dict[str, Any], *, create: bool = True) -> Paths:
    """
    cfg から出力パスを確定し、標準出力構造を返す。

    期待する cfg["paths"]:
      root: "C:/moinfo"
      output_root: "C:/moinfo/_artifacts/runs"
      run_mode: "deterministic" | "timestamped"
      allow_overwrite: bool
    """
    paths_cfg = cfg.get("paths", {}) or {}

    root = _norm(paths_cfg.get("root", infer_repo_root()))
    output_root = _norm(paths_cfg.get("output_root", root / "_artifacts" / "runs"))

    run_mode = str(paths_cfg.get("run_mode", "deterministic")).lower()
    allow_overwrite = bool(paths_cfg.get("allow_overwrite", True))

    base_run_id = _build_run_id(cfg)

    if run_mode == "timestamped":
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{base_run_id}-{ts}"
    else:
        # deterministic（同設定→同ディレクトリ）
        run_id = base_run_id

    run_dir = output_root / run_id

    if run_dir.exists() and (run_mode == "deterministic") and (not allow_overwrite):
        raise FileExistsError(
            f"出力先が既に存在します（allow_overwrite=false）: {run_dir}\n"
            f"同設定で上書きしたい場合は cfg['paths']['allow_overwrite']=true にしてください。"
        )

    tables_dir = run_dir / "tables"
    figs_dir = run_dir / "figs"
    logs_dir = run_dir / "logs"
    api_dir = run_dir / "api"
    config_dir = run_dir / "config"

    bundle = Paths(
        root=root,
        output_root=output_root,
        run_id=run_id,
        run_dir=run_dir,
        tables_dir=tables_dir,
        figs_dir=figs_dir,
        logs_dir=logs_dir,
        api_dir=api_dir,
        config_dir=config_dir,
    )

    if create:
        bundle.ensure_dirs()

    return bundle
