# C:\moinfo\common\moinfo_core\logging.py
from __future__ import annotations

import json
import logging
import os
import platform
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class _ContextFilter(logging.Filter):
    """
    ログレコードに固定のコンテキスト値を注入する（run_id など）。
    """
    def __init__(self, context: Dict[str, Any]):
        super().__init__()
        self._context = context

    def filter(self, record: logging.LogRecord) -> bool:
        for k, v in self._context.items():
            setattr(record, k, v)
        return True


def _level_from_str(level: str) -> int:
    s = str(level).upper().strip()
    return getattr(logging, s, logging.INFO)


def setup_logging(
    *,
    logs_dir: str | Path,
    run_id: str,
    experiment_name: str = "default",
    fingerprint: str = "nohash",
    level: str = "INFO",
    console: bool = True,
    logger_name: str = "moinfo",
) -> logging.Logger:
    """
    - logs_dir に run.log / error.log を作る
    - フォーマットに run_id を埋める
    - Notebook で再実行してもハンドラが増殖しないようにガード
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(_level_from_str(level))
    logger.propagate = False  # rootへ二重出力を避ける

    # 既存ハンドラの増殖を防ぐ：同一logger_nameで再初期化する想定
    #（Notebookでよく起きる）
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # コンテキスト注入
    ctx = {
        "run_id": run_id,
        "exp": experiment_name,
        "fp": fingerprint,
    }
    logger.addFilter(_ContextFilter(ctx))

    fmt = "%(asctime)s | %(levelname)s | %(run_id)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # run.log（全ログ）
    run_log_path = logs_dir / "run.log"
    fh_all = logging.FileHandler(run_log_path, encoding="utf-8")
    fh_all.setLevel(_level_from_str(level))
    fh_all.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(fh_all)

    # error.log（ERROR以上）
    err_log_path = logs_dir / "error.log"
    fh_err = logging.FileHandler(err_log_path, encoding="utf-8")
    fh_err.setLevel(logging.ERROR)
    fh_err.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(fh_err)

    # console（Notebook表示）
    if console:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(_level_from_str(level))
        sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(sh)

    # 起動メタ情報も残す（追跡の要）
    meta = {
        "timestamp_utc": _utc_now_iso(),
        "run_id": run_id,
        "experiment_name": experiment_name,
        "fingerprint": fingerprint,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "user": os.environ.get("USERNAME") or os.environ.get("USER"),
    }
    _write_json(logs_dir / "run_meta.json", meta)

    logger.info("logger initialized")
    return logger


@dataclass
class CrashContext:
    """
    例外時に「どの設定/どのデータで落ちたか」をJSONへ残すための入れ物。
    """
    run_id: str
    run_dir: str
    logs_dir: str
    effective_config_path: Optional[str] = None
    data_path: Optional[str] = None
    step: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


def write_crash_context(
    *,
    crash: CrashContext,
    exc: BaseException,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    crash_context_YYYYmmdd-HHMMSS.json を logs_dir に出す。
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(crash.logs_dir) / f"crash_context_{ts}.json"

    payload: Dict[str, Any] = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "run_id": crash.run_id,
        "run_dir": crash.run_dir,
        "logs_dir": crash.logs_dir,
        "effective_config_path": crash.effective_config_path,
        "data_path": crash.data_path,
        "step": crash.step,
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "traceback": traceback.format_exc(),
        "env": {
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "platform": platform.platform(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "extra": crash.extra or {},
    }

    _write_json(out, payload)

    if logger:
        logger.error("exception captured -> %s", str(out))
    return out


class experiment_guard:
    """
    with experiment_guard(...):
        ...  # 例外が起きたら crash_context を出してログに残す

    reraise=True なら例外を再送出（CI/バッチ向け）
    Notebookで続行したいなら reraise=False か try/except を使う
    """
    def __init__(
        self,
        *,
        logger: logging.Logger,
        run_id: str,
        run_dir: str | Path,
        logs_dir: str | Path,
        effective_config_path: Optional[str] = None,
        data_path: Optional[str] = None,
        step: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        reraise: bool = True,
    ):
        self.logger = logger
        self.crash = CrashContext(
            run_id=str(run_id),
            run_dir=str(run_dir),
            logs_dir=str(logs_dir),
            effective_config_path=effective_config_path,
            data_path=data_path,
            step=step,
            extra=extra,
        )
        self.reraise = reraise

    def __enter__(self):
        self.logger.info("experiment start | step=%s | data=%s", self.crash.step, self.crash.data_path)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            self.logger.info("experiment end (success)")
            return False

        # 例外あり
        self.logger.exception("experiment end (failed): %s", str(exc))
        write_crash_context(crash=self.crash, exc=exc, logger=self.logger)

        # reraise=False なら例外を握りつぶしてNotebook継続
        return (not self.reraise)
