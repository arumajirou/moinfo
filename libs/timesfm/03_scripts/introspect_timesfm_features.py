# C:\moinfo\libs\timesfm\03_scripts\introspect_timesfm_features.py
from __future__ import annotations

import argparse
import os
import sys

# C:\moinfo\common を import 可能にする（どこから実行しても動くように）
THIS_FILE = os.path.abspath(__file__)
MOINF0_ROOT = os.path.abspath(os.path.join(os.path.dirname(THIS_FILE), "..", "..", ".."))
COMMON_DIR = os.path.join(MOINF0_ROOT, "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from moinfo_introspect.public_api import collect_public_api, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Introspect a Python package public API and export JSON.")
    ap.add_argument("--pkg", default="timesfm", help="Root package name (default: timesfm)")
    ap.add_argument(
        "--out",
        default=os.path.join(MOINF0_ROOT, "libs", "timesfm", "04_outputs", "api", "timesfm_public_api.json"),
        help="Output JSON file path",
    )
    ap.add_argument("--no-submodules", action="store_true", help="Do not scan submodules")
    ap.add_argument("--print", action="store_true", help="Print compact list")
    args = ap.parse_args()

    data = collect_public_api(args.pkg, include_submodules=not args.no_submodules)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_json(data, args.out)

    if args.print:
        print(f"== Public API items: {len(data['api'])} ==")
        for it in data["api"]:
            sig = it["signature"] or ""
            print(f"{it['kind']:8} {it['qualname']}{sig}")

        if data["modules_failed"]:
            print("\n== Modules failed to import (optional deps likely missing) ==")
            for mf in data["modules_failed"]:
                print(f"- {mf['module']}: {mf['error']}")

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
