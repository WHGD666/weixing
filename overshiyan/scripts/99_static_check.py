from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    compiled = compileall.compile_dir(ROOT / "src", quiet=1)
    compiled &= compileall.compile_dir(ROOT / "scripts", quiet=1)
    compiled &= compileall.compile_dir(ROOT / "submit/app", quiet=1)
    if not compiled:
        raise SystemExit("Python compilation failed")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    print("static_check=passed")


if __name__ == "__main__":
    main()
