"""Provenance fields recorded with every WB2 artifact (implementation.md §0 rule 8)."""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": _run("status", "--porcelain") not in ("", "unknown"),
    }


def environment() -> dict:
    import matplotlib
    import numpy
    import pandas
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
