"""
crawler/scheduler_windows.ps1 — Phase 2.2 Windows Task Scheduler entry point

Run this once to register a daily job at 11:30 PM that runs the ingestion
engine (crawler/ingestion.py), which regenerates the synthetic catalogue and
appends new price-history rows.

Setup (one-time command prompt, run as administrator):
    # 1. Make sure the project root is added to the system PATH or pass it as
    #    $env:PROJECT_ROOT (default: the folder that holds this script).
    # 2. Register the task by double-clicking this file, or run the command below.

    $root = Split-Path -Parent $PSScriptRoot
    $action = New-ScheduledTaskAction -Execute "python.exe" -Argument "-u $root\crawler\intake.py" -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -Daily -At 23:30
    Register-ScheduledTask -TaskName "FlipkartAnalytics-Ingestion" -Action $action -Trigger $trigger -Description "Daily Flipkart analytics ingestion"

To run manually:
    python crawler\intake.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make the parent directory importable (same fix as ingestion.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler.ingestion import main as _main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Flipkart analytics ingestion (Windows Task Scheduler)")
    parser.add_argument("--db", default=str(Path("data/flipkart.db")), help="Path to the SQLite database")
    args = parser.parse_args()

    print(f"[{datetime.now().isoformat(sep=' ')}] Flipkart analytics ingestion started")
    _main()  # ingestion.py's own main() already prints the summary
    print(f"[{datetime.now().isoformat(sep=' ')}] Flipkart analytics ingestion finished")


if __name__ == "__main__":
    main()
