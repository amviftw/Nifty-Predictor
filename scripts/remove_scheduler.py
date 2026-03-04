#!/usr/bin/env python3
"""
Remove all Nifty Predictor scheduled tasks from Windows Task Scheduler.

Usage (run as Administrator):
    python -m scripts.remove_scheduler
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.setup_scheduler import TASKS, remove_task


def main():
    print("=" * 60)
    print("REMOVING NIFTY PREDICTOR SCHEDULED TASKS")
    print("=" * 60)

    for task in TASKS:
        remove_task(task["name"])

    print("\nAll tasks removed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
