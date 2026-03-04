#!/usr/bin/env python3
"""
Register Windows Task Scheduler tasks for the Nifty Predictor system.

Creates 3 scheduled tasks:
  1. NiftyPredictor_DailyPredict  - Weekdays 9:05 AM
  2. NiftyPredictor_WeeklyTrain   - Sundays 10:00 PM
  3. NiftyPredictor_WeeklyEval    - Saturdays 6:00 PM

Usage (run as Administrator):
    python -m scripts.setup_scheduler
    python -m scripts.setup_scheduler --remove   # Remove all tasks instead
"""

import sys
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_DIR / "storage" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Find Python executable
PYTHON_EXE = sys.executable
if not PYTHON_EXE:
    PYTHON_EXE = shutil.which("python") or shutil.which("python3")

TASK_PREFIX = "NiftyPredictor"

TASKS = [
    {
        "name": f"{TASK_PREFIX}_DailyPredict",
        "description": "Nifty 50 daily prediction - runs T-10 min before market open",
        "schedule_type": "WEEKLY",
        "days": "MON,TUE,WED,THU,FRI",
        "time": "09:05",
        "command": f'"{PYTHON_EXE}" -m scripts.daily_predict --email',
        "log_file": "daily_predict_scheduler.log",
    },
    {
        "name": f"{TASK_PREFIX}_WeeklyTrain",
        "description": "Nifty 50 weekly model retraining",
        "schedule_type": "WEEKLY",
        "days": "SUN",
        "time": "22:00",
        "command": f'"{PYTHON_EXE}" -m scripts.train_models --email',
        "log_file": "train_models_scheduler.log",
    },
    {
        "name": f"{TASK_PREFIX}_WeeklyEval",
        "description": "Nifty 50 weekly signal evaluation report",
        "schedule_type": "WEEKLY",
        "days": "SAT",
        "time": "18:00",
        "command": f'"{PYTHON_EXE}" -m scripts.evaluate_signals --days 7 --email',
        "log_file": "evaluate_signals_scheduler.log",
    },
]


def create_task(task: dict):
    """Register a single task using schtasks.exe."""
    name = task["name"]
    log_path = LOGS_DIR / task["log_file"]

    # Build a batch wrapper that sets the working directory and logs output
    bat_path = PROJECT_DIR / "scripts" / f"{name}.bat"
    bat_content = (
        f"@echo off\r\n"
        f"cd /d \"{PROJECT_DIR}\"\r\n"
        f"{task['command']} >> \"{log_path}\" 2>&1\r\n"
    )
    bat_path.write_text(bat_content, encoding="utf-8")

    # schtasks command
    cmd = [
        "schtasks", "/Create",
        "/TN", name,
        "/TR", f'"{bat_path}"',
        "/SC", task["schedule_type"],
        "/D", task["days"],
        "/ST", task["time"],
        "/F",  # Force overwrite if exists
        "/RL", "HIGHEST",
    ]

    print(f"  Creating task: {name}")
    print(f"    Schedule: {task['days']} at {task['time']}")
    print(f"    Command: {task['command']}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"    [OK] Task created successfully")
    else:
        print(f"    [FAIL] {result.stderr.strip()}")
        if "Access is denied" in result.stderr:
            print("    >> Run this script as Administrator!")

    return result.returncode == 0


def remove_task(name: str):
    """Remove a scheduled task."""
    cmd = ["schtasks", "/Delete", "/TN", name, "/F"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  [OK] Removed: {name}")
    else:
        print(f"  [SKIP] {name} - {result.stderr.strip()}")

    # Remove batch file
    bat_path = PROJECT_DIR / "scripts" / f"{name}.bat"
    if bat_path.exists():
        bat_path.unlink()

    return result.returncode == 0


def verify_tasks():
    """Check the status of all scheduled tasks."""
    print("\nTask Verification:")
    print("-" * 60)

    for task in TASKS:
        cmd = ["schtasks", "/Query", "/TN", task["name"], "/FO", "LIST"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Parse key info from output
            lines = result.stdout.strip().split("\n")
            status = next(
                (l.split(":")[-1].strip() for l in lines if "Status" in l),
                "Unknown",
            )
            next_run = next(
                (l.split(":", 1)[-1].strip() for l in lines if "Next Run Time" in l),
                "Unknown",
            )
            print(f"  {task['name']}")
            print(f"    Status: {status} | Next Run: {next_run}")
        else:
            print(f"  {task['name']} - NOT FOUND")

    print("-" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup Windows Task Scheduler for Nifty Predictor"
    )
    parser.add_argument(
        "--remove", action="store_true",
        help="Remove all scheduled tasks instead of creating them",
    )
    args = parser.parse_args()

    print("=" * 60)

    if args.remove:
        print("REMOVING NIFTY PREDICTOR SCHEDULED TASKS")
        print("=" * 60)
        for task in TASKS:
            remove_task(task["name"])
        print("\nAll tasks removed.")
    else:
        print("SETTING UP NIFTY PREDICTOR SCHEDULED TASKS")
        print("=" * 60)
        print(f"\nPython: {PYTHON_EXE}")
        print(f"Project: {PROJECT_DIR}")
        print(f"Logs: {LOGS_DIR}")
        print()

        all_ok = True
        for task in TASKS:
            ok = create_task(task)
            if not ok:
                all_ok = False
            print()

        if all_ok:
            verify_tasks()
            print("\nAll tasks created successfully!")
            print("\nSchedule Summary:")
            print("  Mon-Fri 9:05 AM  -> Daily prediction email")
            print("  Saturday 6:00 PM -> Weekly evaluation email")
            print("  Sunday 10:00 PM  -> Weekly model retrain email")
        else:
            print("\nSome tasks failed. Try running as Administrator.")

    print("=" * 60)


if __name__ == "__main__":
    main()
