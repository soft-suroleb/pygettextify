"""Utility functions for the task manager."""

import os
import json
import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DATA_DIR = os.path.join(os.path.expanduser("~"), ".taskmaster")
SAVE_FILE = "tasks.json"

logger = logging.getLogger(__name__)


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("Created data directory.")


def save_tasks(tasks):
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, SAVE_FILE)
    data = []
    for task in tasks:
        data.append({
            "title": task.title,
            "description": task.description,
            "priority": task.priority,
            "done": task.done,
        })
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Tasks saved to {filepath}.")
    except OSError as e:
        print(f"Failed to save tasks: {e}")
        logger.error("Save failed: %s", e)


def load_tasks(filepath):
    if not os.path.exists(filepath):
        print("No saved tasks found. Starting fresh.")
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} tasks from file.")
        return data
    except json.JSONDecodeError:
        print("Warning: Task file is corrupted. Starting with an empty list.")
        return []
    except OSError as e:
        print(f"Error reading task file: {e}")
        return []


def format_date(dt):
    return dt.strftime(DATE_FORMAT)


def confirm_action(message):
    """Ask user for confirmation before a destructive action."""
    response = input(f"{message} (yes/no): ").strip().lower()
    if response in ("yes", "y"):
        return True
    print("Action cancelled.")
    return False
