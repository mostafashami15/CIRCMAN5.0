#!/usr/bin/env python3
"""
This script recursively replaces all occurrences of "timestamp" with "timestamp"
in all .py files in the project, while excluding specified directories.

Usage:
    python standardize_timestamp.py
"""

import pathlib

# Directories to exclude from processing; adjust as needed
EXCLUDE_DIRS = {".venv", "__pycache__", "node_modules", "logs", "dist", "build"}


def should_exclude(path: pathlib.Path) -> bool:
    """
    Return True if the file is located in one of the excluded directories.
    """
    return any(part in EXCLUDE_DIRS for part in path.parts)


def replace_in_file(filepath: pathlib.Path, old: str, new: str) -> None:
    """
    Read the file, replace occurrences of old with new,
    and write back the file if changes were made.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"Skipping non-UTF8 file: {filepath}")
        return

    if old in content:
        new_content = content.replace(old, new)
        filepath.write_text(new_content, encoding="utf-8")
        print(f"Modified: {filepath}")


def main():
    root = pathlib.Path(".")
    # Recursively find all .py files in the project
    for filepath in root.rglob("*.py"):
        if should_exclude(filepath):
            continue
        replace_in_file(filepath, "timestamp", "timestamp")


if __name__ == "__main__":
    main()
