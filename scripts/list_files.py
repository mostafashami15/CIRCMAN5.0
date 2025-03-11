# list_file.py
from pathlib import Path


def list_files(directory, exclude_dirs=None, exclude_ext=None):
    """List all files in the directory, excluding specified directories and file types."""
    if exclude_dirs is None:
        exclude_dirs = {".conda", ".git", ".venv"}  # Directories to ignore
    if exclude_ext is None:
        exclude_ext = {".pyc", ".log"}  # File extensions to ignore

    file_list = []

    for file in Path(directory).rglob("*"):
        if file.is_file():
            # Exclude specific directories
            if any(excluded in file.parts for excluded in exclude_dirs):
                continue
            # Exclude specific file extensions
            if file.suffix in exclude_ext:
                continue
            file_list.append(str(file))

    return file_list


# Root directory to scan
directory = "."  # Change this to the desired root path

# Get the filtered file list
all_files = list_files(directory)

# Save the output to a file
output_file = "file_list.txt"
with open(output_file, "w") as f:
    for file in all_files:
        f.write(file + "\n")

print(f"✅ File list saved to {output_file}")
