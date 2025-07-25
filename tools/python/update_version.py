#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import re
import sys
from pathlib import Path

# --- Helper Functions for Updating Files ---

def update_versioning_md(file_path: Path, new_version: str):
    """Updates the version table in Versioning.md."""
    print(f"Checking '{file_path.name}' for version updates...")
    content = file_path.read_text()
    
    # Find the first version number in the markdown table
    match = re.search(r"^\| ([\d.]+) \|", content, re.MULTILINE)
    if not match:
        print(f"Warning: Could not find current version in '{file_path.name}'. Skipping.")
        return

    current_version = match.group(1)
    print(f"Found current version: {current_version}")

    if new_version != current_version:
        print(f"Updating version in '{file_path.name}' to {new_version}...")
        # Prepare the new row by duplicating the header separator line's structure
        header_separator_match = re.search(r"(\r\n?|\n)(\|---\|.*)", content)
        if not header_separator_match:
            print(f"Warning: Could not find table header separator in '{file_path.name}'. Skipping.")
            return

        header_separator = header_separator_match.group(2)
        # Create a new row based on the separator, replacing dashes with spaces and adding the version
        new_row_parts = [" " + part.replace("-", " ") + " " for part in header_separator.split('|')]
        new_row_parts[1] = f" {new_version} " # Set the new version
        new_row = "|".join(new_row_parts)

        # Insert the new row right after the header separator line
        insertion_point = header_separator_match.end(0)
        new_content = content[:insertion_point] + "\n" + new_row + content[insertion_point:]
        file_path.write_text(new_content)
        print("Update complete.")
    else:
        print("Version is already up to date.")


def update_readme_rst(file_path: Path, new_version: str):
    """Updates the release history in the Python README.rst."""
    print(f"Checking '{file_path.name}' for version updates...")
    content = file_path.read_text()

    # Find the first version header in the file
    match = re.search(r"^([\d.]+)", content, re.MULTILINE)
    if not match:
        print(f"Warning: Could not find current version in '{file_path.name}'. Skipping.")
        return

    current_version = match.group(1)
    print(f"Found current version: {current_version}")

    if new_version != current_version:
        print(f"Updating version in '{file_path.name}' to {new_version}...")
        new_header = f"{new_version}\n{'^' * len(new_version)}"
        release_notes = f"Release Notes : https://github.com/Microsoft/onnxruntime/releases/tag/v{new_version}"
        new_section = f"{new_header}\n\n{release_notes}\n\n"
        
        # Insert the new section before the first version header found
        insertion_point = match.start(0)
        new_content = content[:insertion_point] + new_section + content[insertion_point:]
        file_path.write_text(new_content)
        print("Update complete.")
    else:
        print("Version is already up to date.")


def update_init_py(file_path: Path, new_version: str):
    """Updates the __version__ variable in the project's __init__.py."""
    print(f"Checking '{file_path.name}' for version updates...")
    content = file_path.read_text()

    # Find the __version__ line
    match = re.search(r"__version__\s*=\s*[\"']([\d.]+)[\"']", content)
    if not match:
        print(f"Warning: Could not find __version__ in '{file_path.name}'. Skipping.")
        return

    current_version = match.group(1)
    print(f"Found current version: {current_version}")

    if new_version != current_version:
        print(f"Updating version in '{file_path.name}' to {new_version}...")
        new_content = re.sub(
            r"__version__\s*=\s*[\"'][\d.]+[\"']",
            f'__version__ = "{new_version}"',
            content
        )
        file_path.write_text(new_content)
        print("Update complete.")
    else:
        print("Version is already up to date.")


def update_npm_packages(js_root: Path, new_version: str):
    """Updates versions for all NPM packages in the js directory."""
    print("\nUpdating NPM package versions...")
    
    # Nested import and function to keep it self-contained as in original script
    from util import is_windows, run as run_command

    def run_npm(args, cwd):
        if is_windows():
            # In modern shell environments on Windows, `cmd /c` is often not needed
            # but we keep it for compatibility with the original script's intent.
            run_command("cmd", "/c", *args, cwd=cwd)
        else:
            run_command(*args, cwd=cwd)

    # Check if node and npm are installed
    try:
        run_npm(["node", "--version"], cwd=js_root)
        run_npm(["npm", "--version"], cwd=js_root)
    except (FileNotFoundError, Exception) as e:
        print(f"Error: node or npm not found. Skipping NPM package updates. Details: {e}", file=sys.stderr)
        return

    packages = ["common", "node", "web", "react_native"]

    for package in packages:
        package_dir = js_root / package
        print(f"\n--- Updating package: {package} ---")
        run_npm(["npm", "version", new_version], cwd=package_dir)
        run_npm(["npm", "install", "--package-lock-only", "--ignore-scripts"], cwd=package_dir)

    print("\n--- Finalizing JS versions and formatting ---")
    run_npm(["npm", "ci"], cwd=js_root)
    for package in packages:
        run_npm(["npm", "run", "update-version", package], cwd=js_root)
    
    run_npm(["npm", "run", "format"], cwd=js_root)
    print("NPM package updates complete.")


def update_version():
    """Main function to read the new version and orchestrate updates across the project."""
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent

    # 1. Read and validate the new version from VERSION_NUMBER
    version_file = root_dir / "VERSION_NUMBER"
    print(f"Reading new version from '{version_file}'...")
    try:
        new_version = version_file.read_text().strip()
    except FileNotFoundError:
        print(f"Error: '{version_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Validate that the version is in x.y.z format
    if not re.fullmatch(r"\d+\.\d+\.\d+", new_version):
        print(
            f"Error: Version '{new_version}' from '{version_file.name}' is not a valid x.y.z semantic version.",
            file=sys.stderr
        )
        sys.exit(1)
    
    print(f"Target version to set: {new_version}\n")

    # 2. Update files
    update_versioning_md(root_dir / "docs" / "Versioning.md", new_version)
    update_readme_rst(root_dir / "docs" / "python" / "README.rst", new_version)
    update_init_py(root_dir / "onnxruntime" / "__init__.py", new_version)

    # 3. Update all NPM packages
    update_npm_packages(root_dir / "js", new_version)


if __name__ == "__main__":
    update_version()