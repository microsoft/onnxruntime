#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: please install azcli and run "az login" before running this script!

import re
import shutil
import sys
from pathlib import Path

# --- Helper Functions for Updating Files ---


def update_versioning_md(file_path: Path, new_version: str):
    """Updates the version table in Versioning.md."""
    print(f"Checking '{file_path.name}' for version updates...")
    if not file_path.exists():
        print(f"Warning: File not found at '{file_path}'. Skipping.")
        return
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
        new_row_parts = [" " + part.replace("-", " ") + " " for part in header_separator.split("|")]
        new_row_parts[1] = f" {new_version} "  # Set the new version
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
    if not file_path.exists():
        print(f"Warning: File not found at '{file_path}'. Skipping.")
        return
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
    if not file_path.exists():
        print(f"Warning: File not found at '{file_path}'. Skipping.")
        return
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
        new_content = re.sub(r"__version__\s*=\s*[\"'][\d.]+[\"']", f'__version__ = "{new_version}"', content)
        file_path.write_text(new_content)
        print("Update complete.")
    else:
        print("Version is already up to date.")


def update_npm_packages(js_root: Path, new_version: str):
    """Updates versions for all NPM packages in the js directory."""
    print("\nUpdating NPM package versions...")

    # This script assumes a 'util' module is available in the search path.
    try:
        from util import is_windows
        from util import run as run_command
    except ImportError:
        print("Error: Could not import 'is_windows' and 'run' from a 'util' module.", file=sys.stderr)
        print("Please ensure the 'util' module is in Python's search path.", file=sys.stderr)
        return

    command_prefix = []
    # Check if node and npm are directly available in the system's PATH.
    if shutil.which("node") and shutil.which("npm"):
        print("Found node and npm in PATH.")
    # If not, and if on Linux, check if 'fnm' is available.
    elif sys.platform.startswith("linux") and shutil.which("fnm"):
        print("node/npm not in PATH. Found 'fnm' on Linux, will use it to run commands.")
        nvmrc_path = js_root / ".nvmrc"
        # Check for .nvmrc file.
        if not nvmrc_path.exists():
            print(f"Error: 'fnm' is being used, but the version file '{nvmrc_path}' was not found.", file=sys.stderr)
            print(
                "Please create a .nvmrc file in the 'js' directory with the desired Node.js version.", file=sys.stderr
            )
            return

        node_version = nvmrc_path.read_text().strip()
        print(f"Found node version '{node_version}' in .nvmrc.")

        # Ensure the required node version is installed by fnm.
        print(f"Ensuring Node.js version '{node_version}' is installed via fnm...")
        run_command("fnm", "install", node_version, cwd=js_root)

        print(f"Using Node.js version '{node_version}' with fnm.")
        command_prefix = ["fnm", "exec", f"--using={node_version}", "--"]
    # If neither is available, skip the NPM updates.
    else:
        print("Error: Could not find 'node' and 'npm' in your PATH.", file=sys.stderr)
        if sys.platform.startswith("linux"):
            print("Hint: Install 'fnm' (Fast Node Manager) to manage Node.js versions.", file=sys.stderr)
        print("Skipping NPM package updates.", file=sys.stderr)
        return

    def run_npm(args, cwd):
        """Helper to run npm commands, prepending fnm if necessary."""
        full_command = command_prefix + list(args)

        if is_windows():
            run_command("cmd", "/c", *full_command, cwd=cwd)
        else:
            run_command(*full_command, cwd=cwd)

    packages = ["common", "node", "web", "react_native"]

    for package in packages:
        print(f"\n--- Updating package: {package} ---")
        # Use npm's --prefix argument and run from js_root.
        # --allow-same-version prevents an error if the version is already correct.
        run_npm(["npm", "--prefix", package, "version", new_version, "--allow-same-version"], cwd=js_root)
        run_npm(["npm", "--prefix", package, "install", "--package-lock-only", "--ignore-scripts"], cwd=js_root)

    print("\n--- Finalizing JS versions and formatting ---")
    run_npm(["npm", "ci"], cwd=js_root)
    for package in packages:
        run_npm(["npm", "run", "update-version", package], cwd=js_root)

    run_npm(["npm", "run", "format"], cwd=js_root)
    print("NPM package updates complete.")


# Define repository root relative to the script's location
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent


def update_version():
    """Main function to read the new version and orchestrate updates across the project."""
    # Read and validate the new version from VERSION_NUMBER
    version_file = REPO_DIR / "VERSION_NUMBER"
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
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Target version to set: {new_version}\n")

    # Update files using absolute paths from REPO_DIR
    update_versioning_md(REPO_DIR / "docs" / "Versioning.md", new_version)
    update_readme_rst(REPO_DIR / "docs" / "python" / "README.rst", new_version)
    update_init_py(REPO_DIR / "onnxruntime" / "__init__.py", new_version)

    # Update all NPM packages
    update_npm_packages(REPO_DIR / "js", new_version)


if __name__ == "__main__":
    update_version()
