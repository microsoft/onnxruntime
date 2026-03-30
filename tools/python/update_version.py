#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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


def _insert_end_of_version_marker(
    file_path: Path, struct_var_regex: str, old_api_ver: int, end_marker_suffix: str = "",
):
    """Inserts an 'End of Version N' marker before the struct's closing };."""
    content = file_path.read_text()
    name = file_path.name

    struct_match = re.search(struct_var_regex, content)
    if not struct_match:
        print(f"  [{name}] Could not find struct declaration. Skipping.")
        return

    close_match = re.search(r"^};", content[struct_match.start():], re.MULTILINE)
    if not close_match:
        print(f"  [{name}] Could not find struct closing " + "'};'. Skipping.")
        return

    struct_body = content[struct_match.start():struct_match.start() + close_match.start()]

    markers = list(re.finditer(r"//\s*End of Version (\d+)", struct_body))
    if not markers:
        print(f"  [{name}] No 'End of Version' markers found. Skipping.")
        return

    last_marker_ver = int(markers[-1].group(1))
    if last_marker_ver >= old_api_ver:
        print(f"  [{name}] Version {old_api_ver} already finalized (last marker: v{last_marker_ver}).")
        return

    new_entries = re.findall(r"&\w+::(\w+)", struct_body[markers[-1].end():])
    if not new_entries:
        print(f"  [{name}] No new entries after 'End of Version {last_marker_ver}'. Skipping.")
        return

    print(f"  [{name}] Capping {len(new_entries)} new entries as version {old_api_ver}.")
    marker_line = f"    // End of Version {old_api_ver} - DO NOT MODIFY ABOVE{end_marker_suffix}\n"
    close_abs_pos = struct_match.start() + close_match.start()
    content = content[:close_abs_pos] + marker_line + content[close_abs_pos:]
    file_path.write_text(content)


def _insert_api_size_assert(file_path: Path, struct_type: str, struct_var_regex: str, old_api_ver: int):
    """Counts all API entries in the struct and adds a static_assert for the version size."""
    content = file_path.read_text()
    name = file_path.name

    struct_match = re.search(struct_var_regex, content)
    if not struct_match:
        return

    close_match = re.search(r"^};", content[struct_match.start():], re.MULTILINE)
    if not close_match:
        return

    struct_body = content[struct_match.start():struct_match.start() + close_match.start()]

    marker = re.search(rf"//\s*End of Version {old_api_ver}\b", struct_body)
    if not marker:
        return

    all_entries = re.findall(r"&\w+::(\w+)", struct_body[:marker.start()])
    if not all_entries:
        return

    last_entry_name = all_entries[-1]
    last_offset = len(all_entries) - 1

    assert_matches = list(re.finditer(
        rf"static_assert\(offsetof\({re.escape(struct_type)},.*?;\n", content, re.DOTALL,
    ))
    if assert_matches:
        insert_pos = assert_matches[-1].end()
        assert_text = (
            f"static_assert(offsetof({struct_type}, {last_entry_name})"
            f" / sizeof(void*) == {last_offset},\n"
            f'              "Size of version {old_api_ver} API cannot change");\n'
        )
        print(f"  [{name}] Inserting static_assert for {last_entry_name} at offset {last_offset}.")
        content = content[:insert_pos] + assert_text + content[insert_pos:]
        file_path.write_text(content)
    else:
        print(f"  [{name}] Warning: No existing static_assert for {struct_type}. Skipping.")


def update_onnxruntime_c_api_cc(file_path: Path, new_version: str):
    """Updates version references in onnxruntime_c_api.cc."""
    print(f"\nChecking for version updates...")
    if not file_path.exists():
        print(f"Warning: File not found at '{file_path}'. Skipping.")
        return
    content = file_path.read_text()
    new_api_ver = new_version.split(".")[1]
    old_api_ver = int(new_api_ver) - 1

    # Rename ort_api_1_to_XX to match the new version
    decl_match = re.search(r"static constexpr OrtApi ort_api_1_to_(\d+)\b", content)
    if decl_match:
        print(f"Renaming 'ort_api_1_to_{decl_match.group(1)}' -> 'ort_api_1_to_{new_api_ver}'...")
        content = re.sub(rf"\bort_api_1_to_{decl_match.group(1)}\b", f"ort_api_1_to_{new_api_ver}", content)

    print(f"Updating ORT_VERSION static_assert to '{new_version}'...")
    content = re.sub(
        r'(static_assert\(std::string_view\(ORT_VERSION\) == )"[\d.]+"',
        rf'\g<1>"{new_version}"',
        content,
    )

    file_path.write_text(content)

    ort_api_regex = r"static constexpr OrtApi ort_api_1_to_\d+\s*=\s*\{"
    _insert_end_of_version_marker(file_path, ort_api_regex, old_api_ver, " (see above text for more information)")
    _insert_api_size_assert(file_path, "OrtApi", ort_api_regex, old_api_ver)

    print("Update complete.")


def update_cpp_api_files(repo_dir: Path, new_version: str):
    """Updates all C++ API struct files for the version bump."""
    old_api_ver = int(new_version.split(".")[1]) - 1
    session_dir = repo_dir / "onnxruntime" / "core" / "session"

    print("\nUpdating C++ API files...")

    update_onnxruntime_c_api_cc(session_dir / "onnxruntime_c_api.cc", new_version)

    api_files = [
        (session_dir / "plugin_ep" / "ep_api.cc", "OrtEpApi", r"static constexpr OrtEpApi \w+\s*=\s*\{"),
        (session_dir / "compile_api.cc", "OrtCompileApi", r"static constexpr OrtCompileApi \w+\s*=\s*\{"),
        (session_dir / "interop_api.cc", "OrtInteropApi", r"static constexpr OrtInteropApi \w+\s*=\s*\{"),
        (session_dir / "model_editor_c_api.cc", "OrtModelEditorApi", r"static constexpr OrtModelEditorApi \w+\s*=\s*\{"),
    ]

    for file_path, struct_type, pattern in api_files:
        if not file_path.exists():
            print(f"  Warning: {file_path.name} not found. Skipping.")
            continue
        _insert_end_of_version_marker(file_path, pattern, old_api_ver)
        _insert_api_size_assert(file_path, struct_type, pattern, old_api_ver)


def update_npm_packages(js_root: Path, new_version: str):
    """Updates versions for all NPM packages in the js directory."""
    print("\nUpdating NPM package versions...")

    # This script assumes a 'util' module is available in the search path.
    try:
        from util import is_windows  # noqa: PLC0415
        from util import run as run_command  # noqa: PLC0415
    except ImportError:
        print("Error: Could not import 'is_windows' and 'run' from a 'util' module.", file=sys.stderr)
        print("Please ensure the 'util' module is in Python's search path.", file=sys.stderr)
        return

    command_prefix = []
    # Check if node and npm are directly available in the system's PATH.
    if shutil.which("node") and shutil.which("npm"):
        print("Found node and npm in PATH.")
    # If not, and if on Linux, check if 'fnm' is available.
    elif shutil.which("fnm"):
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
        print(full_command)
        run_command(*full_command, cwd=cwd)

    npm_exe = "npm.cmd" if is_windows() else "npm"
    packages = ["common", "node", "web", "react_native"]

    for package in packages:
        print(f"\n--- Updating package: {package} ---")
        # Use npm's --prefix argument and run from js_root.
        # --allow-same-version prevents an error if the version is already correct.
        run_npm([npm_exe, "--prefix", package, "version", new_version, "--allow-same-version"], cwd=js_root)
        run_npm([npm_exe, "--prefix", package, "install", "--package-lock-only", "--ignore-scripts"], cwd=js_root)

    print("\n--- Finalizing JS versions and formatting ---")
    run_npm([npm_exe, "ci"], cwd=js_root)
    for package in packages:
        run_npm([npm_exe, "run", "update-version", package], cwd=js_root)

    run_npm([npm_exe, "run", "format"], cwd=js_root)
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

    # Update C++ API files
    update_cpp_api_files(REPO_DIR, new_version)

    # Update all NPM packages
    update_npm_packages(REPO_DIR / "js", new_version)


if __name__ == "__main__":
    update_version()
