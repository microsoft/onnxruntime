# /----------------------------------------------------------------------------------\
# |                                                                                  |
# |                      VCPKG Automatic Update Script                               |
# |                                                                                  |
# \----------------------------------------------------------------------------------/
#
# Description:
#   This script automates the process of updating the VCPKG dependency to its
#   latest release within the onnxruntime repository. It performs the following
#   end-to-end workflow:
#
#   1.  Checks for prerequisites (Git, GitHub CLI).
#   2.  Syncs the local 'main' branch with the remote repository.
#   3.  Fetches the latest VCPKG release tag and its specific commit SHA from GitHub.
#   4.  Creates a new branch for the update (e.g., 'dev/update-vcpkg-2025.06.13').
#   5.  Calculates the SHA512 hash of the new release's .zip archive.
#   6.  Updates the version and hash in all relevant YAML workflow files.
#   7.  Updates the 'baseline' commit hash in 'cmake/vcpkg-configuration.json'.
#   8.  Updates the git clone tag in 'tools/ci_build/build.py'.
#   9.  (Windows Only) Runs the TerrapinRetrievalTool to register the new artifact.
#   10. Commits all tracked changes, pushes the new branch, and creates a
#       pull request using the GitHub CLI.
#
# Prerequisites:
#   - Python 3.13+
#   - Git installed and configured.
#   - GitHub CLI ('gh') installed and authenticated (`gh auth login`).
#   - 'requests' library for Python (`pip install requests`).
#


import hashlib
import json
import re
import ssl
import sys
import urllib.request
import subprocess
import shutil
import tempfile
import requests
from pathlib import Path

# --- Constants ---
VCPKG_REPO_API_URL = "https://api.github.com/repos/microsoft/vcpkg"
GITHUB_DIR = Path(".github")
VCPKG_CONFIG_JSON = Path("cmake/vcpkg-configuration.json")
BUILD_PY_SCRIPT = Path("tools/ci_build/build.py")
TERRAPIN_TOOL_PATH = r"C:\local\Terrapin\TerrapinRetrievalTool.exe"

# --- Regex Patterns ---
VCPKG_VERSION_PATTERN = re.compile(r"(vcpkg-version:\s*['\"]?)(\d{4}\.\d{2}\.\d{2})(['\"]?)")
VCPKG_HASH_PATTERN = re.compile(r"(vcpkg-hash:\s*['\"]?)([a-fA-F0-9]{128})(['\"]?)")
BUILD_PY_CLONE_PATTERN = re.compile(r'("git",\s*"clone",\s*"-b",\s*")(\d{4}\.\d{2}\.\d{2})(")')


def run_command(command: list[str]):
    """Runs a command and exits if it fails."""
    print(f"\n> {' '.join(command)}")
    try:
        result = subprocess.run(
            command, check=True, text=True, capture_output=True, encoding="utf-8"
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is it installed and in your PATH?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        print(f"--- STDOUT ---\n{e.stdout}", file=sys.stderr)
        print(f"--- STDERR ---\n{e.stderr}", file=sys.stderr)
        sys.exit(1)


def check_gh_installed():
    """Checks if the GitHub CLI is installed."""
    if not shutil.which("gh"):
        print("Error: The GitHub CLI ('gh') is not installed or not in the system's PATH.", file=sys.stderr)
        print("Please install it to continue: https://cli.github.com/", file=sys.stderr)
        sys.exit(1)


def get_latest_vcpkg_info() -> tuple[str, str]:
    """
    Fetches the latest vcpkg release tag and its specific commit SHA using a robust method.
    """
    print("Fetching latest vcpkg release information...")
    try:
        latest_release_url = f"{VCPKG_REPO_API_URL}/releases/latest"
        response = requests.get(latest_release_url, timeout=15)
        response.raise_for_status()
        release_data = response.json()
        latest_version = release_data["tag_name"]
        print(f"  - Latest Version Tag: {latest_version}")

        tag_ref_url = f"{VCPKG_REPO_API_URL}/git/refs/tags/{latest_version}"
        response = requests.get(tag_ref_url, timeout=15)
        response.raise_for_status()
        ref_data = response.json()
        tag_sha = ref_data["object"]["sha"]
        tag_type = ref_data["object"]["type"]

        commit_sha: str
        if tag_type == "commit":
            commit_sha = tag_sha
            print(f"  - Found lightweight tag, commit SHA: {commit_sha}")
        elif tag_type == "tag":
            print(f"  - Found annotated tag, resolving tag object: {tag_sha}")
            annotated_tag_url = f"{VCPKG_REPO_API_URL}/git/tags/{tag_sha}"
            response = requests.get(annotated_tag_url, timeout=15)
            response.raise_for_status()
            commit_sha = response.json()["object"]["sha"]
            print(f"  - Resolved to commit SHA: {commit_sha}")
        else:
            raise ValueError(f"Unknown tag type '{tag_type}' encountered.")

        return latest_version, commit_sha

    except requests.RequestException as e:
        print(f"Error fetching data from GitHub API: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error parsing GitHub API response: {e}", file=sys.stderr)
        sys.exit(1)


def get_zip_hash(version: str) -> str:
    """
    Downloads the vcpkg release zip file and computes its SHA512 hash.
    """
    zip_url = f"https://github.com/microsoft/vcpkg/archive/refs/tags/{version}.zip"
    print(f"\nDownloading and hashing {zip_url}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(zip_url, context=ctx) as response:
            if response.status != 200:
                print(f"Error: Failed to download file. Status: {response.status}", file=sys.stderr)
                sys.exit(1)
            file_content = response.read()
            sha512_hash = hashlib.sha512(file_content).hexdigest()
            print(f"  - SHA512 Hash: {sha512_hash}")
            return sha512_hash
    except urllib.error.URLError as e:
        print(f"Error downloading file: {e}", file=sys.stderr)
        sys.exit(1)


def update_github_workflows(new_version: str, new_hash: str):
    """
    Scans and updates vcpkg versions and hashes in YAML files under .github.
    """
    print("\nScanning .github/workflows for updates...")
    if not GITHUB_DIR.is_dir():
        print(f"Warning: Directory '{GITHUB_DIR}' not found. Skipping.", file=sys.stderr)
        return

    for yaml_file in GITHUB_DIR.rglob("*.yml"):
        print(f"  - Checking {yaml_file}...")
        try:
            content = yaml_file.read_text(encoding="utf-8")
            match = VCPKG_VERSION_PATTERN.search(content)
            if not match:
                continue

            current_version = match.group(2)
            print(f"    - Found vcpkg version: {current_version}")

            if current_version == new_version:
                print("    - Already up to date. Skipping.")
                continue

            content = VCPKG_VERSION_PATTERN.sub(rf"\g<1>{new_version}\g<3>", content)
            content = VCPKG_HASH_PATTERN.sub(rf"\g<1>{new_hash}\g<3>", content)
            yaml_file.write_text(content, encoding="utf-8")
            print(f"    - Updated to version {new_version}.")
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}", file=sys.stderr)


def update_vcpkg_config(new_commit_sha: str):
    """
    Updates the vcpkg baseline commit hash in cmake/vcpkg-configuration.json.
    """
    print(f"\nUpdating {VCPKG_CONFIG_JSON}...")
    if not VCPKG_CONFIG_JSON.is_file():
        print(f"Warning: File '{VCPKG_CONFIG_JSON}' not found. Skipping.", file=sys.stderr)
        return

    try:
        with VCPKG_CONFIG_JSON.open("r+", encoding="utf-8") as f:
            data = json.load(f)
            default_registry = data.get("default-registry")

            if not (default_registry and "baseline" in default_registry):
                print(f"Error: Could not find 'default-registry' with a 'baseline' key in JSON.", file=sys.stderr)
                return

            current_baseline = default_registry.get("baseline")
            print(f"  - Found baseline: {current_baseline}")

            if current_baseline == new_commit_sha:
                print("  - Already up to date. Skipping.")
            else:
                data["default-registry"]["baseline"] = new_commit_sha
                f.seek(0)
                json.dump(data, f, indent=4)
                f.write('\n')
                f.truncate()
                print(f"  - Updated baseline to {new_commit_sha}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing {VCPKG_CONFIG_JSON}: {e}", file=sys.stderr)


def update_build_script(new_version: str):
    """
    Updates the vcpkg git clone tag in tools/ci_build/build.py.
    """
    print(f"\nUpdating {BUILD_PY_SCRIPT}...")
    if not BUILD_PY_SCRIPT.is_file():
        print(f"Warning: File '{BUILD_PY_SCRIPT}' not found. Skipping.", file=sys.stderr)
        return

    try:
        content = BUILD_PY_SCRIPT.read_text(encoding="utf-8")
        match = BUILD_PY_CLONE_PATTERN.search(content)
        if not match:
            print("Warning: Could not find vcpkg clone command in build script.", file=sys.stderr)
            return

        current_version = match.group(2)
        print(f"  - Found version tag: {current_version}")

        if current_version == new_version:
            print("  - Already up to date. Skipping.")
        else:
            new_content = BUILD_PY_CLONE_PATTERN.sub(rf'\g<1>{new_version}\g<3>', content)
            BUILD_PY_SCRIPT.write_text(new_content, encoding="utf-8")
            print(f"  - Updated clone tag to {new_version}")
    except Exception as e:
        print(f"Error processing {BUILD_PY_SCRIPT}: {e}", file=sys.stderr)

def run_terrapin_tool(version: str, sha512_hash: str):
    """
    Runs the Terrapin Retrieval Tool to register the vcpkg artifact.
    This function is Windows-specific.
    """
    print("\n--- Running Terrapin Retrieval Tool ---")
    
    terrapin_path = Path(TERRAPIN_TOOL_PATH)
    if not terrapin_path.exists():
        print(f"Warning: Terrapin tool not found at '{terrapin_path}'. Skipping.", file=sys.stderr)
        return

    # Create a temporary file path for the download
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / f"vcpkg-{version}.zip"
        
        command = [
            str(terrapin_path),
            "-b", "https://vcpkg.storage.devpackages.microsoft.io/artifacts/",
            "-a", "true",
            "-u", "Environment",
            "-p", f"https://github.com/microsoft/vcpkg/archive/refs/tags/{version}.zip",
            "-s", sha512_hash,
            "-d", str(temp_file_path),
        ]
        
        run_command(command)
        print("--- Terrapin Tool Execution Complete ---")


def main():
    """Main function to run the entire update and PR creation process."""
    check_gh_installed()

    print("--- Starting VCPKG Update and PR Creation Script ---")

    # Step 1: Sync with remote and get on a clean main branch
    run_command(["git", "remote", "update"])
    run_command(["git", "checkout", "main"])
    run_command(["git", "reset", "--hard", "origin/main"])

    # Step 2: Get latest VCPKG info
    latest_version, latest_commit_sha = get_latest_vcpkg_info()

    # Step 3: Create a new branch
    branch_name = f"dev/update-vcpkg-{latest_version}"
    run_command(["git", "checkout", "-b", branch_name])

    # Step 4: Perform file modifications
    zip_hash = get_zip_hash(latest_version)
    update_github_workflows(latest_version, zip_hash)
    update_vcpkg_config(latest_commit_sha)
    update_build_script(latest_version)
    
    # Step 5: (Windows Only) Run the Terrapin artifact retrieval tool
    if sys.platform == "win32":
        run_terrapin_tool(latest_version, zip_hash)
    else:
        print("\nSkipping Terrapin Retrieval Tool: not running on Windows.")

    # Step 6: Commit, Push, and Create PR
    commit_title = f"Update vcpkg to version {latest_version}"
    commit_body = (
        f"This automated commit updates the vcpkg dependency to version {latest_version} "
        f"and its corresponding commit hash {latest_commit_sha[:12]}."
    )
    
    # Use 'git commit -a' to automatically stage all tracked, modified files
    run_command(["git", "commit", "-a", "-m", commit_title, "-m", commit_body])
    
    run_command(["git", "push", "--set-upstream", "origin", branch_name])
    run_command(["gh", "pr", "create", "--fill"])

    print("\n--- Successfully Created Pull Request! ---")


if __name__ == "__main__":
    main()