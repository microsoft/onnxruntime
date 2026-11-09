import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
import truststore

truststore.inject_into_ssl()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = (SCRIPT_DIR / ".." / "..").resolve()

sys.path.insert(0, str(REPO_DIR / "tools" / "python"))

from build_tool_info import get_tool_info  # noqa: E402
from util import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, verify, and run the setup-build-tools GitHub action.")
    parser.add_argument("action_version", help="Action release tag.")
    parser.add_argument("action_sha512", help="Expected SHA-512 of the action source ZIP.")
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-fA-F]{128}", args.action_sha512):
        parser.error("Action SHA-512 must contain exactly 128 hexadecimal characters.")

    action_version = args.action_version
    expected_sha512 = args.action_sha512.lower()

    cmake = get_tool_info("cmake")
    vcpkg = get_tool_info("vcpkg")

    # --- Conditionally set Terrapin and define action inputs ---
    disable_terrapin_value = "true"
    terrapin_tool_path_str = "C:\\local\\Terrapin\\TerrapinRetrievalTool.exe"

    action_inputs = {
        "INPUT_CMAKE-VERSION": cmake.version,
        "INPUT_CMAKE-HASH": cmake.sha512,
        "INPUT_VCPKG-VERSION": vcpkg.version,
        "INPUT_VCPKG-HASH": vcpkg.sha512,
        "INPUT_ADD-CMAKE-TO-PATH": "true",
    }

    if sys.platform == "win32" and Path(terrapin_tool_path_str).exists():
        disable_terrapin_value = "false"
        action_inputs["INPUT_TERRAPIN-TOOL-PATH"] = terrapin_tool_path_str
        print("Terrapin tool found. Setting INPUT_DISABLE-TERRAPIN to 'false' and providing tool path.")

    action_inputs["INPUT_DISABLE-TERRAPIN"] = disable_terrapin_value

    # --- Download and Extract the Action to a Temporary Directory ---
    zip_url = f"https://github.com/microsoft/onnxruntime-github-actions/archive/refs/tags/{action_version}.zip"

    # Use AGENT_TEMPDIRECTORY, with a fallback to the system's default temp directory.
    temp_dir = Path(os.environ.get("AGENT_TEMPDIRECTORY", tempfile.gettempdir())).resolve()
    zip_path = temp_dir / "action.zip"
    extract_dir = temp_dir / "action-unzipped"

    print(f"Using temporary directory: {temp_dir}")

    # --- Locate, Run, and Cleanup the Action Script ---
    try:
        print(f"Downloading action source from: {zip_url}")
        digest = hashlib.sha512()
        with requests.get(zip_url, stream=True) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    digest.update(chunk)

        actual_sha512 = digest.hexdigest()
        if actual_sha512 != expected_sha512:
            raise ValueError(f"Action archive SHA-512 mismatch: expected {expected_sha512}, got {actual_sha512}.")

        print(f"Extracting {zip_path} to {extract_dir}")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        try:
            action_base_path = next(extract_dir.glob("onnxruntime-github-actions-*"))
            print(f"Found action base path: {action_base_path}")
        except StopIteration as e:
            raise FileNotFoundError(f"Could not find extracted action directory in '{extract_dir}'") from e

        action_script_path = action_base_path / "setup-build-tools" / "dist" / "index.js"
        if not action_script_path.exists():
            raise FileNotFoundError(f"Action script not found at expected path: {action_script_path}")

        env = os.environ.copy()
        env.update(action_inputs)

        if "AGENT_TOOLSDIRECTORY" in env:
            env["RUNNER_TOOL_CACHE"] = env["AGENT_TOOLSDIRECTORY"]
            print(f"Mapped RUNNER_TOOL_CACHE to AGENT_TOOLSDIRECTORY: {env['RUNNER_TOOL_CACHE']}")
        if "AGENT_TEMPDIRECTORY" in env:
            env["RUNNER_TEMP"] = env["AGENT_TEMPDIRECTORY"]
            print(f"Mapped RUNNER_TEMP to AGENT_TEMPDIRECTORY: {env['RUNNER_TEMP']}")

        run("node", str(action_script_path), env=env)

    finally:
        # --- Cleanup ---
        # This block ensures the zip file and extracted directory are always removed.
        print("\nStarting cleanup...")
        if zip_path.exists():
            print(f"Removing temporary zip file: {zip_path}")
            zip_path.unlink()

        if extract_dir.exists():
            print(f"Removing extracted action directory: {extract_dir}")
            shutil.rmtree(extract_dir)

        print("Cleanup complete.")


if __name__ == "__main__":
    main()
