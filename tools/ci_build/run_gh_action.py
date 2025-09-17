import os
import platform
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = (SCRIPT_DIR / ".." / "..").resolve()

sys.path.insert(0, str(REPO_DIR / "tools" / "python"))

from util import run  # noqa: E402

# Hash structure for platform-specific binaries
CMAKE_HASHES = {
    "windows": {
        "x64": "807b774fcb12defff8ce869e602fc5b6279d5b7bf7229ebcf3f7490da3f887d516b9c49a00d50f9179e552ed8737d19835a19ef8f366d1ffda1ad6f3352a90c2",
        "arm64": "86937dc89deabe0ff2a08fe198fcfc70764476b865cca4c6dc3bfc7fb9f7d44d4929af919e26e84aaedef17ad01ffb9683e42c39cb38b409100f723bc5ef1cc0",
    },
    "linux": {
        "x64": "7939260931098c3f00d2b36de3bee6a0ee3bcae2dba001598c492ed5c82d295c9aa9969654f1ff937fec4d71679541238baaa648c5246f36e14f28f0a62337a0",
        "arm64": "8eeb07e966a5340c122979dd2e371708a78adccc85200b22bc7e66028e65513bce5ced6c37fe65aedb94000d970186c5c7562d1ab3dbda911061de46b75345d9",
    },
    "macos": "99cc9c63ae49f21253efb5921de2ba84ce136018abf08632c92c060ba91d552e0f6acc214e9ba8123dee0cf6d1cf089ca389e321879fd9d719a60d975bcffcc8",
}


def get_platform_keys() -> tuple[str | None, str | None]:
    """Detects the OS and CPU architecture and returns normalized keys."""
    os_key: str | None = None
    match sys.platform:
        case "win32":
            os_key = "windows"
        case "linux":
            os_key = "linux"
        case "darwin":
            os_key = "macos"

    arch_key: str | None = None
    match platform.machine().lower():
        case "amd64" | "x86_64":
            arch_key = "x64"
        case "arm64" | "aarch64":
            arch_key = "arm64"

    return os_key, arch_key


def main() -> None:
    if len(sys.argv) < 2:
        print("::error::Action version argument was not provided.")
        sys.exit(1)

    action_version = sys.argv[1]

    # --- Platform Detection and Variable Setup ---
    os_key, arch_key = get_platform_keys()
    if not os_key or not arch_key:
        print(
            f"::error::Could not determine a supported platform from OS '{sys.platform}' and Arch '{platform.machine()}'."
        )
        sys.exit(1)

    print(f"Detected Platform: OS='{os_key}', Architecture='{arch_key}'")

    try:
        if os_key == "macos":
            cmake_hash = CMAKE_HASHES[os_key]
        else:
            cmake_hash = CMAKE_HASHES[os_key][arch_key]

        print(f"Selected CMake hash for '{os_key}'.")
    except KeyError:
        print(f"::error::Unsupported platform or missing hash for OS='{os_key}' and Arch='{arch_key}'.")
        sys.exit(1)

    # --- Conditionally set Terrapin and define action inputs ---
    disable_terrapin_value = "true"
    terrapin_tool_path_str = "C:\\local\\Terrapin\\TerrapinRetrievalTool.exe"

    action_inputs = {
        "INPUT_CMAKE-VERSION": "3.31.8",
        "INPUT_CMAKE-HASH": cmake_hash,
        "INPUT_VCPKG-VERSION": "2025.06.13",
        "INPUT_VCPKG-HASH": "735923258c5187966698f98ce0f1393b8adc6f84d44fd8829dda7db52828639331764ecf41f50c8e881e497b569f463dbd02dcb027ee9d9ede0711102de256cc",
        "INPUT_ADD-CMAKE-TO-PATH": "true",
    }

    if os_key == "windows" and Path(terrapin_tool_path_str).exists():
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
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

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
