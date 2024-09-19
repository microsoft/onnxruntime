# If deps.txt is updated, run this file to update and upload the dependencies so that CI can use them.
#
# Before running the script, find the latest version number at:
# https://aiinfra.visualstudio.com/Lotus/_artifacts/feed/Lotus/UPack/onnxruntime_build_dependencies/versions
# Increment it to obtain a new version number to use.
#
# Run without --do-upload once to verify downloading. Use --do-upload when you are ready to publish.
# E.g.:
#   python cmake/deps_update_and_upload.py --root-path C:/temp/onnxruntime_deps --version 1.0.82
#   # check contents of C:/temp/onnxruntime_deps
#   python cmake/deps_update_and_upload.py --root-path C:/temp/onnxruntime_deps --version 1.0.82 --no-download --do-upload
#
# Next, update the version number in tools/ci_build/github/azure-pipelines/templates/download-deps.yml.

import argparse
import contextlib
import pathlib
import re
import subprocess
import tempfile

script_dir = pathlib.Path(__file__).parent

parser = argparse.ArgumentParser(description="Update dependencies and publish to Azure Artifacts")
parser.add_argument(
    "--root-path",
    type=pathlib.Path,
    help="Target root path for downloaded files. If not provided, a temporary directory is used.",
)
parser.add_argument(
    "--version",
    type=str,
    help="Package version to publish",
)
parser.add_argument(
    "--do-upload",
    action="store_true",
    dest="upload",
    help="Upload the package to Azure Artifacts",
)
parser.add_argument(
    "--no-download",
    action="store_false",
    dest="download",
    help="Skip downloading the dependency files. "
    "Use with '--do-upload' and '--root-path' to upload the package from existing dependency files.",
)
args = parser.parse_args()

if args.upload:
    assert args.version is not None, "'--version' must be specified if uploading."

if args.upload != args.download:
    assert args.root_path is not None, "'--root-path' must be specified if only downloading or uploading."

deps_path = script_dir / "deps.txt"
with open(deps_path) as file:
    text = file.read()

lines = [line for line in text.split("\n") if not line.startswith("#") and ";" in line]

with contextlib.ExitStack() as context_stack:
    if args.root_path is not None:
        root_path = args.root_path.resolve()
        root_path.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_name = context_stack.enter_context(tempfile.TemporaryDirectory())
        root_path = pathlib.Path(temp_dir_name)

    if args.download:
        print(f"Downloading dependencies to directory: {root_path}")

        dep_pattern = re.compile(r"^[^;]+;https://([^;]+);.*$")

        for line in lines:
            match = dep_pattern.fullmatch(line)
            if match is None:
                continue

            dep_path = match[1]
            url = f"https://{dep_path}"
            full_path = root_path / dep_path

            subprocess.run(["curl", "-sSL", "--create-dirs", "-o", str(full_path), url], check=True)

    package_name = "onnxruntime_build_dependencies"
    version = args.version if args.version is not None else "VERSION_PLACEHOLDER"

    if args.upload:
        # Check if the user is logged in to Azure
        result = subprocess.run("az account show", shell=True, capture_output=True, text=True, check=False)
        if "No subscriptions found" in result.stderr:
            # Prompt the user to log in to Azure
            print("You are not logged in to Azure. Please log in to continue.")
            subprocess.run("az login", shell=True, check=True)

    # Publish the package to Azure Artifacts if --do-upload is specified

    cmd = f'az artifacts universal publish --organization https://dev.azure.com/onnxruntime --feed onnxruntime --name {package_name} --version {version} --description "onnxruntime build time dependencies" --path {root_path}'
    if args.upload:
        subprocess.run(cmd, shell=True, check=True)
    else:
        print("would have run: " + cmd)

    cmd = f'az artifacts universal publish --organization https://dev.azure.com/aiinfra --feed Lotus --name {package_name} --version {version} --description "onnxruntime build time dependencies" --path {root_path}'
    if args.upload:
        subprocess.run(cmd, shell=True, check=True)
    else:
        print("would have run: " + cmd)
