# in case deps.txt is updated, run this file to update and upload the dependencies so that CI can use them.
#
# Before running the script, find the latest version number at:
# https://aiinfra.visualstudio.com/Lotus/_artifacts/feed/Lotus/UPack/onnxruntime_build_dependencies/versions
# Increment it to obtain a new version number to use.
#
# Run without --do-upload once to verify downloading. Use --do-upload when you are ready to publish.
# python cmake/deps_update_and_upload.py --root-path C:/temp/onnxruntime_deps --version 1.0.82 --do-upload
# update version number in tools\ci_build\github\azure-pipelines\templates\download-deps.yml

import argparse
import contextlib
import pathlib
import re
import subprocess
import tempfile

script_dir = pathlib.Path(__file__).parent

parser = argparse.ArgumentParser(description="Update dependencies and publish to Azure Artifacts")
parser.add_argument("--root-path", type=pathlib.Path,
                    help="Target root path for downloaded files. If not provided, a temporary directory is used.")
parser.add_argument("--version", type=str, help="Package version to publish")
parser.add_argument("--do-upload", action="store_true", help="Upload the package to Azure Artifacts")
args = parser.parse_args()

deps_path = script_dir / "deps.txt"
with open(deps_path) as file:
    text = file.read()

lines = [line for line in text.split("\n") if not line.startswith("#") and ";" in line]

with contextlib.ExitStack() as context_stack:
    if args.root_path is not None:
        root_path = args.root_path
        root_path.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_name = context_stack.enter_context(tempfile.TemporaryDirectory())
        root_path = pathlib.Path(temp_dir_name)

    for line in lines:
        url = re.sub("^[^;]+?;https://([^;]+?);.*", r"https://\1", line)
        filename = re.sub("^[^;]+?;https://([^;]+?);.*", r"\1", line)
        full_path = root_path / filename
        subprocess.run(["curl", "-sSL", "--create-dirs", "-o", str(full_path), url], check=True)

    package_name = "onnxruntime_build_dependencies"

    assert not args.do_upload or args.version is not None, "'--version' must be specified if '--do-upload' is specified."
    version = args.version if args.version is not None else "VERSION_PLACEHOLDER"

    if args.do_upload:
        # Check if the user is logged in to Azure
        result = subprocess.run("az account show", shell=True, capture_output=True, text=True, check=False)
        if "No subscriptions found" in result.stderr:
            # Prompt the user to log in to Azure
            print("You are not logged in to Azure. Please log in to continue.")
            subprocess.run("az login", shell=True, check=True)

    # Publish the package to Azure Artifacts if --do-upload is specified

    cmd = f'az artifacts universal publish --organization https://dev.azure.com/onnxruntime --feed onnxruntime --name {package_name} --version {version} --description "onnxruntime build time dependencies" --path {root_path}'
    if args.do_upload:
        subprocess.run(cmd, shell=True, check=True)
    else:
        print("would have run: " + cmd)

    cmd = f'az artifacts universal publish --organization https://dev.azure.com/aiinfra --feed Lotus --name {package_name} --version {version} --description "onnxruntime build time dependencies" --path {root_path}'
    if args.do_upload:
        subprocess.run(cmd, shell=True, check=True)
    else:
        print("would have run: " + cmd)
