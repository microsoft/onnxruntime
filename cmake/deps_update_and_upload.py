# in case deps.txt is updated, run this file to update and upload the dependencies so that CI can use them.
# Before running the script, increase the version number found at:
# https://aiinfra.visualstudio.com/Lotus/_artifacts/feed/Lotus/UPack/onnxruntime_build_dependencies/versions
# Run without --do-upload once to verify downloading. Use --do-upload when you are ready to publish.
# python cmake/deps_update_and_upload.py --root-path C:/temp/onnxruntime_deps --version 1.0.82 --do-upload
# update version number in tools\ci_build\github\azure-pipelines\templates\download-deps.yml
import re
import subprocess
import os
import argparse
import tempfile

parser = argparse.ArgumentParser(description="Update dependencies and publish to Azure Artifacts")
parser.add_argument(
    "--root-path", type=str, default=tempfile.gettempdir(), help="Target root path for downloaded files"
)
parser.add_argument("--version", type=str, default="1.0.82", help="Package version to publish")
parser.add_argument("--do-upload", action="store_true", help="Upload the package to Azure Artifacts")
args = parser.parse_args()

with open("cmake/deps.txt") as file:
    text = file.read()

lines = [line for line in text.split("\n") if not line.startswith("#") and ";" in line]

root_path = args.root_path

for line in lines:
    url = re.sub("^[^;]+?;https://([^;]+?);.*", r"https://\1", line)
    filename = re.sub("^[^;]+?;https://([^;]+?);.*", r"\1", line)
    full_path = os.path.join(root_path, filename)
    subprocess.run(["curl", "-sSL", "--create-dirs", "-o", full_path, url])  # noqa: PLW1510

package_name = "onnxruntime_build_dependencies"
version = args.version

# Check if the user is logged in to Azure
result = subprocess.run("az account show", shell=True, capture_output=True, text=True)  # noqa: PLW1510
if "No subscriptions found" in result.stderr:
    # Prompt the user to log in to Azure
    print("You are not logged in to Azure. Please log in to continue.")
    subprocess.run("az login", shell=True)  # noqa: PLW1510

# Publish the package to Azure Artifacts if --no-upload is not specified

cmd = f'az artifacts universal publish --organization https://dev.azure.com/onnxruntime --feed onnxruntime --name {package_name} --version {version} --description "onnxruntime build time dependencies" --path {root_path}'
if args.do_upload:
    subprocess.run(cmd, shell=True)  # noqa: PLW1510
else:
    print("would have run: " + cmd)

cmd = f'az artifacts universal publish --organization https://dev.azure.com/aiinfra --feed Lotus --name {package_name} --version {version} --description "onnxruntime build time dependencies" --path {root_path}'
if args.do_upload:
    subprocess.run(cmd, shell=True)  # noqa: PLW1510
else:
    print("would have run: " + cmd)
