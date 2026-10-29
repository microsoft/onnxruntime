# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Download a checksum-verified file and publish it as an Azure Artifacts Universal Package."""

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Publishing requires Azure CLI with the azure-devops extension "
            "(az extension add --name azure-devops), and authentication via az login "
            "or AZURE_DEVOPS_EXT_PAT with feed publishing permission. "
            "Obtain the expected SHA256 independently from the upstream source. "
            "The package contains only the downloaded file. Package versions are immutable. "
        ),
    )

    parser.add_argument("--url", required=True, help="HTTP or HTTPS URL of the file to download.")
    parser.add_argument("--name", required=True, help="Azure Artifacts Universal Package name.")
    parser.add_argument("--version", required=True, help="Package version (e.g. 1.0.0 or 1.0.0-rc.1).")
    parser.add_argument("--sha256", required=True, help="Expected SHA256 of the downloaded file.")
    parser.add_argument("--filename", help="Filename inside the package; defaults to the URL path's final component.")
    parser.add_argument(
        "--check-download-only",
        action="store_true",
        help="Download, verify the checksum, and clean up without publishing. No Azure CLI or authentication required.",
    )
    parser.add_argument(
        "--organization", default="https://dev.azure.com/aiinfra", help="Azure DevOps organization URL."
    )
    parser.add_argument(
        "--feed", default="Lotus", help="Organization-scoped Azure Artifacts feed name or ID (default: Lotus)."
    )

    args = parser.parse_args()

    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9a-z.-]+)?", args.version):
        parser.error("--version must have the form X.Y.Z with an optional lowercase prerelease suffix.")

    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
        parser.error("--sha256 must contain exactly 64 hexadecimal characters.")

    url = urllib.parse.urlsplit(args.url)
    if url.scheme not in ("http", "https") or not url.hostname:
        parser.error("--url must be an HTTP or HTTPS URL.")

    if args.filename is None:
        args.filename = urllib.parse.unquote(url.path.rsplit("/", 1)[-1])

    if (
        not args.filename
        or args.filename in (".", "..")
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', args.filename)
        or args.filename.endswith((" ", "."))
    ):
        parser.error("A valid filename without directory components is required; use --filename to override it.")

    return args


def main():
    args = parse_args()
    az = None
    if not args.check_download_only:
        az = shutil.which("az")
        if az is None:
            raise FileNotFoundError("Azure CLI was not found on PATH. Install it and the azure-devops extension.")

    with tempfile.TemporaryDirectory(prefix="universal-package-") as directory:
        downloaded_file = Path(directory) / args.filename
        digest = hashlib.sha256()

        print(f"Downloading {args.url}", flush=True)

        with urllib.request.urlopen(args.url, timeout=60) as response, downloaded_file.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)

        actual_hash = digest.hexdigest()
        if actual_hash != args.sha256.lower():
            raise ValueError(f"Downloaded file SHA256 mismatch: expected {args.sha256}, got {actual_hash}")

        print(f"Verified SHA256: {actual_hash}", flush=True)

        if args.check_download_only:
            print("Download check completed; skipping publication.", flush=True)
            return

        print(f"Publishing Universal Package: {args.name} version {args.version}", flush=True)

        command = [
            az,
            "artifacts",
            "universal",
            "publish",
            "--organization",
            args.organization,
            "--scope",
            "organization",
            "--feed",
            args.feed,
            "--name",
            args.name,
            "--version",
            args.version,
            "--path",
            directory,
            "--description",
            f"Mirrored file: {args.filename}",
            "--detect",
            "false",
        ]

        subprocess.run(command, check=True)

        print("Mirrored downloaded file as Universal Package.", flush=True)


if __name__ == "__main__":
    main()
