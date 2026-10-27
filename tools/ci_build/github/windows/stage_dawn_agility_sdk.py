# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def stage_dawn_agility_sdk(source_dir: Path, mirror_dir: Path) -> Path:
    source_dir = source_dir.resolve()
    entries = [
        line
        for line in (source_dir / "cmake" / "deps.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.startswith("dawn_agility_sdk;")
    ]
    if len(entries) != 1:
        raise ValueError("Expected exactly one dawn_agility_sdk entry in cmake/deps.txt.")

    fields = entries[0].split(";")
    if len(fields) != 3 or not re.fullmatch(r"[0-9a-fA-F]{40}", fields[2]):
        raise ValueError("The dawn_agility_sdk entry must contain a URL and SHA1 hash.")
    _, url, expected_hash = fields
    package = re.fullmatch(
        r"https://www\.nuget\.org/api/v2/package/([A-Za-z0-9][A-Za-z0-9_.-]*)/([0-9][A-Za-z0-9.+-]*)",
        url,
    )
    if package is None:
        raise ValueError("Expected a NuGet v2 package URL for dawn_agility_sdk.")
    package_id, version = package.groups()

    nuget_config = source_dir / "NuGet.config"
    if not nuget_config.is_file():
        raise FileNotFoundError(f"NuGet configuration not found: {nuget_config}")
    nuget = shutil.which("nuget.exe")
    if nuget is None:
        raise FileNotFoundError(
            "nuget.exe must be on PATH. Provision it from an approved source before staging the Dawn Agility SDK."
        )

    with tempfile.TemporaryDirectory(prefix="ort-dawn-agility-sdk-") as package_dir:
        print(f"Acquiring {package_id} {version} using {nuget_config}", flush=True)
        subprocess.run(
            [
                nuget,
                "install",
                package_id,
                "-Version",
                version,
                "-ConfigFile",
                str(nuget_config),
                "-OutputDirectory",
                package_dir,
                "-PackageSaveMode",
                "nupkg",
                "-ExcludeVersion",
                "-DependencyVersion",
                "Ignore",
                "-NonInteractive",
            ],
            check=True,
        )
        archives = [path for path in (Path(package_dir) / package_id).glob("*.nupkg") if path.is_file()]
        if len(archives) != 1:
            raise ValueError(f"Expected exactly one retained .nupkg archive for {package_id} {version}.")
        archive = archives[0]
        with archive.open("rb") as package_file:
            actual_hash = hashlib.file_digest(package_file, "sha1").hexdigest()
        if actual_hash != expected_hash.lower():
            raise ValueError(f"Agility SDK SHA1 mismatch. Expected {expected_hash}, got {actual_hash}.")

        # Match the HTTPS URL-to-local-path mapping in onnxruntime_external_deps.cmake.
        mirror_path = mirror_dir.joinpath(*url.removeprefix("https://").split("/"))
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, mirror_path)
        print(f"Staged Dawn Agility SDK at {mirror_path} (SHA1 {actual_hash})")
        return mirror_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage the pinned Dawn Agility SDK in a local CMake dependency mirror")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    arguments = parser.parse_args()
    stage_dawn_agility_sdk(arguments.source_dir, arguments.mirror_dir)


if __name__ == "__main__":
    main()
