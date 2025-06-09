#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import hashlib
import logging
import os
import shutil
import ssl
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import certifi
import tqdm
import yaml

QCOM_ROOT = Path(__file__).parent.parent.parent
PACKAGE_CONFIG = QCOM_ROOT / "packages.yml"
DEFAULT_PACKAGE_CACHE_DIR = Path(
    os.environ.get(
        "ORT_BUILD_PACKAGE_CACHE_PATH",
        str((Path("~") / ".ort-package-cache").expanduser()),
    )
)
DEFAULT_MAX_CACHE_SIZE_BYTES = int(os.environ.get("ORT_BUILD_PACKAGE_CACHE_SIZE", f"{5 * 1024 * 1024 * 1024}"))  # 5 GiB


class FileCache:
    def __init__(
        self,
        cache_dir: Path = DEFAULT_PACKAGE_CACHE_DIR,
        max_cache_size_bytes: int = DEFAULT_MAX_CACHE_SIZE_BYTES,
    ) -> None:
        self.__cache_dir = cache_dir
        self.__cache_dir.mkdir(exist_ok=True)
        self.__max_cache_size_bytes = max_cache_size_bytes

    def fetch(self, cache_key: str, url: str, expected_sha256: str | None) -> Path:
        """
        Get path to a local copy of the given URL.

        :param cache_key: A string unique to the requested download's content.
        :param url: The URL of the item to fetch.
        :param expected_sha256: If not None, raise ValueError if the download does not match this sha256 hash.
        """
        url_parts = urllib.parse.urlparse(url)
        url_path = PurePosixPath(url_parts.path)
        cache_dir = self.__cache_dir / cache_key
        cache_file_path = cache_dir / url_path.name
        if not cache_file_path.exists():
            logging.info(f"Downloading {url} to {cache_file_path}")

            # Defer writing the final file so we don't leave partial downloads if we get killed.
            with tempfile.SpooledTemporaryFile(mode="wr+b") as tmp_file:
                with urllib.request.urlopen(
                    url, context=ssl.create_default_context(cafile=certifi.where())
                ) as response:
                    length = int(response.getheader("content-length")) / 1024 / 1024
                    with tqdm.tqdm(
                        total=length,
                        unit="MiB",
                        bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{remaining}, {rate_fmt}{postfix}]",
                    ) as pbar:
                        while True:
                            chunk = response.read(64 * 1024)
                            if not chunk:
                                break
                            tmp_file.write(chunk)
                            pbar.update(len(chunk) / 1024 / 1024)

                # Make sure the download's hash matches
                if expected_sha256 is not None:
                    tmp_file.seek(0)
                    actual_sha256 = hashlib.sha256()
                    for bytes in iter(lambda: tmp_file.read(32768), b""):
                        actual_sha256.update(bytes)
                    logging.debug(f"SHA256 hash for {cache_file_path.name}: {actual_sha256.hexdigest()}")
                    if expected_sha256 != actual_sha256.hexdigest():
                        raise ValueError(f"sha256 mismatch for {cache_file_path.name}")

                # Write the final file. Note that SpooledTemporaryFile doesn't necessarily create a file
                # so there's a good chance we're writing to disk for the first time.
                cache_dir.mkdir(exist_ok=True)
                tmp_file.seek(0)
                with cache_file_path.open(mode="wb") as cache_file_stream:
                    shutil.copyfileobj(tmp_file, cache_file_stream)

        else:
            logging.info(f"{url} already exists at {cache_file_path}")
        return cache_file_path

    def prune(self) -> None:
        """Prune old entries from the cache until it's below our maximum size."""

        # Collect package info into a list of tuple(name, create time, size bytes), sorted by size.
        pkg_dirs = [
            (d, d.stat().st_ctime, sum(f.stat().st_size for f in d.glob("*"))) for d in self.__cache_dir.glob("*")
        ]
        pkg_dirs.sort(key=lambda d: d[1])

        # Determine what we need to prune.
        to_prune = []
        total_size = sum(d[2] for d in pkg_dirs)
        logging.info(f"Cache size before pruning: {total_size / 1024.0 / 1024.0:.2f} MiB.")
        for pkg_dir in pkg_dirs:
            if total_size <= self.__max_cache_size_bytes:
                break
            to_prune.append(pkg_dir)
            total_size -= pkg_dir[2]

        # Prune
        for pkg_dir in to_prune:
            logging.debug(f"Pruning {pkg_dir[0]} to reclaim {pkg_dir[2] / 1024.0 / 1024.0:.2f} MiB.")
            shutil.rmtree(pkg_dir[0])
        logging.info(f"Cache size after pruning: {total_size / 1024.0 / 1024.0:.2f} MiB.")


class PackageManager:
    """
    A simple package manager.
    """

    def __init__(self, package: str, package_root: Path) -> None:
        full_config = self.__parse_config(PACKAGE_CONFIG)
        if package not in full_config:
            raise ValueError(f"Unknown package {package}.")
        self.__cache = FileCache()
        self.__config = full_config[package]
        self.__package = package
        self.__package_root = package_root
        self.__package_root.mkdir(parents=True, exist_ok=True)

    def get_bindir(self, assert_exists: bool = True) -> Path:
        """
        Get the binary directory of this package.

        :param assert_exists: If True, raise FileNotFoundError if binary directory does not exist.
        """
        bindir = self.get_content_dir()
        if "bindir" in self.__config:
            bindir = bindir / self.__config["bindir"]
        if assert_exists and not bindir.is_dir():
            raise FileNotFoundError(f"{bindir} is not a directory. Has {self.__package} been installed?")
        return bindir

    def get_content_dir(self) -> Path:
        """Get the absolute path of the package's contents."""
        return self.__package_root / self.get_rel_content_dir()

    def get_rel_content_dir(self) -> Path:
        """
        Get the relative path to a package's content, which includes its unique subdirectory and any common root
        directory found in its archive.
        """
        rootdir = self.get_rel_package_dir()
        content_root = self.__config.get("content_root", None)
        if content_root is not None:
            rootdir = rootdir / Path(self.__format(content_root))
        return rootdir

    def get_rel_package_dir(self) -> Path:
        """
        Get the name of a package-unique directory.
        """
        package_version = self.__config["version"]
        return Path(f"{self.__package}-{package_version}")

    def get_root_dir(self) -> Path:
        """Get the path of the directory in which a package is extracted."""
        return self.__package_root / self.get_rel_package_dir()

    def install(self) -> None:
        """Ensure this package is installed."""
        pkg_rootdir = self.get_content_dir()
        if pkg_rootdir.exists():
            logging.info(f"{pkg_rootdir} already exists.")
            return

        # Fetch the package archive
        cache_key = str(self.get_rel_package_dir())
        url = self.__format(self.__config["url"])
        package_path = self.__cache.fetch(cache_key, url, self.__config.get("sha256", None))

        # Similar to downloads, we extract to a temporary directory and rename on
        # success to avoid partial extrations if we get killed.
        with tempfile.TemporaryDirectory(dir=self.__package_root) as tmp_dir:
            # Extract it to tmp-dir/{package}-{version}
            tmp_rootdir = tmp_dir / self.get_rel_package_dir()
            if tarfile.is_tarfile(package_path):
                logging.info(f"Extracting tarball to {tmp_rootdir}")
                with tarfile.open(package_path, "r") as t:
                    t.extractall(tmp_rootdir)

            elif zipfile.is_zipfile(package_path):
                logging.info(f"Extracting zip file to {tmp_rootdir}")
                with zipfile.ZipFile(package_path, "r") as z:
                    z.extractall(tmp_rootdir)
                # zipfile.extractall does not preserve executable bits
                # https://github.com/python/cpython/issues/59999
                for root, _, files in os.walk(tmp_rootdir):
                    for file in files:
                        file_path = Path(root) / file
                        file_path.chmod(0o755)

            elif package_path.suffix == ".exe":
                # Example: nuget.exe, which is not in an archive
                tmp_rootdir.mkdir()
                shutil.copyfile(package_path, tmp_rootdir / package_path.name)

            else:
                raise ValueError(f"{package_path.name} has unknown archive format.")

            # Move fully extracted package to final location
            logging.info(f"Moving {tmp_rootdir} to {self.__package_root}")
            try:
                Path(tmp_rootdir).rename(self.get_root_dir())
            except OSError:
                # The fast path didn't work, perhaps we crossed mount boundaries.
                shutil.move(tmp_rootdir, self.get_root_dir())
        self.__cache.prune()

    def __format(self, fmt_str: str) -> str:
        """Format a config file string, performing any necessary substitutions."""
        simple_substitutions = ["major_version", "version"]
        return fmt_str.format_map({key: self.__config.get(key) for key in simple_substitutions})

    @staticmethod
    def __parse_config(config_path: Path) -> dict[str, dict[str, Any]]:
        with config_path.open() as config_file:
            return yaml.safe_load(config_file)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--package", action="store", help="Specify the package to select", required=True)
    parser.add_argument(
        "--package-root",
        action="store",
        help="Path to the package installation directory",
        required=True,
    )

    action_group = parser.add_argument_group("Actions").add_mutually_exclusive_group(required=True)
    action_group.add_argument("--install", action="store_true", help="Install the selected package")
    action_group.add_argument(
        "--print-bin-dir",
        action="store_true",
        help="Print the path of the selected package's bin directory",
    )
    action_group.add_argument(
        "--print-content-dir",
        action="store_true",
        help="Print the path of the selected package's content directory",
    )

    return parser


if __name__ == "__main__":
    log_format = "[%(asctime)s] [package_manager.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)
    parser = make_parser()
    args = parser.parse_args()
    packager = PackageManager(args.package, Path(args.package_root))

    if args.install:
        packager.install()
    elif args.print_bin_dir:
        print(packager.get_bindir())
    elif args.print_content_dir:
        print(packager.get_content_dir())
