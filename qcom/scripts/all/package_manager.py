#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import hashlib
import logging
import os
import shutil
import ssl
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import certifi
import tqdm
import yaml

QCOM_ROOT = Path(__file__).parent.parent.parent
REPO_ROOT = QCOM_ROOT.parent
PACKAGE_CONFIG = QCOM_ROOT / "packages.yml"
DEFAULT_PACKAGE_CACHE_DIR = Path(
    os.environ.get(
        "ORT_BUILD_PACKAGE_CACHE_PATH",
        str((Path("~") / ".ort-package-cache").expanduser()),
    )
)

AUTOPRUNE = os.environ.get("ORT_BUILD_PRUNE_PACKAGES", "1") == "1"

DEFAULT_MAX_CACHE_SIZE_BYTES = int(
    os.environ.get("ORT_BUILD_PACKAGE_CACHE_SIZE", f"{10 * 1024 * 1024 * 1024}")
)  # 10 GiB

DEFAULT_TOOLS_DIR = Path(os.environ.get("ORT_BUILD_TOOLS_PATH", REPO_ROOT / "build" / "tools"))

CAFILE = os.environ.get("REQUESTS_CA_BUNDLE", certifi.where())


class FileCache:
    def __init__(
        self,
        cache_dir: Path = DEFAULT_PACKAGE_CACHE_DIR,
        max_cache_size_bytes: int = DEFAULT_MAX_CACHE_SIZE_BYTES,
    ) -> None:
        self.__cache_dir = cache_dir
        self.__cache_dir.mkdir(exist_ok=True)
        self.__max_cache_size_bytes = max_cache_size_bytes

    def fetch(
        self,
        cache_key: str,
        url: str,
        expected_sha256: str | None = None,
        expected_sha1: str | None = None,
        expected_md5: str | None = None,
    ) -> Path:
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
                with urllib.request.urlopen(url, context=ssl.create_default_context(cafile=CAFILE)) as response:
                    content_length = response.getheader("content-length")
                    length = (
                        int(response.getheader("content-length")) / 1024 / 1024 if content_length is not None else 0
                    )
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
                for expected_sha, sha_name, sha_fn in [
                    (expected_sha1, "SHA1", hashlib.sha1),
                    (expected_sha256, "SHA256", hashlib.sha256),
                    (expected_md5, "MD5", hashlib.md5),
                ]:
                    if expected_sha is not None:
                        tmp_file.seek(0)
                        actual_sha = sha_fn()
                        for bytes in iter(lambda: tmp_file.read(32768), b""):
                            actual_sha.update(bytes)
                        logging.debug(f"{sha_name} hash for {cache_file_path.name}: {actual_sha.hexdigest()}")
                        if expected_sha != actual_sha.hexdigest():
                            raise ValueError(f"{sha_name} mismatch for {cache_file_path.name}")

                # Write the final file. Note that SpooledTemporaryFile doesn't necessarily create a file
                # so there's a good chance we're writing to disk for the first time.
                cache_dir.mkdir(exist_ok=True)
                tmp_file.seek(0)
                with cache_file_path.open(mode="wb") as cache_file_stream:
                    shutil.copyfileobj(tmp_file, cache_file_stream)

        else:
            logging.debug(f"{url} already exists at {cache_file_path}")
        return cache_file_path

    def prune(self) -> None:
        """Prune old entries from the cache until it's below our maximum size."""

        # Prepare a list of files in the package directory sorted by access time
        pkg_files = sorted(
            ((pf, pf.stat().st_atime, pf.stat().st_size) for pf in self.__cache_dir.rglob("*") if pf.is_file()),
            key=lambda f: f[1],
        )

        # Determine what we need to prune
        to_prune: list[tuple[Path, float, int]] = []
        total_size = sum(pf[2] for pf in pkg_files)
        logging.info(f"Cache size before pruning: {total_size / 1024.0 / 1024.0:.2f} MiB.")
        for pkg_file in pkg_files:
            if total_size <= self.__max_cache_size_bytes:
                break
            to_prune.append(pkg_file)
            total_size -= pkg_file[2]

        # Prune
        for pkg_file in to_prune:
            logging.debug(f"Pruning {pkg_file[0]} to reclaim {pkg_file[2] / 1024.0 / 1024.0:.2f} MiB.")
            pkg_file[0].unlink()

        # Remove empty directories
        for pkg_dir in self.__cache_dir.iterdir():
            if len([f for f in pkg_dir.rglob("*") if f.is_file()]) == 0:
                logging.debug(f"Removing empty directory {pkg_dir}")
                shutil.rmtree(pkg_dir)

        logging.info(f"Cache size after pruning: {total_size / 1024.0 / 1024.0:.2f} MiB.")


class PackageManager:
    """
    A simple package manager.
    """

    def __init__(self, package: str, package_root: Path | None = None) -> None:
        full_config = self.__parse_config(PACKAGE_CONFIG)
        if package not in full_config:
            raise ValueError(f"Unknown package {package}.")
        self.__cache = FileCache()
        self.__config = full_config[package]
        self.__package = package
        self.__package_root = package_root if package_root is not None else DEFAULT_TOOLS_DIR
        self.__package_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def clean(cls, package_root: Path) -> None:
        config = cls.__parse_config(PACKAGE_CONFIG)

        known = [f"{cls.__format_package_dir(name, atts['version'])}" for name, atts in config.items()]

        for subdir in package_root.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name in known:
                logging.debug(f"{subdir.name} is up to date")
            else:
                cls.__uninstall(subdir)
        FileCache().prune()

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
        return Path(self.__format_package_dir(self.__package, self.__config["version"]))

    def get_root_dir(self) -> Path:
        """Get the path of the directory in which a package is extracted."""
        return self.__package_root / self.get_rel_package_dir()

    def install(self) -> None:
        """Ensure this package is installed."""
        pkg_rootdir = self.get_root_dir()
        if pkg_rootdir.exists():
            logging.debug(f"{pkg_rootdir} already exists.")
            return
        package_path = self.__fetch()

        if package_path.suffix == ".exe":
            self.__run_installer(package_path)
        else:
            # Similar to downloads, we extract to a temporary directory and rename on
            # success to avoid partial extractions if we get killed.
            with tempfile.TemporaryDirectory(dir=self.__package_root) as tmp_dir:
                # Extract it to tmp-dir/{package}-{version}
                tmp_rootdir = tmp_dir / self.get_rel_package_dir()
                if tarfile.is_tarfile(package_path):
                    self.__install_tar(package_path, tmp_rootdir)
                elif zipfile.is_zipfile(package_path):
                    self.__install_zip(package_path, tmp_rootdir)
                else:
                    raise ValueError(f"{package_path.name} has unknown archive format.")

                # Move fully extracted package to final location
                logging.info(f"Moving {tmp_rootdir} to {self.__package_root}")
                shutil.move(tmp_rootdir, self.get_root_dir())
        if AUTOPRUNE:
            self.__cache.prune()

    def repair(self) -> None:
        package_path = self.__fetch()
        if package_path.suffix != ".exe":
            raise RuntimeError("Cannot repair non-installer package.")
        self.__run_installer(package_path, repair=True)

    def __fetch(self) -> Path:
        """Fetch the package archive."""
        cache_key = str(self.get_rel_package_dir())
        url = self.__format(self.__config["url"])
        package_path = self.__cache.fetch(
            cache_key,
            url,
            self.__config.get("sha256", None),
            self.__config.get("sha1", None),
            self.__config.get("md5", None),
        )
        return package_path

    def __format(self, fmt_str: str) -> str:
        """Format a config file string, performing any necessary substitutions."""
        replacements = {key: self.__config.get(key) for key in ["major_version", "version"]}
        replacements["root_dir"] = self.get_root_dir()
        return fmt_str.format_map(replacements)

    def __run_installer(self, installer_path: Path, repair: bool = False) -> None:
        install_args = [self.__format(a) for a in self.__config.get("install_args", [])]
        uninstall_args = [self.__format(a) for a in self.__config.get("uninstall_args", [])]
        repair_args = [self.__format(a) for a in self.__config.get("repair_args", [])]

        # Run the installer
        self.__execute(installer_path, repair_args if repair else install_args)

        # Save the installer and enough info to run it
        uninstaller_path = self.get_root_dir() / installer_path.name
        shutil.copyfile(installer_path, uninstaller_path)
        uninstall_info = {"uninstaller_path": str(uninstaller_path), "args": uninstall_args}
        with (self.get_root_dir() / "uninstall.yml").open("wt") as uninstall_file:
            yaml.safe_dump(uninstall_info, uninstall_file)

    @staticmethod
    def __execute(exe_path: Path, args: Iterable[str]) -> None:
        logging.debug(f"Running {[exe_path, *args]}")
        subprocess.run([exe_path, *args], check=True, timeout=60 * 3)

    @staticmethod
    def __format_package_dir(package_name: str, package_version: str) -> str:
        return f"{package_name}-{package_version}"

    @staticmethod
    def __install_tar(archive_path: Path, destination: Path) -> None:
        logging.info(f"Extracting tarball to {destination}")
        with tarfile.open(archive_path, "r") as t:
            t.extractall(destination)

    @staticmethod
    def __install_zip(archive_path: Path, destination: Path) -> None:
        logging.info(f"Extracting zip file to {destination}")
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(destination)
        # zipfile.extractall does not preserve executable bits
        # https://github.com/python/cpython/issues/59999
        for root, _, files in os.walk(destination):
            for file in files:
                file_path = Path(root) / file
                file_path.chmod(0o755)

    @staticmethod
    def __parse_config(config_path: Path) -> dict[str, dict[str, Any]]:
        with config_path.open() as config_file:
            return yaml.safe_load(config_file)

    @classmethod
    def __uninstall(cls, subdir: Path) -> None:
        logging.info(f"Removing unknown/outdated package in {subdir.name}")
        uninstall_yml_path = subdir / "uninstall.yml"
        if uninstall_yml_path.exists():
            with uninstall_yml_path.open("rt") as uninstall_file:
                uninstall_config = yaml.safe_load(uninstall_file)
            uninstaller_path = Path(uninstall_config["uninstaller_path"])
            cls.__execute(uninstaller_path, uninstall_config.get("args", []))

            # Windows sometimes takes a moment to close the uninstaller. Ug.
            for i in range(5):
                try:
                    logging.debug(f"Trying to remove {uninstaller_path}")
                    uninstaller_path.unlink()
                    break
                except PermissionError:
                    sleep_duration = 1 + (i * 5)
                    logging.warning(f"Could not delete {uninstaller_path}; trying again in {sleep_duration} seconds.")
                    time.sleep(sleep_duration)
        # If we couldn't delete the uninstaller, this will throw for us:
        logging.debug(f"Clobbering {subdir}")
        shutil.rmtree(subdir)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--package", action="store", help="Specify the package to select")
    parser.add_argument(
        "--package-root",
        action="store",
        help="Path to the package installation directory",
        type=Path,
    )

    action_group = parser.add_argument_group("Actions").add_mutually_exclusive_group(required=True)
    action_group.add_argument("--clean", action="store_true", help="Uninstall all outdated packages")
    action_group.add_argument("--install", action="store_true", help="Install the selected package")
    action_group.add_argument("--repair", action="store_true", help="Repair the selected package")
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

    if args.clean:
        PackageManager.clean(args.package_root)
    else:
        if "package" not in args:
            raise ValueError("--package is required for package installation or inspection.")
        packager = PackageManager(args.package, args.package_root)
        if args.install:
            packager.install()
        elif args.repair:
            packager.repair()
        elif args.print_bin_dir:
            print(packager.get_bindir())
        elif args.print_content_dir:
            print(packager.get_content_dir())
