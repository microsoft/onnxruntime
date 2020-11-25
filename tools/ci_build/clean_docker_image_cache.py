#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import collections
import datetime
import json
import os
import re
import sys
import tempfile
from logger import get_logger


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.append(os.path.join(REPO_DIR, "tools", "python"))


from util import run  # noqa: E402


log = get_logger("clean_docker_image_cache")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleans the docker image cache container registry. "
        "This assumes a fairly specific setup - an Azure container registry "
        "and a storage account that receives "
        "ContainerRegistryRepositoryEvents logs from that registry. "
        "The logs are searched to see whether an image was used (pushed or "
        "pulled) recently enough in order to determine whether the image "
        "should be kept or cleaned up.")

    parser.add_argument(
        "--container-registry", required=True,
        help="The container registry name.")

    parser.add_argument(
        "--log-storage-account", required=True,
        help="The storage account name.")
    parser.add_argument(
        "--log-storage-account-container", required=True,
        help="The storage account container name.")
    parser.add_argument(
        "--log-storage-path-pattern", default="*.json",
        help="The log path pattern in the storage account container.")

    parser.add_argument(
        "--cache-lifetime-days", type=int, default=7,
        help="How long an image can be cached without being used, in days.")

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Do a dry-run and don't actually clean anything up.")

    parser.add_argument(
        "--az-path", default="az", help="Path to the az client.")

    return parser.parse_args()


def az(*args, parse_output=True, az_path):
    proc = run(az_path, *args, "--output", "json", capture_stdout=parse_output)
    if parse_output:
        return json.loads(proc.stdout.decode())
    return None


def download_logs(storage_account, container, log_path_pattern, target_dir, az_path):
    log_paths = az(
        "storage", "blob", "download-batch",
        "--destination", target_dir,
        "--source", container,
        "--account-name", storage_account,
        "--pattern", log_path_pattern,
        az_path=az_path)
    return [os.path.join(target_dir, log_path) for log_path in log_paths]


ImageInfo = collections.namedtuple("ImageInfo", ["repository", "digest"])


def get_image_name(image_info):
    return "{}@{}".format(image_info.repository, image_info.digest)


timestamp_pattern = re.compile(
    r"^(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)T(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)")


def parse_timestamp(timestamp_str):
    match = timestamp_pattern.match(timestamp_str)
    if match is None:
        return None

    return datetime.datetime(
        year=int(match['year']), month=int(match['month']), day=int(match['day']),
        hour=int(match['hour']), minute=int(match['minute']), second=int(match['second']),
        tzinfo=datetime.timezone.utc)


def parse_log_line(line, min_datetime):
    entry = json.loads(line)

    for field_name, expected_value in [
            ("category", "ContainerRegistryRepositoryEvents"),
            ("resultType", "HttpStatusCode"),
            ("resultDescription", "200")]:
        if entry.get(field_name) != expected_value:
            return None

    timestamp = parse_timestamp(entry.get("time", ""))
    if timestamp is None or timestamp < min_datetime:
        return None

    if entry.get("operationName") not in ["Pull", "Push"]:
        return None

    props = entry.get("properties", {})
    repo, digest = props.get("repository"), props.get("digest")

    if repo is None or digest is None:
        return None

    return ImageInfo(repo, digest)


def get_valid_images_from_logs(log_paths, min_datetime):
    valid_images = set()  # set of ImageInfo

    for log_path in log_paths:
        log.debug("Processing log file: {}".format(log_path))
        with open(log_path, mode="r") as log_file:
            for line in log_file:
                image_info = parse_log_line(line, min_datetime)
                if image_info is not None:
                    valid_images.add(image_info)

    return valid_images


def get_registry_images(container_registry, az_path):
    registry_images = set()  # set of ImageInfo

    repositories = az(
        "acr", "repository", "list", "--name", container_registry,
        az_path=az_path)

    for repository in repositories:
        digests = az(
            "acr", "repository", "show-manifests",
            "--repository", repository, "--name", container_registry,
            "--query", "[*].digest",
            az_path=az_path)

        registry_images.update(
            [ImageInfo(repository, digest) for digest in digests])

    return registry_images


def clean_images(container_registry, image_names, az_path):
    for image_name in image_names:
        az("acr", "repository", "delete", "--name", container_registry,
           "--image", image_name, "--yes",
           az_path=az_path,
           parse_output=False)


def main():
    args = parse_args()

    valid_images = set()

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_paths = download_logs(
            args.log_storage_account,
            args.log_storage_account_container,
            args.log_storage_path_pattern,
            tmp_dir,
            args.az_path)

        cache_lifetime = datetime.timedelta(days=args.cache_lifetime_days)

        min_timestamp = \
            datetime.datetime.now(tz=datetime.timezone.utc) - cache_lifetime

        valid_images = get_valid_images_from_logs(log_paths, min_timestamp)

    all_images = get_registry_images(args.container_registry, args.az_path)

    def sorted_image_names(image_infos):
        return sorted([get_image_name(image_info) for image_info in image_infos])

    log.debug("All images:\n{}".format(
        "\n".join(sorted_image_names(all_images))))
    log.debug("Valid images:\n{}".format(
        "\n".join(sorted_image_names(valid_images))))

    images_to_clean = all_images - valid_images
    image_names_to_clean = sorted_image_names(images_to_clean)

    log.info("Images to clean:\n{}".format(
        "\n".join(image_names_to_clean)))

    if args.dry_run:
        log.info("Dry run, no images will be cleaned.")
        return 0

    clean_images(args.container_registry, image_names_to_clean, args.az_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
