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
        "The logs are searched in order to determine whether images should be "
        "retained or removed. "
        "For an image to be retained, it must have been accessed at least N "
        "times (specified by --cache-min-access-count) over the past K days "
        "(specified by --cache-history-days)."
    )

    parser.add_argument("--container-registry", required=True, help="The container registry name.")

    parser.add_argument("--log-storage-account", required=True, help="The storage account name.")
    parser.add_argument("--log-storage-account-container", required=True, help="The storage account container name.")
    parser.add_argument(
        "--log-storage-path-pattern", default="*.json", help="The log path pattern in the storage account container."
    )

    parser.add_argument("--cache-history-days", type=int, default=7, help="The length of the cache history in days.")

    parser.add_argument(
        "--cache-min-access-count", type=int, default=1, help="The minimum access count over the cache history."
    )

    parser.add_argument("--dry-run", action="store_true", help="Do a dry-run and do not remove any images.")

    parser.add_argument("--az-path", default="az", help="Path to the az client.")

    return parser.parse_args()


def az(*args, parse_output=True, az_path):
    proc = run(az_path, *args, "--output", "json", capture_stdout=parse_output)
    if parse_output:
        return json.loads(proc.stdout.decode())
    return None


def download_logs(storage_account, container, log_path_pattern, target_dir, az_path):
    log_paths = az(
        "storage",
        "blob",
        "download-batch",
        "--destination",
        target_dir,
        "--source",
        container,
        "--account-name",
        storage_account,
        "--pattern",
        log_path_pattern,
        az_path=az_path,
    )
    return [os.path.join(target_dir, log_path) for log_path in log_paths]


ImageInfo = collections.namedtuple("ImageInfo", ["repository", "digest"])


def get_image_name(image_info):
    return f"{image_info.repository}@{image_info.digest}"


timestamp_pattern = re.compile(
    r"^(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)T(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)"
)


def parse_timestamp(timestamp_str):
    match = timestamp_pattern.match(timestamp_str)
    if match is None:
        return None

    return datetime.datetime(
        year=int(match["year"]),
        month=int(match["month"]),
        day=int(match["day"]),
        hour=int(match["hour"]),
        minute=int(match["minute"]),
        second=int(match["second"]),
        tzinfo=datetime.timezone.utc,
    )


def parse_log_line(line, min_datetime):
    entry = json.loads(line)

    def check_time(value):
        timestamp = parse_timestamp(value)
        return timestamp is not None and timestamp >= min_datetime

    for field_name, expected_value_or_checker in [
        ("category", "ContainerRegistryRepositoryEvents"),
        ("operationName", lambda value: value in ["Pull", "Push"]),
        ("resultType", "HttpStatusCode"),
        ("resultDescription", lambda value: value in ["200", "201"]),
        ("time", check_time),
    ]:
        value = entry.get(field_name, "")
        if callable(expected_value_or_checker):
            if not expected_value_or_checker(value):
                return None
        else:
            if value != expected_value_or_checker:
                return None

    props = entry.get("properties", {})
    repo, digest = props.get("repository"), props.get("digest")

    if repo is None or digest is None:
        return None

    return ImageInfo(repo, digest)


def get_valid_images_from_logs(log_paths, min_datetime, min_access_count):
    image_counts = dict()  # dict of {ImageInfo -> count}

    for log_path in log_paths:
        log.debug(f"Processing log file: {log_path}")
        with open(log_path) as log_file:
            for line in log_file:
                image_info = parse_log_line(line, min_datetime)
                if image_info is not None:
                    image_counts[image_info] = image_counts.get(image_info, 0) + 1

    return {image for image, count in image_counts.items() if count >= min_access_count}


def get_registry_images(container_registry, az_path):
    registry_images = set()  # set of ImageInfo

    repositories = az("acr", "repository", "list", "--name", container_registry, az_path=az_path)

    for repository in repositories:
        digests = az(
            "acr",
            "repository",
            "show-manifests",
            "--repository",
            repository,
            "--name",
            container_registry,
            "--query",
            "[*].digest",
            az_path=az_path,
        )

        registry_images.update([ImageInfo(repository, digest) for digest in digests])

    return registry_images


def clean_images(container_registry, image_names, az_path):
    for image_name in image_names:
        az(
            "acr",
            "repository",
            "delete",
            "--name",
            container_registry,
            "--image",
            image_name,
            "--yes",
            az_path=az_path,
            parse_output=False,
        )


# Note:
# the log download and parsing could be replaced by a log analytics query
"""
let cache_history = 7d;
let cache_min_access_count = 1;
ContainerRegistryRepositoryEvents
| where TimeGenerated >= ago(cache_history)
| where OperationName in ("Pull", "Push")
| where ResultDescription in ("200", "201")
| summarize AccessCount = count() by Repository, Digest
| where AccessCount >= cache_min_access_count
| project Repository, Digest
"""
# need to figure out how run the query the programmatically though


def main():
    args = parse_args()

    valid_images = set()

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_paths = download_logs(
            args.log_storage_account,
            args.log_storage_account_container,
            args.log_storage_path_pattern,
            tmp_dir,
            args.az_path,
        )

        cache_history = datetime.timedelta(days=args.cache_history_days)

        min_timestamp = datetime.datetime.now(tz=datetime.timezone.utc) - cache_history

        valid_images = get_valid_images_from_logs(log_paths, min_timestamp, args.cache_min_access_count)

    all_images = get_registry_images(args.container_registry, args.az_path)

    def sorted_image_names(image_infos):
        return sorted([get_image_name(image_info) for image_info in image_infos])

    log.debug("All images:\n{}".format("\n".join(sorted_image_names(all_images))))  # noqa: G001
    log.debug("Valid images:\n{}".format("\n".join(sorted_image_names(valid_images))))  # noqa: G001

    images_to_clean = all_images - valid_images
    image_names_to_clean = sorted_image_names(images_to_clean)

    log.info("Images to clean:\n{}".format("\n".join(image_names_to_clean)))  # noqa: G001

    if args.dry_run:
        log.info("Dry run, no images will be cleaned.")
        return 0

    clean_images(args.container_registry, image_names_to_clean, args.az_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
