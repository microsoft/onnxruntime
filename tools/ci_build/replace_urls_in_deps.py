import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dep:
    name: str
    url: str
    sha1_hash: str


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_dir", required=True)

    return parser.parse_args()


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

args = parse_arguments()
new_dir = None
if args.new_dir:
    new_dir = args.new_dir
else:
    BUILD_BINARIESDIRECTORY = os.environ.get("BUILD_BINARIESDIRECTORY", "")
    new_dir = os.path.join(BUILD_BINARIESDIRECTORY, "deps")

deps = []
with open(os.path.join(REPO_DIR, "cmake", "deps.txt")) as f:
    depfile_reader = csv.reader(f, delimiter=";")
    for row in depfile_reader:
        if len(row) != 3:
            continue
        deps.append(Dep(row[0], row[1], row[2]))

with open(os.path.join(REPO_DIR, "cmake", "deps.txt"), "w", newline="") as f:
    depfile_writer = csv.writer(f, delimiter=";")
    for dep in deps:
        if dep.url.startswith("https://"):
            new_url = str(Path(new_dir) / dep.url[8:])
            if sys.platform.startswith("win"):
                new_url = new_url.replace("\\", "/")
            depfile_writer.writerow([dep.name, new_url, dep.sha1_hash])
