from pathlib import Path
import argparse
import configparser
import json
import re

import pygit2


def format_component(submod):
    return {"component": {"type": "git", "git": {"commitHash": str(submod.head_id), "repositoryUrl": submod.url}}}


def lookup_submodule(repo, submodule_path):
    submodule = repo.lookup_submodule(submodule_path)
    try:
        # Some submodules have names which don't correspond to the actual path in the repo
        # (e.g. 'git submodule init' was called with the --name option, or the submodule
        # was moved and the old name was kept). listall_submodules() returns submodule paths,
        # but pygit up to 1.0.1 requires the submodule name (not the path) in lookup_submodule
        # to be able to access the URL and other properties.
        # This seems to be a bug in pygit2, since its documentation says the submodules can
        # be opened by path.
        # If accessing the URL throws a RuntimeError, we get the submodule name manually from
        # .gitmodules.
        submodule.url
        return submodule
    except RuntimeError:
        pass

    config = configparser.ConfigParser()
    config.read(Path(repo.workdir, '.gitmodules'))
    for section in config.sections():
        if config[section]['path'] == submodule_path:
            name = re.fullmatch('submodule "(.*)"', section).group(1)
            submodule = repo.lookup_submodule(name)
            return submodule
    raise NotImplementedError()  # should not be reached


def process_component(repo):
    return [lookup_submodule(repo, submod) for submod in repo.listall_submodules()]


def recursive_process(base_repo):
    processed_subs = []
    repos_to_process = [base_repo]
    while repos_to_process:
        repo = repos_to_process.pop()
        submodules = process_component(repo)
        processed_subs.extend(submodules)
        repos_to_process.extend([mod.open() for mod in submodules])
    return {"Registrations": [format_component(component) for component in processed_subs]}


def main(repo_path, output_file):
    repo = pygit2.Repository(repo_path)
    registrations = recursive_process(repo)
    with open(output_file, 'w') as f:
        json.dump(registrations, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_repository", help="path to base repository to get registrations for.")
    parser.add_argument("-o", "--output", help="output file name.", default="cgmanifest.json")
    args = parser.parse_args()
    main(args.base_repository, args.output)
