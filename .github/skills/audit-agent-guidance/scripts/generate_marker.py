#!/usr/bin/env python3

import argparse
import re
import subprocess


COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


def parse_commit(value: str) -> str:
    if not COMMIT_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("commit must be a full 40-character hexadecimal object ID")

    return value.lower()


def current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise SystemExit("git was not found on PATH") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise SystemExit(f"unable to resolve HEAD: {message}") from error

    return parse_commit(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an agent-guidance audit PR marker.")
    parser.add_argument(
        "--commit",
        type=parse_commit,
        help="Audited commit. Defaults to the current HEAD.",
    )
    args = parser.parse_args()
    commit = args.commit or current_commit()

    print(f"Agent-Guidance-Audit: version=1; audited-commit={commit}")


if __name__ == "__main__":
    main()
