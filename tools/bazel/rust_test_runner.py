# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run selected native Rust libtest cases, rejecting empty/ignored selections."""

import os
import re
import subprocess
import sys
from pathlib import Path

from python.runfiles import runfiles


def main():
    if len(sys.argv) < 5:
        raise ValueError("Expected Rust test binary, runtime library, crate manifest and default test names.")
    resolver = runfiles.Create()

    def resolve(name):
        path = resolver.Rlocation(name)
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Missing Rust test runfile {name!r}; build the Bazel test target with its data.")
        return str(Path(path).absolute())

    binary, runtime, manifest = map(resolve, sys.argv[1:4])
    os.chdir(Path(manifest).parents[2])
    env = os.environ.copy()
    env["RUST_ONNXRUNTIME_LIBRARY_PATH"] = runtime
    requested = [os.environ["TESTBRIDGE_TEST_ONLY"]] if os.environ.get("TESTBRIDGE_TEST_ONLY") else sys.argv[4:]
    listing = subprocess.run([binary, "--list", "--format=terse"], env=env, text=True, capture_output=True, check=False)
    if listing.returncode:
        raise RuntimeError(
            f"Rust test discovery failed for {binary!r} (exit {listing.returncode}). "
            f"Inspect the libtest diagnostics and rebuild this Bazel test target:\n{listing.stdout}{listing.stderr}"
        )
    available = {line.removesuffix(": test") for line in listing.stdout.splitlines() if line.endswith(": test")}
    if not available:
        raise ValueError(
            f"Rust test binary {binary!r} exposed no runnable libtest cases; "
            "select a Rust test target containing #[test] functions, not compile-time layout assertions."
        )
    if not requested or any(not name or name not in available for name in requested):
        raise ValueError(
            f"Unknown or empty Rust test selection {requested!r}. "
            f"Use an exact libtest name from: {', '.join(sorted(available))}"
        )
    for name in requested:
        result = subprocess.run(
            [binary, name, "--exact", "--test-threads=1", "--nocapture"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout, end="", flush=True)
        if result.returncode or not re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", result.stdout):
            raise RuntimeError(
                f"Rust test {name!r} did not execute and pass (exit {result.returncode}); "
                "inspect the libtest output above and select a non-ignored test with available fixtures."
            )


if __name__ == "__main__":
    main()
