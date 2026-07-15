#!/usr/bin/env python3
"""Resolve and topologically sort Abseil library dependencies."""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tempfile
import zipfile
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from collections.abc import Sequence


class AbslDependencyResolver:
    """Resolves Abseil dependencies and provides a topologically sorted list for static linking."""

    def __init__(self, dependency_graph: dict[str, list[str]]) -> None:
        """Initialize the resolver with a dependency graph.

        Args:
            dependency_graph: Dictionary mapping library names to their dependencies.
        """
        self.graph = dependency_graph
        self.resolved_order: list[str] = []
        self.visiting: set[str] = set()  # For detecting circular dependencies
        self.visited: set[str] = set()  # For tracking already resolved libraries

    def _visit(self, library_name: str) -> None:
        """Recursively visit each library and its dependencies (DFS).

        Args:
            library_name: The name of the library to visit.

        Raises:
            RuntimeError: If a circular dependency is detected.
        """
        if library_name in self.visited:
            return
        if library_name in self.visiting:
            msg = f"Circular dependency detected at: {library_name}"
            raise RuntimeError(msg)

        self.visiting.add(library_name)

        # Visit all dependencies of the current library first.
        for dependency in self.graph.get(library_name, []):
            self._visit(dependency)

        self.visiting.remove(library_name)
        self.visited.add(library_name)
        self.resolved_order.append(library_name)

    def resolve(self, initial_libs: Sequence[str]) -> list[str]:
        """Resolve dependencies and return a topologically sorted list.

        Args:
            initial_libs: List of starting libraries to resolve.

        Returns:
            A flat, topologically sorted list of all direct and indirect dependencies.
        """
        # Clear state for potential re-use
        self.resolved_order.clear()
        self.visiting.clear()
        self.visited.clear()

        for lib in initial_libs:
            if lib not in self.graph:
                print(
                    f"Warning: Library '{lib}' not found in the dependency graph. Treating as a leaf.",
                    file=sys.stderr,
                )
            self._visit(lib)

        # For linking, the order must be reversed (e.g., [... -lstrings, -lbase]).
        return self.resolved_order[::-1]


def generate_dependency_graph(source_dir: str) -> dict[str, list[str]]:
    """Parse CMakeLists.txt files to build a dependency graph.

    Parses CMakeLists.txt files in the extracted Abseil source directory
    to build a dependency graph. Handles the modern `absl_cc_library` CMake function.

    Args:
        source_dir: Path to the extracted Abseil source directory.

    Returns:
        Dictionary mapping library names to their dependencies.

    Raises:
        FileNotFoundError: If the 'absl' directory is not found in source_dir.
    """
    print("Parsing CMake files to build dependency graph...")
    graph: dict[str, list[str]] = {}

    # Regex to find absl_cc_library blocks with NAME and DEPS
    # This captures the entire function call including all parameters
    cc_lib_regex = re.compile(
        r"absl_cc_library\s*\((.*?)\n\)",
        re.DOTALL | re.IGNORECASE,
    )

    # Regex to find the NAME field
    name_regex = re.compile(r"NAME\s+(\S+)", re.IGNORECASE)

    # Regex to find all absl:: library names
    absl_target_regex = re.compile(r"(absl::\w+)")

    absl_source_path = os.path.join(source_dir, "absl")
    if not os.path.isdir(absl_source_path):
        msg = f"Could not find 'absl' directory in '{source_dir}'"
        raise FileNotFoundError(msg)

    for root, _, files in os.walk(absl_source_path):
        for file in files:
            if file == "CMakeLists.txt":
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    content = f.read()
                    matches = cc_lib_regex.finditer(content)
                    for match in matches:
                        content_block = match.group(1)

                        # Extract the target name from the NAME field
                        name_match = name_regex.search(content_block)
                        if not name_match:
                            continue

                        target_name = name_match.group(1).strip()
                        target_alias = f"absl::{target_name}"

                        # Find the DEPS section
                        deps_match = re.search(
                            r"DEPS\s+(.*?)(?:\n\s*(?:PUBLIC|PRIVATE|TESTONLY|COPTS|DEFINES|LINKOPTS|$))",
                            content_block,
                            re.DOTALL | re.IGNORECASE,
                        )

                        dependencies = []
                        if deps_match:
                            deps_block = deps_match.group(1)
                            dependencies = absl_target_regex.findall(deps_block)

                        if target_alias not in graph:
                            graph[target_alias] = []
                        graph[target_alias].extend(dep for dep in dependencies if dep != target_alias)

    # Ensure all found dependencies also exist as keys in the graph for completeness
    all_deps = {dep for deps in graph.values() for dep in deps}
    for dep in all_deps:
        if dep not in graph:
            graph[dep] = []

    print(f"Dependency graph built successfully with {len(graph)} libraries.")
    return graph


def fetch_and_generate_graph(version: str) -> dict[str, list[str]]:
    """Download and extract a specific version of Abseil, then generate the dependency graph.

    Args:
        version: The Abseil version (Git tag) to download.

    Returns:
        Dictionary mapping library names to their dependencies.

    Raises:
        SystemExit: If the download fails or the version is invalid.
    """
    url = f"https://github.com/abseil/abseil-cpp/archive/refs/tags/{version}.zip"
    print(f"Downloading Abseil version {version} from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to download Abseil version '{version}'.", file=sys.stderr)
        print(
            "Please check if the version tag is correct and you have an internet connection.",
            file=sys.stderr,
        )
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    zip_file = io.BytesIO(response.content)

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Extracting source to temporary directory...")
        with zipfile.ZipFile(zip_file) as z:
            source_root = os.path.join(temp_dir, z.namelist()[0].split("/")[0])
            z.extractall(temp_dir)

        return generate_dependency_graph(source_root)


def main() -> None:
    """Parse arguments, generate the graph, resolve dependencies, and print the result."""
    parser = argparse.ArgumentParser(
        description="Resolve and topologically sort Abseil library dependencies for a given version.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="The Abseil version (Git tag) to use, e.g., '20240116.2'.",
    )
    parser.add_argument(
        "--input-file",
        default="input.txt",
        help="Path to the input file containing the list of direct dependencies.",
    )
    args = parser.parse_args()

    dependency_graph = fetch_and_generate_graph(args.version)

    try:
        with open(args.input_file) as f:
            initial_libs = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    if not initial_libs:
        print("Warning: Input file is empty. No dependencies to resolve.", file=sys.stderr)
        return

    print(f"\n--- Resolving dependencies for: {', '.join(initial_libs)} ---")

    resolver = AbslDependencyResolver(dependency_graph)
    try:
        sorted_deps = resolver.resolve(initial_libs)
        for lib in sorted_deps:
            print(lib)
    except (RuntimeError, KeyError) as e:
        print(f"An error occurred during resolution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
