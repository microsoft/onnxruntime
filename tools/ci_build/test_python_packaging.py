#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = (
    "tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt",
    "tools/ci_build/github/linux/docker/inference/aarch64/python/cpu/scripts/requirements.txt",
    "tools/ci_build/github/linux/docker/scripts/lort/requirements.txt",
    "tools/ci_build/requirements/transformers-test/requirements.txt",
)
EXCLUDED = {
    REQUIREMENTS[0]: {"triton", "neural-compressor"},
    REQUIREMENTS[1]: set(),
    REQUIREMENTS[2]: {"h5py"},
    REQUIREMENTS[3]: {"torch", "torchvision", "sentencepiece"},
}


def marker_environment(version):
    return {
        **default_environment(),
        "python_version": version,
        "python_full_version": version + ".0",
        "implementation_name": "cpython",
        "implementation_version": version + ".0",
        "platform_python_implementation": "CPython",
    }


def read_requirements(path):
    return [
        Requirement(line)
        for line in (ROOT / path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


class PythonPackagingRequirementsTest(unittest.TestCase):
    def test_numpy_selection_preserves_older_versions(self):
        for path in REQUIREMENTS:
            requirements = read_requirements(path)
            self.assertIn('numpy==2.2.6; python_version < "3.11"', (ROOT / path).read_text().splitlines())
            for version, pin in (("3.11", "==2.4.2"), ("3.14", "==2.4.2"), ("3.15", "==2.5.4")):
                with self.subTest(path=path, version=version):
                    selected = [
                        str(req.specifier)
                        for req in requirements
                        if req.name == "numpy"
                        and (req.marker is None or req.marker.evaluate(marker_environment(version)))
                    ]
                    self.assertEqual(selected, [pin])

    def test_unsupported_dependencies_are_selected_only_before_python315(self):
        for path, excluded in EXCLUDED.items():
            requirements = read_requirements(path)
            for version in ("3.11", "3.14", "3.15"):
                with self.subTest(path=path, version=version):
                    active = {
                        req.name
                        for req in requirements
                        if req.marker is None or req.marker.evaluate(marker_environment(version))
                    }
                    self.assertEqual(active & excluded, excluded if version != "3.15" else set())


if __name__ == "__main__":
    unittest.main()
