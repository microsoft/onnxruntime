#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Test runner for the WGSL template Python port.

Discovers ``test_*.py`` siblings and aggregates them into one suite.
Invoked manually or via ``ctest`` (see CMake's ``add_test`` wiring).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the wgsl_template package importable regardless of cwd.
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))


def load_tests(loader, standard_tests, pattern):
    discovered = loader.discover(
        start_dir=str(_THIS_DIR),
        pattern="test_*.py",
        top_level_dir=str(_THIS_DIR),
    )
    standard_tests.addTests(discovered)
    return standard_tests


if __name__ == "__main__":  # pragma: no cover
    unittest.main(module=__name__, verbosity=2)
