#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "github" / "android"))

import build_aar_package


class BuildAarPackageTest(unittest.TestCase):
    def test_no_exceptions_preserves_telemetry_packaging(self):
        for disable_exceptions in (False, True):
            for no_telemetry in (False, True):
                with self.subTest(disable_exceptions=disable_exceptions, no_telemetry=no_telemetry):
                    build_params = ["--minimal_build"]
                    if disable_exceptions:
                        build_params.append("--disable_exceptions")
                    if no_telemetry:
                        build_params.append("--no_telemetry")
                    with tempfile.TemporaryDirectory(prefix="ort-aar-telemetry-") as temporary:
                        settings = Path(temporary) / "settings.json"
                        settings.write_text(json.dumps({"build_params": build_params}), encoding="utf-8")
                        parsed = build_aar_package._parse_build_settings(
                            argparse.Namespace(build_settings_file=settings)
                        )
                    self.assertEqual(parsed["use_telemetry"], not no_telemetry)


if __name__ == "__main__":
    unittest.main()
