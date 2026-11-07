# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import tempfile
import unittest
from pathlib import Path

from release_inventory import inventory


class ReleaseInventoryTest(unittest.TestCase):
    def test_hashes_actual_bytes_and_does_not_claim_consumer_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "package.zip"
            path.write_bytes(b"actual artifact bytes")
            result = inventory([f"//:package={path}"], "1.32.0", "linux-x64", "cpu")
            self.assertEqual(result["artifacts"][0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result["artifacts"][0]["bytes"], 21)
            self.assertIn("recorded separately", result["validation"])

    def test_rejects_empty_or_duplicate_inventory(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            inventory([], "1.32.0", "linux-x64", "cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "package.zip"
            path.write_bytes(b"payload")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                inventory([f"//:package={path}"] * 2, "1.32.0", "linux-x64", "cpu")

    def test_inventories_tree_artifact_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory) / "wheel"
            tree.mkdir()
            (tree / "onnxruntime.whl").write_bytes(b"wheel bytes")
            result = inventory([f"//:python_wheel={tree}"], "1.32.0", "linux-x64", "cpu")
            self.assertEqual(result["artifacts"][0]["file"], "wheel/onnxruntime.whl")
            self.assertEqual(result["artifacts"][0]["bytes"], 11)

    def test_rejects_empty_tree(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "empty"):
            inventory([f"//:python_wheel={directory}"], "1.32.0", "linux-x64", "cpu")

    def test_hashes_declared_sandbox_symlink_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "package.zip"
            path.write_bytes(b"declared artifact")
            link = Path(directory) / "sandbox.zip"
            link.symlink_to(path)
            result = inventory([f"//:package={link}"], "1.32.0", "linux-x64", "cpu")
            self.assertEqual(result["artifacts"][0]["file"], "sandbox.zip")
            self.assertEqual(result["artifacts"][0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_hashes_sandbox_symlink_within_declared_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            path.write_bytes(b"outside package tree")
            tree = Path(directory) / "package"
            tree.mkdir()
            (tree / "linked").symlink_to(path)
            result = inventory([f"//:package={tree}"], "1.32.0", "linux-x64", "cpu")
            self.assertEqual(result["artifacts"][0]["file"], "package/linked")
            self.assertEqual(result["artifacts"][0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
