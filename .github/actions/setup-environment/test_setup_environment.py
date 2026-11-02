# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import re
import unittest
from pathlib import Path

import yaml


class SetupEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        cls.action = yaml.safe_load(Path(__file__).with_name("action.yml").read_text(encoding="utf-8"))

    def test_metadata_and_pinned_actions(self):
        self.assertEqual(self.action["runs"]["using"], "composite")
        self.assertEqual(set(self.action["inputs"]), {"python-version", "architecture", "node-version", "registry-url"})
        for definition in self.action["inputs"].values():
            self.assertFalse(definition["required"])
            self.assertEqual(definition["default"], "")
            self.assertTrue(definition["description"])
        self.assertEqual(len(self.action["runs"]["steps"]), 2)
        for tool, step in zip(("python", "node"), self.action["runs"]["steps"], strict=True):
            self.assertRegex(step["uses"], rf"^actions/setup-{tool}@[0-9a-f]{{40}}$")

    def test_python_node_both_and_empty_routes(self):
        cases = [
            ({}, []),
            ({"python-version": "3.12"}, ["python"]),
            ({"node-version": "24.x"}, ["node"]),
            ({"python-version": "3.14", "node-version": "24.x"}, ["python", "node"]),
        ]
        for supplied, expected in cases:
            with self.subTest(inputs=supplied):
                selected = []
                for step in self.action["runs"]["steps"]:
                    tool = step["uses"].split("@")[0].removeprefix("actions/setup-")
                    self.assertEqual(step["if"], f"inputs.{tool}-version != ''")
                    if supplied.get(f"{tool}-version", "") != "":
                        selected.append(tool)
                self.assertEqual(selected, expected)

    def test_only_observed_options_are_forwarded_without_shell_interpolation(self):
        python, node = self.action["runs"]["steps"]
        self.assertEqual(
            python["with"],
            {"python-version": "${{ inputs.python-version }}", "architecture": "${{ inputs.architecture }}"},
        )
        self.assertEqual(
            node["with"],
            {"node-version": "${{ inputs.node-version }}", "registry-url": "${{ inputs.registry-url }}"},
        )
        for step in (python, node):
            self.assertEqual(set(step), {"name", "if", "uses", "with"})

    def test_workflow_callers_have_checkout_and_supported_inputs(self):
        count = 0
        for path in (self.root / ".github" / "workflows").glob("*.y*ml"):
            workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
            for job in workflow.get("jobs", {}).values():
                checked_out = False
                for step in job.get("steps", []):
                    if step.get("uses", "").startswith("actions/checkout@"):
                        options = step.get("with", {})
                        if not options.get("path") and not options.get("repository") and not step.get("if"):
                            checked_out = True
                    if step.get("uses") != "./.github/actions/setup-environment":
                        continue
                    count += 1
                    with self.subTest(workflow=path.name, step=step.get("name")):
                        self.assertTrue(checked_out, "Local action requires checkout at the workspace root")
                        self.assertLessEqual(set(step.get("with", {})), set(self.action["inputs"]))
                        self.assertTrue(step["with"].get("python-version") or step["with"].get("node-version"))
        self.assertGreater(count, 0)

    def test_direct_workflow_setup_does_not_bypass_shared_pins(self):
        for path in (self.root / ".github" / "workflows").glob("*.y*ml"):
            with self.subTest(workflow=path.name):
                self.assertIsNone(
                    re.search(r"uses:\s*actions/setup-(?:python|node)@", path.read_text(encoding="utf-8"))
                )


if __name__ == "__main__":
    unittest.main()
