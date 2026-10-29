# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fnmatch
import re
import unittest
from pathlib import Path

import yaml


class JarPipelineTriggersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline_dir = Path(__file__).parent / "github" / "azure-pipelines"
        cls.pipeline = yaml.safe_load((cls.pipeline_dir / "jar_package_testing.yml").read_text(encoding="utf-8"))
        cls.resources = cls.pipeline["resources"]["pipelines"]

    def _triggered_resources(self, source, branch):
        matches = []
        for resource in self.resources:
            if resource["source"] != source:
                continue
            filters = resource["trigger"]["branches"]
            included = any(fnmatch.fnmatchcase(branch, pattern) for pattern in filters.get("include", ["*"]))
            excluded = any(fnmatch.fnmatchcase(branch, pattern) for pattern in filters.get("exclude", []))
            if included and not excluded:
                matches.append(resource["pipeline"])
        return matches

    def test_release_completion_uses_legacy_resource(self):
        branch = "refs/heads/rel-1.30.1"
        self.assertEqual(self._triggered_resources("Zip-Nuget-Java-Nodejs Packaging Pipeline", branch), ["build"])
        self.assertEqual(self._triggered_resources("Nuget - Packaging - CUDA13", branch), [])

    def test_other_branches_use_cuda13_resource(self):
        for branch in ("refs/heads/main", "refs/heads/dev/test-jar-routing", "refs/heads/rel-1.31.0"):
            with self.subTest(branch=branch):
                self.assertEqual(self._triggered_resources("Zip-Nuget-Java-Nodejs Packaging Pipeline", branch), [])
                self.assertEqual(self._triggered_resources("Nuget - Packaging - CUDA13", branch), ["buildCuda13"])

    def test_current_stages_select_cuda13_resource(self):
        stages = self.pipeline["extends"]["parameters"]["stages"]
        self.assertEqual(len(stages), 4)
        for stage in stages:
            with self.subTest(template=stage["template"]):
                self.assertEqual(stage["parameters"]["PipelineResource"], "buildCuda13")

    def test_downloads_and_workspace_paths_follow_selected_resource(self):
        def strings(value):
            if isinstance(value, dict):
                for child in value.values():
                    yield from strings(child)
            elif isinstance(value, list):
                for child in value:
                    yield from strings(child)
            elif isinstance(value, str):
                yield value

        def downloads(value):
            if isinstance(value, dict):
                if "download" in value:
                    yield value["download"]
                for child in value.values():
                    yield from downloads(child)
            elif isinstance(value, list):
                for child in value:
                    yield from downloads(child)

        for template in ("win", "linux", "gpu"):
            path = self.pipeline_dir / "templates" / f"final-jar-testing-{template}.yml"
            source = path.read_text(encoding="utf-8")
            parameters = yaml.safe_load(source)["parameters"]
            resource_parameter = next(parameter for parameter in parameters if parameter["name"] == "PipelineResource")
            self.assertEqual(resource_parameter["default"], "build")
            for resource in ("build", "buildCuda13"):
                with self.subTest(template=template, resource=resource):
                    expanded = yaml.safe_load(source.replace("${{ parameters.PipelineResource }}", resource))
                    self.assertEqual(list(downloads(expanded)), [resource] * (2 if template == "gpu" else 1))
                    workspace_paths = [value for value in strings(expanded) if "$(Pipeline.Workspace)" in value]
                    self.assertTrue(workspace_paths)
                    for value in workspace_paths:
                        normalized = value.replace("\\", "/")
                        roots = re.findall(r"""\$\(Pipeline\.Workspace\)/([^/\s:'"]+)""", normalized)
                        self.assertEqual(set(roots), {resource})


if __name__ == "__main__":
    unittest.main()
