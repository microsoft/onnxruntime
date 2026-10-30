# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import get_docker_image


class GetDockerImageTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.context = Path(self.directory.name)
        (self.context / "scripts").mkdir()
        (self.context / "scripts" / "deps.txt").write_text("", encoding="utf-8")
        self.arguments = [
            "get_docker_image.py",
            "--dockerfile",
            "Dockerfile.test",
            "--context",
            str(self.context),
            "--repository",
            "test-image",
            "--docker-path",
            "test-docker",
            "--docker-build-args",
            '--no-cache --build-arg "BASEIMAGE=registry/base:version"',
        ]

    def test_registry_build_pulls_base_and_preserves_cache_and_argument_forwarding(self):
        image = "test-registry.azurecr.io/test-image:latest"
        with (
            patch.object(sys, "argv", [*self.arguments, "--container-registry", "test-registry"]),
            patch.object(get_docker_image, "run") as run,
        ):
            self.assertEqual(get_docker_image.main(), 0)
        self.assertEqual(
            run.call_args_list,
            [
                call(
                    "test-docker",
                    "--log-level",
                    "error",
                    "buildx",
                    "build",
                    "--pull",
                    "--load",
                    "--tag",
                    image,
                    "--cache-from=type=registry,ref=" + image,
                    "--build-arg",
                    "BUILDKIT_INLINE_CACHE=1",
                    "--no-cache",
                    "--build-arg",
                    "BASEIMAGE=registry/base:version",
                    "-f",
                    "Dockerfile.test",
                    str(self.context),
                ),
                call("test-docker", "push", image),
                call("test-docker", "tag", image, "test-image"),
            ],
        )

    def test_non_registry_build_preserves_pull_and_does_not_push(self):
        with patch.object(sys, "argv", self.arguments), patch.object(get_docker_image, "run") as run:
            self.assertEqual(get_docker_image.main(), 0)
        self.assertEqual(
            run.call_args_list,
            [
                call(
                    "test-docker",
                    "build",
                    "--pull",
                    "--no-cache",
                    "--build-arg",
                    "BASEIMAGE=registry/base:version",
                    "--tag",
                    "test-image:latest",
                    "--file",
                    "Dockerfile.test",
                    str(self.context),
                ),
                call("test-docker", "tag", "test-image:latest", "test-image"),
            ],
        )

    def test_registry_build_keeps_layer_cache_enabled_by_default(self):
        with (
            patch.object(sys, "argv", [*self.arguments[:-2], "--container-registry", "test-registry"]),
            patch.object(get_docker_image, "run") as run,
        ):
            self.assertEqual(get_docker_image.main(), 0)
        build_arguments = run.call_args_list[0].args
        self.assertIn("--pull", build_arguments)
        self.assertIn("--cache-from=type=registry,ref=test-registry.azurecr.io/test-image:latest", build_arguments)
        self.assertNotIn("--no-cache", build_arguments)

    def test_failed_build_is_not_pushed_or_tagged(self):
        for registry_args in ([], ["--container-registry", "test-registry"]):
            with (
                self.subTest(registry_args=registry_args),
                patch.object(sys, "argv", [*self.arguments, *registry_args]),
                patch.object(
                    get_docker_image,
                    "run",
                    side_effect=subprocess.CalledProcessError(1, "test-docker"),
                ) as run,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    get_docker_image.main()
                run.assert_called_once()

    def test_failed_push_is_not_tagged(self):
        with (
            patch.object(sys, "argv", [*self.arguments, "--container-registry", "test-registry"]),
            patch.object(
                get_docker_image,
                "run",
                side_effect=[None, subprocess.CalledProcessError(1, "push")],
            ) as run,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            get_docker_image.main()
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
