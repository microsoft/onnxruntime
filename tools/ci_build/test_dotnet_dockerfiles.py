# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from pathlib import Path


class DotnetDockerfilesTest(unittest.TestCase):
    def test_cached_images_install_sdk_without_external_script(self):
        inference = Path(__file__).parent / "github" / "linux" / "docker" / "inference"
        for architecture, image in (
            ("aarch64", "cpu"),
            ("x86_64", "cpu"),
            ("x86_64", "cuda12"),
            ("x86_64", "cuda13"),
        ):
            with self.subTest(architecture=architecture, image=image):
                dockerfile = inference / architecture / "default" / image / "Dockerfile"
                content = dockerfile.read_text()
                self.assertNotIn("/tmp/scripts/install_dotnet.sh", content)
                install = "RUN dnf install -y dotnet-sdk-10.0 perl-core && dnf clean all"
                self.assertIn(install, content)
                self.assertLess(content.index(install), content.index("USER $BUILD_USER"))
                if "exclude=dotnet" in content:
                    self.assertLess(content.index(install), content.index("exclude=dotnet"))


if __name__ == "__main__":
    unittest.main()
