# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

from ..vcpkg_helpers import generate_triplet_for_posix_platform


class PosixTripletTest(unittest.TestCase):
    def test_binskim_flags_match_target_platform(self):
        targets = (("linux", "x64"), ("linux", "arm64"), ("osx", "x64"), ("osx", "arm64"), ("osx", "universal2"))
        for os_name, target_abi in targets:
            for enable_binskim in (False, True):
                with (
                    self.subTest(os_name=os_name, target_abi=target_abi, enable_binskim=enable_binskim),
                    tempfile.TemporaryDirectory() as build_dir,
                ):
                    generate_triplet_for_posix_platform(
                        build_dir=build_dir,
                        configs={"Release", "RelWithDebInfo"},
                        os_name=os_name,
                        enable_rtti=True,
                        enable_exception=True,
                        enable_binskim=enable_binskim,
                        enable_asan=False,
                        enable_minimal_build=False,
                        crt_linkage="dynamic",
                        target_abi=target_abi,
                        osx_deployment_target="14.0",
                        use_full_protobuf=False,
                    )
                    folder = "binskim" if enable_binskim else "default"
                    for config in ("Release", "RelWithDebInfo"):
                        triplet = (Path(build_dir) / config / folder / f"{target_abi}-{os_name}.cmake").read_text(
                            encoding="utf-8"
                        )
                        expect_x64_linux_flags = enable_binskim and os_name == "linux" and target_abi == "x64"
                        for flag in ("-fstack-clash-protection", "-fcf-protection"):
                            self.assertEqual(flag in triplet, expect_x64_linux_flags, triplet)
                        self.assertEqual("-fstack-protector-strong" in triplet, enable_binskim, triplet)


if __name__ == "__main__":
    unittest.main()
