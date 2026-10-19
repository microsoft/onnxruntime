# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

from ..vcpkg_helpers import (
    generate_android_triplets,
    generate_linux_triplets,
    generate_macos_triplets,
    generate_triplet_for_posix_platform,
    generate_windows_triplets,
)


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

    def test_noexception_triplets_strip_unwind_tables_only_for_minimal_build(self):
        generators = (
            ("x64-linux.cmake", lambda d: generate_linux_triplets(d, {"Release"}, use_full_protobuf=False)),
            ("arm64-android.cmake", lambda d: generate_android_triplets(d, {"Release"}, False, 27, False)),
            ("arm64-osx.cmake", lambda d: generate_macos_triplets(d, {"Release"}, "14.0", False)),
        )
        for file_name, generate in generators:
            with tempfile.TemporaryDirectory() as build_dir:
                generate(build_dir)
                for folder, expect_unwind_stripped in (("noexception", False), ("noexception_minimal", True)):
                    with self.subTest(file_name=file_name, folder=folder):
                        triplet = (Path(build_dir) / "Release" / folder / file_name).read_text(encoding="utf-8")
                        self.assertIn("-fno-exceptions", triplet)
                        self.assertIn("-DONNX_DISABLE_EXCEPTIONS=ON", triplet)
                        self.assertEqual("-DONNX_MINIMAL_BUILD=ON" in triplet, expect_unwind_stripped, triplet)
                        for flag in ("-fno-unwind-tables", "-fno-asynchronous-unwind-tables"):
                            self.assertEqual(flag in triplet, expect_unwind_stripped, triplet)

    def test_windows_noexception_triplets(self):
        with tempfile.TemporaryDirectory() as build_dir:
            generate_windows_triplets(build_dir, {"Release"}, "14.44", False)
            for folder, minimal in (("noexception", False), ("noexception_minimal", True)):
                with self.subTest(folder=folder):
                    triplet = (Path(build_dir) / "Release" / folder / "x64-windows-static-md.cmake").read_text(
                        encoding="utf-8"
                    )
                    self.assertNotIn("/EHsc", triplet)
                    self.assertIn("-DONNX_DISABLE_EXCEPTIONS=ON", triplet)
                    self.assertEqual("-DONNX_MINIMAL_BUILD=ON" in triplet, minimal, triplet)


if __name__ == "__main__":
    unittest.main()
