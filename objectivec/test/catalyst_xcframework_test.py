# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Synthetic format regressions; these do not replace an Xcode Catalyst build."""

import plistlib
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "bazel"))
from catalyst_xcframework import merge, verify_catalyst


def object_file(cpu, platform=6, version=0x0F0100, with_version=True):
    command = struct.pack("<6I", 0x32, 24, platform, version, 0x120000, 0) if with_version else b""
    return struct.pack("<8I", 0xFEEDFACF, cpu, 0, 1, int(with_version), len(command), 0, 0) + command


def static_archive(payload, name=b"genuine_fixture_object.o"):
    body = name + payload
    header = (
        f"#1/{len(name)}".ljust(16)
        + "0".ljust(12)
        + "0".ljust(6)
        + "0".ljust(6)
        + "644".ljust(8)
        + str(len(body)).ljust(10)
        + "`\n"
    ).encode("ascii")
    return b"!<arch>\n" + header + body + (b"\n" if len(body) & 1 else b"")


def universal(platform=6, version=0x0F0100, with_version=True, wide=False):
    cpus = (0x0100000C, 0x01000007)
    slices = [static_archive(object_file(cpu, platform, version, with_version)) for cpu in cpus]
    offset = 8 + (32 if wide else 20) * 2
    result = struct.pack(">II", 0xCAFEBABF if wide else 0xCAFEBABE, 2)
    for cpu, payload in zip(cpus, slices, strict=True):
        result += (
            struct.pack(">IIQQII", cpu, 0, offset, len(payload), 0, 0)
            if wide
            else struct.pack(">IIIII", cpu, 0, offset, len(payload), 0)
        )
        offset += len(payload)
    return result + b"".join(slices)


def framework_zip(path, payload, identifier="macos-arm64_x86_64"):
    library = {
        "LibraryIdentifier": identifier,
        "LibraryPath": "onnxruntime_objc.framework",
        "SupportedArchitectures": ["arm64", "x86_64"],
        "SupportedPlatform": "macos",
    }
    root = "onnxruntime_objc.xcframework/"
    prefix = root + identifier + "/onnxruntime_objc.framework/"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(root + "Info.plist", plistlib.dumps({"AvailableLibraries": [library]}))
        archive.writestr(prefix + "onnxruntime_objc", payload)
        archive.writestr(
            prefix + "Info.plist",
            plistlib.dumps(
                {
                    "CFBundleExecutable": "onnxruntime_objc",
                    "LSMinimumSystemVersion": "12.0",
                }
            ),
        )
        archive.writestr(prefix + "Headers/onnxruntime.h", b"fixture header")


class CatalystXCFrameworkTest(unittest.TestCase):
    def test_requires_platform_proof_in_each_architecture(self):
        self.assertEqual(verify_catalyst(universal()), "15.1")
        self.assertEqual(verify_catalyst(universal(wide=True)), "15.1")
        for platform in (1, 2, 7):
            with self.subTest(platform=platform), self.assertRaisesRegex(ValueError, "not Catalyst"):
                verify_catalyst(universal(platform=platform))

    def test_rejects_missing_or_inconsistent_minimum(self):
        with self.assertRaisesRegex(ValueError, "LC_BUILD_VERSION"):
            verify_catalyst(universal(with_version=False))
        with self.assertRaisesRegex(ValueError, "15.1"):
            verify_catalyst(universal(version=0x0F0000))
        with self.assertRaisesRegex(ValueError, "both architecture"):
            verify_catalyst(static_archive(object_file(0x0100000C)))

    def test_rejects_truncated_headers_bitcode_and_wrong_cpu(self):
        for payload in (b"!<arch>\nshort", b"\xca\xfe\xba\xbe", b"BC\xc0\xde", universal()[:-10]):
            with self.subTest(payload=payload[:8]), self.assertRaises(ValueError):
                verify_catalyst(payload)
        wrong = bytearray(universal())
        struct.pack_into(">I", wrong, 8, 0x01000007)
        with self.assertRaisesRegex(ValueError, "CPU disagrees"):
            verify_catalyst(bytes(wrong))

    def test_rejects_conflicting_object_even_when_other_objects_are_catalyst(self):
        mixed = universal()
        bad = static_archive(object_file(0x0100000C, platform=1))
        with self.assertRaisesRegex(ValueError, "not Catalyst"):
            verify_catalyst(b"!<arch>\n" + static_archive(mixed)[8:] + bad[8:])

    def test_merges_only_proven_native_slice_and_preserves_base(self):
        with tempfile.TemporaryDirectory() as directory:
            base, catalyst, output = (Path(directory) / name for name in ("base.zip", "catalyst.zip", "out.zip"))
            framework_zip(base, b"unchanged base bytes")
            framework_zip(catalyst, universal())
            merge(base, catalyst, output)
            with zipfile.ZipFile(base) as original, zipfile.ZipFile(output) as merged:
                root = "onnxruntime_objc.xcframework/"
                for name in original.namelist():
                    if name != root + "Info.plist":
                        self.assertEqual(original.read(name), merged.read(name))
                info = plistlib.loads(merged.read(root + "Info.plist"))
                self.assertEqual(info["AvailableLibraries"][-1]["SupportedPlatformVariant"], "maccatalyst")
                prefix = root + "ios-arm64_x86_64-maccatalyst/onnxruntime_objc.framework/"
                framework_info = plistlib.loads(merged.read(prefix + "Info.plist"))
                self.assertEqual(framework_info["MinimumOSVersion"], "15.1")
                self.assertNotIn("LSMinimumSystemVersion", framework_info)
                self.assertEqual(merged.read(prefix + "onnxruntime_objc"), universal())

    def test_relabeling_macos_never_produces_an_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base, catalyst, output = (Path(directory) / name for name in ("base.zip", "catalyst.zip", "out.zip"))
            framework_zip(base, b"unchanged")
            framework_zip(catalyst, universal(platform=1))
            with self.assertRaisesRegex(ValueError, "not Catalyst"):
                merge(base, catalyst, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
