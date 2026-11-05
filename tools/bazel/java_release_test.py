# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Portable artifact-construction checks, not substitutes for native inference."""

import argparse
import hashlib
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import java_release


class JavaReleaseTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name, content in {"VERSION_NUMBER": "1.32.0\n", "LICENSE": "MIT\n", "notices": "notices\n"}.items():
            (self.root / name).write_text(content, encoding="utf-8")
        for name, members in {
            "api.jar": {"ai/onnxruntime/OrtSession.class": b"\xca\xfe\xba\xbe\x00\x00\x00\x34fixture"},
            "native.jar": {
                "ai/onnxruntime/OrtSession.class": b"\xca\xfe\xba\xbe\x00\x00\x00\x34fixture",
                "ai/onnxruntime/native/linux-x64/libonnxruntime.so": b"fixture-only",
            },
            "source.jar": {"ai/onnxruntime/OrtSession.java": b"// source\r\n"},
            "docs.jar": {"index.html": b"index", "ai/onnxruntime/OrtSession.html": b"API"},
        }.items():
            (self.root / name).write_bytes(java_release.zip_bytes(members))
        self.args = argparse.Namespace(
            output=self.root / "maven.zip",
            version_file=self.root / "VERSION_NUMBER",
            license=self.root / "LICENSE",
            notices=self.root / "notices",
            api_jar=self.root / "api.jar",
            native_jar=self.root / "native.jar",
            javadoc_jar=self.root / "docs.jar",
            source=[self.root / "source.jar"],
            classifier="linux-x64",
            variant="cpu",
        )

    def test_repository_has_coordinates_companions_and_checksums(self):
        java_release.build_maven(self.args)
        members = java_release.jar_members(self.args.output)
        manifest = json.loads(members["onnxruntime-release.json"])
        self.assertEqual(manifest["classifier"], "linux-x64")
        prefix = "com/microsoft/onnxruntime/onnxruntime-bazel-linux-x64/1.32.0/onnxruntime-bazel-linux-x64-1.32.0"
        for suffix in (".pom", ".jar", "-sources.jar", "-javadoc.jar"):
            path = prefix + suffix
            self.assertEqual(manifest["artifacts"][path], hashlib.sha256(members[path]).hexdigest())
            for algorithm in ("sha256", "sha512", "sha1"):
                self.assertEqual(
                    members[f"{path}.{algorithm}"].decode().strip(),
                    hashlib.new(algorithm, members[path], usedforsecurity=False).hexdigest(),
                )
        pom = ET.fromstring(members[prefix + ".pom"])
        namespace = {"m": java_release.POM_NAMESPACE}
        self.assertEqual(pom.findtext("m:groupId", namespaces=namespace), java_release.GROUP)
        self.assertEqual(pom.findtext("m:artifactId", namespaces=namespace), "onnxruntime-bazel-linux-x64")
        self.assertEqual(pom.findtext("m:version", namespaces=namespace), "1.32.0")
        with zipfile.ZipFile(io.BytesIO(members[prefix + "-sources.jar"])) as sources:
            self.assertEqual(sources.read("ai/onnxruntime/OrtSession.java"), b"// source\n")
            self.assertIn("META-INF/LICENSE", sources.namelist())
        with zipfile.ZipFile(io.BytesIO(members[prefix + "-javadoc.jar"])) as docs:
            self.assertIn("ai/onnxruntime/OrtSession.html", docs.namelist())
        with zipfile.ZipFile(io.BytesIO(members[prefix + ".jar"])) as binary:
            self.assertIn("ai/onnxruntime/OrtSession.class", binary.namelist())
            self.assertIn("ai/onnxruntime/native/linux-x64/libonnxruntime.so", binary.namelist())

    def test_deterministic_repository(self):
        java_release.build_maven(self.args)
        original = self.args.output.read_bytes()
        java_release.build_maven(self.args)
        self.assertEqual(original, self.args.output.read_bytes())

    def test_native_classifier_must_match_payload(self):
        self.args.classifier = "osx-aarch64"
        with self.assertRaisesRegex(ValueError, "no resources for osx-aarch64"):
            java_release.build_maven(self.args)

    def test_rejects_missing_javadoc(self):
        self.args.javadoc_jar.write_bytes(java_release.zip_bytes({"placeholder.txt": b"not API documentation"}))
        with self.assertRaisesRegex(ValueError, "omits the public API"):
            java_release.build_maven(self.args)

    def test_rejects_missing_sources(self):
        self.args.source[0].write_bytes(java_release.zip_bytes({"readme.txt": b"not Java source"}))
        with self.assertRaisesRegex(ValueError, "No Java sources"):
            java_release.build_maven(self.args)

    def test_rejects_unsafe_source_archive(self):
        self.args.source[0].write_bytes(java_release.zip_bytes({"../OrtSession.java": b"bad"}))
        with self.assertRaisesRegex(ValueError, "unsafe member"):
            java_release.build_maven(self.args)

    def test_rejects_invalid_version(self):
        self.args.version_file.write_text("../not-a-version", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported Maven version"):
            java_release.build_maven(self.args)

    def test_gpu_coordinates_are_distinct(self):
        self.args.variant = "cuda"
        members = java_release.jar_members(self.args.native_jar)
        for name in (
            "libonnxruntime4j_jni.so",
            "libonnxruntime_providers_shared.so",
            "libonnxruntime_providers_cuda.so",
        ):
            members["ai/onnxruntime/native/linux-x64/" + name] = b"fixture-only"
        self.args.native_jar.write_bytes(java_release.zip_bytes(members))
        java_release.build_maven(self.args)
        members = java_release.jar_members(self.args.output)
        manifest = json.loads(members["onnxruntime-release.json"])
        self.assertEqual(manifest["artifact"], "onnxruntime-bazel-gpu-linux-x64")

    def test_gpu_requires_provider_payload(self):
        self.args.variant = "cuda"
        with self.assertRaisesRegex(ValueError, "both provider libraries"):
            java_release.build_maven(self.args)

    def test_java8_metadata_rejects_newer_class_version(self):
        self.args.api_jar.write_bytes(
            java_release.zip_bytes(
                {
                    "ai/onnxruntime/OrtSession.class": b"\xca\xfe\xba\xbe\x00\x00\x00\x3dfixture",
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "exceeds Java 8"):
            java_release.build_maven(self.args)

    def test_classifier_cannot_ship_a_different_api(self):
        members = java_release.jar_members(self.args.native_jar)
        members["ai/onnxruntime/OrtSession.class"] += b"different"
        self.args.native_jar.write_bytes(java_release.zip_bytes(members))
        with self.assertRaisesRegex(ValueError, "mismatched API class"):
            java_release.build_maven(self.args)

    def test_android_aar_has_maven_companions(self):
        self.args.aar = self.root / "android.aar"
        self.args.aar.write_bytes(
            java_release.zip_bytes(
                {
                    "classes.jar": (self.root / "api.jar").read_bytes(),
                    "AndroidManifest.xml": b"<manifest/>",
                    "jni/arm64-v8a/libonnxruntime.so": b"fixture-runtime",
                    "jni/arm64-v8a/libonnxruntime4j_jni.so": b"fixture-jni",
                }
            )
        )
        self.args.native_jar = None
        self.args.classifier = "android-arm64"
        java_release.build_maven(self.args)
        members = java_release.jar_members(self.args.output)
        prefix = "com/microsoft/onnxruntime/onnxruntime-bazel-android/1.32.0/onnxruntime-bazel-android-1.32.0"
        self.assertIn(prefix + ".aar", members)
        self.assertIn(prefix + "-sources.jar", members)
        self.assertIn(prefix + "-javadoc.jar", members)
        pom = ET.fromstring(members[prefix + ".pom"])
        self.assertEqual(pom.findtext(f"{{{java_release.POM_NAMESPACE}}}packaging"), "aar")


if __name__ == "__main__":
    unittest.main()
