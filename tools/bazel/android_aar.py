# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Assemble an Android library archive from Bazel-compiled Java and native code."""

import pathlib
import sys
import zipfile


def package(output, classes, runtime, jni, assets):
    entries = {
        "AndroidManifest.xml": (
            b'<manifest xmlns:android="http://schemas.android.com/apk/res/android" '
            b'package="ai.onnxruntime"><uses-sdk android:minSdkVersion="27" /></manifest>\n'
        ),
        "R.txt": b"",
        # Preserve classes reached only through native JNI calls.
        "proguard.txt": b"-keep class ai.onnxruntime.** { *; }\n",
        "classes.jar": pathlib.Path(classes).read_bytes(),
        "jni/arm64-v8a/libonnxruntime.so": pathlib.Path(runtime).read_bytes(),
        "jni/arm64-v8a/libonnxruntime4j_jni.so": pathlib.Path(jni).read_bytes(),
    }
    for asset in assets:
        path = pathlib.Path(asset)
        name = ("headers/" if path.suffix == ".h" else "META-INF/") + path.name
        if name in entries:
            raise ValueError(f"Duplicate Android archive entry {name}: remove the conflicting package asset.")
        entries[name] = path.read_bytes()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


if __name__ == "__main__":
    package(*sys.argv[1:5], sys.argv[5:])
