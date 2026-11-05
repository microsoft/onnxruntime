# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Package compiled Android Java/JSI and real Metro TypeScript sources."""

import argparse
import gzip
import io
import json
import tarfile
import zipfile
from pathlib import Path


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


def package(args):
    version = Path(args.version).read_text(encoding="utf-8").strip()
    entries = {
        "AndroidManifest.xml": (
            b'<manifest xmlns:android="http://schemas.android.com/apk/res/android" '
            b'package="ai.onnxruntime.reactnative"><uses-sdk android:minSdkVersion="27" /></manifest>\n'
        ),
        "R.txt": b"",
        "proguard.txt": b"-keep class ai.onnxruntime.reactnative.** { *; }\n",
        "classes.jar": Path(args.classes).read_bytes(),
    }
    libraries = {Path(p).name: Path(p).read_bytes() for p in args.native}
    expected = {"libonnxruntime.so", "libonnxruntimejsi.so", "libc++_shared.so"}
    if set(libraries) != expected or len(args.native) != len(expected):
        raise ValueError(f"React Native AAR requires exactly {sorted(expected)}; check its native target inputs.")
    entries.update({f"jni/arm64-v8a/{name}": data for name, data in libraries.items()})
    for path in args.notice:
        entries["META-INF/" + Path(path).name] = Path(path).read_bytes()
    write_zip(args.aar, entries)
    files = {"android/onnxruntime-react-native.aar": Path(args.aar).read_bytes()}
    files["android/build.gradle"] = (
        b'configurations.maybeCreate("default")\nartifacts.add("default", file("onnxruntime-react-native.aar"))\n'
    )
    files["react-native.config.js"] = (
        b"const path = require('path');\nmodule.exports = {dependency: {platforms: {ios: null, android: {\n"
        b"sourceDir: path.join(__dirname, 'android'),\n"
        b"packageImportPath: 'import ai.onnxruntime.reactnative.OnnxruntimePackage;',\n"
        b"packageInstance: 'new OnnxruntimePackage()'\n}}}};\n"
    )
    for path in args.frontend:
        files["lib/" + Path(path).name] = Path(path).read_bytes()
    for path in args.common:
        source = Path(path)
        sources = source.rglob("*") if source.is_dir() else [source]
        for item in sources:
            if item.is_file():
                relative = item.relative_to(source).as_posix() if source.is_dir() else item.name
                files["node_modules/onnxruntime-common/lib/" + relative] = item.read_bytes()
    common = {"name": "onnxruntime-common", "version": version, "main": "lib/index.js", "types": "lib/index.d.ts"}
    files["node_modules/onnxruntime-common/package.json"] = json.dumps(common).encode()
    metadata = {
        "name": "onnxruntime-react-native",
        "version": version,
        "main": "lib/index.ts",
        "react-native": "lib/index.ts",
        "types": "lib/index.ts",
        "license": "MIT",
        "dependencies": {"onnxruntime-common": version},
        "bundledDependencies": ["onnxruntime-common"],
        "peerDependencies": {"react": "^18.2.0", "react-native": "0.73.11"},
        "onnxruntimeBazelAar": "android/onnxruntime-react-native.aar",
    }
    files["package.json"] = json.dumps(metadata, indent=2).encode()
    for path in args.notice:
        files[Path(path).name] = Path(path).read_bytes()
    with (
        Path(args.npm).open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo("package/" + name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("aar", "npm", "classes", "version"):
        parser.add_argument("--" + key, required=True)
    for key in ("native", "frontend", "common", "notice"):
        parser.add_argument("--" + key, action="append", default=[])
    package(parser.parse_args())


if __name__ == "__main__":
    main()
