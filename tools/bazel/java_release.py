# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create unsigned Maven repository artifacts without Gradle or native rebuilding."""

import argparse
import hashlib
import io
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

GROUP = "com.microsoft.onnxruntime"
POM_NAMESPACE = "http://maven.apache.org/POM/4.0.0"
CLASSIFIERS = {f"{system}-{arch}" for system in ("linux", "osx", "win") for arch in ("x64", "aarch64")} | {
    "android-arm64"
}


def jar_members(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"JAR {path} contains duplicate members; rebuild its producing target.")
        members = {}
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename:
                raise ValueError(f"JAR {path} contains unsafe member {item.filename!r}; do not distribute it.")
            if not item.is_dir():
                members[item.filename] = archive.read(item)
        return members


def zip_bytes(members):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return output.getvalue()


def source_members(paths):
    members = {}
    for path in paths:
        for name, data in jar_members(path).items():
            if not name.endswith(".java"):
                continue
            # Java source JARs have already stripped the Java source-root prefix.
            if name in members:
                raise ValueError(f"Repeated Java source {name!r} in {paths}; package each source exactly once.")
            members[name] = data.replace(b"\r\n", b"\n")
    if not members:
        raise ValueError(f"No Java sources found in {paths}; pass java_binding's direct source JARs.")
    return members


def validate_api_classes(members):
    classes = {name: data for name, data in members.items() if name.endswith(".class")}
    if not classes:
        raise ValueError("Java API JAR contains no compiled classes; build java_binding before packaging.")
    for name, data in classes.items():
        if len(data) < 8 or data[:4] != b"\xca\xfe\xba\xbe":
            raise ValueError(f"Invalid Java class {name}; rebuild the Java API JAR.")
        major = int.from_bytes(data[6:8], "big")
        if major > 52:
            raise ValueError(f"Java class {name} has major {major}, exceeds Java 8 (52); compile with --release 8.")


def build_javadoc(java_home, sources, output):
    executable = java_home.resolve() / "bin" / "javadoc"
    if not executable.is_file():
        executable = executable.with_suffix(".exe")
    if not executable.is_file():
        raise FileNotFoundError(f"Javadoc is missing from JDK {java_home}; use a complete Bazel Java runtime.")
    with tempfile.TemporaryDirectory(prefix="ort-javadoc-") as directory:
        root = Path(directory)
        source_root = root / "sources"
        files = []
        for name, data in source_members(sources).items():
            destination = source_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            files.append(str(destination))
        docs = root / "docs"
        arguments = root / "sources.args"
        arguments.write_text("\n".join('"' + name.replace("\\", "/") + '"' for name in files), encoding="utf-8")
        result = subprocess.run(
            [
                str(executable),
                "-quiet",
                "-notimestamp",
                "-Xdoclint:none",
                "--no-platform-links",
                "--release",
                "8",
                "-encoding",
                "UTF-8",
                "-charset",
                "UTF-8",
                "-docencoding",
                "UTF-8",
                "-d",
                str(docs),
                "@" + str(arguments),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"Javadoc failed using {executable} (exit {result.returncode}); fix the Java source diagnostics:\n"
                f"{result.stdout}\n{result.stderr}"
            )
        members = {path.relative_to(docs).as_posix(): path.read_bytes() for path in docs.rglob("*") if path.is_file()}
        if "index.html" not in members or "ai/onnxruntime/OrtSession.html" not in members:
            raise RuntimeError("Javadoc omitted the public API/index; inspect java_binding's source JAR inputs.")
        output.write_bytes(zip_bytes(members))


def maven_pom(version, artifact, packaging="jar"):
    ET.register_namespace("", POM_NAMESPACE)

    def element(parent, name, text=None):
        return ET.SubElement(parent, f"{{{POM_NAMESPACE}}}{name}") if text is None else add_text(parent, name, text)

    def add_text(parent, name, text):
        child = ET.SubElement(parent, f"{{{POM_NAMESPACE}}}{name}")
        child.text = text
        return child

    root = ET.Element(f"{{{POM_NAMESPACE}}}project")
    for key, value in {
        "modelVersion": "4.0.0",
        "groupId": GROUP,
        "artifactId": artifact,
        "version": version,
        "packaging": packaging,
        "name": "ONNX Runtime native Bazel distribution",
        "description": "Source-built single-platform ONNX Runtime inference SDK.",
        "url": "https://onnxruntime.ai",
    }.items():
        element(root, key, value)
    license_element = element(element(root, "licenses"), "license")
    element(license_element, "name", "MIT License")
    element(license_element, "url", "https://opensource.org/licenses/MIT")
    scm = element(root, "scm")
    element(scm, "connection", "scm:git:https://github.com/microsoft/onnxruntime.git")
    element(scm, "url", "https://github.com/microsoft/onnxruntime")
    developer = element(element(root, "developers"), "developer")
    element(developer, "id", "onnxruntime")
    element(developer, "name", "ONNX Runtime")
    properties = element(root, "properties")
    element(properties, "project.build.sourceEncoding", "UTF-8")
    element(properties, "maven.compiler.release", "8")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def build_maven(args):
    version = args.version_file.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-.][A-Za-z0-9.]+)?", version):
        raise ValueError(f"Unsupported Maven version {version!r} in {args.version_file}; supply a release version.")
    if args.classifier not in CLASSIFIERS:
        raise ValueError(f"Unknown native classifier {args.classifier!r}; select one of {sorted(CLASSIFIERS)}.")
    aar = getattr(args, "aar", None)
    if bool(aar) == bool(args.native_jar):
        raise ValueError("Supply exactly one native JAR or Android AAR for the Maven distribution.")
    if aar and (args.classifier != "android-arm64" or args.variant != "cpu"):
        raise ValueError("Android AAR requires classifier android-arm64 and CPU variant.")
    artifact = (
        "onnxruntime-bazel-android"
        if aar
        else "onnxruntime-bazel" + ("-gpu" if args.variant == "cuda" else "") + "-" + args.classifier
    )
    pom = maven_pom(version, artifact, "aar" if aar else "jar")
    metadata = {
        "META-INF/LICENSE": args.license.read_bytes(),
        "META-INF/ThirdPartyNotices.txt": args.notices.read_bytes(),
        f"META-INF/maven/{GROUP}/{artifact}/pom.xml": pom,
        f"META-INF/maven/{GROUP}/{artifact}/pom.properties": (
            f"groupId={GROUP}\nartifactId={artifact}\nversion={version}\n"
        ).encode(),
    }
    api = jar_members(args.api_jar)
    validate_api_classes(api)
    api.update(metadata)
    api["META-INF/MANIFEST.MF"] = (
        "Manifest-Version: 1.0\r\nAutomatic-Module-Name: com.microsoft.onnxruntime\r\n"
        f"Implementation-Title: onnxruntime\r\nImplementation-Version: {version}\r\n\r\n"
    ).encode()
    native = jar_members(aar or args.native_jar)
    if not aar:
        validate_api_classes(native)
        for name, data in api.items():
            if name.endswith(".class") and native.get(name) != data:
                raise ValueError(f"Native classifier contains mismatched API class {name}; rebuild both Java targets.")
    native_prefix = f"ai/onnxruntime/native/{args.classifier}/"
    if aar:
        required = {
            "classes.jar",
            "AndroidManifest.xml",
            "jni/arm64-v8a/libonnxruntime.so",
            "jni/arm64-v8a/libonnxruntime4j_jni.so",
        }
        if not required.issubset(native):
            raise ValueError(f"Android AAR omits required Java/native members: {sorted(required - native.keys())}")
        validate_api_classes(jar_members(io.BytesIO(native["classes.jar"])))
    elif not any(name.startswith(native_prefix) for name in native):
        raise ValueError(
            f"Native JAR {args.native_jar} has no resources for {args.classifier}; "
            "build the Java JAR for the requested target platform."
        )
    if args.variant == "cuda":
        required = {
            native_prefix + name
            for name in (
                "libonnxruntime.so",
                "libonnxruntime4j_jni.so",
                "libonnxruntime_providers_shared.so",
                "libonnxruntime_providers_cuda.so",
            )
        }
        if args.classifier != "linux-x64" or not required.issubset(native):
            raise ValueError("CUDA Maven packaging requires linux-x64 runtime, JNI and both provider libraries.")
    docs = jar_members(args.javadoc_jar)
    if "index.html" not in docs or "ai/onnxruntime/OrtSession.html" not in docs:
        raise ValueError(f"Javadoc JAR {args.javadoc_jar} omits the public API; build :java_javadoc_jar.")
    sources = source_members(args.source)
    sources.update(metadata)
    docs.update(metadata)
    native.update(metadata)
    native["META-INF/MANIFEST.MF"] = api["META-INF/MANIFEST.MF"]
    stem = f"{artifact}-{version}"
    prefix = f"{GROUP.replace('.', '/')}/{artifact}/{version}/"
    artifacts = {
        f"{stem}.pom": pom,
        f"{stem}.jar": zip_bytes(native),
        f"{stem}-sources.jar": zip_bytes(sources),
        f"{stem}-javadoc.jar": zip_bytes(docs),
    }
    if aar:
        del artifacts[f"{stem}.jar"]
        artifacts[f"{stem}.aar"] = zip_bytes(native)
    members = {}
    for name, data in artifacts.items():
        members[prefix + name] = data
        for algorithm in ("sha256", "sha512", "sha1"):
            digest = hashlib.new(algorithm, data, usedforsecurity=False).hexdigest()
            members[f"{prefix}{name}.{algorithm}"] = (digest + "\n").encode()
    members["onnxruntime-release.json"] = (
        json.dumps(
            {
                "schema_version": 1,
                "language": "java",
                "version": version,
                "variant": args.variant,
                "group": GROUP,
                "artifact": artifact,
                "classifier": args.classifier,
                "artifacts": {prefix + name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
            },
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    args.output.write_bytes(zip_bytes(members))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    docs = commands.add_parser("javadoc")
    docs.add_argument("--java-home", type=Path, required=True)
    docs.add_argument("--source", type=Path, action="append", required=True)
    docs.add_argument("--output", type=Path, required=True)
    maven = commands.add_parser("maven")
    for name in ("output", "version-file", "api-jar", "javadoc-jar", "license", "notices"):
        maven.add_argument("--" + name, type=Path, required=True)
    maven.add_argument("--native-jar", type=Path)
    maven.add_argument("--aar", type=Path)
    maven.add_argument("--source", type=Path, action="append", required=True)
    maven.add_argument("--classifier", required=True)
    maven.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "javadoc":
        build_javadoc(args.java_home, args.source, args.output)
    else:
        build_maven(args)


if __name__ == "__main__":
    main()
