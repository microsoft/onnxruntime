# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create deterministic managed and per-RID native NuGet packages without MSBuild."""

import argparse
import html
import plistlib
import zipfile
from pathlib import Path, PurePosixPath

MANAGED_ID = "Microsoft.ML.OnnxRuntime.Managed.Bazel"
MANAGED_FRAMEWORKS = {
    "netstandard2.0": "managed_netstandard20",
    "net8.0": "managed",
    "net9.0-android35.0": "managed_android",
    "net9.0-ios18.0": "managed_ios",
    "net9.0-maccatalyst18.0": "managed_maccatalyst",
}
SYMBOL_FRAMEWORKS = {
    "netstandard2.0": "symbol_netstandard20",
    "net8.0": "symbol",
    "net9.0-android35.0": "symbol_android",
    "net9.0-ios18.0": "symbol_ios",
    "net9.0-maccatalyst18.0": "symbol_maccatalyst",
}
APPLE_REQUIRED_SLICES = {
    ("ios", "", ("arm64",)),
    ("ios", "simulator", ("arm64", "x86_64")),
    ("ios", "maccatalyst", ("arm64", "x86_64")),
}


def xml(value):
    return html.escape(value, quote=True)


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(entries.items()):
            if "\\" in name or name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"Unsafe NuGet member path {name}; use a normalized package-relative path.")
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


def validate_android_archive(path):
    try:
        with zipfile.ZipFile(path) as archive:
            runtime_libraries = {name for name in archive.namelist() if name.endswith("/libonnxruntime.so")}
            expected_runtime = {"jni/arm64-v8a/libonnxruntime.so"}
            if runtime_libraries != expected_runtime:
                raise ValueError(
                    f"Android AAR {path} must contain exactly the ARM64 runtime {sorted(expected_runtime)}; "
                    f"got {sorted(runtime_libraries)}."
                )
            try:
                manifest = archive.read("AndroidManifest.xml").decode()
            except KeyError as error:
                raise ValueError(f"Android AAR {path} is missing AndroidManifest.xml.") from error
            if 'android:minSdkVersion="27"' not in manifest:
                raise ValueError(
                    f'Android AAR {path} must declare android:minSdkVersion="27" to match the source build.'
                )
    except zipfile.BadZipFile as error:
        raise ValueError(f"Android native input {path} is not a valid AAR ZIP archive.") from error


def validate_apple_xcframework(path):
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            plist_names = [name for name in names if name.endswith(".xcframework/Info.plist")]
            if len(plist_names) != 1:
                raise ValueError(f"Apple XCFramework {path} must contain exactly one Info.plist; got {plist_names}.")
            try:
                info = plistlib.loads(archive.read(plist_names[0]))
                libraries = info["AvailableLibraries"]
                libraries_by_slice = {
                    (
                        library["SupportedPlatform"],
                        library.get("SupportedPlatformVariant", ""),
                        tuple(sorted(library["SupportedArchitectures"])),
                    ): library
                    for library in libraries
                }
            except (KeyError, plistlib.InvalidFileException, TypeError, ValueError) as error:
                raise ValueError(
                    f"Apple XCFramework {path} has invalid library metadata in {plist_names[0]}."
                ) from error
            missing_slices = APPLE_REQUIRED_SLICES.difference(libraries_by_slice)
            if missing_slices:
                raise ValueError(
                    f"Apple XCFramework {path} cannot be packaged for iOS and Mac Catalyst because it is missing "
                    f"required slices {sorted(missing_slices)}. Build genuine Mac Catalyst slices instead of using "
                    "an iOS or macOS binary."
                )
            root = PurePosixPath(plist_names[0]).parent
            minimum_versions = {}
            for slice_key in APPLE_REQUIRED_SLICES:
                library = libraries_by_slice[slice_key]
                framework = root / library["LibraryIdentifier"] / library["LibraryPath"]
                if framework.suffix != ".framework":
                    raise ValueError(
                        f"Apple XCFramework {path} slice {slice_key} uses {framework.name}; "
                        "the managed mobile package requires a static framework with explicit deployment metadata."
                    )
                framework_plist = str(framework / "Info.plist")
                if framework_plist not in names:
                    raise ValueError(
                        f"Apple XCFramework {path} slice {slice_key} is missing framework Info.plist "
                        f"{framework_plist}; its minimum deployment version cannot be verified."
                    )
                try:
                    minimum_version = plistlib.loads(archive.read(framework_plist))["MinimumOSVersion"]
                except (KeyError, plistlib.InvalidFileException, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Apple XCFramework {path} slice {slice_key} does not declare a valid MinimumOSVersion."
                    ) from error
                if not isinstance(minimum_version, str) or not minimum_version:
                    raise ValueError(
                        f"Apple XCFramework {path} slice {slice_key} has invalid MinimumOSVersion "
                        f"{minimum_version!r}; expected a non-empty version string."
                    )
                minimum_versions[slice_key] = minimum_version
            ios_versions = {
                minimum_versions[slice_key] for slice_key in APPLE_REQUIRED_SLICES if slice_key[1] != "maccatalyst"
            }
            if len(ios_versions) != 1:
                raise ValueError(
                    f"Apple XCFramework {path} has inconsistent iOS deployment minimums {sorted(ios_versions)}."
                )
            catalyst_version = minimum_versions[("ios", "maccatalyst", ("arm64", "x86_64"))]
            return ios_versions.pop(), catalyst_version
    except zipfile.BadZipFile as error:
        raise ValueError(f"Apple native input {path} is not a valid XCFramework ZIP archive.") from error


def metadata_files(package_id, nuspec_name):
    return {
        "[Content_Types].xml": b"""<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="nuspec" ContentType="application/octet-stream" />
  <Default Extension="dll" ContentType="application/octet-stream" />
  <Default Extension="xml" ContentType="text/xml" />
  <Default Extension="so" ContentType="application/octet-stream" />
  <Default Extension="dylib" ContentType="application/octet-stream" />
  <Default Extension="aar" ContentType="application/octet-stream" />
  <Default Extension="zip" ContentType="application/octet-stream" />
  <Default Extension="pdb" ContentType="application/octet-stream" />
  <Default Extension="h" ContentType="text/plain" />
  <Default Extension="txt" ContentType="text/plain" />
  <Default Extension="md" ContentType="text/markdown" />
  <Default Extension="png" ContentType="image/png" />
</Types>
""",
        "_rels/.rels": (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            f'  <Relationship Type="http://schemas.microsoft.com/packaging/2010/07/manifest" '
            f'Target="/{xml(nuspec_name)}" Id="manifest" />\n'
            "</Relationships>\n"
        ).encode(),
    }


def dependency_groups(kind, version):
    if kind in {"managed", "symbols"}:
        groups = """    <dependencies>
      <group targetFramework="netstandard2.0">
        <dependency id="System.Memory" version="[4.5.5]" />
      </group>
      <group targetFramework="net8.0">
        <dependency id="System.Memory" version="[4.5.5]" />
        <dependency id="System.Numerics.Tensors" version="[9.0.0]" />
      </group>
"""
        for framework in ("net9.0-android35.0", "net9.0-ios18.0", "net9.0-maccatalyst18.0"):
            groups += f"""      <group targetFramework="{framework}">
        <dependency id="System.Memory" version="[4.5.5]" />
        <dependency id="System.Numerics.Tensors" version="[9.0.0]" />
      </group>
"""
        return (
            groups
            + """    </dependencies>
"""
        )
    if kind == "android":
        return f"""    <dependencies>
      <group targetFramework="net9.0-android35.0">
        <dependency id="{MANAGED_ID}" version="[{xml(version)}]" />
      </group>
    </dependencies>
"""
    if kind == "apple":
        return f"""    <dependencies>
      <group targetFramework="net9.0-ios18.0">
        <dependency id="{MANAGED_ID}" version="[{xml(version)}]" />
      </group>
      <group targetFramework="net9.0-maccatalyst18.0">
        <dependency id="{MANAGED_ID}" version="[{xml(version)}]" />
      </group>
    </dependencies>
"""
    return f"""    <dependencies>
      <group targetFramework="netstandard2.0">
        <dependency id="{MANAGED_ID}" version="[{xml(version)}]" />
      </group>
      <group targetFramework="net8.0">
        <dependency id="{MANAGED_ID}" version="[{xml(version)}]" />
      </group>
    </dependencies>
"""


def package_description(kind, rid, flavor):
    descriptions = {
        "android": "Source-built ONNX Runtime Android ARM64 native runtime for API 27 or later, packaged by Bazel.",
        "apple": (
            "Source-built ONNX Runtime static frameworks for iOS ARM64, iOS Simulator ARM64/x86_64, "
            "and Mac Catalyst ARM64/x86_64, packaged by Bazel."
        ),
        "managed": "Source-built ONNX Runtime managed bindings compiled directly with Bazel rules_dotnet.",
        "symbols": "Portable symbols for source-built ONNX Runtime managed bindings.",
    }
    return descriptions.get(
        kind,
        f"Source-built ONNX Runtime {flavor.upper()} native runtime for {rid}; install only on that RID.",
    )


def nuspec(package_id, version, kind, rid, flavor, release_notes):
    description = package_description(kind, rid, flavor)
    dependencies = dependency_groups(kind, version)
    release_notes_xml = f"    <releaseNotes>{xml(release_notes)}</releaseNotes>\n" if release_notes else ""
    package_files = (
        """    <license type="file">LICENSE</license>
    <readme>README.md</readme>
    <icon>ORT_icon_for_light_bg.png</icon>
"""
        if kind != "symbols"
        else ""
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">
  <metadata>
    <id>{xml(package_id)}</id>
    <version>{xml(version)}</version>
    <authors>Microsoft</authors>
    <owners>Microsoft</owners>
    <copyright>Microsoft Corporation. All rights reserved.</copyright>
    <description>{xml(description)}</description>
    <tags>ONNX ONNXRuntime Machine Learning Bazel</tags>
    <projectUrl>https://github.com/Microsoft/onnxruntime</projectUrl>
    <repository type="git" url="https://github.com/Microsoft/onnxruntime.git" />
{package_files}{release_notes_xml}{dependencies}  </metadata>
</package>
""".encode()


def package(args):
    version = Path(args.version_file).read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"VERSION_NUMBER input {args.version_file} is empty.")
    if args.kind == "native" and (not args.rid or args.rid == "unsupported"):
        raise ValueError(f"Native NuGet packaging requires a supported RID, got {args.rid!r}.")
    package_ids = {
        "android": "Microsoft.ML.OnnxRuntime.Bazel.Android.Arm64",
        "apple": "Microsoft.ML.OnnxRuntime.Bazel.Apple",
        "managed": MANAGED_ID,
        "symbols": MANAGED_ID,
    }
    package_id = package_ids.get(args.kind, f"Microsoft.ML.OnnxRuntime.Bazel.{args.flavor.capitalize()}.{args.rid}")
    spec_name = package_id + ".nuspec"
    entries = metadata_files(package_id, spec_name)
    release_notes = ""
    docs_by_name = {
        "LICENSE": "LICENSE",
        "ThirdPartyNotices.txt": "ThirdPartyNotices.txt",
        "Privacy.md": "Privacy.md",
        "nupkg.README.md": "README.md",
        "ORT_icon_for_light_bg.png": "ORT_icon_for_light_bg.png",
    }
    if args.kind != "symbols":
        for item in map(Path, args.doc):
            destination = docs_by_name.get(item.name)
            if destination:
                entries[destination] = item.read_bytes()
        missing_docs = {
            "LICENSE",
            "ThirdPartyNotices.txt",
            "Privacy.md",
            "README.md",
            "ORT_icon_for_light_bg.png",
        }.difference(entries)
        if missing_docs:
            raise ValueError(f"NuGet package metadata inputs are incomplete: missing {sorted(missing_docs)}.")
    if args.kind == "managed":
        required = {"Microsoft.ML.OnnxRuntime.dll", "Microsoft.ML.OnnxRuntime.xml"}
        for framework, attribute in MANAGED_FRAMEWORKS.items():
            items = getattr(args, attribute)
            managed = {Path(item).name: Path(item) for item in items}
            if set(managed) != required:
                raise ValueError(
                    f"Managed package requires exactly {sorted(required)} for {framework}, got {sorted(managed)}."
                )
            for name, source in managed.items():
                entries[f"lib/{framework}/{name}"] = source.read_bytes()
    elif args.kind == "symbols":
        required = {"Microsoft.ML.OnnxRuntime.pdb"}
        for framework, attribute in SYMBOL_FRAMEWORKS.items():
            items = getattr(args, attribute)
            symbols = {Path(item).name: Path(item) for item in items}
            if set(symbols) != required:
                raise ValueError(
                    f"Symbols package requires exactly {sorted(required)} for {framework}, got {sorted(symbols)}."
                )
            pdb = symbols["Microsoft.ML.OnnxRuntime.pdb"]
            if not pdb.read_bytes().startswith(b"BSJB"):
                raise ValueError(f"Symbol input {pdb} is not a portable PDB (missing BSJB metadata signature).")
            entries[f"lib/{framework}/Microsoft.ML.OnnxRuntime.pdb"] = pdb.read_bytes()
    elif args.kind == "native":
        native = {Path(item).name: Path(item) for item in args.native}
        required = (
            {"onnxruntime.dll"}
            if args.rid.startswith("win-")
            else ({"libonnxruntime.dylib"} if args.rid.startswith("osx-") else {"libonnxruntime.so"})
        )
        if not required.issubset(native):
            raise ValueError(
                f"Native package for {args.rid} is missing runtime {sorted(required)}; got {sorted(native)}."
            )
        if args.flavor == "cuda" and "libonnxruntime_providers_cuda.so" not in native:
            raise ValueError("CUDA NuGet packaging requires the explicit source-built CUDA provider library.")
        for name, source in native.items():
            entries[f"runtimes/{args.rid}/native/{name}"] = source.read_bytes()
        for source in map(Path, args.header):
            marker = "include/onnxruntime/"
            normalized = source.as_posix()
            index = normalized.find(marker)
            if index < 0:
                raise ValueError(f"Public header {source} is outside include/onnxruntime.")
            entries["build/native/" + normalized[index:]] = source.read_bytes()
        entries["build/native/Microsoft.ML.OnnxRuntime.Bazel.props"] = (
            f"<Project><PropertyGroup><OnnxRuntimeBazelRid>{xml(args.rid)}</OnnxRuntimeBazelRid>"
            f"<OnnxRuntimeBazelFlavor>{xml(args.flavor)}</OnnxRuntimeBazelFlavor></PropertyGroup></Project>\n"
        ).encode()
        entries["build/native/Microsoft.ML.OnnxRuntime.Bazel.targets"] = (
            b'<Project><Target Name="ValidateOnnxRuntimeBazelRid" BeforeTargets="Build" '
            b"Condition=\"'$(RuntimeIdentifier)' != '' and '$(RuntimeIdentifier)' != '$(OnnxRuntimeBazelRid)'\">"
            b'<Error Text="This ONNX Runtime Bazel package contains native assets only for '
            b'$(OnnxRuntimeBazelRid); set RuntimeIdentifier accordingly or install the matching RID package." />'
            b"</Target></Project>\n"
        )
    elif args.kind == "android":
        mobile_native = list(map(Path, args.mobile_native))
        if len(mobile_native) != 1 or mobile_native[0].suffix != ".aar":
            raise ValueError(f"Android NuGet package requires exactly one source-built AAR, got {mobile_native}.")
        validate_android_archive(mobile_native[0])
        release_notes = "Android API 27 or later; arm64-v8a native runtime."
        targets = list(map(Path, args.mobile_targets))
        if len(targets) != 1:
            raise ValueError(f"Android NuGet package requires exactly one targets file, got {targets}.")
        target_bytes = targets[0].read_bytes()
        asset = mobile_native[0]
        entries[f"runtimes/android/native/{asset.name}"] = asset.read_bytes()
        target_name = "Microsoft.ML.OnnxRuntime.Bazel.Android.targets"
        for root in ("build", "buildTransitive"):
            entries[f"{root}/net9.0-android35.0/{target_name}"] = target_bytes
    else:
        mobile_native = list(map(Path, args.mobile_native))
        if len(mobile_native) != 1 or mobile_native[0].suffix != ".zip":
            raise ValueError(
                f"Apple NuGet package requires exactly one source-built XCFramework zip, got {mobile_native}."
            )
        ios_minimum, catalyst_minimum = validate_apple_xcframework(mobile_native[0])
        release_notes = (
            f"iOS {ios_minimum} or later (device arm64; simulator arm64/x86_64); "
            f"Mac Catalyst {catalyst_minimum} or later (arm64/x86_64)."
        )
        targets = list(map(Path, args.mobile_targets))
        if len(targets) != 1:
            raise ValueError(f"Apple NuGet package requires exactly one NativeReference targets file, got {targets}.")
        target_bytes = targets[0].read_bytes()
        asset = mobile_native[0]
        entries[f"runtimes/apple/native/{asset.name}"] = asset.read_bytes()
        for framework in ("net9.0-ios18.0", "net9.0-maccatalyst18.0"):
            target_name = "Microsoft.ML.OnnxRuntime.Bazel.Apple.targets"
            for root in ("build", "buildTransitive"):
                entries[f"{root}/{framework}/{target_name}"] = target_bytes
    entries[spec_name] = nuspec(package_id, version, args.kind, args.rid, args.flavor, release_notes)
    if args.kind != "symbols":
        scope = package_description(args.kind, args.rid, args.flavor)
        support = f"\n\n**Packaged platform support:** {release_notes}" if release_notes else ""
        upstream_readme = entries["README.md"].decode()
        entries["README.md"] = f"# {package_id}\n\n{scope}{support}\n\n{upstream_readme}".encode()
    write_zip(args.output, entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=("android", "apple", "managed", "native", "symbols"), required=True)
    parser.add_argument("--version-file", required=True)
    parser.add_argument("--rid", default="")
    parser.add_argument("--flavor", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--managed", action="append", default=[])
    parser.add_argument("--managed-netstandard20", action="append", default=[])
    parser.add_argument("--managed-android", action="append", default=[])
    parser.add_argument("--managed-ios", action="append", default=[])
    parser.add_argument("--managed-maccatalyst", action="append", default=[])
    parser.add_argument("--native", action="append", default=[])
    parser.add_argument("--mobile-native", action="append", default=[])
    parser.add_argument("--mobile-targets", action="append", default=[])
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--symbol-netstandard20", action="append", default=[])
    parser.add_argument("--symbol-android", action="append", default=[])
    parser.add_argument("--symbol-ios", action="append", default=[])
    parser.add_argument("--symbol-maccatalyst", action="append", default=[])
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--doc", action="append", default=[])
    package(parser.parse_args())


if __name__ == "__main__":
    main()
