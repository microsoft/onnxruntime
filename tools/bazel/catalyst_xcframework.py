# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Add a Catalyst slice only after proving the actual archive's Mach-O platform."""

import argparse
import plistlib
import struct
import zipfile
from pathlib import Path, PurePosixPath

ARCHITECTURES = {0x01000007: "x86_64", 0x0100000C: "arm64"}
CATALYST_PLATFORM = 6
MINIMUM_OS = (15, 1, 0)


def _records(data, depth=0):
    if depth > 16:
        raise ValueError("Nested native archive exceeds 16 levels; supply a regular non-LTO static archive.")
    if data.startswith(b"!<arch>\n"):
        offset = 8
        records = []
        while offset < len(data):
            header = data[offset : offset + 60]
            if len(header) != 60 or header[58:] != b"`\n":
                raise ValueError("Malformed static archive member header; rebuild the Catalyst library.")
            name = header[:16].decode("ascii").strip()
            size = int(header[48:58].decode("ascii").strip())
            start = offset + 60
            end = start + size
            if size < 0 or end > len(data):
                raise ValueError("Truncated static archive member; rebuild the Catalyst library.")
            payload = data[start:end]
            if name.startswith("#1/"):
                name_size = int(name[3:])
                if name_size < 0 or name_size > len(payload):
                    raise ValueError("Invalid BSD archive name length; rebuild the Catalyst library.")
                name = payload[:name_size].rstrip(b"\0").decode("utf-8")
                payload = payload[name_size:]
            if not name.startswith("__.SYMDEF") and name not in ("/", "//", "/SYM64/"):
                records.extend(_records(payload, depth + 1))
            offset = end + (size & 1)
        return records
    fat = {
        b"\xca\xfe\xba\xbe": (">", False),
        b"\xbe\xba\xfe\xca": ("<", False),
        b"\xca\xfe\xba\xbf": (">", True),
        b"\xbf\xba\xfe\xca": ("<", True),
    }.get(data[:4])
    if fat:
        endian, wide = fat
        if len(data) < 8:
            raise ValueError("Truncated universal archive header; rebuild the Catalyst library.")
        count = struct.unpack_from(endian + "I", data, 4)[0]
        entry_size = 32 if wide else 20
        if not count or count > (len(data) - 8) // entry_size:
            raise ValueError("Invalid universal archive architecture count; rebuild the Catalyst library.")
        records = []
        seen = set()
        for index in range(count):
            entry = 8 + index * entry_size
            cpu = struct.unpack_from(endian + "I", data, entry)[0]
            start, size = struct.unpack_from(endian + ("QQ" if wide else "II"), data, entry + 8)
            if cpu in seen or start < 8 + count * entry_size or not size or start + size > len(data):
                raise ValueError("Invalid universal archive slice; rebuild distinct complete Catalyst architectures.")
            seen.add(cpu)
            members = _records(data[start : start + size], depth + 1)
            if not members or any(member[0] != cpu for member in members):
                raise ValueError("Universal archive CPU disagrees with its objects; rebuild the Catalyst library.")
            records.extend(members)
        return records
    endian = {b"\xcf\xfa\xed\xfe": "<", b"\xfe\xed\xfa\xcf": ">"}.get(data[:4])
    if not endian or len(data) < 32:
        raise ValueError("Expected 64-bit Mach-O objects, not relabeled macOS data or bitcode; rebuild without LTO.")
    cpu, _, filetype, count, size = struct.unpack_from(endian + "IIIII", data, 4)
    if filetype != 1 or 32 + size > len(data) or count > size // 8:
        raise ValueError("Invalid Mach-O object load commands; supply a complete static library.")
    offset = 32
    platform = None
    minimum = None
    for _ in range(count):
        if offset + 8 > 32 + size:
            raise ValueError("Missing Mach-O load command; rebuild the Catalyst library.")
        command, length = struct.unpack_from(endian + "II", data, offset)
        if length < 8 or offset + length > 32 + size:
            raise ValueError("Truncated Mach-O load command; rebuild the Catalyst library.")
        if command == 0x32:
            if length < 24 or platform is not None:
                raise ValueError("Invalid LC_BUILD_VERSION; rebuild the Catalyst library.")
            platform, minimum = struct.unpack_from(endian + "II", data, offset + 8)
        elif command in (0x24, 0x25, 0x2F, 0x30):
            raise ValueError("Legacy non-Catalyst deployment load command; rebuild every object for macabi.")
        offset += length
    if offset != 32 + size:
        raise ValueError("Mach-O load-command sizes disagree; rebuild the Catalyst library.")
    return [(cpu, platform, minimum)]


def verify_catalyst(data):
    records = _records(data)
    if {cpu for cpu, _, _ in records} != set(ARCHITECTURES):
        raise ValueError("Catalyst requires genuine arm64 and x86_64 objects; rebuild both architecture splits.")
    minimum = (MINIMUM_OS[0] << 16) | (MINIMUM_OS[1] << 8) | MINIMUM_OS[2]
    for cpu, architecture in ARCHITECTURES.items():
        versions = set()
        for object_cpu, platform, version in records:
            if object_cpu != cpu or platform is None:
                continue
            if platform != CATALYST_PLATFORM:
                raise ValueError(f"{architecture} contains platform {platform}, not Catalyst (6); rebuild for macabi.")
            versions.add(version)
        if versions != {minimum}:
            raise ValueError(
                f"{architecture} lacks consistent Catalyst 15.1 LC_BUILD_VERSION proof; rebuild all sources."
            )
    return "15.1"


def _read_archive(path):
    with zipfile.ZipFile(path) as archive:
        entries = {}
        modes = {}
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in entry.filename:
                raise ValueError(f"Unsafe XCFramework entry {entry.filename!r}; supply a normal Bazel archive.")
            if not entry.is_dir():
                if entry.filename in entries:
                    raise ValueError(f"Duplicate XCFramework entry {entry.filename}; rebuild the archive.")
                entries[entry.filename] = archive.read(entry)
                modes[entry.filename] = entry.external_attr
    roots = [name for name in entries if name.endswith(".xcframework/Info.plist") and name.count("/") == 1]
    if len(roots) != 1:
        raise ValueError("Expected one XCFramework root Info.plist; supply a Bazel XCFramework ZIP.")
    root = roots[0].removesuffix("Info.plist")
    return entries, modes, root, plistlib.loads(entries[roots[0]])


def merge(base, catalyst, output):
    entries, modes, root, info = _read_archive(base)
    source, source_modes, source_root, source_info = _read_archive(catalyst)
    libraries = source_info.get("AvailableLibraries", [])
    if len(libraries) != 1:
        raise ValueError("Catalyst input must contain exactly one universal library; use the dedicated native target.")
    library = libraries[0]
    if library.get("SupportedPlatform") != "macos" or set(library.get("SupportedArchitectures", [])) != {
        "arm64",
        "x86_64",
    }:
        raise ValueError("Expected the MacOSX-SDK arm64/x86_64 build; supply the dedicated Catalyst target.")
    framework = library["LibraryPath"]
    identifier = library["LibraryIdentifier"]
    for name in (framework, identifier):
        if "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("Invalid XCFramework library path; rebuild the dedicated Catalyst target.")
    prefix = source_root + identifier + "/"
    framework_prefix = prefix + framework + "/"
    plists = [name for name in source if name.startswith(framework_prefix) and name.endswith("/Info.plist")]
    if len(plists) != 1:
        raise ValueError("Catalyst framework requires one real Info.plist; rebuild its framework metadata.")
    framework_info = plistlib.loads(source[plists[0]])
    executable = framework_info.get("CFBundleExecutable")
    if not isinstance(executable, str) or "/" in executable or "\\" in executable:
        raise ValueError("Catalyst Info.plist has no valid CFBundleExecutable; rebuild its framework metadata.")
    binary = framework_prefix + executable
    if binary not in source:
        raise ValueError("Catalyst framework binary is missing; rebuild the dedicated native target.")
    minimum = verify_catalyst(source[binary])
    framework_info["MinimumOSVersion"] = minimum
    framework_info.pop("LSMinimumSystemVersion", None)
    source[plists[0]] = plistlib.dumps(framework_info, sort_keys=True)
    source[framework_prefix + "Info.plist"] = source[plists[0]]
    destination_identifier = "ios-arm64_x86_64-maccatalyst"
    destination = root + destination_identifier + "/"
    if any(entry.get("SupportedPlatformVariant") == "maccatalyst" for entry in info["AvailableLibraries"]):
        raise ValueError("Base XCFramework already contains Catalyst; merge only the base iOS/macOS archive.")
    for name, payload in source.items():
        if name.startswith(prefix):
            target = destination + name.removeprefix(prefix)
            if target in entries:
                raise ValueError(
                    f"XCFramework merge would overwrite {target}; rebuild distinct base and Catalyst inputs."
                )
            entries[target] = payload
            modes[target] = source_modes.get(name, 0o100644 << 16)
    info["AvailableLibraries"].append(
        {
            "LibraryIdentifier": destination_identifier,
            "LibraryPath": framework,
            "SupportedArchitectures": ["arm64", "x86_64"],
            "SupportedPlatform": "ios",
            "SupportedPlatformVariant": "maccatalyst",
        }
    )
    entries[root + "Info.plist"] = plistlib.dumps(info, sort_keys=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(entries.items()):
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.external_attr = modes.get(name, 0o100644 << 16)
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--catalyst", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merge(args.base, args.catalyst, args.output)


if __name__ == "__main__":
    main()
