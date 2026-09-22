# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Explicit migration for Dawn's switch to DXC's internal validator."""

import re
import subprocess

AZURE = "tools/ci_build/github/azure-pipelines/"
FILES = [
    ".github/workflows/windows_webgpu.yml",
    "cgmanifests/webgpu/cgmanifest.webgpu.json",
    "cmake/external/onnxruntime_external_deps.cmake",
    "cmake/onnxruntime_java.cmake",
    "cmake/onnxruntime_nodejs.cmake",
    "cmake/onnxruntime_providers_webgpu.cmake",
    "cmake/onnxruntime_python.cmake",
    "java/src/main/java/ai/onnxruntime/OnnxRuntime.java",
    "onnxruntime/test/webgpu/delay_load/main.cc",
    "plugin-ep-webgpu/README.md",
    "plugin-ep-webgpu/csharp/pack_nuget.py",
    "plugin-ep-webgpu/python/build_wheel.py",
    "setup.py",
    AZURE + "stages/nodejs-npm-packaging-stage.yml",
    AZURE + "stages/nodejs-win-packaging-stage.yml",
    AZURE + "stages/plugin-win-webgpu-stage.yml",
    AZURE + "templates/c-api-artifacts-package-and-publish-steps-windows.yml",
]
TOKEN = re.compile(r"dxil\.dll|copy_dxil_dll|Microsoft::DXIL|ONNXRUNTIME_LIBRARY_WEBGPU_DXC_DXIL_NAME")


def remove_dxil(path: str, text: str) -> str:
    """Edit known ORT packaging forms; reject unfamiliar occurrences for review."""
    if path.endswith(".py"):
        text = text.replace('"dxil.dll", ', "").replace(', "dxil.dll"', "")
    text = text.replace("`dxil.dll`, ", "").replace("dxil.dll and dxcompiler.dll", "dxcompiler.dll")
    text = text.replace("# Ensure dxcompiler.dll exist in", "# Ensure dxcompiler.dll exists in")
    text = text.replace("DXC DLLs successfully copied", "DXC DLL successfully copied")
    text = re.sub(
        r"  /\*\* The short name of the WebGPU DXC library \"dxil\.dll\" \*/\n"
        r"  static final String ONNXRUNTIME_LIBRARY_WEBGPU_DXC_DXIL_NAME = \"dxil\";\n\n",
        "",
        text,
    )
    text = text.replace(
        "because dxcompiler.dll ends up in the default output path instead of CMAKE_BINARY_DIR/$<CONFIG>,\n"
        "          #   and the copy_dxil_dll target copies dxil.dll to a different location.",
        "because dxcompiler.dll ends up in the default output path instead of CMAKE_BINARY_DIR/$<CONFIG>.",
    )
    result = []
    for line in text.splitlines(keepends=True):
        if TOKEN.search(line):
            stripped = line.strip()
            # These are single list items, copy statements, or an entire
            # dependency statement. No arbitrary source line is discarded.
            allowed = (
                stripped.startswith(
                    (
                        "#",
                        "//",
                        'fs::copy_file(L"dxil.dll"',
                        'Copy-Item -Path "$sourcePath\\dxil.dll"',
                        "copy $(Build.BinariesDirectory)",
                    )
                )
                or stripped
                in (
                    '"dxil.dll",',
                    "dxil.dll",
                    "$<TARGET_FILE:Microsoft::DXIL>",
                    "$<TARGET_FILE_DIR:dxcompiler>/dxil.dll",
                )
                or re.fullmatch(r"\$\{\{ github.workspace \}\}/[\w/]+/dxil\.dll", stripped)
                or re.fullmatch(r'list\(APPEND \w+ "\$<TARGET_FILE:Microsoft::DXIL>"\)', stripped)
                or re.fullmatch(r'list\(APPEND \w+ "\$<TARGET_FILE_DIR:dxcompiler>/dxil.dll"\)', stripped)
                or re.fullmatch(r"add_dependencies\(\w+ copy_dxil_dll\)", stripped)
                or re.fullmatch(r"target_link_libraries\(\w+ Microsoft::DXIL\)", stripped)
                or stripped == "extractProviderLibrary(ONNXRUNTIME_LIBRARY_WEBGPU_DXC_DXIL_NAME);"
            )
            if not allowed:
                raise ValueError(f"unrecognized DXIL reference in {path}: {stripped}")
            continue
        result.append(line)
    return "".join(result)


def plan_migration(plan, source, enabled: bool) -> None:
    native_cmake = source.read("src/dawn/native/CMakeLists.txt")
    third_party = source.read("third_party/CMakeLists.txt")
    if "copy_dxil_dll" in native_cmake + third_party:
        if enabled:
            plan.report["blockers"].append("--drop-dxil is incompatible with Dawn's copy_dxil_dll dependency")
        return
    affected = [p for p in FILES if TOKEN.search(plan.read(p))]
    plan.report["dxilMigration"] = {"affectedFiles": affected, "requested": enabled}
    if not affected:
        return
    if not enabled:
        plan.report["blockers"].append(
            "Dawn removed copy_dxil_dll. Review DXC compatibility, then rerun with --drop-dxil or adapt packaging manually."
        )
        return
    loader = source.read("src/dawn/native/d3d12/PlatformFunctionsD3D12.cpp")
    backend = source.read("src/dawn/native/d3d12/BackendD3D12.cpp")
    if (
        "dxil.dll" in loader
        or "dxcompiler.dll" not in loader
        or not re.search(r"dxcCreateInstance\(\s*CLSID_DxcValidator", backend)
    ):
        plan.report["blockers"].append(
            "Dawn's internal DXC validator contract is not recognized; review DXIL migration"
        )
        return
    # DXC #6853 enabled internal-validator hashing before this release. Other
    # redistributed versions need their own validation before applying this recipe.
    for path in [AZURE + "stages/plugin-win-webgpu-stage.yml", AZURE + "stages/nodejs-win-packaging-stage.yml"]:
        releases = re.findall(
            r"\$dxcZipUrl = \"https://github.com/microsoft/DirectXShaderCompiler/releases/download/([^/]+)/",
            plan.read(path),
        )
        if releases != ["v1.8.2502"]:
            plan.report["blockers"].append(f"DXIL migration has not been verified for the redistributed DXC in {path}")
            return
    for path in affected:
        plan.edit(path, remove_dxil(path, plan.read(path)))
    found = subprocess.run(
        ["git", "grep", "-l", "-I", "-F", "-e", "dxil.dll", "-e", "copy_dxil_dll", "-e", "Microsoft::DXIL", "--", "."],
        cwd=plan.root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if found.returncode not in (0, 1):
        raise ValueError(f"could not audit remaining DXIL references: {found.stderr}")
    remaining = [
        path
        for path in found.stdout.splitlines()
        if path not in FILES and not path.startswith(("plugin-ep-webgpu/tools/dawn/", "cmake/vcpkg-ports/dawn/"))
    ]
    if remaining:
        plan.report["blockers"].append(f"DXIL references outside known migration paths need review: {remaining}")
    plan.report["dxilMigration"]["status"] = "prepared"
    plan.report["notes"].append(
        "DXIL migration requires package load/shader compilation tests; source checks do not replace them."
    )
