# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NuGet archives containing the directly compiled managed and native runtime."""

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:zip.bzl", "pkg_zip")
load("@rules_python//python:py_test.bzl", "py_test")

def _assembly_impl(ctx):
    files = [file for file in ctx.attr.library[DefaultInfo].files.to_list() if file.basename == "Microsoft.ML.OnnxRuntime.dll"]
    if len(files) != 1:
        fail("Expected one Microsoft.ML.OnnxRuntime.dll from %s; inspect its rules_dotnet outputs before packaging." % ctx.attr.library.label)
    return [DefaultInfo(files = depset(files))]

_assembly = rule(
    implementation = _assembly_impl,
    attrs = {"library": attr.label(mandatory = True)},
)

def csharp_nuget_package():
    """Package one desktop CPU RID without dotnet pack or MSBuild."""
    compatible = select({
        "//bazel:windows": [],
        "//bazel:linux": [],
        "//bazel:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:x86_64": [],
        "//bazel:aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:cuda_enabled": ["@platforms//:incompatible"],
        "//conditions:default": [],
    })
    rids = {}
    for os, rid_os in [("windows", "win"), ("linux", "linux"), ("macos", "osx")]:
        for cpu, rid_cpu in [("x86_64", "x64"), ("aarch64", "arm64")]:
            name = "nuget_" + os + "_" + cpu
            native.config_setting(
                name = name,
                constraint_values = ["@platforms//os:" + os, "@platforms//cpu:" + cpu],
            )
            rids[":" + name] = rid_os + "-" + rid_cpu
    rids["//conditions:default"] = "unsupported"
    metadata = {
        "Microsoft.ML.OnnxRuntime.Bazel.nuspec": [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">',
            "  <metadata>",
            "    <id>Microsoft.ML.OnnxRuntime.Bazel</id>",
            "    <version>1.32.0</version>",
            "    <authors>Microsoft</authors>",
            "    <description>Experimental source-built single-platform CPU ONNX Runtime package.</description>",
            '    <license type="expression">MIT</license>',
            "    <dependencies>",
            '      <group targetFramework="net8.0">',
            '        <dependency id="System.Numerics.Tensors" version="[9.0.0]" />',
            "      </group>",
            "    </dependencies>",
            "  </metadata>",
            "</package>",
        ],
        "[Content_Types].xml": [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />',
            '  <Default Extension="nuspec" ContentType="application/octet-stream" />',
            '  <Default Extension="dll" ContentType="application/octet-stream" />',
            '  <Default Extension="so" ContentType="application/octet-stream" />',
            '  <Default Extension="dylib" ContentType="application/octet-stream" />',
            '  <Default Extension="txt" ContentType="text/plain" />',
            '  <Override PartName="/LICENSE" ContentType="text/plain" />',
            "</Types>",
        ],
        "_rels/.rels": [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
            '  <Relationship Type="http://schemas.microsoft.com/packaging/2010/07/manifest" Target="/Microsoft.ML.OnnxRuntime.Bazel.nuspec" Id="manifest" />',
            "</Relationships>",
        ],
    }
    for index, (filename, content) in enumerate(metadata.items()):
        write_file(
            name = "nuget_metadata_%d" % index,
            out = "nuget_metadata/" + filename,
            content = content,
        )
    _assembly(name = "csharp_package_assembly", library = ":csharp_binding")
    pkg_files(
        name = "nuget_managed",
        srcs = [":csharp_package_assembly"],
        prefix = "lib/net8.0",
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_files(
        name = "nuget_native",
        srcs = [":runtime_library", ":providers_shared_library"],
        prefix = select({label: "runtimes/" + rid + "/native" for label, rid in rids.items()}),
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_files(
        name = "nuget_metadata",
        srcs = [":nuget_metadata_%d" % index for index in range(len(metadata))],
        strip_prefix = strip_prefix.from_pkg("nuget_metadata"),
    )
    pkg_zip(
        name = "csharp_nuget",
        srcs = [":nuget_managed", ":nuget_native", ":nuget_metadata", "LICENSE", "ThirdPartyNotices.txt"],
        out = "Microsoft.ML.OnnxRuntime.Bazel.1.32.0.nupkg",
        target_compatible_with = compatible,
    )
    py_test(
        name = "csharp_nuget_test",
        srcs = ["tools/bazel/nuget_package_test.py"],
        main = "tools/bazel/nuget_package_test.py",
        args = [
            "$(rootpath :csharp_nuget)",
            "$(rootpath VERSION_NUMBER)",
            "$(rootpath :csharp_package_assembly)",
            "$(rootpath :runtime_library)",
            "$(rootpath :providers_shared_library)",
        ] + select({label: [rid] for label, rid in rids.items()}),
        data = [":csharp_nuget", ":csharp_package_assembly", ":runtime_library", ":providers_shared_library", "VERSION_NUMBER"],
        target_compatible_with = compatible,
    )
