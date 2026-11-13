# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Version-driven managed and honest per-RID native NuGet release artifacts."""

load("@rules_dotnet//dotnet:defs.bzl", "csharp_library", "csharp_test")
load("@rules_dotnet//dotnet/private:providers.bzl", "DotnetAssemblyRuntimeInfo")
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

def _assembly_impl(ctx):
    runtime_info = ctx.attr.library[DotnetAssemblyRuntimeInfo]
    files = [
        file
        for file in runtime_info.libs + runtime_info.pdbs + runtime_info.xml_docs
        if file.basename in [
            "Microsoft.ML.OnnxRuntime.dll",
            "Microsoft.ML.OnnxRuntime.pdb",
            "Microsoft.ML.OnnxRuntime.xml",
        ]
    ]
    basenames = {file.basename: True for file in files}
    for required in [
        "Microsoft.ML.OnnxRuntime.dll",
        "Microsoft.ML.OnnxRuntime.pdb",
        "Microsoft.ML.OnnxRuntime.xml",
    ]:
        if required not in basenames:
            fail("Expected %s from %s; inspect its rules_dotnet outputs before packaging." % (required, ctx.attr.library.label))
    by_name = {file.basename: file for file in files}
    return [
        DefaultInfo(files = depset(files)),
        OutputGroupInfo(
            dll = depset([by_name["Microsoft.ML.OnnxRuntime.dll"]]),
            pdb = depset([by_name["Microsoft.ML.OnnxRuntime.pdb"]]),
            xml = depset([by_name["Microsoft.ML.OnnxRuntime.xml"]]),
        ),
    ]

_assembly = rule(
    implementation = _assembly_impl,
    attrs = {"library": attr.label(providers = [DotnetAssemblyRuntimeInfo], mandatory = True)},
)

def _platform_attributes_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".cs")
    ctx.actions.write(
        output,
        """using System.Runtime.Versioning;
[assembly: TargetFramework(".NETCoreApp,Version=v9.0", FrameworkDisplayName = ".NET 9.0")]
[assembly: TargetPlatform("%s")]
""" % ctx.attr.platform,
    )
    return [DefaultInfo(files = depset([output]))]

_platform_attributes = rule(
    implementation = _platform_attributes_impl,
    attrs = {"platform": attr.string(mandatory = True)},
)

def _package_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output)
    args = ctx.actions.args()
    args.add("--output", output)
    args.add("--kind", ctx.attr.kind)
    args.add("--version-file", ctx.file.version)
    args.add("--rid", ctx.attr.rid)
    args.add("--flavor", ctx.attr.flavor)
    args.add_all(ctx.files.managed, before_each = "--managed")
    args.add_all(ctx.files.managed_netstandard20, before_each = "--managed-netstandard20")
    args.add_all(ctx.files.managed_android, before_each = "--managed-android")
    args.add_all(ctx.files.managed_ios, before_each = "--managed-ios")
    args.add_all(ctx.files.managed_maccatalyst, before_each = "--managed-maccatalyst")
    args.add_all(ctx.files.native, before_each = "--native")
    args.add_all(ctx.files.mobile_native, before_each = "--mobile-native")
    args.add_all(ctx.files.mobile_targets, before_each = "--mobile-targets")
    args.add_all(ctx.files.symbols, before_each = "--symbol")
    args.add_all(ctx.files.symbols_netstandard20, before_each = "--symbol-netstandard20")
    args.add_all(ctx.files.symbols_android, before_each = "--symbol-android")
    args.add_all(ctx.files.symbols_ios, before_each = "--symbol-ios")
    args.add_all(ctx.files.symbols_maccatalyst, before_each = "--symbol-maccatalyst")
    args.add_all(ctx.files.headers, before_each = "--header")
    args.add_all(ctx.files.docs, before_each = "--doc")
    inputs = (
        [ctx.file.version] +
        ctx.files.managed +
        ctx.files.managed_netstandard20 +
        ctx.files.managed_android +
        ctx.files.managed_ios +
        ctx.files.managed_maccatalyst +
        ctx.files.native +
        ctx.files.mobile_native +
        ctx.files.mobile_targets +
        ctx.files.symbols +
        ctx.files.symbols_netstandard20 +
        ctx.files.symbols_android +
        ctx.files.symbols_ios +
        ctx.files.symbols_maccatalyst +
        ctx.files.headers +
        ctx.files.docs
    )
    ctx.actions.run(
        executable = ctx.executable.packager,
        arguments = [args],
        inputs = inputs,
        outputs = [output],
        mnemonic = "NuGetPackage",
    )
    return [DefaultInfo(files = depset([output]))]

_package = rule(
    implementation = _package_impl,
    attrs = {
        "output": attr.string(mandatory = True),
        "kind": attr.string(values = ["android", "apple", "managed", "native", "symbols"], mandatory = True),
        "version": attr.label(allow_single_file = True, mandatory = True),
        "rid": attr.string(default = ""),
        "flavor": attr.string(values = ["cpu", "cuda"], default = "cpu"),
        "managed": attr.label_list(allow_files = True),
        "managed_netstandard20": attr.label_list(allow_files = True),
        "managed_android": attr.label_list(allow_files = True),
        "managed_ios": attr.label_list(allow_files = True),
        "managed_maccatalyst": attr.label_list(allow_files = True),
        "native": attr.label_list(allow_files = True),
        "mobile_native": attr.label_list(allow_files = True),
        "mobile_targets": attr.label_list(allow_files = True),
        "symbols": attr.label_list(allow_files = True),
        "symbols_netstandard20": attr.label_list(allow_files = True),
        "symbols_android": attr.label_list(allow_files = True),
        "symbols_ios": attr.label_list(allow_files = True),
        "symbols_maccatalyst": attr.label_list(allow_files = True),
        "headers": attr.label_list(allow_files = True),
        "docs": attr.label_list(allow_files = True),
        "packager": attr.label(executable = True, cfg = "exec", mandatory = True),
    },
)

def csharp_nuget_package():
    """Create a managed package plus one explicitly identified native RID package."""
    native_platforms = select({
        "//bazel:windows": [],
        "//bazel:linux": [],
        "//bazel:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:x86_64": [],
        "//bazel:aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
    android_arm64 = ["@platforms//os:android", "@platforms//cpu:aarch64"]
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
    py_binary(
        name = "nuget_packager",
        srcs = ["tools/bazel/nuget_package.py"],
        main = "tools/bazel/nuget_package.py",
    )
    csharp_library(
        name = "csharp_package_binding_netstandard20",
        srcs = native.glob(["csharp/src/Microsoft.ML.OnnxRuntime/**/*.shared.cs"]),
        allow_unsafe_blocks = True,
        keyfile = "csharp/OnnxRuntime.snk",
        out = "Microsoft.ML.OnnxRuntime",
        target_frameworks = ["netstandard2.0"],
        deps = ["@ort_dotnet_memory//:memory"],
    )
    mobile_sources = native.glob(["csharp/src/Microsoft.ML.OnnxRuntime/**/*.shared.cs"])
    for platform, target_platform, defines, deps in [
        (
            "android",
            "Android35.0",
            ["__MOBILE__", "__ANDROID__"],
            [
                "@ort_dotnet_android_ref//:java_interop",
                "@ort_dotnet_android_ref//:mono_android",
                "@ort_dotnet_android_ref//:mono_android_export",
                "@ort_dotnet_android_ref//:mono_android_runtime",
            ],
        ),
        (
            "ios",
            "iOS18.0",
            ["__MOBILE__", "__IOS__", "__ENABLE_COREML__"],
            ["@ort_dotnet_ios_ref//:ios"],
        ),
        (
            "maccatalyst",
            "MacCatalyst18.0",
            ["__MOBILE__", "__IOS__", "__ENABLE_COREML__"],
            ["@ort_dotnet_maccatalyst_ref//:maccatalyst"],
        ),
    ]:
        attributes = "csharp_package_%s_attributes" % platform
        _platform_attributes(name = attributes, platform = target_platform)
        csharp_library(
            name = "csharp_package_binding_" + platform,
            srcs = mobile_sources + [":" + attributes],
            allow_unsafe_blocks = True,
            defines = defines,
            keyfile = "csharp/OnnxRuntime.snk",
            out = "Microsoft.ML.OnnxRuntime",
            target_frameworks = ["net9.0"],
            deps = [
                "@ort_dotnet_memory//:memory",
                "@ort_dotnet_tensors//:tensors",
            ] + deps,
        )
    _assembly(name = "csharp_package_assembly", library = ":csharp_binding")
    _assembly(
        name = "csharp_package_assembly_netstandard20",
        library = ":csharp_package_binding_netstandard20",
    )
    for platform in ["android", "ios", "maccatalyst"]:
        _assembly(
            name = "csharp_package_assembly_" + platform,
            library = ":csharp_package_binding_" + platform,
        )
    native.filegroup(
        name = "csharp_package_dll",
        srcs = [":csharp_package_assembly"],
        output_group = "dll",
    )
    native.filegroup(
        name = "csharp_package_pdb",
        srcs = [":csharp_package_assembly"],
        output_group = "pdb",
    )
    native.filegroup(
        name = "csharp_package_xml",
        srcs = [":csharp_package_assembly"],
        output_group = "xml",
    )
    native.filegroup(
        name = "csharp_package_dll_netstandard20",
        srcs = [":csharp_package_assembly_netstandard20"],
        output_group = "dll",
    )
    native.filegroup(
        name = "csharp_package_pdb_netstandard20",
        srcs = [":csharp_package_assembly_netstandard20"],
        output_group = "pdb",
    )
    native.filegroup(
        name = "csharp_package_xml_netstandard20",
        srcs = [":csharp_package_assembly_netstandard20"],
        output_group = "xml",
    )
    for platform in ["android", "ios", "maccatalyst"]:
        for output_group in ["dll", "pdb", "xml"]:
            native.filegroup(
                name = "csharp_package_%s_%s" % (output_group, platform),
                srcs = [":csharp_package_assembly_" + platform],
                output_group = output_group,
            )
    docs = [
        "LICENSE",
        "ThirdPartyNotices.txt",
        "docs/Privacy.md",
        "tools/nuget/nupkg.README.md",
        "ORT_icon_for_light_bg.png",
    ]
    _package(
        name = "csharp_nuget_managed",
        docs = docs,
        kind = "managed",
        managed = [
            ":csharp_package_dll",
            ":csharp_package_xml",
        ],
        managed_netstandard20 = [
            ":csharp_package_dll_netstandard20",
            ":csharp_package_xml_netstandard20",
        ],
        managed_android = [
            ":csharp_package_dll_android",
            ":csharp_package_xml_android",
        ],
        managed_ios = [
            ":csharp_package_dll_ios",
            ":csharp_package_xml_ios",
        ],
        managed_maccatalyst = [
            ":csharp_package_dll_maccatalyst",
            ":csharp_package_xml_maccatalyst",
        ],
        output = "Microsoft.ML.OnnxRuntime.Managed.Bazel.nupkg",
        packager = ":nuget_packager",
        version = "VERSION_NUMBER",
    )
    _package(
        name = "csharp_nuget_symbols",
        kind = "symbols",
        output = "Microsoft.ML.OnnxRuntime.Managed.Bazel.snupkg",
        packager = ":nuget_packager",
        symbols = [":csharp_package_pdb"],
        symbols_netstandard20 = [":csharp_package_pdb_netstandard20"],
        symbols_android = [":csharp_package_pdb_android"],
        symbols_ios = [":csharp_package_pdb_ios"],
        symbols_maccatalyst = [":csharp_package_pdb_maccatalyst"],
        version = "VERSION_NUMBER",
    )
    py_test(
        name = "csharp_nuget_symbols_test",
        srcs = ["tools/bazel/nuget_symbols_test.py"],
        main = "tools/bazel/nuget_symbols_test.py",
        args = [
            "$(rootpath :csharp_nuget_managed)",
            "$(rootpath :csharp_nuget_symbols)",
            "$(rootpath VERSION_NUMBER)",
            "$(rootpath :csharp_package_pdb)",
            "$(rootpath :csharp_package_pdb_netstandard20)",
            "$(rootpath :csharp_package_dll)",
            "$(rootpath :csharp_package_xml)",
            "$(rootpath :csharp_package_dll_netstandard20)",
            "$(rootpath :csharp_package_xml_netstandard20)",
            "$(rootpath :csharp_package_pdb_android)",
            "$(rootpath :csharp_package_dll_android)",
            "$(rootpath :csharp_package_xml_android)",
            "$(rootpath :csharp_package_pdb_ios)",
            "$(rootpath :csharp_package_dll_ios)",
            "$(rootpath :csharp_package_xml_ios)",
            "$(rootpath :csharp_package_pdb_maccatalyst)",
            "$(rootpath :csharp_package_dll_maccatalyst)",
            "$(rootpath :csharp_package_xml_maccatalyst)",
        ],
        data = [
            ":csharp_nuget_managed",
            ":csharp_nuget_symbols",
            ":csharp_package_dll",
            ":csharp_package_xml",
            ":csharp_package_pdb",
            ":csharp_package_dll_netstandard20",
            ":csharp_package_xml_netstandard20",
            ":csharp_package_pdb_netstandard20",
            ":csharp_package_pdb_android",
            ":csharp_package_dll_android",
            ":csharp_package_xml_android",
            ":csharp_package_pdb_ios",
            ":csharp_package_dll_ios",
            ":csharp_package_xml_ios",
            ":csharp_package_pdb_maccatalyst",
            ":csharp_package_dll_maccatalyst",
            ":csharp_package_xml_maccatalyst",
            "VERSION_NUMBER",
        ],
    )
    csharp_test(
        name = "csharp_nuget_mobile_metadata_test",
        srcs = ["tools/bazel/NuGetMobileMetadataTest.cs"],
        args = [
            "$(rootpath :csharp_package_dll_android)",
            "$(rootpath :csharp_package_dll_ios)",
            "$(rootpath :csharp_package_dll_maccatalyst)",
        ],
        data = [
            ":csharp_package_dll_android",
            ":csharp_package_dll_ios",
            ":csharp_package_dll_maccatalyst",
        ],
        target_frameworks = ["net10.0"],
    )
    native_files = [":runtime_release_library", ":providers_shared_release_library"] + select({
        "//bazel:cuda_enabled": [":cuda_provider_release_library"],
        "//conditions:default": [],
    })
    _package(
        name = "csharp_nuget",
        docs = docs,
        flavor = select({
            "//bazel:cuda_enabled": "cuda",
            "//conditions:default": "cpu",
        }),
        headers = native.glob(["include/onnxruntime/core/session/*.h"]),
        kind = "native",
        native = native_files,
        output = "Microsoft.ML.OnnxRuntime.Bazel.nupkg",
        packager = ":nuget_packager",
        rid = select(rids),
        target_compatible_with = native_platforms,
        version = "VERSION_NUMBER",
    )
    _package(
        name = "csharp_nuget_android",
        docs = docs,
        kind = "android",
        mobile_native = [":android_aar"],
        mobile_targets = ["csharp/src/Microsoft.ML.OnnxRuntime/targets/net9.0-android/targets.xml"],
        output = "Microsoft.ML.OnnxRuntime.Bazel.Android.Arm64.nupkg",
        packager = ":nuget_packager",
        target_compatible_with = android_arm64,
        version = "VERSION_NUMBER",
    )
    py_test(
        name = "csharp_nuget_android_test",
        srcs = ["tools/bazel/nuget_mobile_native_test.py"],
        main = "tools/bazel/nuget_mobile_native_test.py",
        args = [
            "android",
            "$(rootpath :csharp_nuget_android)",
            "$(rootpath VERSION_NUMBER)",
            "$(rootpath :android_aar)",
        ],
        data = [
            ":android_aar",
            ":csharp_nuget_android",
            "VERSION_NUMBER",
        ],
        target_compatible_with = android_arm64,
    )
    _package(
        name = "csharp_nuget_apple",
        docs = docs,
        kind = "apple",
        mobile_native = ["//objectivec:onnxruntime_objc_xcframework.xcframework.zip"],
        mobile_targets = ["tools/bazel/Microsoft.ML.OnnxRuntime.Bazel.Apple.targets"],
        output = "Microsoft.ML.OnnxRuntime.Bazel.Apple.nupkg",
        packager = ":nuget_packager",
        target_compatible_with = ["@platforms//os:macos"],
        version = "VERSION_NUMBER",
    )
    py_test(
        name = "csharp_nuget_apple_test",
        srcs = ["tools/bazel/nuget_mobile_native_test.py"],
        main = "tools/bazel/nuget_mobile_native_test.py",
        args = [
            "apple",
            "$(rootpath :csharp_nuget_apple)",
            "$(rootpath VERSION_NUMBER)",
            "$(rootpath //objectivec:onnxruntime_objc_xcframework.xcframework.zip)",
        ],
        data = [
            "//objectivec:onnxruntime_objc_xcframework.xcframework.zip",
            ":csharp_nuget_apple",
            "VERSION_NUMBER",
        ],
        target_compatible_with = ["@platforms//os:macos"],
    )
    py_test(
        name = "csharp_nuget_test",
        srcs = ["tools/bazel/nuget_package_test.py"],
        main = "tools/bazel/nuget_package_test.py",
        args = [
            "$(rootpath :csharp_nuget_managed)",
            "$(rootpath :csharp_nuget)",
            "$(rootpath VERSION_NUMBER)",
            "$(rootpath :csharp_package_dll)",
            "$(rootpath :csharp_package_xml)",
            "$(rootpath :csharp_package_dll_netstandard20)",
            "$(rootpath :csharp_package_xml_netstandard20)",
            "$(rootpath :csharp_package_dll_android)",
            "$(rootpath :csharp_package_xml_android)",
            "$(rootpath :csharp_package_dll_ios)",
            "$(rootpath :csharp_package_xml_ios)",
            "$(rootpath :csharp_package_dll_maccatalyst)",
            "$(rootpath :csharp_package_xml_maccatalyst)",
            "$(rootpath :runtime_release_library)",
            "$(rootpath :providers_shared_release_library)",
        ] + select({
            "//bazel:cuda_enabled": ["$(rootpath :cuda_provider_release_library)"],
            "//conditions:default": [],
        }) + select({label: [rid] for label, rid in rids.items()}) + select({
            "//bazel:cuda_enabled": ["cuda"],
            "//conditions:default": ["cpu"],
        }),
        data = [
            ":csharp_nuget",
            ":csharp_nuget_managed",
            ":csharp_package_assembly",
            ":csharp_package_dll",
            ":csharp_package_xml",
            ":csharp_package_dll_netstandard20",
            ":csharp_package_xml_netstandard20",
            ":csharp_package_dll_android",
            ":csharp_package_xml_android",
            ":csharp_package_dll_ios",
            ":csharp_package_xml_ios",
            ":csharp_package_dll_maccatalyst",
            ":csharp_package_xml_maccatalyst",
            ":runtime_release_library",
            ":providers_shared_release_library",
            "VERSION_NUMBER",
        ] + select({
            "//bazel:cuda_enabled": [":cuda_provider_release_library"],
            "//conditions:default": [],
        }),
        target_compatible_with = native_platforms,
    )
