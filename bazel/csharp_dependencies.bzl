# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pinned NuGet assemblies for the existing desktop C# tests."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _dependencies_impl(ctx):
    for name, version, framework, dll, sha256 in [
        ("xunit.assert", "2.9.0", "net6.0", "xunit.assert", "dc6519d25a036bfdd60d746e05076086c29713f6d9ac2049e41a29af8dd2b950"),
        ("xunit.extensibility.core", "2.9.0", "netstandard1.1", "xunit.core", "758ba58f863e904b367698c6bca3ee22193782cce5568fb23755d27e3ae33f1c"),
        ("xunit.extensibility.execution", "2.9.0", "netstandard1.1", "xunit.execution.dotnet", "d7044cb02f9919d3f853fb6eac1e2645f3b4ca1eae378ed59f50c91f237f377b"),
        ("xunit.abstractions", "2.0.3", "netstandard2.0", "xunit.abstractions", "d03d72fc2df8880448f7a81bddb00e1bcb5c18f323c7e7cc69b4cfa727469403"),
        ("xunit.skippablefact", "1.4.13", "netstandard2.0", "Xunit.SkippableFact", "a4bb71d3fda84ca60ed58d5583793f1258b63961d3e6ea2b19d069dae5ef15fc"),
        ("google.protobuf", "3.33.6", "net5.0", "Google.Protobuf", "c592a57e56603821eb20098765b5b52152e861d7bf89820f79b1d3426decc790"),
        ("newtonsoft.json", "13.0.3", "net6.0", "Newtonsoft.Json", "872fc189e638ab1056555b03aaa38f68bcb54286e221aa646eb1129babf63c77"),
    ]:
        http_archive(
            name = name.replace(".", "_"),
            urls = [
                "https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-public/nuget/v3/flat2/{0}/{1}/{0}.{1}.nupkg".format(name, version),
                "https://api.nuget.org/v3-flatcontainer/{0}/{1}/{0}.{1}.nupkg".format(name, version),
            ],
            sha256 = sha256,
            type = "zip",
            build_file_content = """
load("@rules_dotnet//dotnet:defs.bzl", "import_dll")
import_dll(
    name = "assembly",
    dll = "lib/%s/%s.dll",
    visibility = ["//visibility:public"],
)
""" % (framework, dll),
        )
    return ctx.extension_metadata(reproducible = True)

csharp_test_dependencies = module_extension(implementation = _dependencies_impl)
