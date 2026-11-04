# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Public, checksum-pinned libclang for native Rust binding generation."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _rust_clang_dependencies_impl(module_ctx):
    os = module_ctx.os.name
    arch = module_ctx.os.arch
    if "windows" in os and arch in ["amd64", "x86_64"]:
        archive = ("0b/2d/3f480b1e1d31eb3d6de5e3ef641954e5c67430d5ac93b7fa7e07589576c7", "win_amd64", "4dd2d3b82fab35e2bf9ca717d7b63ac990a3519c7e312f19fa8e86dcc712f7fb", "libclang.dll")
    elif os == "linux" and arch in ["amd64", "x86_64"]:
        archive = ("1d/fc/716c1e62e512ef1c160e7984a73a5fc7df45166f2ff3f254e71c58076f7c", "manylinux2010_x86_64", "c533091d8a3bbf7460a00cb6c1a71da93bffe148f172c7d03b1c31fbf8aa2a0b", "libclang.so")
    elif "mac" in os and arch in ["aarch64", "arm64"]:
        archive = ("db/ed/1df62b44db2583375f6a8a5e2ca5432bbdc3edb477942b9b7c848c720055", "macosx_11_0_arm64", "83ce5045d101b669ac38e6da8e58765f12da2d3aafb3b9b98d88b286a60964d8", "libclang.dylib")
    elif "mac" in os and arch in ["amd64", "x86_64"]:
        archive = ("e2/e5/fc61bbded91a8830ccce94c5294ecd6e88e496cc85f6704bf350c0634b70", "macosx_10_9_x86_64", "6f14c3f194704e5d09769108f03185fce7acaf1d1ae4bbb2f30a72c2400cb7c5", "libclang.dylib")
    else:
        fail("Rust binding generation requires Windows/Linux x86-64 or macOS x86-64/ARM64; unsupported host: %s %s" % (os, arch))
    http_archive(
        name = "ort_rust_libclang",
        urls = ["https://files.pythonhosted.org/packages/%s/libclang-18.1.1-py2.py3-none-%s.whl" % (archive[0], archive[1])],
        sha256 = archive[2],
        type = "zip",
        build_file_content = """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
cc_import(
    name = "libclang",
    shared_library = glob(["**/native/%s"], allow_empty = False)[0],
    visibility = ["//visibility:public"],
)
""" % archive[3],
    )
    return module_ctx.extension_metadata(reproducible = True)

rust_clang_dependencies = module_extension(
    implementation = _rust_clang_dependencies_impl,
    os_dependent = True,
    arch_dependent = True,
)
