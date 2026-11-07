# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Public, checksum-pinned libclang for native Rust binding generation."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _xcode_clang_impl(ctx):
    clang = ctx.execute(["/usr/bin/xcrun", "--find", "clang"])
    resource = ctx.execute(["/usr/bin/xcrun", "clang", "-print-resource-dir"])
    sdk = ctx.execute(["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"])
    if clang.return_code or resource.return_code or sdk.return_code:
        fail("Rust bindgen requires Xcode clang and builtin headers; select a complete Xcode installation with xcode-select. %s %s" % (clang.stderr, resource.stderr))
    library = ctx.path(clang.stdout.strip()).dirname.dirname.get_child("lib", "libclang.dylib")
    headers = ctx.path(resource.stdout.strip()).get_child("include")
    if not library.exists or not headers.exists:
        fail("Selected Xcode is missing libclang or builtin headers (%s, %s); install the complete Xcode toolchain." % (library, headers))
    ctx.symlink(library.dirname, "xcode_lib")
    ctx.symlink(headers, "builtin")
    ctx.symlink(ctx.path(sdk.stdout.strip()).get_child("usr", "include"), "sdk")
    ctx.file("BUILD.bazel", """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_import(name = "libclang", shared_library = "xcode_lib/libclang.dylib")
cc_library(name = "headers", hdrs = glob(["builtin/**", "sdk/**"]), includes = ["builtin", "sdk"])
""")

_xcode_clang = repository_rule(
    implementation = _xcode_clang_impl,
    local = True,
    configure = True,
    environ = ["DEVELOPER_DIR"],
)

def _rust_clang_dependencies_impl(module_ctx):
    os = module_ctx.os.name
    arch = module_ctx.os.arch
    if "mac" in os:
        _xcode_clang(name = "ort_rust_libclang")
        return module_ctx.extension_metadata(reproducible = True)
    if "windows" in os and arch in ["amd64", "x86_64"]:
        archive = ("0b/2d/3f480b1e1d31eb3d6de5e3ef641954e5c67430d5ac93b7fa7e07589576c7", "win_amd64", "4dd2d3b82fab35e2bf9ca717d7b63ac990a3519c7e312f19fa8e86dcc712f7fb", "libclang.dll")
    elif os == "linux" and arch in ["amd64", "x86_64"]:
        archive = ("1d/fc/716c1e62e512ef1c160e7984a73a5fc7df45166f2ff3f254e71c58076f7c", "manylinux2010_x86_64", "c533091d8a3bbf7460a00cb6c1a71da93bffe148f172c7d03b1c31fbf8aa2a0b", "libclang.so")
    else:
        fail("Rust binding generation requires Windows/Linux x86-64 or macOS x86-64/ARM64; unsupported host: %s %s" % (os, arch))
    http_archive(
        name = "ort_rust_libclang",
        urls = ["https://files.pythonhosted.org/packages/%s/libclang-18.1.1-py2.py3-none-%s.whl" % (archive[0], archive[1])],
        sha256 = archive[2],
        type = "zip",
        build_file_content = """
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(name = "headers", visibility = ["//visibility:public"])
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
