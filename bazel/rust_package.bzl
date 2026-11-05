# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Platform-specific Rust source bundle with its Bazel-built native SDK."""

load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:zip.bzl", "pkg_zip")
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

def rust_release_package(cargo_tool):
    """Package both Cargo crates and the selected source-built desktop runtime."""
    cpu_only = select({
        "//bazel:cuda_enabled": ["@platforms//:incompatible"],
        "//conditions:default": [],
    })
    py_binary(
        name = "crate_packager",
        srcs = ["//:tools/bazel/rust_crate_package.py"],
        main = "//:tools/bazel/rust_crate_package.py",
    )
    native.genrule(
        name = "onnxruntime_crate",
        srcs = [
            "README.md",
            "RELEASE_PACKAGE.md",
            "LICENSE-APACHE",
            "LICENSE-MIT",
            "//rust/onnxruntime:Cargo.toml",
            "//rust/onnxruntime:crate_package_files",
        ],
        outs = ["onnxruntime-0.0.14.crate"],
        cmd = " ".join([
            "$(location :crate_packager)",
            "--manifest $(location //rust/onnxruntime:Cargo.toml)",
            "--readme $(location README.md)",
            "--release-notes $(location RELEASE_PACKAGE.md)",
            "--license $(location LICENSE-APACHE) $(location LICENSE-MIT)",
            "--source $(locations //rust/onnxruntime:crate_package_files)",
            "--output $@",
        ]),
        tools = [":crate_packager"],
        target_compatible_with = cpu_only,
    )
    native.genrule(
        name = "onnxruntime_sys_crate",
        srcs = [
            "README.md",
            "RELEASE_PACKAGE.md",
            "LICENSE-APACHE",
            "LICENSE-MIT",
            "//rust/onnxruntime-sys:Cargo.toml",
            "//rust/onnxruntime-sys:crate_package_files",
        ],
        outs = ["onnxruntime-sys-0.0.14.crate"],
        cmd = " ".join([
            "$(location :crate_packager)",
            "--manifest $(location //rust/onnxruntime-sys:Cargo.toml)",
            "--readme $(location README.md)",
            "--release-notes $(location RELEASE_PACKAGE.md)",
            "--license $(location LICENSE-APACHE) $(location LICENSE-MIT)",
            "--source $(locations //rust/onnxruntime-sys:crate_package_files)",
            "--output $@",
        ]),
        tools = [":crate_packager"],
        target_compatible_with = cpu_only,
    )
    native.filegroup(
        name = "crates_io_artifacts",
        srcs = [
            ":onnxruntime_crate",
            ":onnxruntime_sys_crate",
        ],
        target_compatible_with = cpu_only,
        visibility = ["//:__pkg__"],
    )
    py_test(
        name = "crates_io_artifacts_test",
        srcs = ["//:tools/bazel/rust_crate_package_test.py"],
        main = "//:tools/bazel/rust_crate_package_test.py",
        args = [
            "$(rootpath :onnxruntime_crate)",
            "$(rootpath :onnxruntime_sys_crate)",
        ],
        data = [
            ":onnxruntime_crate",
            ":onnxruntime_sys_crate",
        ],
        target_compatible_with = cpu_only,
    )
    py_test(
        name = "crates_io_consumer_test",
        srcs = ["//:tools/bazel/rust_crate_consumer_test.py"],
        main = "//:tools/bazel/rust_crate_consumer_test.py",
        args = [
            "$(rootpath " + cargo_tool + ")",
            "$(rootpath :onnxruntime_crate)",
            "$(rootpath :onnxruntime_sys_crate)",
            "$(rootpath :release_archive)",
            "$(rootpath //:go/testdata/add_f32.onnx)",
        ],
        data = [
            cargo_tool,
            ":onnxruntime_crate",
            ":onnxruntime_sys_crate",
            ":release_archive",
            "//:go/testdata/add_f32.onnx",
        ],
        tags = ["manual"],
    )
    compatible = select({
        "//bazel:windows": [],
        "//bazel:linux": [],
        "//bazel:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + select({
        "//bazel:x86_64": [],
        "//bazel:aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }) + cpu_only
    pkg_files(
        name = "release_workspace_sources",
        srcs = native.glob([
            "Cargo.toml",
            "Cargo.lock",
            "README.md",
            "RELEASE_PACKAGE.md",
            "LICENSE-*",
        ]),
        prefix = "source",
        strip_prefix = strip_prefix.from_pkg(""),
    )
    pkg_files(
        name = "release_onnxruntime_sources",
        srcs = [
            "//rust/onnxruntime:Cargo.toml",
            "//rust/onnxruntime:crate_package_files",
        ],
        prefix = "source/onnxruntime",
        strip_prefix = strip_prefix.from_root("rust/onnxruntime"),
    )
    pkg_files(
        name = "release_onnxruntime_sys_sources",
        srcs = [
            "//rust/onnxruntime-sys:Cargo.toml",
            "//rust/onnxruntime-sys:crate_package_files",
        ],
        prefix = "source/onnxruntime-sys",
        strip_prefix = strip_prefix.from_root("rust/onnxruntime-sys"),
    )
    pkg_filegroup(
        name = "release_headers",
        srcs = ["//:runtime_sdk_headers"],
        prefix = "native",
    )
    pkg_files(
        name = "release_libraries",
        srcs = [
            "//:providers_shared_release_library",
            "//:runtime_release_library",
        ],
        attributes = pkg_attributes(mode = "0755"),
        prefix = "native/lib",
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_files(
        name = "release_import_library",
        srcs = ["//:runtime_interface_library"],
        prefix = "native/lib",
        renames = {"//:runtime_interface_library": "onnxruntime.lib"},
        target_compatible_with = ["@platforms//os:windows"],
    )
    pkg_files(
        name = "release_notices",
        srcs = [
            "//:LICENSE",
            "//:ThirdPartyNotices.txt",
            "//:VERSION_NUMBER",
        ],
        prefix = "native",
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_zip(
        name = "release_archive",
        srcs = [
            ":release_headers",
            ":release_libraries",
            ":release_notices",
            ":release_onnxruntime_sources",
            ":release_onnxruntime_sys_sources",
            ":release_workspace_sources",
        ] + select({
            "//bazel:windows": [":release_import_library"],
            "//conditions:default": [],
        }),
        out = "onnxruntime-rust-bindings.zip",
        package_dir = "onnxruntime-rust",
        target_compatible_with = compatible,
        visibility = ["//:__pkg__"],
    )
    py_test(
        name = "release_archive_test",
        srcs = ["//:tools/bazel/rust_package_test.py"],
        main = "//:tools/bazel/rust_package_test.py",
        args = [
            "$(rootpath :release_archive)",
            "$(rootpath //:runtime_release_library)",
            "$(rootpath //:providers_shared_release_library)",
            "$(rootpath //:VERSION_NUMBER)",
        ],
        data = [
            ":release_archive",
            "//:providers_shared_release_library",
            "//:runtime_release_library",
            "//:VERSION_NUMBER",
        ],
        target_compatible_with = compatible,
    )
    py_test(
        name = "release_consumer_test",
        srcs = ["//:tools/bazel/rust_package_consumer_test.py"],
        main = "//:tools/bazel/rust_package_consumer_test.py",
        args = [
            "$(rootpath " + cargo_tool + ")",
            "$(rootpath :release_archive)",
            "$(rootpath //:go/testdata/add_f32.onnx)",
        ],
        data = [
            cargo_tool,
            ":release_archive",
            "//:go/testdata/add_f32.onnx",
        ],
        tags = ["manual"],
        target_compatible_with = compatible,
    )
