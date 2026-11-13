# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Go source/native bundle and portable module-proxy artifacts."""

load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:zip.bzl", "pkg_zip")
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

def _go_module_proxy_impl(ctx):
    output = ctx.actions.declare_directory("go_proxy")
    args = ctx.actions.args()
    args.add("--module", ctx.file.module)
    args.add("--version-file", ctx.file.version_file)
    args.add("--source")
    args.add_all(ctx.files.sources)
    args.add("--legal")
    args.add_all(ctx.files.legal)
    args.add("--output-dir", output.path)
    ctx.actions.run(
        executable = ctx.executable.packager,
        arguments = [args],
        inputs = depset(
            direct = [ctx.file.module, ctx.file.version_file] + ctx.files.sources + ctx.files.legal,
        ),
        outputs = [output],
        mnemonic = "GoModuleProxyPackage",
        progress_message = "Creating version-derived Go module proxy artifacts",
    )
    return [DefaultInfo(files = depset([output]))]

_go_module_proxy = rule(
    implementation = _go_module_proxy_impl,
    attrs = {
        "legal": attr.label_list(allow_files = True),
        "module": attr.label(allow_single_file = True, mandatory = True),
        "packager": attr.label(cfg = "exec", executable = True, mandatory = True),
        "sources": attr.label_list(allow_files = True),
        "version_file": attr.label(allow_single_file = True, mandatory = True),
    },
)

def go_release_package(go_tool):
    """Define root-package Go release targets without making go/ a Bazel package."""
    py_binary(
        name = "go_module_proxy_packager",
        srcs = ["tools/bazel/go_module_proxy_package.py"],
        main = "tools/bazel/go_module_proxy_package.py",
    )
    native.filegroup(
        name = "go_module_proxy_sources",
        srcs = native.glob(
            [
                "go/onnxruntime/*.c",
                "go/onnxruntime/*.go",
                "go/onnxruntime/*.h",
            ],
            exclude = ["go/onnxruntime/*_test.go"],
        ),
    )
    _go_module_proxy(
        name = "go_module_proxy_artifacts",
        legal = [
            "LICENSE",
            "ThirdPartyNotices.txt",
        ],
        module = "go/go.mod",
        packager = ":go_module_proxy_packager",
        sources = [":go_module_proxy_sources"],
        version_file = "VERSION_NUMBER",
    )
    py_test(
        name = "go_module_proxy_artifacts_test",
        srcs = ["tools/bazel/go_module_proxy_package_test.py"],
        main = "tools/bazel/go_module_proxy_package_test.py",
        args = ["$(rootpath :go_module_proxy_artifacts)"],
        data = [":go_module_proxy_artifacts"],
    )
    py_test(
        name = "go_module_proxy_consumer_test",
        srcs = ["tools/bazel/go_module_proxy_consumer_test.py"],
        main = "tools/bazel/go_module_proxy_consumer_test.py",
        args = [
            "$(rootpath " + go_tool + ")",
            "$(rootpath :go_module_proxy_artifacts)",
            "$(rootpath :go_release_archive)",
            "$(rootpath //:go/testdata/add_f32.onnx)",
        ],
        data = [
            go_tool,
            ":go_module_proxy_artifacts",
            ":go_release_archive",
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
    }) + select({
        "//bazel:cuda_enabled": ["@platforms//:incompatible"],
        "//conditions:default": [],
    })
    pkg_files(
        name = "go_release_sources",
        srcs = native.glob(
            [
                "go/go.mod",
                "go/RELEASE_PACKAGE.md",
                "go/onnxruntime/*.c",
                "go/onnxruntime/*.go",
                "go/onnxruntime/*.h",
            ],
            exclude = ["go/onnxruntime/*_test.go"],
        ),
        prefix = "source",
        strip_prefix = strip_prefix.from_pkg("go"),
    )
    pkg_files(
        name = "go_release_libraries",
        srcs = [
            "//:providers_shared_release_library",
            "//:runtime_release_library",
        ],
        attributes = pkg_attributes(mode = "0755"),
        prefix = "native/lib",
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_files(
        name = "go_release_notices",
        srcs = [
            "//:LICENSE",
            "//:ThirdPartyNotices.txt",
            "//:VERSION_NUMBER",
        ],
        prefix = "native",
        strip_prefix = strip_prefix.files_only(),
    )
    pkg_zip(
        name = "go_release_archive",
        srcs = [
            ":go_release_libraries",
            ":go_release_notices",
            ":go_release_sources",
        ],
        out = "onnxruntime-go-bindings.zip",
        package_dir = "onnxruntime-go",
        target_compatible_with = compatible,
    )
    py_test(
        name = "go_release_archive_test",
        srcs = ["tools/bazel/go_package_test.py"],
        main = "tools/bazel/go_package_test.py",
        args = [
            "$(rootpath :go_release_archive)",
            "$(rootpath //:runtime_release_library)",
            "$(rootpath //:providers_shared_release_library)",
            "$(rootpath //:VERSION_NUMBER)",
        ],
        data = [
            ":go_release_archive",
            "//:providers_shared_release_library",
            "//:runtime_release_library",
            "//:VERSION_NUMBER",
        ],
        target_compatible_with = compatible,
    )
    py_test(
        name = "go_release_consumer_test",
        srcs = ["tools/bazel/go_package_consumer_test.py"],
        main = "tools/bazel/go_package_consumer_test.py",
        args = [
            "$(rootpath " + go_tool + ")",
            "$(rootpath :go_release_archive)",
            "$(rootpath //:go/testdata/add_f32.onnx)",
        ],
        data = [
            go_tool,
            ":go_release_archive",
            "//:go/testdata/add_f32.onnx",
        ],
        tags = ["manual"],
        target_compatible_with = compatible,
    )
