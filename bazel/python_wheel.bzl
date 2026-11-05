# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native CPython wheel packaging without invoking setup.py or build.py."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_cuda//cuda/private:providers.bzl", "CudaToolkitInfo")
load("@rules_python//python:py_test.bzl", "py_test")

def _single_file(target, description):
    files = target[DefaultInfo].files.to_list()
    if len(files) != 1:
        fail("{} must provide exactly one file, got {}".format(description, files))
    return files[0]

def _is_constraint(ctx, attribute):
    return ctx.target_platform_has_constraint(
        getattr(ctx.attr, attribute)[platform_common.ConstraintValueInfo],
    )

def _platform_tag(ctx):
    x86_64 = _is_constraint(ctx, "_x86_64")
    aarch64 = _is_constraint(ctx, "_aarch64")
    if _is_constraint(ctx, "_windows") and x86_64:
        return "win_amd64"
    if _is_constraint(ctx, "_linux") and x86_64:
        return "linux_x86_64"
    if _is_constraint(ctx, "_linux") and aarch64:
        return "linux_aarch64"
    if _is_constraint(ctx, "_macos") and x86_64:
        return "macosx_11_0_x86_64"
    if _is_constraint(ctx, "_macos") and aarch64:
        return "macosx_11_0_arm64"
    fail("Python wheels support Windows x86-64, Linux x86-64/aarch64, and macOS x86-64/arm64.")

def _python_release_wheel_impl(ctx):
    runtime = ctx.toolchains["@rules_python//python:toolchain_type"].py3_runtime
    version_info = runtime.interpreter_version_info
    if not version_info or not version_info.major or not version_info.minor:
        fail("The selected Python toolchain must report a major and minor version.")
    if version_info.major != 3 or version_info.minor not in [11, 12, 13, 14]:
        fail("ONNX Runtime release wheels support CPython 3.11 through 3.14.")
    python_tag = "cp{}{}".format(version_info.major, version_info.minor)
    cuda_version = ""
    if ctx.attr._cuda_enabled[BuildSettingInfo].value:
        cuda_toolkit = ctx.toolchains["@rules_cuda//cuda:toolchain_type"].cuda_toolkit[CudaToolkitInfo]
        if cuda_toolkit.version_major < 0 or cuda_toolkit.version_minor < 0:
            fail("CUDA Python packaging requires a detected CUDA toolkit version.")
        cuda_version = "{}.{}".format(cuda_toolkit.version_major, cuda_toolkit.version_minor)

    output = ctx.actions.declare_directory(ctx.label.name + "_dist")
    args = ctx.actions.args()
    args.add(ctx.file._packager)
    args.add("--package-tree", ctx.file.package_tree.path)
    args.add("--version-file", ctx.file.version_file)
    args.add("--requirements-file", ctx.file.requirements_file)
    args.add("--description-file", ctx.file.description_file)
    for license_file in ctx.files.license_files:
        args.add("--license-file", license_file)
    args.add("--output", output.path)
    args.add("--distribution", ctx.attr.distribution)
    args.add("--python-tag", python_tag)
    args.add("--abi", python_tag)
    args.add("--platform", _platform_tag(ctx))
    args.add("--python-requires", ">=3.11")
    args.add("--cuda-version", cuda_version)

    runtime_inputs = runtime.files if runtime.files else depset()
    ctx.actions.run(
        executable = runtime.interpreter,
        arguments = [args],
        inputs = depset(
            direct = [
                ctx.file._packager,
                ctx.file.description_file,
                ctx.file.package_tree,
                ctx.file.requirements_file,
                ctx.file.version_file,
            ] + ctx.files.license_files,
            transitive = [runtime_inputs],
        ),
        outputs = [output],
        mnemonic = "BuildOrtPythonWheel",
    )
    return [DefaultInfo(files = depset([output]), runfiles = ctx.runfiles([output]))]

_python_release_wheel = rule(
    implementation = _python_release_wheel_impl,
    attrs = {
        "description_file": attr.label(allow_single_file = True, mandatory = True),
        "distribution": attr.string(mandatory = True),
        "license_files": attr.label_list(allow_files = True),
        "package_tree": attr.label(allow_single_file = True, mandatory = True),
        "requirements_file": attr.label(allow_single_file = True, mandatory = True),
        "version_file": attr.label(allow_single_file = True, mandatory = True),
        "_aarch64": attr.label(default = "@platforms//cpu:aarch64"),
        "_cuda_enabled": attr.label(default = "//bazel:cuda"),
        "_linux": attr.label(default = "@platforms//os:linux"),
        "_macos": attr.label(default = "@platforms//os:macos"),
        "_packager": attr.label(
            allow_single_file = True,
            default = "//:tools/bazel/python_wheel_packager.py",
        ),
        "_windows": attr.label(default = "@platforms//os:windows"),
        "_x86_64": attr.label(default = "@platforms//cpu:x86_64"),
    },
    toolchains = [
        "@rules_cuda//cuda:toolchain_type",
        "@rules_python//python:toolchain_type",
    ],
)

def python_wheel_package():
    """Package the complete source-assembled CPU or CUDA inference release."""
    compatible = select({
        "//bazel:linux_x86_64_cuda": [],
        "//bazel:cuda_enabled": ["@platforms//:incompatible"],
        "//bazel:windows": [],
        "//bazel:linux": [],
        "//bazel:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
    _python_release_wheel(
        name = "python_wheel",
        description_file = "docs/python/README.rst",
        distribution = select({
            "//bazel:cuda_enabled": "onnxruntime-gpu",
            "//conditions:default": "onnxruntime",
        }),
        license_files = [
            "LICENSE",
            "ThirdPartyNotices.txt",
        ],
        package_tree = ":python_runtime_files",
        requirements_file = "requirements.txt",
        target_compatible_with = compatible,
        version_file = "VERSION_NUMBER",
    )
    py_test(
        name = "python_wheel_test",
        srcs = ["tools/bazel/python_wheel_test.py"],
        main = "tools/bazel/python_wheel_test.py",
        args = [
            "$(rootpath :python_wheel)",
            "$(rootpath VERSION_NUMBER)",
            "$(rootpath go/testdata/add_f32.onnx)",
        ] + select({
            "//bazel:cuda_enabled": ["onnxruntime-gpu"],
            "//conditions:default": ["onnxruntime"],
        }),
        data = [
            ":python_wheel",
            "VERSION_NUMBER",
            "go/testdata/add_f32.onnx",
        ],
        target_compatible_with = compatible,
        deps = ["@ort_pip//numpy"],
    )
    py_test(
        name = "python_wheel_packager_test",
        srcs = [
            "tools/bazel/python_package_tree.py",
            "tools/bazel/python_wheel_packager.py",
            "tools/bazel/python_wheel_packager_test.py",
        ],
        main = "tools/bazel/python_wheel_packager_test.py",
        imports = ["tools/bazel"],
    )
    py_test(
        name = "python_wheel_subprocess_test",
        srcs = [
            "tools/bazel/python_wheel_test.py",
            "tools/bazel/python_wheel_test_test.py",
        ],
        main = "tools/bazel/python_wheel_test_test.py",
        imports = ["tools/bazel"],
    )
