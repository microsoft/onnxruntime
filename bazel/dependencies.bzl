# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pinned source repositories for the native Bazel build."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

def _headers(name, url, sha256, strip_prefix, include_dir):
    http_archive(
        name = name,
        urls = [url],
        sha256 = sha256,
        strip_prefix = strip_prefix,
        build_file_content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "headers",
    hdrs = glob(["**/*.h", "**/*.hpp", "include/gsl/*", "Eigen/**", "unsupported/Eigen/**"], allow_empty = True),
    includes = [%r],
    visibility = ["//visibility:public"],
)
""" % include_dir,
    )

def _dependencies_impl(_ctx):
    for name, artifact, version, sha256 in [
        ("ort_junit", "org/junit/platform/junit-platform-console-standalone", "1.11.4", "b016ef6b1c3454d6d7c2c88ce081dabf289699686af6622d6e4e2e1b54b4a2fc"),
        ("ort_test_protobuf_java", "com/google/protobuf/protobuf-java", "3.25.5", "8540247fad9e06baefa8fb45eb313802d019f485f14300e0f9d6b556ed88e753"),
    ]:
        jar = artifact.split("/")[-1] + "-" + version + ".jar"
        http_file(
            name = name,
            urls = ["https://repo.maven.apache.org/maven2/" + artifact + "/" + version + "/" + jar],
            sha256 = sha256,
            downloaded_file_path = jar,
        )
    http_archive(
        name = "ort_mingw",
        urls = ["https://github.com/mstorsjo/llvm-mingw/releases/download/20261104/llvm-mingw-20261104-ucrt-x86_64.zip"],
        sha256 = "552191b5754c49a5a20723276a6fc56541ea1922c4c50b7b669022333fc1b40f",
        strip_prefix = "llvm-mingw-20261104-ucrt-x86_64",
        build_file = "//bazel:mingw.BUILD.bazel",
    )
    http_file(
        name = "ort_node_import_library",
        urls = ["https://nodejs.org/dist/v20.19.0/win-x64/node.lib"],
        sha256 = "72f7ca3b33f0991e93e25521f7e78c8adf187df7d5223bf0efcb5e005420f327",
        downloaded_file_path = "node.lib",
    )
    _headers(
        "ort_wil",
        "https://github.com/microsoft/wil/archive/refs/tags/v1.0.250325.1.zip",
        "48f7fdc3c7d0b284efae783c1266836b61b63fd0c94749098a60dc8ec78b0725",
        "wil-1.0.250325.1",
        "include",
    )
    http_archive(
        name = "ort_dotnet_tensors",
        urls = [
            "https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-public/nuget/v3/flat2/system.numerics.tensors/9.0.0/system.numerics.tensors.9.0.0.nupkg",
            "https://api.nuget.org/v3-flatcontainer/system.numerics.tensors/9.0.0/system.numerics.tensors.9.0.0.nupkg",
        ],
        sha256 = "b750243c36002a62b28b1ac5d3fbc284ad340ba1494cc36aca110611a0b1f959",
        type = "zip",
        build_file_content = """
load("@rules_dotnet//dotnet:defs.bzl", "import_dll")
import_dll(
    name = "tensors",
    dll = "lib/net8.0/System.Numerics.Tensors.dll",
    version = "9.0.0",
    visibility = ["//visibility:public"],
)
""",
    )
    _headers(
        "ort_node_headers",
        "https://nodejs.org/dist/v20.19.0/node-v20.19.0-headers.tar.gz",
        "800194e32cef4ee77f5a2e8c0e5dd2b4acb38b39ef2ea544929c181b949494bb",
        "node-v20.19.0",
        "include/node",
    )
    _headers(
        "ort_node_addon_api",
        "https://github.com/nodejs/node-addon-api/archive/refs/tags/v6.1.0.zip",
        "72783c170b711a0e977a886ba7a1ca275c3b1322b416f8c6acb37dbcf24604e9",
        "node-addon-api-6.1.0",
        ".",
    )
    _headers(
        "ort_pybind11",
        "https://github.com/pybind/pybind11/archive/refs/tags/v3.0.2.zip",
        "e3754d16ea536bdc968ce1464fa822e198a1217881d31bd9bf3d287fed2bbf4c",
        "pybind11-3.0.2",
        "include",
    )
    _headers(
        "ort_cutlass",
        "https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.7.0.zip",
        "e88f72932f830ee40cb46d398319336e4e9f6dfc235de4b7844be712cf4e6ab4",
        "cutlass-4.7.0",
        "include",
    )
    _headers(
        "ort_cudnn_frontend",
        "https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.27.0.zip",
        "0190120bfc8b1f8288c19beffa5a21af643650d6f65e8bad89ab5d7ed74f4273",
        "cudnn-frontend-1.27.0",
        "include",
    )
    _headers(
        "ort_date",
        "https://github.com/HowardHinnant/date/archive/refs/tags/v3.0.1.zip",
        "f4300b96f7a304d4ef9bf6e0fa3ded72159f7f2d0f605bdde3e030a0dba7cf9f",
        "date-3.0.1",
        "include",
    )
    _headers(
        "ort_gsl",
        "https://github.com/microsoft/GSL/archive/refs/tags/v4.2.1.zip",
        "c9291d95f5f6e5c561990d06a589b01d89e553d0f366f0ce723dd1788e7a6076",
        "GSL-4.2.1",
        "include",
    )
    _headers(
        "ort_safeint",
        "https://github.com/dcleblanc/SafeInt/archive/refs/tags/3.0.28.zip",
        "3ffbd9a2fdff45da77da3e7269e9aa512ea43bed5c38ce8fd8f3d1068a032c3f",
        "SafeInt-3.0.28",
        ".",
    )
    _headers(
        "ort_mp11",
        "https://github.com/boostorg/mp11/archive/refs/tags/boost-1.82.0.zip",
        "81431bdc44c439a324e02c07ed067f8f556419fd86f2d8b486ff568df6aac899",
        "mp11-boost-1.82.0",
        "include",
    )
    _headers(
        "ort_eigen",
        "https://github.com/eigen-mirror/eigen/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33/eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip",
        "6a60d76351f97132669daeeb721d6bf14b008101883ad2d687a3201c5c461eb0",
        "eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33",
        ".",
    )
    _headers(
        "ort_flatbuffers",
        "https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.zip",
        "57bd580c0772fd1a726c34ab8bf05325293bc5f9c165060a898afa1feeeb95e1",
        "flatbuffers-23.5.26",
        "include",
    )
    _headers(
        "ort_fp16",
        "https://github.com/Maratyszcza/FP16/archive/0a92994d729ff76a58f692d3028ca1b64b145d91.zip",
        "e66e65515fa09927b348d3d584c68be4215cfe664100d01c9dbc7655a5716d70",
        "FP16-0a92994d729ff76a58f692d3028ca1b64b145d91",
        "include",
    )
    http_archive(
        name = "ort_onnx",
        urls = ["https://github.com/onnx/onnx/archive/refs/tags/v1.23.2.zip"],
        sha256 = "8e1d61e61314c15fcb5a0b2ee512e7113428a47b4e7f0e525e22e9300981f938",
        strip_prefix = "onnx-1.23.2",
        patches = ["//:cmake/patches/onnx/onnx.patch"],
        patch_args = ["-p1"],
        build_file = "//bazel:onnx.BUILD.bazel",
    )

ort_dependencies = module_extension(implementation = _dependencies_impl)
