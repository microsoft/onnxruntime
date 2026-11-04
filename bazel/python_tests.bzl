# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native Python package assembly and bounded existing inference suites."""

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

def python_runtime_tests():
    """Assemble the source package alongside the native extension, not a wheel."""
    sources = {
        "onnxruntime/__init__.py": "onnxruntime/__init__.py",
        "onnxruntime/capi/__init__.py": "onnxruntime/python/__init__.py",
        "onnxruntime/capi/_ld_preload.py": "onnxruntime/python/_ld_preload.py",
        "onnxruntime/capi/onnxruntime_inference_collection.py": "onnxruntime/python/onnxruntime_inference_collection.py",
        "onnxruntime/capi/onnxruntime_validation.py": "onnxruntime/python/onnxruntime_validation.py",
    }
    for dest, src in sources.items():
        copy_file(
            name = "python_package_" + dest.replace("/", "_").replace(".", "_"),
            src = src,
            out = "python_runtime/" + dest,
        )
    expand_template(
        name = "python_package_loader",
        template = "onnxruntime/python/_pybind_state.py.in",
        out = "python_runtime/onnxruntime/capi/_pybind_state.py",
        substitutions = {
            "@ONNXRUNTIME_IMPORT_PYTORCH_TO_RESOLVE_DLLS@": "",
            "@ONNXRUNTIME_SETDLOPENFLAGS_GLOBAL@": "",
            "@ONNXRUNTIME_SETDLOPENFLAGS_LOCAL@": "",
        },
    )
    write_file(
        name = "python_package_version",
        out = "python_runtime/onnxruntime/capi/version_info.py",
        content = ["vs2019 = True"],
    )
    write_file(
        name = "python_package_info",
        out = "python_runtime/onnxruntime/capi/build_and_package_info.py",
        content = [
            "from .onnxruntime_pybind11_state import get_version_string",
            "package_name = 'onnxruntime'",
            "__version__ = get_version_string()",
        ],
    )
    for platform, src, extension in [
        ("windows", ":python_windows_extension", "pyd"),
        ("unix", select({
            "//bazel:macos": ":python_macos_extension",
            "//conditions:default": ":onnxruntime_pybind11_state.so",
        }), "so"),
    ]:
        copy_file(
            name = "python_package_extension_" + platform,
            src = src,
            out = "python_runtime/onnxruntime/capi/onnxruntime_pybind11_state." + extension,
            target_compatible_with = ["@platforms//os:windows"] if platform == "windows" else select({
                "//bazel:windows": ["@platforms//:incompatible"],
                "//conditions:default": [],
            }),
        )
    py_library(
        name = "python_runtime_package",
        srcs = ["python_runtime/" + dest for dest in sources] + [
            ":python_package_loader",
            ":python_package_version",
            ":python_package_info",
        ],
        data = select({
            "//bazel:windows": [":python_package_extension_windows"],
            "//conditions:default": [":python_package_extension_unix"],
        }),
        imports = ["python_runtime"],
        target_compatible_with = select({
            "//bazel:cuda_enabled": ["@platforms//:incompatible"],
            "//conditions:default": [],
        }),
        deps = ["@ort_pip//numpy"],
    )
    suites = {
        "python_inference_test": ("onnxruntime_test_python.TestInferenceSession", [
            "test_get_version_string",
            "test_get_build_info",
            "test_get_providers",
            "test_run_model",
            "test_run_model_from_bytes",
            "test_run_model2",
            "test_run_model2_contiguous",
            "test_run_model_multiple_threads",
            "test_run_model_symbolic_input",
            "test_boolean_inputs",
            "test_string_input1",
            "test_string_input2",
            "test_raise_wrong_num_inputs",
            "test_graph_optimization_level",
            "test_ort_execution_mode",
            "test_session_options_add_config_entry",
            "test_invalid_session_options_config_entry",
            "test_ort_value",
            "test_ort_device",
            "test_ort_memory_info",
            "test_check_and_normalize_provider_args",
        ]),
        "python_global_threadpool_test": ("onnxruntime_test_python_global_threadpool.TestGlobalThreadPool", [
            "test_global_threadpool",
            "test_raise_error_if_use_per_session_threads",
        ]),
    }
    for name, (test_class, methods) in suites.items():
        py_test(
            name = name,
            srcs = ["tools/bazel/python_test_runner.py"],
            main = "tools/bazel/python_test_runner.py",
            args = [test_class + "." + method for method in methods],
            data = [
                "onnxruntime/test/python/helper.py",
                "onnxruntime/test/python/" + test_class.split(".")[0] + ".py",
            ] + ["onnxruntime/test/testdata/" + fixture for fixture in ([
                "mnist.onnx",
            ] if name == "python_global_threadpool_test" else [
                "identity_string.onnx",
                "logicaland.onnx",
                "matmul_1.onnx",
                "matmul_2.onnx",
                "mul_1.onnx",
            ])],
            deps = [
                ":python_runtime_package",
                "@rules_python//python/runfiles",
            ],
            legacy_create_init = False,
        )
    native.test_suite(
        name = "python_tests",
        tests = sorted([":" + name for name in suites]),
    )
    py_test(
        name = "python_test_runner_test",
        srcs = [
            "tools/bazel/python_test_runner.py",
            "tools/bazel/python_test_runner_test.py",
        ],
        main = "tools/bazel/python_test_runner_test.py",
        deps = ["@rules_python//python/runfiles"],
    )
