# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Native Python package assembly and bounded existing inference suites."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_cuda//cuda/private:providers.bzl", "CudaToolkitInfo")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

def _single_file(target, description):
    files = target[DefaultInfo].files.to_list()
    if len(files) != 1:
        fail("{} must provide exactly one file, got {}".format(description, files))
    return files[0]

def _is_constraint(ctx, attribute):
    return ctx.target_platform_has_constraint(
        getattr(ctx.attr, attribute)[platform_common.ConstraintValueInfo],
    )

def _python_package_tree_impl(ctx):
    runtime = ctx.toolchains["@rules_python//python:toolchain_type"].py3_runtime
    version_info = runtime.interpreter_version_info
    if not version_info or not version_info.major or not version_info.minor:
        fail("The selected Python toolchain must report a major and minor version.")

    files = []
    inputs = []
    for target, destination in ctx.attr.files.items():
        source = _single_file(target, destination)
        files.append({
            "destination": destination,
            "source": source.path,
        })
        inputs.append(source)

    windows = _is_constraint(ctx, "_windows")
    extension = _single_file(ctx.attr.extension, "Python native extension")
    files.append({
        "destination": "onnxruntime/capi/onnxruntime_pybind11_state." + ("pyd" if windows else "so"),
        "source": extension.path,
    })
    inputs.append(extension)

    provider_bridge = _single_file(ctx.attr.provider_bridge, "provider bridge")
    bridge_name = (
        "onnxruntime_providers_shared.dll" if windows else "libonnxruntime_providers_shared.dylib" if _is_constraint(ctx, "_macos") else "libonnxruntime_providers_shared.so"
    )
    files.append({
        "destination": "onnxruntime/capi/" + bridge_name,
        "source": provider_bridge.path,
    })
    inputs.append(provider_bridge)

    if ctx.attr.cuda_provider:
        cuda_provider = _single_file(ctx.attr.cuda_provider, "CUDA provider library")
        files.append({
            "destination": "onnxruntime/capi/libonnxruntime_providers_cuda.so",
            "source": cuda_provider.path,
        })
        inputs.append(cuda_provider)

    cuda_enabled = ctx.attr._cuda_enabled[BuildSettingInfo].value
    cuda_version = ""
    if cuda_enabled:
        cuda_toolchain = ctx.toolchains["@rules_cuda//cuda:toolchain_type"]
        if cuda_toolchain == None:
            fail("CUDA Python packaging requires a registered CUDA toolchain; configure the CUDA toolkit before enabling //bazel:cuda.")
        cuda_toolkit = cuda_toolchain.cuda_toolkit[CudaToolkitInfo]
        if cuda_toolkit.version_major < 0 or cuda_toolkit.version_minor < 0:
            fail("CUDA Python packaging requires a detected CUDA toolkit version.")
        cuda_version = "{}.{}".format(cuda_toolkit.version_major, cuda_toolkit.version_minor)

    manifest = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(
        output = manifest,
        content = json.encode_indent({
            "cuda_enabled": cuda_enabled,
            "cuda_version": cuda_version,
            "files": files,
            "linux": _is_constraint(ctx, "_linux"),
            "package_name": ctx.attr.package_name,
            "tools_init_append": ctx.file.tools_init_append.path,
            "version_file": ctx.file.version_file.path,
        }),
    )
    output = ctx.actions.declare_directory("python_runtime")
    args = ctx.actions.args()
    args.add(ctx.file._assembler)
    args.add("--manifest", manifest)
    args.add("--output", output.path)
    runtime_inputs = runtime.files if runtime.files else depset()
    ctx.actions.run(
        executable = runtime.interpreter,
        arguments = [args],
        inputs = depset(
            direct = inputs + [
                manifest,
                ctx.file._assembler,
                ctx.file.tools_init_append,
                ctx.file.version_file,
            ],
            transitive = [runtime_inputs],
        ),
        outputs = [output],
        mnemonic = "AssembleOrtPythonPackage",
    )
    return [DefaultInfo(files = depset([output]), runfiles = ctx.runfiles([output]))]

_python_package_tree = rule(
    implementation = _python_package_tree_impl,
    attrs = {
        "cuda_provider": attr.label(allow_single_file = True),
        "extension": attr.label(allow_single_file = True, mandatory = True),
        "files": attr.label_keyed_string_dict(allow_files = True),
        "package_name": attr.string(mandatory = True),
        "provider_bridge": attr.label(allow_single_file = True, mandatory = True),
        "tools_init_append": attr.label(allow_single_file = True, mandatory = True),
        "version_file": attr.label(allow_single_file = True, mandatory = True),
        "_assembler": attr.label(
            allow_single_file = True,
            default = "//:tools/bazel/python_package_tree.py",
        ),
        "_cuda_enabled": attr.label(default = "//bazel:cuda"),
        "_linux": attr.label(default = "@platforms//os:linux"),
        "_macos": attr.label(default = "@platforms//os:macos"),
        "_windows": attr.label(default = "@platforms//os:windows"),
    },
    toolchains = [
        config_common.toolchain_type("@rules_cuda//cuda:toolchain_type", mandatory = False),
        "@rules_python//python:toolchain_type",
    ],
)

def _add_tree(files, source_root, destination_root, patterns = ["*.py"]):
    for source in native.glob(
        [source_root + "/" + pattern for pattern in patterns],
        allow_empty = True,
    ):
        relative = source[len(source_root) + 1:]
        files[source] = destination_root + "/" + relative

def _python_package_files():
    files = {
        "LICENSE": "onnxruntime/LICENSE",
        "ThirdPartyNotices.txt": "onnxruntime/ThirdPartyNotices.txt",
        "docs/Privacy.md": "onnxruntime/Privacy.md",
        "onnxruntime/__init__.py": "onnxruntime/__init__.py",
        "onnxruntime/py.typed": "onnxruntime/py.typed",
        "onnxruntime/python/_pybind_state.py.in": "onnxruntime/capi/_pybind_state.py",
        "tools/ci_build/github/android/nnapi_supported_ops.md": "onnxruntime/tools/mobile_helpers/nnapi_supported_ops.md",
        "tools/ci_build/github/apple/coreml_supported_mlprogram_ops.md": "onnxruntime/tools/mobile_helpers/coreml_supported_mlprogram_ops.md",
        "tools/ci_build/github/apple/coreml_supported_neuralnetwork_ops.md": "onnxruntime/tools/mobile_helpers/coreml_supported_neuralnetwork_ops.md",
        "tools/python/remove_initializer_from_input.py": "onnxruntime/tools/remove_initializer_from_input.py",
    }
    for source_root, destination_root, patterns in [
        ("onnxruntime/python", "onnxruntime/capi", ["*.py"]),
        ("onnxruntime/python/backend", "onnxruntime/backend", ["*.py"]),
        ("onnxruntime/python/datasets", "onnxruntime/datasets", ["*.py", "*.onnx", "*.pb"]),
        ("onnxruntime/python/tools", "onnxruntime/tools", ["*.py"]),
        ("onnxruntime/python/tools/qnn", "onnxruntime/tools/qnn", ["*.py"]),
        ("onnxruntime/python/tools/quantization", "onnxruntime/quantization", ["*.py"]),
        ("onnxruntime/python/tools/quantization/operators", "onnxruntime/quantization/operators", ["*.py"]),
        ("onnxruntime/python/tools/quantization/CalTableFlatBuffers", "onnxruntime/quantization/CalTableFlatBuffers", ["*.py"]),
        ("onnxruntime/python/tools/quantization/fusions", "onnxruntime/quantization/fusions", ["*.py"]),
        ("onnxruntime/python/tools/quantization/execution_providers/qnn", "onnxruntime/quantization/execution_providers/qnn", ["*.py"]),
        ("onnxruntime/python/tools/quantization/neural_compressor", "onnxruntime/quantization/neural_compressor", ["*.py"]),
        ("onnxruntime/python/tools/transformers", "onnxruntime/transformers", ["*.py"]),
        ("tools/python/util/ort_format_model", "onnxruntime/tools/ort_format_model", ["*.py"]),
        ("tools/python/util/qdq_helpers", "onnxruntime/tools/qdq_helpers", ["*.py"]),
        ("tools/python/util/mobile_helpers", "onnxruntime/tools/mobile_helpers", ["*.py", "*.md", "*.config"]),
        ("onnxruntime/core/flatbuffers/ort_flatbuffers_py", "onnxruntime/tools/ort_format_model/ort_flatbuffers_py", ["**/*.py"]),
    ]:
        _add_tree(files, source_root, destination_root, patterns)
    for model in [
        "bart",
        "bert",
        "gpt2",
        "llama",
        "longformer",
        "phi2",
        "sam2",
        "stable_diffusion",
        "t5",
        "whisper",
    ]:
        _add_tree(
            files,
            "onnxruntime/python/tools/transformers/models/" + model,
            "onnxruntime/transformers/models/" + model,
        )
    _add_tree(
        files,
        "tools/python/util",
        "onnxruntime/tools",
        patterns = [
            "check_onnx_model_mobile_usability.py",
            "convert_onnx_models_to_ort.py",
            "file_utils.py",
            "logger.py",
            "make_dynamic_shape_fixed.py",
            "onnx_model_utils.py",
            "optimize_onnx_model.py",
            "pytorch_export_helpers.py",
            "reduced_build_config_parser.py",
            "update_onnx_opset.py",
        ],
    )
    return files

def python_runtime_tests():
    """Assemble the source package alongside the native extension, not a wheel."""
    package_name = select({
        "//bazel:cuda_enabled": "onnxruntime-gpu",
        "//conditions:default": "onnxruntime",
    })
    _python_package_tree(
        name = "python_package_tree",
        cuda_provider = select({
            "//bazel:cuda_enabled": ":cuda_provider_release_library",
            "//conditions:default": None,
        }),
        extension = ":python_native_release_library",
        files = _python_package_files(),
        package_name = package_name,
        provider_bridge = ":providers_shared_release_library",
        tools_init_append = "tools/python/util/__init__append.py",
        version_file = "VERSION_NUMBER",
        target_compatible_with = select({
            "//bazel:linux_x86_64_cuda": [],
            "//bazel:cuda_enabled": ["@platforms//:incompatible"],
            "//bazel:windows": [],
            "//bazel:linux": [],
            "//bazel:macos": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    py_library(
        name = "python_runtime_package",
        data = [":python_package_tree"],
        imports = ["python_runtime"],
        deps = ["@ort_pip//numpy"],
    )
    native.filegroup(
        name = "python_runtime_files",
        srcs = [":python_package_tree"],
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
