# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Source-native, single-threaded Emscripten runtime and selected WASM tests."""

load("@aspect_rules_js//js:defs.bzl", "js_binary", "js_run_binary", "js_test")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load(":js_env.bzl", "WINDOWS_JS_ENV")

def _transition_impl(settings, attr):
    return {
        "//command_line_option:platforms": ["@emsdk//:platform_wasm"],
        "//command_line_option:compiler": "emscripten",
        "//command_line_option:cpu": "wasm",
        "//command_line_option:dynamic_mode": "off",
        "//command_line_option:custom_malloc": "@emsdk//emscripten_toolchain:malloc",
        # Host platform .bazelrc options (notably MSVC /std:) must not reach emcc.
        "//command_line_option:cxxopt": ["-std=c++20"],
        "//command_line_option:copt": ["-sDISABLE_EXCEPTION_CATCHING=0"],
        "//command_line_option:features": settings["//command_line_option:features"] + [
            "-use_pthreads",
            "exceptions",
            "wasm_simd" if attr.simd else "-wasm_simd",
        ],
        "//bazel:wasm_simd": attr.simd,
    }

_wasm_transition = transition(
    implementation = _transition_impl,
    inputs = ["//command_line_option:features"],
    outputs = [
        "//command_line_option:platforms",
        "//command_line_option:compiler",
        "//command_line_option:cpu",
        "//command_line_option:dynamic_mode",
        "//command_line_option:custom_malloc",
        "//command_line_option:cxxopt",
        "//command_line_option:copt",
        "//command_line_option:features",
        "//bazel:wasm_simd",
    ],
)

def _module_impl(ctx):
    args = ctx.actions.args()
    args.add_all("--archive", ctx.files.binary)
    args.add_joined("--outputs", ctx.outputs.outputs, join_with = ",")
    ctx.actions.run(
        executable = ctx.executable._extractor,
        arguments = [args],
        inputs = ctx.files.binary,
        outputs = ctx.outputs.outputs,
    )
    return [DefaultInfo(
        files = depset(ctx.outputs.outputs),
        runfiles = ctx.runfiles(files = ctx.outputs.outputs),
    )]

_module = rule(
    implementation = _module_impl,
    attrs = {
        "binary": attr.label(cfg = _wasm_transition, mandatory = True),
        "simd": attr.bool(),
        "outputs": attr.output_list(mandatory = True),
        "_extractor": attr.label(
            default = "@emsdk//emscripten_toolchain:wasm_binary",
            cfg = "exec",
            executable = True,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def wasm_targets():
    """Declare CPU inference modules without changing the native default graph."""
    cc_binary(
        name = "ort_wasm",
        srcs = ["onnxruntime/wasm/api.cc"],
        additional_linker_inputs = [
            "onnxruntime/wasm/pre.js",
            "onnxruntime/wasm/js_post_js.js",
        ],
        linkopts = [
            "--no-entry",
            "-sALLOW_MEMORY_GROWTH=1",
            "-sMAXIMUM_MEMORY=4294967296",
            "-sMODULARIZE=1",
            "-sEXPORT_ES6=1",
            "-sEXPORT_NAME=createOrtModule",
            "-sENVIRONMENT=web,node",
            "-sFILESYSTEM=0",
            "-sWASM_BIGINT=1",
            "-sDISABLE_EXCEPTION_CATCHING=0",
            "-sEXPORTED_FUNCTIONS=_malloc,_free",
            "-sEXPORTED_RUNTIME_METHODS=stackAlloc,stackRestore,stackSave,UTF8ToString,stringToUTF8,lengthBytesUTF8,getValue,setValue,HEAP8,HEAPU8,HEAP32,HEAPU32",
            "-sINCOMING_MODULE_JS_API=locateFile,instantiateWasm,wasmBinary",
            "--pre-js=$(location onnxruntime/wasm/pre.js)",
            "--post-js=$(location onnxruntime/wasm/js_post_js.js)",
        ],
        target_compatible_with = [
            "@platforms//cpu:wasm32",
            "@platforms//os:emscripten",
        ],
        deps = [":runtime"],
    )
    js_binary(
        name = "wasm_package_assembler",
        env = WINDOWS_JS_ENV,
        entry_point = "tools/bazel/wasm_package.cjs",
        data = native.glob([
            "js/common/lib/*.ts",
            "js/web/lib/wasm/*.ts",
        ]) + [
            "//bazel/node:compiler",
            "js/web/lib/backend-wasm.ts",
            "js/web/lib/validate-session-options.ts",
            "js/web/lib/version.ts",
            "js/web/package.json",
            "js/tsconfig.json",
            "VERSION_NUMBER",
            "LICENSE",
            "ThirdPartyNotices.txt",
        ],
        tags = ["manual"],
    )
    cc_binary(
        name = "ort_wasm_test",
        testonly = True,
        srcs = [
            "onnxruntime/test/wasm/test_main.cc",
            "onnxruntime/test/wasm/test_inference.cc",
        ],
        additional_linker_inputs = ["onnxruntime/test/testdata/mul_1.onnx"],
        linkopts = [
            "-sALLOW_MEMORY_GROWTH=1",
            "-sEXIT_RUNTIME=1",
            "-sDISABLE_EXCEPTION_CATCHING=0",
            "-sENVIRONMENT=node",
            "--preload-file",
            "$(location onnxruntime/test/testdata/mul_1.onnx)@/testdata/mul_1.onnx",
        ],
        target_compatible_with = [
            "@platforms//cpu:wasm32",
            "@platforms//os:emscripten",
        ],
        deps = [":runtime", "@googletest//:gtest"],
    )
    for suffix, simd in [("scalar", False), ("simd", True)]:
        _module(
            name = "wasm_" + suffix,
            binary = ":ort_wasm",
            simd = simd,
            outputs = [
                "wasm_" + suffix + "/ort_wasm.js",
                "wasm_" + suffix + "/ort_wasm.wasm",
            ],
            tags = ["manual"],
        )
        js_run_binary(
            name = "web_" + suffix + "_package",
            tool = ":wasm_package_assembler",
            srcs = native.glob([
                "js/common/lib/*.ts",
                "js/web/lib/wasm/*.ts",
            ]) + [
                "js/web/lib/backend-wasm.ts",
                "js/web/lib/validate-session-options.ts",
                "js/web/lib/version.ts",
                "js/web/package.json",
                "js/tsconfig.json",
                "VERSION_NUMBER",
                "LICENSE",
                "ThirdPartyNotices.txt",
                ":wasm_" + suffix + "/ort_wasm.js",
                ":wasm_" + suffix + "/ort_wasm.wasm",
            ],
            out_dirs = ["web_" + suffix + "_package"],
            args = [
                "web_" + suffix + "_package",
                "$(rootpath :wasm_" + suffix + "/ort_wasm.js)",
                "$(rootpath :wasm_" + suffix + "/ort_wasm.wasm)",
                suffix,
            ],
            tags = ["manual"],
        )
        pkg_tar(
            name = "web_" + suffix + "_npm",
            srcs = [":web_" + suffix + "_package"],
            out = "onnxruntime-web-bazel-" + suffix + ".tgz",
            extension = "tar.gz",
            strip_prefix = "web_" + suffix + "_package",
            tags = ["manual"],
        )
        js_test(
            name = "web_" + suffix + "_npm_test",
            env = WINDOWS_JS_ENV,
            entry_point = "tools/bazel/wasm_npm_test.cjs",
            include_npm = True,
            args = [
                "$(rootpath :web_" + suffix + "_npm)",
                "$(rootpath onnxruntime/test/testdata/mul_1.onnx)",
                "$(rootpath VERSION_NUMBER)",
                suffix,
            ],
            data = [
                ":web_" + suffix + "_npm",
                "//bazel/node:compiler",
                "onnxruntime/test/testdata/mul_1.onnx",
                "VERSION_NUMBER",
            ],
            tags = ["manual"],
        )
        js_test(
            name = "web_" + suffix + "_smoke_test",
            env = WINDOWS_JS_ENV,
            entry_point = "tools/bazel/wasm_web_smoke_test.cjs",
            args = ["$(rootpath :web_" + suffix + "_package)"],
            data = [
                ":web_" + suffix + "_package",
                "onnxruntime/test/testdata/mul_1.onnx",
                "onnxruntime/test/testdata/matmul_1.onnx",
                "onnxruntime/wasm/api.h",
            ],
            tags = ["manual"],
        )
        js_test(
            name = "web_" + suffix + "_browser_test",
            env = WINDOWS_JS_ENV,
            entry_point = "tools/bazel/wasm_browser_test.cjs",
            args = ["$(rootpath :web_" + suffix + "_package)"],
            data = [
                ":web_" + suffix + "_package",
                "tools/bazel/wasm_browser_test.html",
                "onnxruntime/test/testdata/mul_1.onnx",
                "onnxruntime/test/testdata/matmul_1.onnx",
            ],
            tags = ["manual"],
        )
        _module(
            name = "wasm_" + suffix + "_test_module",
            testonly = True,
            binary = ":ort_wasm_test",
            simd = simd,
            outputs = [
                "wasm_" + suffix + "_test/ort_wasm_test.js",
                "wasm_" + suffix + "_test/ort_wasm_test.wasm",
                "wasm_" + suffix + "_test/ort_wasm_test.data",
            ],
            tags = ["manual"],
        )
        js_test(
            name = "wasm_" + suffix + "_test",
            env = WINDOWS_JS_ENV,
            entry_point = "tools/bazel/wasm_test_runner.cjs",
            args = ["$(rootpath :wasm_" + suffix + "_test/ort_wasm_test.js)"],
            data = [
                ":wasm_" + suffix + "_test_module",
                ":wasm_" + suffix + "_test/ort_wasm_test.js",
            ],
            tags = ["manual"],
        )
