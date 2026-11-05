# Native Bazel build (experimental)

This is an independent, native Bazel build graph. Bazel compiles ONNX Runtime,
MLAS, ONNX, protobuf, execution-provider code, and language extensions directly.
It does **not** invoke CMake, `build.py`, Gradle, MSBuild, `cmake-js`, or
`rules_foreign_cc`. Existing ONNX and Abseil **source patches** are reused;
their location under `cmake/` does not make them build-system dependencies.

The native graph targets Linux x86-64, Windows x86-64, and macOS on Intel and
Apple Silicon. Linux and native Windows execution have been validated; macOS
validation is still pending. This is not yet a replacement for the
release build: additional execution providers,
browser GPU execution providers, and complete language-specific release packaging
are not included. Experimental Android ARM64 runtime/JNI/AAR and single-threaded
CPU WebAssembly paths are described below.
`orttraining` build and test migration is intentionally out of scope.

## Final integrated validation

The combined source snapshot `1b05810e36a22445164c9eab29707ee784fd4167`
includes the C++ tests, language bindings, Android, WebAssembly, CPU contrib,
and package targets. On Linux, frozen wildcard analysis
(`bazel build //... --nobuild`) passed with 220 targets, 52,800 configured
targets, and zero build actions. The following final source-native command
then passed **all five targets**:

```sh
bazel test -c opt //:python_wheel_test //:python_smoke_test //:runtime_sdk_archive_test //:java_native_jar_test //:java_smoke_test --jobs=16 --loading_phase_threads=4 --build_python_zip=false --lockfile_mode=error --test_output=all --nocache_test_results
```

The run took 177.945 seconds and executed 577 actions (571 Linux sandbox,
two Java worker, four internal), reusing 930 unchanged actions. Core runtime
and pybind sources recompiled, and runtime/JNI libraries relinked as needed;
no native binary substitutions were used. This confirms the final integration,
not only the individual package-slice snapshots described below.
Final Windows frozen analysis additionally included all six explicit manual
WASM test targets: 226 targets, 74,380 configured targets, zero build actions.
Platform execution coverage and remaining gaps are detailed below; this does
not claim full-suite, native Apple, or Android device validation.

## CPU contrib operators

CPU contrib operators are opt-in; the established default still defines
`DISABLE_CONTRIB_OPS`. Enable them for a runtime, binding, or test invocation
with `--//bazel:contrib`. This compiles the existing contrib schemas, CPU
kernel registration table, and inference implementations directly from source,
using the same MLAS, Eigen (including unsupported special functions), RE2,
ONNX, and protobuf targets as the standard CPU runtime. No external runtime
or CMake-generated artifact is used.

```sh
bazel build -c opt --//bazel:contrib //:runtime_library
bazel test -c opt --//bazel:contrib //:contrib_quantization_test \
  --test_filter=DynamicQuantizeMatMul.Int8
bazel test -c opt --//bazel:contrib //:contrib_attention_test \
  --test_filter=ContribOpLinearAttentionTest.LinearRule_MultiToken
bazel test -c opt --//bazel:contrib //:contrib_sparse_test \
  --test_filter=SparseToDenseMatMul.TestCsr
```

The selectable tests compile existing sources under `onnxruntime/test/contrib_ops`
and use the existing `OpTester` infrastructure. These targets are incompatible
without the contrib flag, rather than silently running a no-contrib binary.
The attention target includes linear attention and its fused gate tests;
the quantization target includes dynamic quantized matrix multiplication.
The sparse target includes sparse-to-dense matrix multiplication.
This is representative coverage, not migration of the complete contrib test suite.

The source selection in `bazel/contrib.bzl` follows CPU inference categories:
top-level kernels, attention LSTM, BERT/attention, math, MoE, quantization,
sparse, tensor, generation/transformers, and utilities. It includes the
contrib-dependent GQA graph transformers. It does not select CUDA, JS,
WebGPU, or `orttraining` sources. ATen interop is not enabled and its source
directory is excluded; training-only `ShrunkenGather` remains behind its
existing `ENABLE_TRAINING_OPS` guard. The QMoE baseline translation unit
includes its ISA-specific `.cc` implementations, which are declared as textual
inputs instead of compiled twice. Optional MLAS QMoE offload and KleidiAI
configurations are not enabled. Existing platform and feature guards govern
the registered inference kernels; this is not a reduced hand-picked registration
table or a claim that every contrib schema has a CPU implementation.

Linux x86-64 validation used Bazel 8.4.2, GCC 13.2.0, and the frozen module
lock on Azure Linux 3.0. The native source build and this focused invocation
passed **four existing cases** (two attention/gate, one quantization, one sparse),
with zero skipped cases:

```sh
bazel test -c opt --//bazel:contrib --jobs=24 \
  //:contrib_attention_test //:contrib_quantization_test //:contrib_sparse_test \
  --test_filter='ContribOpLinearAttentionTest.LinearRule_MultiToken:ContribOpGatedRMSNormTest.Float_PerHead:DynamicQuantizeMatMul.Int8:SparseToDenseMatMul.TestCsr' \
  --nocache_test_results --test_output=summary
```

The default configuration, with no contrib flag, also source-built and passed
**one existing standard CPU case**, with zero skipped cases:

```sh
bazel test -c opt --jobs=24 //:cpu_provider_test \
  --test_filter=MathOpTest.Add_float --test_output=summary
```

Explicitly requesting `//:contrib_sparse_test` without the flag failed analysis
as an incompatible target, as intended. No full suites were executed. The new
contrib configuration has not been validated on macOS; Windows compilation was
stopped before completion when validation moved to the larger Linux host.

After integration, frozen Windows analysis passed both the default wildcard
graph (214 targets, 16,493 newly configured targets, zero build actions) and
the three enabled contrib targets (9,653 configured targets, zero build actions):

```sh
bazel build //... -c opt --nobuild --lockfile_mode=error --build_python_zip=false --jobs=2
bazel build -c opt --//bazel:contrib //:contrib_attention_test //:contrib_quantization_test //:contrib_sparse_test --nobuild --lockfile_mode=error --build_python_zip=false --jobs=2
```

The shared activation parser is selected once when either Android FP16 Conv or
CPU contrib needs it, rather than duplicated when both flags are selected.
Android-plus-contrib native execution has not been validated.

## Getting started

Install [Bazelisk](https://github.com/bazelbuild/bazelisk), which reads the pinned
version from `.bazelversion`, or install that Bazel version directly. Install
the host's native development tools: GCC/Clang and development headers on Linux,
Visual Studio 2022's Desktop development with C++ workload and Windows SDK on
Windows, or Xcode command-line tools on macOS. Windows also needs Git for Windows
and a Bash location visible to Bazel:

```powershell
$env:BAZEL_SH = 'C:\Program Files\Git\bin\bash.exe'
```

```sh
bazel build -c opt //:runtime_library
bazel test -c opt //:runtime_smoke_test
bazel test -c opt //:common_test --test_filter=NarrowTest.Basic
# Analyze every target without compiling or executing full suites:
bazel build -c opt //... --nobuild
```

These same Bazel commands run natively on each host; Windows does not require
WSL. The selected shared library is `bazel-bin/libonnxruntime.so` on Linux,
`bazel-bin/onnxruntime.dll` on Windows, or `bazel-bin/libonnxruntime.dylib`
on macOS. The public headers remain
under `include/onnxruntime/`; the `//:api_headers` target exports their include
paths. `//:runtime_shared` exposes the shared library to other Bazel C/C++
targets. `//:runtime` is the static runtime library.

Dependencies are pinned in `MODULE.bazel` and `bazel/dependencies.bzl`. Source
archives have SHA-256 checksums. Protobuf sources and configuration headers are
generated by Bazel actions. MLAS ISA flags are confined to dispatched kernel
targets rather than applied to the entire runtime.

Local commands and CI enforce `--lockfile_mode=error`: stale or incomplete
module lock state fails instead of being updated implicitly. After an intentional
dependency change, run `bazel mod deps --lockfile_mode=update`, then review and
commit the resulting `MODULE.bazel.lock` changes. Normal builds and tests must
not override the frozen-lock setting.

### Standalone native SDK archive

```sh
bazel build -c opt //:runtime_sdk_archive
bazel test -c opt //:runtime_sdk_archive_test --test_output=errors
```

`bazel-bin/onnxruntime-sdk.zip` contains a single `onnxruntime/` directory:
standalone public C/C++ session headers in `include/`, the source-built shared
runtime and shared-provider bridge in `lib/` (`bin/` on Windows), the Windows import library
`lib/onnxruntime.lib`, version and license notices, and a small Add inference
example with its model. CUDA-enabled Linux archives also include the
source-built CUDA and shared-provider libraries; CUDA/cuDNN SDK libraries are
not redistributed. Consumers must supply the matching system runtime
dependencies, including the Visual C++ runtime on Windows or CUDA/cuDNN for
GPU execution. The archive is specific to the selected target platform and
build flags, not a universal or officially signed release package. It does
not produce Maven publications, npm, NuGet, or Apple frameworks. The separate CPU
Python wheel and Android AAR targets are described below.

The archive test checks its layout, extracts it to a temporary directory, and
loads that extracted runtime to verify the version through `OrtGetApiBase`.
It does not load a runtime from the Bazel output directory. To compile and run
the packaged example outside Bazel, extract the archive and run from its
`onnxruntime/` directory:

```sh
# Linux:
c++ -std=c++17 -Iinclude examples/sdk_inference_example.cc -Llib -lonnxruntime -Wl,-rpath,'$ORIGIN/lib' -o sdk_inference_example
./sdk_inference_example examples/add_f32.onnx
# macOS: use -Wl,-rpath,@loader_path/lib instead of the Linux rpath.
```

From a Visual Studio x64 developer prompt:

```bat
cl /nologo /EHsc /std:c++17 /Iinclude examples\sdk_inference_example.cc lib\onnxruntime.lib /Fe:bin\sdk_inference_example.exe
bin\sdk_inference_example.exe examples\add_f32.onnx
```

The committed graph passed **one Windows archive test** without substitutions:

```sh
bazel test -c opt //:runtime_sdk_archive_test --build_python_zip=false --test_output=all --nocache_test_results --jobs=8
```

The ZIP was newly assembled from this worktree's cached source-built runtime
and import library; the test extracted it and passed layout/version/runtime-load
checks. Bazel reused 1,764 unchanged source-build actions, rather than rebuilding
the runtime. The Windows-only launcher flag avoids an unnecessary Python ZIP.

Earlier package-only validation temporarily reused a separately verified
source-built native DLL/import library. For that archive, the documented `cl`
command compiled the bundled example using only extracted headers/import library
and passed Add inference using the extracted DLL. That standalone example check
is separate from the source-native archive test above; all temporary graph
substitutions were removed. Subsequent source-native Linux archive validation
is recorded with the wheel results below; macOS and CUDA archive execution
remain unvalidated.

The extracted Linux SDK also passed standalone compilation and Add inference,
using only its headers, library, source, and model. From its `examples/`
directory:

```sh
g++ -std=c++20 -I../include sdk_inference_example.cc -L../lib -lonnxruntime '-Wl,-rpath,$ORIGIN/../lib' -o sdk_inference_example
./sdk_inference_example add_f32.onnx
```

After integrating the desktop shared-provider bridge, the following source-native
Windows command also passed: one SDK archive test and eight Rust selector-runner
regressions. Only the small shared-provider bridge was freshly compiled; the
runtime reused verified source-build actions, with no binary substitutions:

```sh
bazel test -c opt //:runtime_sdk_archive_test //rust:test_runner_test --build_python_zip=false --jobs=2 --local_test_jobs=1 --test_output=all --nocache_test_results
```

### Native Python wheel

```sh
bazel build -c opt //:python_wheel
bazel test -c opt //:python_wheel_test --build_python_zip=false --test_output=all
```

`//:python_wheel` packages the same source-assembled CPU inference API and
source-built extension used by `//:python_runtime_package`. It uses
`rules_python`'s wheel rule directly, not `setup.py`, `build.py`, or an
external wheel download. NumPy remains a declared dependency rather than
being bundled. The wheel contains versioned metadata, native CPython 3.12
ABI/platform tags, license notices, and a hashed `RECORD`. Its explicit
version in `bazel/python_wheel.bzl` must match `VERSION_NUMBER`; the package
test rejects a mismatch.

The initial wheel targets are Windows x86-64 (`win_amd64`) and Linux x86-64
(`linux_x86_64`). Linux is deliberately **not** tagged `manylinux`: that would
require an appropriate build environment and dependency audit/repair.
macOS deployment-target tagging and CUDA provider packaging are not enabled
for this wheel. This is the CPU inference package, not the complete optional
Python tooling/transformers distribution. Training is intentionally excluded.

For example, from a dedicated CPython 3.12 virtual environment on Windows:

```powershell
python -m pip install .\bazel-bin\onnxruntime-1.32.0-cp312-cp312-win_amd64.whl
```

Use the corresponding `linux_x86_64` filename on Linux. Installation needs
the normal platform runtime dependencies, including the Visual C++ runtime
on Windows. `//:python_wheel_test` verifies metadata, every `RECORD` hash and
size, absence of bundled dependencies/test sources, and high-level
`InferenceSession` Add inference from the wheel extracted into a temporary
directory. A fresh subprocess verifies it imported that extracted package,
not the source/runfiles package.

Non-minimal Python initializes the shared-provider bridge even in a CPU-only
build. The native graph therefore builds that small library on desktop
platforms without enabling CUDA, exports `Provider_GetHost` and
`Provider_SetHost` on Windows, and installs it beside the Python extension.
The SDK archive also includes it. The wheel test rejects the previously
observed `Init provider bridge failed` import warning instead of suppressing
it. CUDA's existing shared-provider target uses the same library.

Windows validation: `bazel test -c opt //:python_wheel_test
--build_python_zip=false --test_output=all --nocache_test_results` passed one
test. The wheel used a temporary local substitution of the previously
source-built native extension, while its shared-provider bridge was freshly
compiled from source. An offline `pip install --no-deps --target ...` of that
wheel also passed high-level Add inference under the managed CPython 3.12
interpreter with the declared NumPy dependency; the bridge initialization
warning was absent. The native extension substitution was then removed.
`bazel build -c opt //:python_wheel_test //:runtime_sdk_archive_test --nobuild
--jobs=2 --loading_phase_threads=2` passed analysis of the final source-extension
and SDK graphs with zero build actions. That Windows package-only check did not
rebuild the full native extension. The integrated Windows worktree has no warm
native Python-extension objects, so no additional heavy local extension build
was started.

Subsequent **fresh source-native Linux x86-64 validation** of integrated commit
`43bef85f0d` passed all three targets below, using a task-private repository cache:

```sh
bazel test -c opt //:python_wheel_test //:python_smoke_test //:runtime_sdk_archive_test --jobs=16 --loading_phase_threads=4 --build_python_zip=false --lockfile_mode=error --test_output=all --nocache_test_results
```

The build executed 1,462 actions (1,138 Linux sandbox actions), compiling the
runtime, pybind extension, and shared-provider bridge from source without
substitutions. Wheel metadata/every RECORD hash/isolated high-level Add
inference, low-level NumPy Add inference, and extracted SDK runtime loading
all passed; no bridge initialization warning occurred. This is not complete
Python suite or manylinux publication validation.

An offline `pip install --no-index --no-deps` of the generated Linux wheel
into an isolated managed CPython 3.12 virtual environment also passed
high-level Add inference, using NumPy from the pinned Bazel dependency.

The consolidated source graph, including Apple, Rust, wheel, SDK, and C++ test
targets, passed frozen-lock Windows analysis without executing build actions:

```sh
bazel build //... -c opt --nobuild --lockfile_mode=error --jobs=2
```

This analyzed 196 targets (60,443 configured targets), with zero build actions.
It does not establish native Apple, Rust, or wheel execution.

### Android ARM64 runtime, JNI, and AAR

On a **Linux x86-64 host**, install Android NDK **r27d (27.3.13750724)** and
set `ANDROID_NDK_HOME` to its root. The official Linux archive is
`https://dl.google.com/android/repository/android-ndk-r27d-linux.zip`
(SHA-256 `601246087a682d1944e1e16dd85bc6e49560fe8b6d61255be2829178c8ed15d9`).
The pinned `rules_android_ndk` 0.1.5 toolchain uses Android API **27** and
the NDK's static libc++ runtime. A Java or Android SDK/Gradle installation is
not required: Bazel obtains the host JDK and compiles the Java sources.

```sh
export ANDROID_NDK_HOME=/path/to/android-ndk-r27d
bazel build --config=android_arm64 -c opt //:android_aar --jobs=16
# Check real cross-compiled artifact structure on the host (not device execution):
python3 tools/bazel/android_aar_verify.py bazel-bin/onnxruntime-android-arm64.aar
# Portable Java and packaging regressions use the HOST platform, not the Android config:
bazel test //:android_java_util_test //:android_aar_packager_test --jobs=16
```

`bazel-bin/onnxruntime-android-arm64.aar` is a single-ABI Android library,
with `classes.jar`, a manifest declaring `minSdkVersion=27`, consumer R8/ProGuard
keep rules, `jni/arm64-v8a/libonnxruntime.so`,
`jni/arm64-v8a/libonnxruntime4j_jni.so`, standalone C/C++ API headers in
`headers/`, and license/third-party/version assets in `META-INF/`.
Java compilation uses `java/src/main/java` and `java/src/main/android`,
**not** the desktop JVM FP16 implementation. JNI C sources are compiled
against the NDK's `jni.h`, using headers generated from those Java classes.
Both native shared objects have 16 KB ELF load-segment alignment; the final
APK's ZIP alignment and Java desugaring remain the consuming Android
application build's responsibility.

The individual cross-build targets are `//:android/libonnxruntime.so` and
`//:android/libonnxruntime4j_jni.so`; `//:android_runtime_shared` exposes the
native library to Bazel C/C++ consumers. `//:android_java` can also be compiled
on the host. Android toolchain selection is **opt-in**: no Android defines,
NDK registration, or ARM ISA flags are applied to desktop targets. ARM64
MLAS uses baseline NEON with separately compiled FP16, dot-product, i8mm,
and bfloat16 kernels rather than host x86 code. Standard FP16 Conv also links
the shared activation attribute parser; this does not enable contrib ops.

Validation on Linux x86-64 with Bazel 8.4.2 and NDK r27d:
the exact build command above **passed with actual C/C++/assembly/JNI/Java
compilation and linking** under the frozen module lock. The verifier command
**passed**, checking ARM64 ELF headers, 16 KB load segments, archive assets,
and Android Java classes. NDK `llvm-nm -D --defined-only` confirmed
`OrtGetApiBase` in the runtime and
`Java_ai_onnxruntime_OnnxRuntime_initialiseAPIBase` in JNI.
`llvm-readelf -d` confirmed that JNI depends on `libonnxruntime.so`, and
both libraries require only Android system libraries otherwise (no
`libc++_shared.so` to bundle). The host test command **passed** (three JUnit
utility tests and three packaging regressions). With `ANDROID_NDK_HOME`
unset, `bazel build //... --nobuild --jobs=16` **passed desktop analysis**;
that check is not a desktop compilation or test run.

After integration with the Apple/Rust/wheel targets, Windows desktop analysis
also passed with no NDK configured: `bazel build //... -c opt --nobuild
--lockfile_mode=error --build_python_zip=false --jobs=2` analyzed 210 targets
(60,516 configured targets), with zero build actions. This host-only command
then compiled the 37 Android Java sources and passed one selected existing
JUnit case plus three AAR packager regressions, without native compilation:

```sh
bazel test -c opt //:android_java_util_test //:android_aar_packager_test --test_filter="ai.onnxruntime.UtilTest#reshapeTest" --build_python_zip=false --jobs=2 --local_test_jobs=1 --test_output=all --nocache_test_results
```

No emulator/device execution was performed; neither tool was available on the
validation host. This slice does not include NNAPI/QNN, CUDA, training,
telemetry, additional ABIs, Maven publication, Android instrumentation tests,
or Windows/macOS-hosted Android cross-build validation. Contrib ops remain
disabled by default.

**React Native Android is not implemented by this AAR.** Its actual bridge
needs source-built ReactAndroid/JSI, CallInvoker and fbjni, Java React bridge
and AndroidX APIs, plus `cpp-adapter.cpp` and the shared JSI C++ sources
listed in `js/react_native/android/CMakeLists.txt`. Those dependencies do not
have a source-native Bazel graph here. The Java/JNI AAR neither builds
`libonnxruntimejsi.so` nor contains `OnnxruntimeModule`/`OnnxruntimePackage`;
it must not be advertised as the React Native package.
### Standalone Java/JNI JAR

```sh
bazel build -c opt //:java_native_jar
bazel test -c opt //:java_native_jar_test --test_output=all
```

`bazel-bin/onnxruntime-native.jar` includes the existing inference Java classes,
the native runtime, JNI library, shared-provider bridge, version and license
notices. It uses Bazel's single-JAR assembler, not Gradle or a downloaded ORT
JAR. This is a single-platform CPU distribution, not a universal or CUDA
package. Its native resources use the existing
`ai/onnxruntime/native/<os>-<architecture>/` layout, so the existing Java loader
extracts and loads the packaged libraries without explicit native paths.

The package test compiles the inference smoke consumer against only the
assembled JAR, clears `java.library.path`, and supplies only a model path.
Unlike the existing three-argument JNI smoke test, it sets no native-library
overrides. Windows, Linux and macOS resource names are defined; native
execution evidence is recorded separately rather than inferred from analysis.

On the Linux x86-64 VM, the following frozen-lock command passed both tests
in 13.505 seconds, with 42 actions. The JNI library and 37 existing Java sources
were compiled from source; the unchanged native runtime reused the source
build from the preceding Python wheel/SDK check:

```sh
bazel test -c opt //:java_native_jar_test //:java_smoke_test --jobs=16 --loading_phase_threads=4 --build_python_zip=false --lockfile_mode=error --test_output=all --nocache_test_results
```

An out-of-Bazel check also passed from a directory containing only the JAR,
`RuntimeSmokeTest.java`, and `add_f32.onnx`, using the managed JDK 11:

```sh
java -Djava.library.path= --class-path onnxruntime-native.jar RuntimeSmokeTest.java add_f32.onnx
```

The Java test exercises the resource-loading path; the existing three-argument
smoke test additionally confirms that explicit native paths still work.
Windows and macOS package execution and Maven publication remain unverified.

After integration with the platform and test layers, frozen Windows
`bazel build //... -c opt --nobuild --lockfile_mode=error
--build_python_zip=false --jobs=2` passed analysis of 217 targets (60,530
configured targets), with zero build actions. No fresh Windows JNI/runtime
build was started for this package check.

### Desktop portability checks

Linux x86-64 portability validation used Bazel 8.4.2 with the frozen module
lock and freshly source-built native artifacts. This focused command passed
all five smoke targets:

```sh
bazel test -c opt //:runtime_smoke_test //:java_smoke_test //:python_smoke_test //:node_smoke_test //:csharp_smoke_test
```

The following commands each passed **one existing selected case** against
those source-built Linux artifacts; no full suite was run:

```sh
bazel test -c opt //:python_inference_test --test_filter=onnxruntime_test_python.TestInferenceSession.test_run_model
bazel test -c opt //:java_tensor_test --test_filter=ai.onnxruntime.OnnxTensorTest#testArrayCreation
bazel test -c opt //:csharp_tests --test_filter=Microsoft.ML.OnnxRuntime.Tests.InferenceTest.CanCreateAndDisposeSessionWithModelPath
bazel test -c opt //:node_model_metadata_test --test_filter='symbol.shape$'
```

The MLAS baseline archive is retained when linking because dispatched ISA
archives call back into baseline packing and quantization helpers. macOS
logging uses an Objective-C++ library, and Intel assembly selects Mach-O
constant sections on Apple targets. Clang 18 on Linux successfully
cross-assembled all 40 selected Intel MLAS assembly files as Mach-O objects;
this is not native macOS build or execution validation. No macOS host was
available, so Intel and Apple Silicon native validation remain pending.
Windows smoke-target analysis (`bazel build -c opt //:runtime_smoke_test
--nobuild`) passed; this portability slice did not rebuild the Windows runtime.

## WebAssembly CPU (experimental)

The opt-in WebAssembly graph compiles the same runtime sources with the pinned
Emscripten 4.0.19 Bazel toolchain. It selects wasm32, POSIX platform sources,
single-threaded MLAS, C++ exceptions, and either scalar kernels or fixed SIMD128
kernels. It does not compile host x86/ARM MLAS kernels for the target. SDK source
and compiler archives are checksummed; no prebuilt ORT WASM or npm runtime is used.
The WebAssembly targets are `manual`, and their underlying C++ binaries are
incompatible with desktop platforms, so normal `//...` remains a desktop build.

```sh
bazel build -c opt //:wasm_scalar //:wasm_simd
bazel build -c opt //:web_scalar_package //:web_simd_package
bazel test -c opt //:wasm_scalar_test --test_filter=WebAssemblyTest.test
bazel test -c opt //:wasm_simd_test --test_filter=WebAssemblyTest.test
bazel test -c opt //:web_scalar_smoke_test //:web_simd_smoke_test
bazel test -c opt //:web_scalar_browser_test //:web_simd_browser_test \
  --test_env=ORT_BAZEL_BROWSER=/absolute/path/to/chromium
```

Windows needs the Python 3 launcher (`py -3`) required by the SDK's compiler
driver, in addition to the desktop prerequisites above. With a per-user Python
installation, pass `--action_env=PATH` so the strict action environment can find
that launcher. Use a short isolated output root to avoid MSVC response-file path
limits while Bazel builds the host `protoc`, for example:

```powershell
bazel --output_user_root=C:\b\ort-wasm test -c opt //:wasm_scalar_test --test_filter=WebAssemblyTest.test --action_env=PATH
```

The package targets source-assemble the checked-in common and WebAssembly
TypeScript frontend and emit `ort-web.js`, `ort_wasm.mjs`, and `ort_wasm.wasm` in
`bazel-bin/web_{scalar,simd}_package/`. In a browser, serve those three assets
together and load `ort-web.js` as a classic script; it exposes the common
`ort.Tensor`, `ort.InferenceSession`, and LoRA APIs. In Node, `require()` the same
`ort-web.js`. The frontend defaults to one thread and selects its companion
native-compiled module, rather than a release CDN artifact. All CPU inference
functions declared in `onnxruntime/wasm/api.h` are retained through the original
`EMSCRIPTEN_KEEPALIVE` declarations. The original `pre.js` and `js_post_js.js` are
linker inputs, preserving external-data mounting and the pointer-size contract.

The `wasm_*_test` targets compile the existing
`onnxruntime/test/wasm/test_inference.cc` and preload its checked-in
`mul_1.onnx` fixture. Gtest case selection is forwarded to the WASM executable;
an invalid or zero-match selection fails. The frontend and browser smoke targets
independently run Mul and MatMul through `ort.InferenceSession`, checking shapes
`[3, 2]` and `[3, 1]` and every output value. The Node smoke also checks invalid-model
rejection and the declared CPU inference API exports. The browser test requires Chromium/Edge and fails
explicitly if none is configured; on Windows it also detects the default Edge
installation. Each browser invocation uses an isolated test profile.

Linux x86-64 validation used Bazel 8.4.2, Emscripten 4.0.19, managed Node
24.22.1, and the frozen module lock. These commands passed with `--jobs=24`:

```sh
bazel test -c opt //:wasm_scalar_test //:wasm_simd_test \
  //:web_scalar_smoke_test //:web_simd_smoke_test
bazel test -c opt //:wasm_scalar_test //:wasm_simd_test \
  --test_filter=WebAssemblyTest.test
bazel test -c opt //:runtime_smoke_test \
  //:web_scalar_smoke_test //:web_simd_smoke_test
bazel test -c opt //:web_scalar_browser_test //:web_simd_browser_test \
  --test_env=ORT_BAZEL_BROWSER=/absolute/path/to/chrome-headless-shell \
  --test_env=LD_LIBRARY_PATH
```

Both browser variants executed Mul and MatMul in Chrome for Testing
157.0.8103.38 with its sandbox enabled. Missing Azure Linux browser libraries
were downloaded/extracted into the task's tools directory, not installed
globally; `LD_LIBRARY_PATH` exposed those libraries to the browser. The
`runtime_smoke_test` command freshly rebuilt and executed the native desktop
runtime as a regression check. Selecting `DefinitelyNoSuchCase` or `-*` on
`//:wasm_scalar_test` failed with Bazel exit 3, as intended. No CMake, `build.py`,
prebuilt ORT runtime, or release npm ORT package was used.

After merging Android, CPU contrib, Rust, Apple, and packaging layers, the
following frozen Windows analysis also passed: 226 targets, 74,380 configured
targets, zero build actions. It checks the explicit manual WASM targets as well
as the desktop wildcard, without claiming Windows WASM execution:

```sh
bazel build //... //:wasm_scalar_test //:wasm_simd_test //:web_scalar_smoke_test //:web_simd_smoke_test //:web_scalar_browser_test //:web_simd_browser_test -c opt --nobuild --lockfile_mode=error --build_python_zip=false --jobs=2
```

Windows compilation reached native Emscripten source compilation but was
stopped for shared-host resource scheduling; it is not an end-to-end Windows
build validation. Local managed Edge redirected fresh profiles to
`edge://force-signin`, so browser validation ran on Linux rather than bypassing
that policy.

This is a CPU inference slice, not complete `onnxruntime-web` release packaging.
Threading, worker proxying, relaxed SIMD, memory64, JSPI/Asyncify, WebGPU, WebNN,
JSEP, WebGL, XNNPACK, training, and the complete JavaScript Web test suite are
not enabled. Requests for threading or proxy mode fail instead of silently
running single-threaded. Browser GPU backends are not registered. Contrib
operators remain excluded, matching the native CPU graph.

## Language extensions

```sh
bazel test -c opt \
  //:go_test \
  //:java_smoke_test \
  //:python_smoke_test \
  //:node_smoke_test \
  //:csharp_smoke_test
```

| Surface | Native Bazel targets | Current scope |
| --- | --- | --- |
| C/C++ | `//:runtime`, `//:runtime_library`, `//:runtime_smoke_test` | Standard ONNX CPU operators and optional `--//bazel:contrib` CPU inference kernels; inference through the public C++ API. |
| Python | `//:python_native_library`, `//:python_runtime_package`, `//:python_tests`, `//:python_wheel` | CPython 3.12 extension, source-assembled high-level CPU inference package, selected existing unittest cases, and CPU Windows/Linux x86-64 wheels. Not a complete Python suite/tooling migration. |
| Java/JNI | `//:java_binding`, `//:java_native_library`, `//:java_smoke_test`, `//:java_native_jar` | Java and JNI sources; JNI headers come from Bazel's Java compiler. Standalone single-platform CPU JAR includes native libraries and uses the existing resource loader. No Gradle invocation; not Maven publication. |
| Android Java/JNI | `//:android_java`, `//:android_aar` | Android Java sources, source-built ARM64 runtime/JNI, and single-ABI AAR; Linux-host cross-build validated. Not React Native or device execution. |
| Go | `//:go_binding`, `//:go_test` | First-party cgo binding and existing fixture-based unit tests. Tests explicitly load the Bazel-built shared library; they do not take the binding's missing-library skip path. External Qwen model tests are excluded. |
| Node.js | `//:onnxruntime_binding.node`, `//:node_js_package`, `//:node_test` | Native Node-API extension, source-assembled common/Node JavaScript packages, and four existing test files with file/case selection. Not release npm packaging or the complete Node suite. |
| C# | `//:csharp_binding`, `//:csharp_smoke_test`, `//:csharp_tests` | .NET 8 binding and existing desktop test sources compiled directly with `rules_dotnet`, exercised on .NET 10. Native xUnit discovery/execution supports exact class/method selection, theories and fixtures. Training APIs are not enabled. Not a NuGet package. |
| Objective-C / Swift import surface | `//objectivec:onnxruntime_objc`, `//objectivec:inference_tests` | Native ARC Objective-C++ inference wrappers, an explicit `onnxruntime_objc` Clang module, and existing hostless XCTest cases/fixtures. macOS only; Apple compilation/execution and Swift import are not yet validated. Not an iOS framework, XCFramework, or Swift package. |
| JavaScript Web / WebAssembly | `//:wasm_scalar`, `//:wasm_simd`, `//:web_scalar_package`, `//:web_simd_package` | Source-native Emscripten CPU runtime and source-assembled frontend; selectable existing WASM inference test and Node/browser smoke targets. Single-threaded scalar/fixed SIMD only, not the complete release or Web test suite. |
| React Native | Not implemented | Requires the Android/iOS cross-compilation and packaging graphs plus the React Native bridge. A working Node addon is not evidence of mobile support. |
| Rust | `//rust/onnxruntime-sys:onnxruntime_sys`, `//rust/onnxruntime:onnxruntime`, `//rust:smoke_test` | Both existing first-party crates, current-header bindgen generation, and selected existing unit/integration tests. Native `rules_rust` compilation and explicit loading of the source-built runtime; no Cargo build or legacy native acquisition. Not crates.io packaging. |

The language toolchains are managed by their Bazel rules, including a pinned
Node.js 24 LTS runtime. Windows cgo uses a pinned LLVM-MinGW compiler in a
separate Bazel platform configuration; ORT itself still uses MSVC and MASM.
Windows Java tests use the managed JDK 25 runtime: the managed JDK 11 bundles
an older MSVC runtime that crashes when loading this MSVC-built native library.
There is no global MinGW installation or compiler switch. Python dependencies
are listed in `bazel/requirements.txt`, without a separate wheel-hash lock;
pip selects the host-compatible wheel from the configured index. Building native extensions
is intentionally separate from producing release packages.

The smoke tests use the checked-in `go/testdata/add_f32.onnx` model and verify
its `[2, 3]` output and every expected value, not just library loading. The
C++ test additionally creates MatMul and Conv models in memory, exercising
MLAS on CPU and cuBLAS/cuDNN on CUDA.

### Apple Objective-C inference bindings

On a macOS host with full Xcode selected (including the macOS SDK and XCTest):

```sh
bazel build -c opt //objectivec:onnxruntime_objc
bazel test -c opt //objectivec:inference_tests --test_filter=ORTSessionTest/testInitAndRunOk
bazel test -c opt //objectivec:inference_tests --test_filter=ORTValueTest
bazel test -c opt //objectivec:inference_tests --test_filter=ORTModuleImportTest/testInferenceModuleImport
```

These are **commands awaiting Apple-host validation**, not passing results.
`objc_library` compiles the existing ARC `.mm` sources and links the source-built
`//:runtime`; `rules_apple` builds the XCTest bundle. A hostless runner invokes
`xcrun xctest` directly, without generating an Xcode project or invoking
CMake, CocoaPods, `build.py`, or `xcodebuild`. Filters accept an exact XCTest
class or `Class/testMethod`; invalid filters and zero executed tests fail.
The bundle includes the existing `single_add.basic.ort` and
`identity_string.ort` fixtures at its resource root for `NSBundle` lookup.
Tests cover the existing environment, value, session, and error utilities.
The module-import test additionally compiles the public inference umbrella.

The explicit module map exports only the inference umbrella and its headers.
It is a Clang module import surface intended for Swift consumers, not a
first-party Swift implementation or evidence of Swift compatibility. This
repository has no standalone Swift inference wrapper to compile in this slice.
Checkpoint/training headers, implementations, tests, and fixtures are excluded.
CoreML's provider-factory include directory is deliberately not exposed:
its wrapper takes the existing unavailable-provider error path. XNNPACK uses
the generic provider-registration C API, which reports its absence at runtime.
Neither optional provider is enabled by these targets.

Apple libraries and XCTest bundles are incompatible on non-macOS platforms,
so wildcard builds do not require an Apple SDK on Windows/Linux. The portable
runner contract tests use a fake `xcrun`; they do not compile or execute Apple
code:

```sh
bazel test //objectivec:test_runner_test
```

`rules_apple` is pinned to the already resolved 3.16.0 release to preserve the
existing non-Apple toolchain graph. Upgrading to 4.2.0 was rejected because its
transitive Swift 2.4 toolchain required `swiftc.exe` even for unrelated Windows
tests. A one-line source patch forwards test platform compatibility to that
release's generated internal bundle target too, preventing wildcard builds
from resolving Apple toolchains on non-Apple hosts. The normal frozen-lock
policy is unchanged.

Validation for this slice used Windows and task-isolated Bazel output roots.
The unzipped Python launcher requires a short output root on Windows: a deeply
nested session path hit `WinError 206` before executing the tests.

| Command | Result |
| --- | --- |
| `bazel mod deps --lockfile_mode=update` | Completed; the direct pin reuses the already resolved dependency, with no semantic lockfile change. |
| `bazel build //... --nobuild` | Passed analysis of all 101 targets; no native compilation or full test suite was run. |
| `bazel build //objectivec/... //:runtime_smoke_test --nobuild --skip_incompatible_explicit_targets` | Passed. Configured-query inspection also confirmed the Objective-C library, test sources, XCTest target, and generated internal bundle are incompatible on Windows. |
| `bazel test //objectivec:test_runner_test --build_python_zip=no --test_output=all` | Passed all seven mocked-runner regressions with a short isolated output root and the frozen module lock. |
| `python objectivec\test\apple_test_runner_test.py -v` | Passed all seven mocked-runner regressions: class/case filters, archive extraction, test environment, failure status, invalid filters, and zero matches. |

No authenticated macOS execution environment was available. Compilation and
linking against the Apple SDK, actual XCTest execution, and Swift import
remain unverified on both Intel and Apple Silicon. Windows graph analysis
and mocked-runner results are not substitutes for those checks. No machine
was provisioned or CI dispatched for this slice.

### Existing desktop C# tests

`//:csharp_tests` compiles the existing common test sources (excluding training)
and the desktop `InferenceTest.netcore.cs` partial class, without reading or
executing project files. SHA-256-pinned public NuGet archives in
`bazel/csharp_dependencies.bzl` supply the existing xUnit 2.9 assertions and
discovery/execution engine, SkippableFact, protobuf and JSON dependencies.
The runner delegates test execution to xUnit, including constructors,
`ITestOutputHelper`, collection fixtures, async tests and theory data; it does
not replace assertions or invoke test methods via a custom reflection loop.
Checked-in models and tensor data are embedded under the original logical
resource names and staged into Bazel's isolated writable test directory for
tests that open files by name.

Select the Bazel target, then an exact fully qualified class or class.method:

```sh
bazel test -c opt //:csharp_tests \
  --test_filter=Microsoft.ML.OnnxRuntime.Tests.OrtFloat16Tests
bazel test -c opt //:csharp_tests \
  --test_filter=Microsoft.ML.OnnxRuntime.Tensors.Tests.TensorTests.ConstructTensorFromArrayRank1
bazel test -c opt //:csharp_tests \
  --test_filter=Microsoft.ML.OnnxRuntime.Tests.InferenceTest.RunInferenceUsingPreAllocatedOutputsAndDictionary
bazel test -c opt //:csharp_tests \
  --test_filter=Microsoft.ML.OnnxRuntime.Tests.InferenceTest.CanCreateAndDisposeSessionWithModelPath
bazel test -c opt //:csharp_tests \
  --test_filter=Microsoft.ML.OnnxRuntime.Tests.OrtValueTests.PopulateAndReadStringTensor
```

Filters use source method names, not `Fact.DisplayName`; wildcards and VSTest
filter expressions are not supported. An unknown class/method, malformed
selection, discovery/execution error, failed assertion, or all-skipped/zero-run
selection fails. With no filter, xUnit selects all discovered tests. The target
is tagged `manual` so `//...` does not accidentally run the entire desktop suite.

Windows validation freshly compiled the binding, runner and existing tests
through Bazel 8.4.2 with `--lockfile_mode=error`. The five commands above passed
9, 2 (both MemberData rows), 1, 1 and 1 existing tests respectively: **14 total**.
Discovery reported 228 cases before deferred theory expansion. Unknown-class
and unknown-method selections both failed (Bazel exit 3). Selecting
`InferenceTest.TestModelSequenceOfMapIntFloat` with
`--test_env=DisableMlOps=ON` also failed with one skipped test and zero executed
non-skipped tests, as required. No full C# suite was
run. To avoid rebuilding unchanged native code, these local runs temporarily
substituted the previously Bazel-built and verified Windows native DLL; that
substitution and binary are not committed. The committed test dependency is
the source-built `//:runtime_library`, not an installed/prebuilt runtime.
This validates fresh managed compilation and native integration, not a fresh
native runtime rebuild or Linux/macOS execution of this target.

Unported desktop prerequisites remain explicit: downloaded pretrained-model
corpora and their legacy `Properties.txt` locator, custom-op and plugin-EP
shared libraries, provider-specific GPU/DirectML configuration, and release
NuGet/package end-to-end testing. Their source tests remain compiled/discoverable
but selecting one does not manufacture missing fixtures or silently pass.
WinML/UWP, device/mobile/MAUI and BrowserStack projects need separate platform
graphs. Training build/test migration is intentionally out of scope.

### Existing Rust crates and tests

`rules_rust` compiles the existing `rust/onnxruntime-sys` and
`rust/onnxruntime` sources directly. `rust_bindgen` generates FFI from the
checked-in C API headers using a checksum-pinned public libclang archive.
An explicit patch to `rules_rust_bindgen` connects its vendored Windows import
library to the bindgen CLI's native link dependencies. Generated bindings
include compile-time layout assertions, rather than separate libtest cases.
The existing Cargo `build.rs` (download/system/CMake native acquisition) is
not executed. The high-level target enables its existing `model-fetching`
feature so the existing integration test source compiles, but the default
selection uses only checked-in fixtures and does not download models.
Cargo manifests remain the dependency source of truth; crate_universe uses
Cargo for dependency metadata/resolution, not for building either crate.
`rust/Cargo.lock` and `rust/Cargo.Bazel.lock` pin resolved crates, checksums,
and Bazel renderings. Intentional Rust dependency changes require
`CARGO_BAZEL_REPIN=true bazel mod deps --lockfile_mode=update` before reviewing
and committing both Rust locks and `MODULE.bazel.lock`.

Windows needs a short output base for Cargo dependency build-script working
directories; Windows `CreateProcess` rejects overlong working-directory paths
even when Rust can read individual long paths. For example, prepend
`--output_base=C:\b\ort-rust` immediately after `bazel`. Use a separate output
base for each checkout.

```sh
bazel test -c opt //rust:smoke_test
bazel test -c opt //rust/onnxruntime:selected_unit_tests
bazel test -c opt //rust/onnxruntime:selected_integration_tests --test_filter=download::upsample
bazel test -c opt //rust/onnxruntime:selected_unit_tests --test_filter=environment::tests::sequential_environment_creation
bazel test -c opt //rust:test_runner_test
```

The smoke target exercises the generated current C API and high-level Add
inference, checking output shape `[2, 3]` and all six values. The unit target
defaults to three existing cases (environment lifecycle, softmax, and native
integer tensor construction). The integration target defaults to the existing
`download::upsample` case, which uses the checked-in model and image despite
its module name. `--test_filter` accepts an exact libtest name, not a Cargo
command or wildcard. Unknown selections, zero executed tests, ignored-only
selections, discovery errors and assertion failures fail. Internal raw libtest
targets are tagged `manual`; use the selection targets for these guarantees.
Downloaded model-zoo integration cases, GPU behavior, release packaging and
training remain outside this bounded validation.

Windows x86-64 validation passed the following bounded command with the frozen
module lock: two smoke cases, three existing unit cases, one existing integration
case, and eight selection-runner regression cases:

```powershell
bazel --output_base=C:\b\orr-f12c test -c opt //rust:smoke_test //rust:test_runner_test //rust/onnxruntime:selected_unit_tests //rust/onnxruntime:selected_integration_tests --build_python_zip=false
```

Both crates and their FFI were compiled from source against the current headers.
To avoid repeating the unchanged native runtime compilation, this execution used
a temporary, uncommitted substitution of the previously verified native
Bazel-source-built `onnxruntime.dll`. That substitution was removed afterward;
the committed graph depends on `//:runtime_library`, not an installed or
checked-in binary. A subsequent `build --nobuild` checked the restored
source-runtime graph. Thus this is executed Rust inference/test evidence, not a
fresh end-to-end native-runtime rebuild in the Rust checkout.

The same bounded unit target with `--test_filter=not_a_test` failed as expected
(Bazel exit 3), listing available exact names. Runner regressions also verified
empty discovery, zero execution, ignored-only execution and failed processes.
Rust execution on Linux/macOS, the full existing Rust suites and remote Rust
compilation have not been validated.

Restored source-graph validation (passed without compiling native code):

```powershell
bazel --output_base=C:\b\orr-f12c build -c opt //rust:smoke_test //rust/onnxruntime:selected_unit_tests //rust/onnxruntime:selected_integration_tests --nobuild --build_python_zip=false --jobs=1
```

## CUDA

CUDA is opt-in and currently restricted to Linux x86-64. Native desktop CPU
support does not imply Windows CUDA support. CPU builds do not need a CUDA toolkit or cuDNN installation.
Use a local development toolkit and a cuDNN SDK with `include/cudnn.h`:

```sh
export CUDA_PATH=/usr/local/cuda
export CUDNN_PATH=/path/to/cudnn
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"

bazel test -c opt \
  --//bazel:cuda \
  --@rules_cuda//cuda:archs=sm_90 \
  --test_env=LD_LIBRARY_PATH \
  //:runtime_cuda_smoke_test

bazel build -c opt \
  --//bazel:cuda \
  --@rules_cuda//cuda:archs=sm_90 \
  //:libonnxruntime.so \
  //:libonnxruntime_providers_shared.so \
  //:libonnxruntime_providers_cuda.so
```

Choose architectures appropriate for the GPU and toolkit; `sm_90` is the H200
example, not a universal deployment setting. `rules_cuda` compiles `.cu`
translation units directly. cuDNN and cuFFT are dynamically loaded using ORT's
existing loaders. Preserve all three ORT shared libraries together, and make
the CUDA/cuDNN runtime libraries discoverable.

The CUDA smoke test disables CPU execution-provider fallback. Passing it
therefore requires GPU execution of Add, MatMul, and Conv, not merely detection of an
installed GPU. GPU tests are tagged `manual`/`requires-gpu` and must be requested
explicitly.

The following combinations have executed the fallback-disabled CUDA test on
an NVIDIA H200 (`sm_90`), driver 580.159.04, with GCC 13.2 on Azure Linux 3:

| Toolkit | cuDNN | Result |
| --- | --- | --- |
| CUDA 12.8, NVCC 12.8.93 | 9.14.0.64 | Add, MatMul, and Conv passed on GPU. |
| CUDA 13.3, NVCC 13.3.73 | 9.27.0.42 | Add, MatMul, and Conv passed on GPU. |

These are tested combinations, not a claim that every CUDA 12/13 minor release,
GPU architecture, or driver/cuDNN combination has passed. In particular,
CUDA 13.0 has not been validated with this graph. The existing Abseil NVCC 13.3
source workaround is included.

The provider links with `--no-undefined` and exports only its `GetProvider`
entry point, so missing implementation dependencies fail the build rather
than surfacing later during provider loading.

Contrib CUDA kernels and optional FlashAttention, memory-efficient attention,
and XQA paths are not enabled. Some shared attention infrastructure is compiled
because standard ONNX operators use it; this does not register contrib ops.

## Tests and remaining migration

### Existing C++ suites

The native C++ test layer reuses existing GoogleTest sources and ORT's existing
test environment, OpTester, graph builders, allocators, and comparison helpers.
It does not replace these suites with smoke tests:

```sh
# Build and run only one binary and one existing test case:
bazel test -c opt //:graph_test --test_filter=GraphViewer.FilteredGraph
# Select multiple cases from one binary (quote patterns containing wildcards):
bazel test -c opt //:cpu_provider_test --test_filter="MathOpTest.MatMulFloatType:ConvTest.Conv2D_1"
# List cases without executing them:
bazel test -c opt //:session_test --test_arg=--gtest_list_tests --test_output=all
```

`//:core_tests` groups `//:framework_test`, `//:session_test`, `//:graph_test`,
`//:optimizer_test`, `//:cpu_provider_test`, `//:runtime_optimizer_test`,
`//:platform_test`, `//:common_runtime_test`, `//:logging_test`, and
`//:provider_infrastructure_test`, `//:flatbuffers_test`, `//:quantization_test`,
`//:lora_test`, `//:opaque_api_test`, `//:ep_graph_test`, and
`//:internal_testing_ep_test`. Session tests are sourced from
`test/framework/`; graph/IR tests are sourced from `test/ir/`. The provider
target uses the existing `test/providers/cpu/` operator tests, not contrib or
other EP suites. These targets statically link the native CPU runtime.
Selecting a target avoids building other test binaries. `--test_filter` selects
GoogleTest suites/cases at runtime; it does not reduce compilation within the
selected binary. `--test_arg=--gtest_filter=Suite.Case` is also supported.
Request `//:core_tests` only when intentionally running all sixteen binaries;
`//...` includes these suites as well as the desktop binding tests.

Checked-in `onnxruntime/test/testdata/` files are declared as Bazel runfiles.
The test entry point copies them into each test's writable `TEST_TMPDIR` before
calling the existing ORT test main. Relative fixture paths are preserved, and
tests that save or modify models cannot write into the checkout or another
test's fixtures.

The runtime's existing `DISABLE_CONTRIB_OPS` configuration and test-source
feature guards remain in effect. The contrib GQA layout transformer test is
excluded because the foundation does not compile that transformer.
`test/providers/cpu/model_tests.cc` is excluded because its ONNX model-corpus
runner needs externally downloaded models and separate runner/protobuf targets.
The additional targets include the optimizer's `runtime_optimization/` source,
remaining top-level common tests, logging tests, top-level platform tests plus
Windows platform/logging sources on Windows, and root-level CPU provider
enumeration, I/O binding, memory-copy, type-control, and partitioning tests.
The logging target uses the existing `SKIP_DEFAULT_LOGGER_TESTS` configuration
because ORT's test environment already owns the process default logger.
The original five narrow utility sources stay in `//:common_test`, without
being duplicated in `//:common_runtime_test`.

Linux PCI/DRM/NPU discovery and POSIX/Windows telemetry-specific subdirectories
are not wired: the foundation still uses default device discovery and disables
optional telemetry transports. Android tests and DirectML provider tests need
their own platform configurations. GPU tests, production dynamic EP libraries,
external model-corpus runners, and benchmarks/fuzzers remain outside this C++
slice. The shared public
API, global-threadpool, and CPU custom-op/plugin additions are described below.

Starlark files use LF even on Windows: frozen module-extension digests depend on
their exact bytes. A CRLF checkout otherwise fails `--lockfile_mode=error` before
compilation; changing the module lock separately per host is not the solution.

On Windows x86-64, Bazel 8.4.2 and Visual Studio 2022/MSVC 14.44 built all five
binaries with the frozen lock unchanged. The following targeted command passed
**17 existing cases** (12 framework, 1 session, 1 graph, 1 optimizer, 2 CPU
operator cases), with a nonzero GoogleTest count verified for every binary:

```sh
bazel test //:framework_test //:session_test //:graph_test //:optimizer_test //:cpu_provider_test -c opt --jobs=12 --test_filter="AllocatorTest.*:GraphViewer.FilteredGraph:FreeDimensionOverrideDenotationTransformerTest.Test:InferenceSessionTests.DisableCPUArena:MathOpTest.MatMulFloatType:ConvTest.Conv2D_1" --test_output=all --nocache_test_results --keep_going
```

These additional selection checks also passed:

```sh
# One existing serialization case passed, including writing its model in TEST_TMPDIR.
bazel test //:graph_test -c opt --test_filter=GraphTest.UnusedValueInfoSerializes --nocache_test_results --test_output=all
# Successfully listed the selected session cases; listing does not execute them.
bazel test //:session_test -c opt --test_arg=--gtest_list_tests --test_filter="InferenceSessionTests.*" --test_output=all
```

That is **18 executed cases**, not a claim that the full suites pass. Full-suite
execution was deliberately not performed; Linux and macOS validation of the new
C++ test layer remains pending. Case filters do not enable excluded runtime
features or supply external model corpora.

The five additional CPU targets compiled and linked from the existing sources
on the same Windows toolchain. This targeted command passed **nine existing
cases** with nonzero counts in every binary: common 1, logging 2, platform 1,
provider infrastructure 4, and runtime optimizer 1.

```sh
bazel test //:common_runtime_test //:logging_test //:platform_test //:provider_infrastructure_test //:runtime_optimizer_test -c opt --jobs=12 --test_filter="DenormalTest.DenormalAsZeroTest:LoggingTests.TestCLogSink:LoggingTestsFixture.TestDefaultFiltering:PlatformEnvTest.DirectoryCreationAndDeletion:GetExecutionProvidersTest.*:InferenceSessionTests.TestBindCpu:GraphRuntimeOptimizationTest.SaveRuntimeOptimizationToOrtFormat" --test_output=all --nocache_test_results --keep_going
```

These are source-native Windows binaries, not substitutions with prebuilt test
executables. Bazel reused 1,811 unchanged dependency/runtime compile actions and
performed 45 local actions to compile/link/run the new sources; it did not
rebuild the entire runtime. Full-suite and non-Windows execution of these new
groups has not been performed. The runtime optimizer case saves and reloads an
ORT-format model; the platform case creates/deletes directories in the isolated
test workspace, and I/O binding executes an existing CPU model fixture.

Six further native targets reuse all existing top-level flatbuffers,
quantization, LoRA, opaque API, and EP graph sources plus the two internal
testing-EP test sources. The internal testing EP is the repository's existing
CPU-backed test double, not a new hardware execution provider. Training-specific
FlatBuffer cases and CUDA-specific LoRA cases remain disabled by their existing
feature guards.

This command passed **six existing cases**, one per target, on Windows with the
same frozen-lock native build. The new test sources compiled and linked against
the cached source-built runtime; no prebuilt test binaries were substituted.
Nonzero GoogleTest counts were verified in all six logs.

```sh
bazel test //:flatbuffers_test //:quantization_test //:lora_test //:opaque_api_test //:ep_graph_test //:internal_testing_ep_test -c opt --jobs=12 --test_filter="FlatbufferUtilsTest.ExternalWriteReadWithLoadInitializers:Quantization.QuantizeLinear_Int8:LoraAdapterTest.Load:OpaqueApiTest.RunModelWithOpaqueInputOutput:EpGraphTest.BasicCApiUse:InternalTestingEP.TestSaveAndLoadOrtModel" --test_output=all --nocache_test_results --keep_going
```

The selected cases include external-data FlatBuffer writes/reads, INT8
quantization, adapter serialization, opaque input/output model execution, public
EP graph access, and save/reload of an ORT model partitioned with the CPU-backed
test double. Full suites and Linux/macOS execution of these six targets remain
unvalidated. Together with the previous selections, **33 distinct existing C++
cases** have executed on Windows; that is not full-suite coverage.

### Shared C/C++ API and CPU plugin tests

`//:shared_api_tests` groups four additional existing-source executables:
`//:shared_api_test`, `//:global_threadpool_test`,
`//:custom_op_registration_test`, and `//:plugin_api_test`. These link through
`//:runtime_shared`, not the static runtime used by `//:core_tests`. The shared
test support compiles only common error/stacktrace and test helpers separately;
it does not link the inference runtime statically. Windows link parameter files
were checked: all four executables consume `onnxruntime.dll.if.lib`, without a
static runtime library. No additional runtime exports were needed for this slice.

The shared API target compiles the existing allocator, environment creation,
EP-context data, data-copy, experimental API, inference, model editor/loading,
non-tensor types, ORT-format model, run/session options, and version sources.
The global-threadpool target retains its existing dedicated main and its global
thread configuration/hooks. Select CPU parameter `/0`; optional-provider `/1`
and `/2` variants return without inference when their provider is disabled.

Native shared libraries are built from the existing custom-op library and all
three example plugin EP implementations (compiling, kernel-registry, and virtual
GPU test double). These are test fixtures, not additional production EP builds.
The custom-op registration executable imports the actual custom-op library.
The plugin executable compiles the existing `autoep/test_execution.cc` suite and
its public API helpers. The fixture wrapper copies dynamic fixtures into the
isolated writable test directory, and POSIX loader search paths include that
directory. It also rejects zero-match/all-skipped execution with nonzero status;
`--gtest_list_tests` remains available.

This exact selected command passed **seven existing cases** on Windows: shared
API 4, dedicated global-threadpool CPU inference 1, imported custom-op function
registration 1, and dynamically loaded plugin EP Mul inference 1.

```sh
bazel test -c opt //:shared_api_test //:global_threadpool_test //:custom_op_registration_test //:plugin_api_test --test_filter="CApiTest.run_options:CApiTest.allocation_info:ModelEditorAPITest.Basic_CxxApi:CApiTest.test_custom_op_library:CustomOpRegistration.TestUsingFuncNameCApi:OrtEpLibrary.PluginEp_AppendV2_MulInference:*.simple/0" --lockfile_mode=error --jobs=8 --test_output=all --nocache_test_results
```

The test sources and custom-op/plugin DLLs were freshly compiled/linked; the
native runtime was reused from this task's source-build cache, with no supplied
binary substitutions. An additional existing shutdown death test passed:

```sh
bazel test -c opt //:shared_api_test --test_filter=EnvCreation.ReleaseEnvFromEarlierRegisteredCleanupAtShutdown --jobs=8 --test_output=all --nocache_test_results
```

Re-executed death-test children reuse the parent's prepared fixtures instead of
attempting to copy over them. The updated wrapper also passed the existing
`DenormalTest.DenormalAsZeroTest` through the statically linked
`//:common_runtime_test`. This brings selected Windows C++ coverage to **41
distinct existing cases across 20 executables**, not full-suite coverage. A preliminary
global-threadpool filter using `/simple/0` instead of `.simple/0` selected zero
cases and correctly failed; the corrected command above passed all four targets.
The external ONNX corpus (`test_io_types.cc`), config-dependent optional
providers, and full test suites remain unvalidated or unwired. No Linux/macOS
execution of these four targets has been performed.

### Internal loader and broader auto-EP suites

`//:runtime_path_test` builds the existing runtime-path helper as a native
shared library and imports its public test function. The internal `Env`
implementation must execute inside that helper module, so this fixture embeds
the current monolithic native runtime archive with restricted symbol exports.
The executable still consumes the actual runtime DLL for its public API
environment; this does not change the DLL-only linkage of the shared API tests.

`//:provider_bridge_test` and `//:optional_provider_probe_test` use the existing
normal-exception internal tests and source-built missing-export/valid/shared
fixture libraries. They deliberately test internal static runtime symbols, not
exported C API replacements. A small Python launcher copies each executable and
its dynamic fixtures into a private `TEST_TMPDIR/runtime/` directory, then
propagates its exit status. Consequently the tests can install/remove libraries
beside their executable without modifying Bazel outputs or other tests. The
optional-provider fixtures are fake loader interfaces, not an OpenVINO build.

`//:runtime_dlopen_test` runs the existing Windows standalone
`LoadLibrary`/`GetProcAddress`/inference/`FreeLibrary` probe against a copied
source-built runtime DLL. The probe has no runtime import-library dependency.
Its attempted CUDA registration fails as expected in the CPU configuration,
then it executes the CPU model and verifies that the DLL unloads. This is one
standalone probe, not a GoogleTest case or CUDA validation.

`//:autoep_tests` compiles the existing top-level `test/autoep/*.cc` sources,
with their existing feature guards, and uses the same isolated launcher plus
the three native example plugin libraries. It links the static internal test
runtime because registration/selection sources access internal types and
`OrtSessionOptions::GetProviderOptionPrefix`; those are not public DLL exports.
This target is tagged `manual`: the full provider-dependent suite is not ready
for wildcard execution.

On Windows, this exact command passed **seven existing GoogleTests plus the
standalone DLL probe**, with nonzero counts in every GoogleTest executable:

```sh
bazel test -c opt //:runtime_path_test //:provider_bridge_test //:optional_provider_probe_test //:runtime_dlopen_test //:autoep_tests --test_filter="GetRuntimePathFromSharedLibraryTest.Basic:ProviderBridgeTest.MissingExportUnloadsLibraryAndAllowsRetry:OptionalProviderProbeTest.MissingExportUnloadsAndAllowsReplacement:OrtEpLibrary.LoadUnloadPluginLibrary:AutoEpSelection.CpuEP:AutoEpSelection.PreferCpu:AutoEpSelection.DuplicateEpCustomOpDomainsAreDeduplicatedForSessionCreation" --build_python_zip=false --jobs=2 --local_test_jobs=1 --test_output=all --nocache_test_results
```

New test/helper/fixture sources compiled natively using the cached source-built
runtime; no binary substitutions were used. The final selected run required
only cached native actions. This brings selected Windows coverage to **48
distinct existing GoogleTests across 24 GoogleTest executables**, plus the
standalone DLL probe. Full-suite and non-Windows execution remain unvalidated.
`bazel test -c opt //:provider_bridge_test --test_filter=BazelMissingCase
--build_python_zip=false --jobs=2 --local_test_jobs=1 --test_output=all
--nocache_test_results` failed with expected Bazel exit 3, confirming that the
isolated launcher preserves the GoogleTest wrapper's zero-match rejection.

One additional attempted existing auto-EP case still fails in this configuration:
`OrtEpLibrary.LoadUnloadPluginLibraryCxxApi` expects nonempty hardware metadata
but default device discovery supplies none. Its fatal assertion precedes the
test's explicit unregister call, leaving the example EP registered. In an
earlier combined invocation, the subsequent
`AutoEpSelection.DuplicateEpCustomOpDomainsAreDeduplicatedForSessionCreation`
then failed to find the `test.Custom_Mul` kernel. The duplicate-device case
passes both alone and in the supported combined command above without the
failing metadata test; it is now counted as passing. Inspection against the
upstream fixture build confirmed the example EP sources, exports, and custom-op
registration callbacks are present; no missing Bazel fixture dependency or
compile define was identified. The metadata case remains unvalidated, not
classified as a preexisting failure. A prior
registration-library path mismatch was corrected by executing from the isolated
directory, not by weakening the assertion. No-exception loader variants and
cpuinfo allocation/refcount probes are not wired; the default runtime does not
enable `CPUINFO_SUPPORTED`. Hardware-discovery configurations remain outside
this slice.

### Existing Java/JNI tests

`//:java_util_test` and `//:java_tensor_test` compile and execute the existing
`UtilTest` and `OnnxTensorTest` JUnit classes. `//:java_tests` groups these two
targets. The tensor target depends on the native Bazel runtime and JNI library;
the utility target needs no native libraries. No Gradle invocation or installed
ORT Java package is involved. JUnit Platform 1.11.4 (including Jupiter) and the
Java test helpers' protobuf 3.25.5 dependency are checksum-pinned downloads.

```sh
# One target and one existing method, not the complete Java suite:
bazel test -c opt //:java_util_test --test_filter='ai.onnxruntime.UtilTest#capacityFromSize'
bazel test -c opt //:java_tensor_test --test_filter='ai.onnxruntime.OnnxTensorTest#testArrayCreation'
# Exercise a model through JNI, rather than only tensor creation:
bazel test -c opt //:java_tensor_test --test_filter='ai.onnxruntime.OnnxTensorTest#testFp32ToFp16'
# Select an entire existing class explicitly:
bazel test -c opt //:java_util_test --test_filter=ai.onnxruntime.UtilTest
```

Filters use JUnit's exact `fully.qualified.Class#method` syntax, not GoogleTest
wildcards. The selected class must match the target. An unknown class/method or
zero-test discovery fails instead of passing silently. Omit the filter to run
the target's complete class; request `//:java_tests` only to run both classes.
Target selection avoids building the other test executable; method selection
does not prune Java compilation or native runtime compilation.
The existing Java test fixtures are loose Bazel runfiles, preserving adjacency
for models with external tensor data. The opt-in `onnxruntime.test.resources`
property lets the existing test helper resolve these files; Gradle/classpath
resource lookup is unchanged when the property is absent.

Windows x86-64 validation used Bazel 8.4.2, managed JDK 25, and the frozen module
lock. `bazel test -c opt //:java_util_test
--test_filter=ai.onnxruntime.UtilTest#capacityFromSize --test_output=all
--nocache_test_results` passed **one existing case**. Substituting
`#missingMethod` failed (Bazel exit 3), confirming invalid selections are not
false greens. `bazel build -c opt //:java_tests --nobuild` successfully analyzed
both native targets; this is analysis, not a native build or test pass.

To avoid another native runtime rebuild during this layer's development,
`testArrayCreation` and `testFp32ToFp16` each passed as **one selected case** using
the Bazel-built `//:java_test_runner` with the foundation's previously built
Windows runtime/JNI DLLs. These are prebuilt-backed checks, **not end-to-end
validation of `//:java_tensor_test` in this worktree**. The runner compiles the
current binding and existing test sources. The reproduction command is:

```powershell
# NativeBin must identify a previously verified native Bazel foundation build.
$env:TESTBRIDGE_TEST_ONLY = 'ai.onnxruntime.OnnxTensorTest#testFp32ToFp16'
bazel run -c opt //:java_test_runner -- "$NativeBin\onnxruntime.dll" "$NativeBin\onnxruntime4j_jni.dll" "$PWD\java\src\test\resources" ai.onnxruntime.OnnxTensorTest
Remove-Item Env:\TESTBRIDGE_TEST_ONLY
```

This layer deliberately does not claim complete Java migration. Inference,
environment/thread-pool, EP, plugin, and compile API suites remain
unwired. Training is intentionally excluded. `SparseTensorTest.testCOO` was
tried against the foundation and failed because its model requires the excluded
contrib `SparseToDenseMatMul` operator;
that class is not registered as a passing native target. Full-class execution
and Linux/macOS validation remain pending. C# additionally has the selectable
desktop tests documented above; Go retains its existing self-contained tests.

### Selectable existing Python inference tests

`//:python_inference_test` registers 21 existing `TestInferenceSession` methods:
model loading from paths/bytes, MatMul and noncontiguous arrays, symbolic shapes,
boolean/string inputs, session options, provider arguments, OrtValue, devices,
and memory info. `//:python_global_threadpool_test` registers both existing
`TestGlobalThreadPool` methods in a separate process because global thread-pool
initialization cannot be mixed with ordinary session initialization.
`//:python_tests` groups those two targets; it is not the complete Python suite.
The exact method inventory is in `bazel/python_tests.bzl`.

```sh
# One existing method:
bazel test -c opt //:python_inference_test --test_filter=onnxruntime_test_python.TestInferenceSession.test_run_model
# One registered class (two thread-pool cases):
bazel test -c opt //:python_global_threadpool_test --test_filter=onnxruntime_test_python_global_threadpool.TestGlobalThreadPool
# All registered methods from one module:
bazel test -c opt //:python_inference_test --test_filter=onnxruntime_test_python
# Runner selection regressions, without compiling ORT:
bazel test -c opt //:python_test_runner_test
```

Filters are exact dotted module/class/method names, not wildcard expressions.
A class or module filter selects **only the registered methods**, not every
method in the upstream class. Unknown, unsupported, or empty selections fail;
load/import failures also fail. Omitting the filter runs all methods registered
for that target. Selection reduces execution, not native extension compilation.

Bazel assembles the existing package initializer, inference collection, and
configured binding loader alongside the source-built extension. Only the six
required checked-in model fixtures are runfiles, preserving the existing
`helper.get_name` relative layout. Tests run in `TEST_TMPDIR`, not the source
tree or read-only runfiles. These high-level targets are currently CPU-only
and incompatible with `--//bazel:cuda=true`; the low-level Python smoke target
retains its existing CUDA behavior.

Windows x86-64 targeted validation used Bazel 8.4.2 and managed CPython 3.12:
`bazel build -c opt //:python_tests --nobuild` passed native graph analysis.
`bazel test -c opt //:python_test_runner_test --test_output=all
--nocache_test_results` passed **six selection regression cases**.
With a temporary local validation-only substitution of the foundation's
previously verified native `.pyd` for `:python_windows_extension`, the following
each passed **one existing case**:

```sh
bazel test -c opt //:python_inference_test --test_filter=onnxruntime_test_python.TestInferenceSession.test_run_model --test_output=all --nocache_test_results
bazel test -c opt //:python_inference_test --test_filter=onnxruntime_test_python.TestInferenceSession.test_run_model_from_bytes --test_output=all --nocache_test_results
bazel test -c opt //:python_inference_test --test_filter=onnxruntime_test_python.TestInferenceSession.test_run_model2_contiguous --test_output=all --nocache_test_results
bazel test -c opt //:python_inference_test --test_filter=onnxruntime_test_python.TestInferenceSession.test_string_input2 --test_output=all --nocache_test_results
bazel test -c opt //:python_inference_test --test_filter=onnxruntime_test_python.TestInferenceSession.test_ort_value --test_output=all --nocache_test_results
```

The class-filter command above passed **two existing thread-pool cases**.
Selecting `onnxruntime_test_python.TestInferenceSession.missing_method` failed
with Bazel exit 3, rather than producing a zero-test success. This is **seven
prebuilt-backed existing cases**, not a fresh end-to-end native extension build
in this worktree. The substitution and binary were removed; committed targets
always build the actual extension from source. No full Python suite was run.

Contrib/custom-op models, external model downloads, GPU/provider-specific tests,
ONNX backend/model-generation dependencies, compile API, I/O binding, symbolic
shape inference, and the remaining unregistered inference methods are not yet
migrated. In particular, `test_model_meta` silently returns without its external
SqueezeNet model, so it is not registered as coverage. Linux/macOS execution of
these high-level Python targets remains unverified. Training is intentionally
out of scope, not planned migration work.

### Selectable existing Node binding tests

The following targets run existing Mocha test files against the common and Node
JavaScript APIs, not a replacement Tensor class or a published ORT npm binary:

| Target | Existing file under `js/node/test/` |
| --- | --- |
| `//:node_model_metadata_test` | `unittests/lib/model-metadata.ts` |
| `//:node_inference_session_test` | `unittests/lib/inference-session.ts` |
| `//:node_tensor_test` | `unittests/lib/tensor.ts` |
| `//:node_simple_api_test` | `api/simple-api-tests.ts` |

`//:node_test` is an explicit aggregate runner for these four files. It is tagged
`manual` so `//...` executes the per-file targets without running them twice.
The existing low-level `//:node_smoke_test` remains available.

```sh
# One file and one existing case:
bazel test -c opt //:node_model_metadata_test --test_filter='symbol shape$'
# Pick a file explicitly and list its full case titles without executing them:
bazel test -c opt //:node_test --test_arg=--file=unittests/lib/model-metadata --test_arg=--list --test_output=all
# Only the existing scalar-type inference checks in the simple API file:
bazel test -c opt //:node_simple_api_test --test_filter='test_types_.*[.]onnx$'
# Selection regressions, without compiling the native runtime:
bazel test -c opt //:node_test_runner_test
```

`--test_filter` is a JavaScript regular expression over Mocha's full case title,
not GoogleTest syntax. Quote expressions containing spaces or shell characters.
`--file` accepts the exact extensionless paths in the table and can be repeated;
explicit files replace the target's default selection. Unknown files/arguments,
malformed expressions, zero matching cases, and selections with no passing cases
(including all-skipped selections) fail. Mixed passing/pending results retain
Mocha's existing skips and report their counts. `--list` is explicitly a discovery
operation, not an execution pass. File/case selection reduces test execution,
not addon or shared runtime compilation.

`//:node_js_package` uses pinned TypeScript to transpile the existing
`js/common/lib/`, `js/node/lib/`, selected tests, and their existing test helper.
The checked-in ONNX protobuf JavaScript, JSON configuration, SqueezeNet model and
reference data, and scalar-type models are declared inputs. This assembly is
transpilation, not a full semantic TypeScript type check or release packaging.
Mocha, TypeScript, and helper dependencies are pinned in
`bazel/node/package.json` and the integrity-checked `pnpm-lock.yaml`; npm lifecycle
hooks are disabled. The addon and runtime remain native Bazel compilation
dependencies. No `npm run build`, install/postinstall script, `cmake-js`, or
downloaded ORT binary participates in the committed graph.

The runner copies package/fixtures into its writable `TEST_TMPDIR`, stages the
Bazel-built addon and shared runtime in the binding's platform/architecture
directory, links Bazel-managed helper dependencies, and registers the actual
Node backend. It does not use the upstream warmup's ignored-error path. On
Windows, explicitly set `BAZEL_SH` to Git for Windows Bash as shown above; WSL's
Bash is not the native Node launcher shell.

Windows x86-64 validation used Bazel 8.4.2, managed Node 24.22.1, and
`--lockfile_mode=error`. These commands passed:

```sh
# JS compilation/assembly and three selector regression cases; no ORT rebuild:
bazel test -c opt //:node_js_package //:node_test_runner_test --test_output=all --nocache_test_results
# Analysis only of the real native dependency graph, not native execution:
bazel build -c opt //:node_test //:node_inference_session_test //:node_model_metadata_test //:node_tensor_test //:node_simple_api_test --nobuild
```

With an explicit temporary local-only substitution of the foundation's previously
verified Windows `onnxruntime_binding.node` and `onnxruntime.dll`, the actual
Bazel launcher also executed:

```sh
bazel test -c opt //:node_test --test_filter='model input/output with|test_types_.*[.]onnx$|run\(\) -|check data and type' --test_output=all --nocache_test_results
bazel test -c opt //:node_model_metadata_test --test_filter='symbol shape$' --test_output=all --nocache_test_results
bazel test -c opt //:node_test --test_arg=--file=unittests/lib/model-metadata --test_arg=--list --test_output=all --nocache_test_results
```

The aggregate passed **41 cases**: three metadata, three SqueezeNet inference
fetch variants, 23 tensor construction/type cases, and 12 scalar-type inference
cases. One preallocated-fetch case was already pending upstream. The individual
metadata filter passed **one case**; listing reported **three cases without
execution**. Unknown `--file=missing`, `--test_filter=no-such-case`, and malformed
`--test_filter=[` each failed with Bazel exit 3. These are prebuilt-backed
checks, **not a fresh end-to-end native addon/runtime rebuild in this worktree**.
The substitutions and temporary binaries were removed; committed targets always
compile the actual native source dependencies. Local registry rewrites used to
access the pinned npm archives are not repository configuration.

The full Node suite was not run. Worker/standalone tests, EPContext callback
tests, the separate repeated-run/WebGPU API file, installer tests, downloaded
ONNX backend corpora, optional providers, and release npm artifacts remain
outside this slice. The WebGPU-option recognition case in `simple-api-tests.ts`
is not evidence of WebGPU execution. Linux/macOS execution of this new layer
remains unverified. Browser/WASM has the separate scalar/SIMD CPU path described
above; React Native remains unimplemented. Native Node coverage alone does not
migrate either surface.

### Foundation validation (before the language test layers)

`bazel test //... -c opt --jobs=16 --keep_going --nocache_test_results` passed
all seven default test targets on Azure Linux 3 with GCC 13.2, with
`CUDA_PATH`, `CUDNN_PATH`, and `LD_LIBRARY_PATH` unset. This includes all five
desktop-language smoke/unit targets and 20 existing C++ utility tests.

On Windows x86-64 with Visual Studio 2022 (MSVC 14.44), Bazel 8.4.2,
and the managed JDK 25 runtime, the following also passed all seven targets:

```powershell
bazel test //... -c opt --jobs=8 --keep_going --test_output=errors --nocache_test_results
```

This includes actual Add/MatMul/Conv inference through C++, the 20 C++ utility
tests, and Go, Java/JNI, Python/NumPy, Node-API, and C# tests. The frozen Bazel
module lock remained enabled. The local company package mirrors were configured
in the user's Bazel RC, not embedded in repository dependencies.

`//:common_test` runs existing narrow, SafeInt, semantic-version, span, and string
utility tests. `//:go_test` runs the existing self-contained Go tests. The
language smoke tests are integration tests, not replacements for each binding's
complete test suite.

The historical wildcard commands above applied to the seven-target foundation,
not the expanded test graph. The `Native Bazel desktop` workflow now performs
frozen wildcard and explicit manual WASM **analysis only**, then executes binding/package
checks and selected existing C++ and managed-language cases. It never uses
`bazel test //...` to run the migrated full suites. Apple native inference is
selected only on macOS, the CPU Python wheel on Windows/Linux, and standalone
DLL loading only on Windows. The Windows runner uses a short output base to
avoid Rust dependency build-script path-length failures.

The workflow retains `ubuntu-24.04`, `windows-2022`, and Apple Silicon
`macos-15`. Intel macOS source support remains, but there is no Intel macOS CI
leg; native macOS verification prioritizes Apple Silicon. This defines CI
coverage, not a claim that the updated workflow or native Mac tests have passed.
Passing Linux tests alone is not evidence that the other hosts work.
The updated workflow YAML parsed successfully and all seven Bash steps passed
`bash -n`; the matrix and absence of wildcard test execution were also checked.

Remaining work includes remaining provider/operator/contrib test coverage,
additional EPs, additional architecture variants, complete release packaging,
cross-compilation, sanitizers, and the CUDA/toolchain matrix. Those are separate
native targets/toolchains to add, not reasons to delegate back to the old build
system.
