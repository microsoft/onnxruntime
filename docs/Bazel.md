# Native Bazel build (experimental)

This is an independent, native Bazel build graph. Bazel compiles ONNX Runtime,
MLAS, ONNX, protobuf, execution-provider code, and language extensions directly.
It does **not** invoke CMake, `build.py`, Gradle, MSBuild, `cmake-js`, or
`rules_foreign_cc`. Existing ONNX and Abseil **source patches** are reused;
their location under `cmake/` does not make them build-system dependencies.

The native graph targets Linux x86-64, Windows x86-64, and macOS on Intel and
Apple Silicon. Foundation runtime and five desktop bindings passed native
Linux, Windows, and Apple Silicon macOS CI at `a6de7b83a5`
([completed run](https://github.com/microsoft/onnxruntime/actions/runs/41977549518)):
all nine selected targets passed on each OS. Dependent migration results are
qualified separately below. Intel macOS execution is not claimed.
This is not yet a replacement for the
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

The later foundation shared-loader fix `7b167b8d41` is integrated additively
above the React Native package work. Its default CPU
`//:runtime_shared_smoke_test` is included in the existing selected
three-host binding/package CI step; the existing C++ filters and manual
Apple/RN targets are retained. Windows frozen wildcard/manual-WASM analysis
passed 254 targets with zero build actions at jobs=2. No native builds were
repeated for this integration.

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

When invoking Bazel from Git Bash, keep any `MSYS2_ARG_CONV_EXCL=*` label
protection on that invocation. Windows JavaScript targets clear the override
only inside their launchers so native Node can resolve `/c/...` entrypoint paths.

The override is shared by all native Aspect `js_binary`/`js_test` declarations:
Node smoke/existing cases, npm tools/consumers, React Native tools/consumer,
and WASM/browser tools/tests. A Windows Git Bash regression retained the
inherited `*` while executing the real common-JavaScript compiler, four
Node test-runner checks (including the native Node environment assertion),
and the offline React Native npm consumer. All passed, and the parent shell
still held `*` afterward. Only JS/tool/test actions ran; no native runtime
rebuild was performed. For example:

```sh
MSYS2_ARG_CONV_EXCL='*' bazel build //:react_native_common --lockfile_mode=error --jobs=2 --build_python_zip=false
MSYS2_ARG_CONV_EXCL='*' bazel test //:node_test_runner_test --lockfile_mode=error --jobs=2 --build_python_zip=false '--test_env=MSYS2_ARG_CONV_EXCL=*' --nocache_test_results --test_output=all
```

After integrating foundation `1550734d3d`, frozen Windows whole-graph analysis
including the six manual WASM labels passed: 254 targets, 74,475 configured
targets, 20.579 seconds, and zero actions (`-c opt --nobuild`,
`--lockfile_mode=error --jobs=2 --build_python_zip=false`).

```sh
bazel build -c opt //:runtime_library
bazel test -c opt //:runtime_smoke_test //:runtime_shared_smoke_test
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

When invoking Bazel from Git Bash on Windows, set
`export MSYS2_ARG_CONV_EXCL='*'` so labels and test filters are not rewritten
as filesystem paths. CI also uses a short Windows output base to avoid path
length limits.

The MLAS baseline archive is retained because ISA archives call back into
its helpers on ELF linkers. macOS logging is compiled as Objective-C++ rather
than placed in a C++-only source list, and Intel constants use Mach-O sections.
These graph fixes do not substitute for native macOS execution evidence.

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
uses matching ReactAndroid/JSI, CallInvoker and fbjni SDKs, Java React bridge
and AndroidX APIs, plus `cpp-adapter.cpp` and the shared JSI C++ sources
listed in `js/react_native/android/CMakeLists.txt`. The separate React Native graph is described below. The Java/JNI AAR neither builds
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

### Native npm and NuGet distributions

```sh
bazel build -c opt //:node_npm //:csharp_nuget
bazel test -c opt //:node_npm_test //:csharp_nuget_test //:node_smoke_test //:csharp_smoke_test --jobs=24 --loading_phase_threads=4 --build_python_zip=false --lockfile_mode=error --test_output=all --nocache_test_results
```

`bazel-bin/onnxruntime-node-bazel.tgz` is a production-only, single-platform CPU
`onnxruntime-node` package. It contains the source-built Node-API addon,
runtime and provider bridge under the existing
`bin/napi-v6/<platform>/<architecture>/` loader path, compiler-produced
JavaScript and declarations, the bundled `onnxruntime-common` API, and notices.
It excludes tests, fixtures, and legacy install/build scripts. TypeScript 5.9.3
checks the production sources before emitting; declaration generation does not
silence semantic diagnostics. Pinned Node 20 types support that compilation.
The installed declaration consumer is also checked with Node types.

Install the tarball with `npm install --offline --ignore-scripts
/absolute/path/to/onnxruntime-node-bazel.tgz`. Its bundled common API needs no
registry fetch. `//:node_npm_test` uses the pinned Node/npm toolchain and a fresh
offline consumer directory, checks installed metadata and declarations, and
runs high-level CPU Add inference through the installed package rather than
the raw addon. It requires Node 20 or newer.

`bazel-bin/Microsoft.ML.OnnxRuntime.Bazel.1.32.0.nupkg` contains the directly
compiled `lib/net8.0/Microsoft.ML.OnnxRuntime.dll`, source-built runtime and
provider bridge under `runtimes/<rid>/native/`, NuGet metadata, and notices.
The experimental package ID is `Microsoft.ML.OnnxRuntime.Bazel`; do not combine
it with another package supplying the same managed assembly. Its declared
dependency is exactly `System.Numerics.Tensors` 9.0.0. Artifact production uses
`rules_dotnet` and `rules_pkg`, not `dotnet pack`, MSBuild, or a downloaded ORT
package. The archive test verifies exact compiled payload bytes, RID layout,
version, dependency metadata, notices, and the absence of unexpected files.

On Linux x86-64, all four targets in the command above passed. The native
runtime, managed assembly, and Node addon were genuinely compiled from source
on the VM; no native artifact substitutions were used. A separate .NET 10
consumer restored this exact NuGet and the pinned Tensor dependency from a
local-only feed, built its own small consumer project, and passed Add inference
using normal RID-native resolution with no DLL resolver or native-library
argument. That consumer's standard `dotnet restore`/`dotnet build` validates
NuGet consumption; neither command builds ORT or assembles the package.
The loader trace confirmed the consumer's own `libonnxruntime.so` was loaded
with `LD_LIBRARY_PATH` and `LD_PRELOAD` unset.

Integration validation used source snapshot
`8390d2f3be7aeafd33bea86179255bbc4284c4cc` plus this additive package slice.
Frozen `bazel build -c opt //... --nobuild --jobs=2
--loading_phase_threads=2 --build_python_zip=false --lockfile_mode=error`
passed all 233 Linux targets with zero actions. The existing
`//:node_model_metadata_test --test_filter='symbol shape$'` also completed
one real case with no skips after the Node declaration dependency change.

Both archive graphs define desktop Windows/Linux/macOS x64 and arm64 layouts,
but package installation/execution evidence here is Linux x64 only. These are
CPU, single-platform experimental artifacts, not universal packages or release
registry publication. CUDA, mobile, and additional runtime variants are not
implicitly included.

After integration, Windows frozen analysis of `//...` plus the six explicit
manual WASM test targets passed at `--jobs=2`: 239 targets, 74,424 configured
targets, and zero actions. No local native compilation was performed for this
package integration. The three-host selected-only CI package-consumer step
now includes `//:node_npm_test` and `//:csharp_nuget_test`; YAML and all seven
Bash blocks passed syntax checks. Windows/macOS package consumption is not
claimed from graph analysis or CI configuration.

CI follow-up fixes pin checkout/upload-artifact v4 to immutable commits and
disable MSYS argument conversion on Windows so package-qualified Bazel labels
and slash-containing test filters reach Bazel unchanged. `apple_support`
is a direct dependency before `rules_cc`, as required by its pinned toolchain
setup documentation: Apple CC must take priority over the generic C++ toolchain
for Objective-C/XCTest analysis. Frozen Windows analysis still passed all
239 targets with zero actions and no lock changes. Native macOS analysis and
execution of this correction await the pushed CI run; Apple tests remain enabled.

The remainder also merges foundation `d473c4e677`, including preservation of
the inner Go test's `RunEnvironmentInfo` and inherited environment while
overriding only `ORT_LIB_PATH`. This retains `GO_TEST_RUN_FROM_BAZEL`, which
the existing Go test uses to select its fixture working directory. Configured
Windows `//:go_test` inspection confirms both variables; merged frozen
wildcard/manual-WASM analysis passed 239 targets with zero actions.
The foundation's fresh Linux source build passed all seven uncached desktop
targets after this fix, and selected `TestGetVersion` executed and passed.
No native rebuild was performed merely to integrate the unchanged package graph.

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
| Node.js | `//:onnxruntime_binding.node`, `//:node_js_package`, `//:node_test`, `//:node_npm` | Native Node-API extension, source-assembled common/Node JavaScript packages, four selectable existing test files, and production-only CPU npm tarball with declarations and offline install/inference validation. Not registry publication or the complete Node suite. |
| C# | `//:csharp_binding`, `//:csharp_smoke_test`, `//:csharp_tests`, `//:csharp_nuget` | .NET 8 binding and existing desktop test sources compiled directly with `rules_dotnet`, exercised on .NET 10. Native xUnit discovery/execution supports exact class/method selection, theories and fixtures. Single-RID CPU NuGet package; training APIs are not enabled. |
| Objective-C / Swift import surface | `//objectivec:onnxruntime_objc`, `//objectivec:inference_tests`, `//objectivec:swift_inference_tests`, `//objectivec:ios_framework`, `//objectivec:ios_inference_tests` | Native ARC inference wrappers, explicit module, real Swift Add-inference XCTest, macOS XCTest and iOS static framework/test-bundle graphs. Native Apple compilation/execution and Swift import remain unverified. Not an XCFramework or Swift package. |
| JavaScript Web / WebAssembly | `//:wasm_scalar`, `//:wasm_simd`, `//:web_scalar_package`, `//:web_simd_package` | Source-native Emscripten CPU runtime and source-assembled frontend; selectable existing WASM inference test and Node/browser smoke targets. Single-threaded scalar/fixed SIMD only, not the complete release or Web test suite. |
| React Native | `//js/react_native:android_package`, `//js/react_native:android_jsi_native_archive`, `//js/react_native:ios_adapter_framework` | Genuine Android ARM64 Java/JNI/JSI AAR and offline-installable npm package with source-native ORT, compiled against matching pinned SDKs. iOS adapter framework graph uses actual React headers; Apple compilation and application/device inference remain unverified. |
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

Completed macOS-15 job `127059252426` at `7806e8c4fd` passed the 20-target
bindings/package step, including the source Rust smoke (`current_c_api` and
`add_inference`). `objectivec:test_runner_test` validates only the runner,
not Objective-C compilation. The two loader-replacement tests failed after
removing a missing-export library and copying a valid fixture to the same
path; fixture replacement now allocates the new file before removing the old
one, preventing inode identity reuse while preserving unload/remove/retry
assertions. Native Apple inference now runs independently of earlier C++ test
failures when the job is not cancelled. The iOS build in that completed job
was cancelled; Objective-C/Swift/iOS verification is still not claimed.

The replacement change passed source-only MSVC `/Zs` checks for both modified
tests using their Bazel compiler response files, and Windows frozen analysis
passed 254 targets in 28.610 seconds with zero actions. Executing the two
native suites would require 1,866 actions after the integrated Abseil patch
invalidated the runtime cache; that build was stopped to respect the local
resource limit. Those source/analysis checks did not establish a passing
macOS regression.

CPU-only Linux execution subsequently passed all four optional-provider
cases but exposed a bridge-fixture issue even in valid-only initialization.
The header-free bridge fixtures had unnecessarily linked
`shared_api_test_common`, bringing exported ONNX/protobuf/Abseil objects and
GNU-unique definitions into an image loaded with `RTLD_GLOBAL`.
That revision removed these dependencies from bridge fixtures only. The valid
probe still requires the real provider headers/runtime support. All unload/retry
assertions remain.

The corrected graph passed all three `ProviderBridgeTest` and all four
`OptionalProviderProbeTest` cases, no skips, on the authorized CPU-only Linux
host with GCC 13.2. The uncached warm rerun took 2.099472 seconds wall
(8 actions, 6 sandbox actions, 608 cache hits); corrected bridge fixtures
contained zero GNU-unique definitions and only intended fixture exports
plus standard runtime/weak symbols. Missing `Provider_SetHost` remained
absent in the negative fixture and resolved in the valid one. No GPU was
built or executed. The initial native build took 210.601 seconds/1,818
actions, followed by an invocation-only selector error and then the genuine
bad-dependency failures; these are staged validation timings, not a cold
successful benchmark. The reproducible suite selector is:

```sh
bazel test //:provider_bridge_test //:optional_provider_probe_test -c opt --lockfile_mode=error --build_python_zip=false --test_filter='ProviderBridgeTest.*:OptionalProviderProbeTest.*' --test_output=all --nocache_test_results
```

After the dependency correction, frozen Windows analysis passed 254 targets
in 24.424 seconds with zero actions. The retained native evidence archive SHA-256
is `76c910733e423f09eb51cde65be373d995bd76fdce775fa066bef07f308a199b`.
The coordinator verified cleanup of the task-private CPU VM root and Bazel
server (2.6 GiB), credentials/configuration and source transfer; retained
evidence remains hash-verified, and shared/global resources were untouched.
That Linux run did not establish macOS fixture execution.

Completed migration run
[`41980869983`](https://github.com/microsoft/onnxruntime/actions/runs/41980869983)
at `e083a400e3` passed the macOS bridge replacement test, but the optional-provider
replacement still failed. Its shared/missing-export variants now omit unused
ORT headers and runtime dependencies; only the valid provider uses the real
provider interface. The completed Linux plugin inference failure was a missing
example library beside the runtime's `_solib` path. Shared API tests now reuse
the lazy shared-library bundle to co-locate example plugins on Linux without
linking them eagerly; CUDA configurations retain the lazy CUDA provider files.

That completed run also exposed actual Apple compilation failures, not just
missing execution evidence: floating-point `std::to_chars` was unavailable at
the macOS 12 deployment target, and Objective-C++ `@import` failed with C++
modules disabled. The existing ONNX iOS fallback now also covers macOS before
13.3, and both Objective-C++ libraries explicitly enable C++ modules. Deployment
minimums remain macOS 12 and iOS 15.1. These corrections are not a claim of
successful macOS fixture execution, Objective-C inference, Swift import, or iOS
framework/XCTest compilation. The Windows job was cancelled during selected
C++ compilation, not a passing native selected-suite result.

The revised graph passed all three bridge cases, all four optional-provider
cases, and actual example-plugin Mul inference on CPU Linux/GCC 13.2: eight
tests, zero skips. The uncached build-and-test invocation took 596.765 seconds
including 2,413 actions in a new task-private build directory; this is not pure
test execution time. Both header-free probe images exported only their intended
fixture symbol and had zero GNU-unique definitions. `readelf` confirmed the
test executable links the runtime but not the example plugins, and all three
example libraries were present beside that runtime in its `_solib` runfiles.
No GPU was built or executed.

```sh
bazel test //:provider_bridge_test //:optional_provider_probe_test //:plugin_api_test -c opt --jobs=12 --lockfile_mode=error --build_python_zip=false --local_test_jobs=1 --test_output=all --nocache_test_results --test_filter='ProviderBridgeTest.*:OptionalProviderProbeTest.*:OrtEpLibrary.PluginEp_AppendV2_MulInference'
```

Full frozen Windows wildcard plus six manual WASM targets passed analysis:
255 targets, 62.471 seconds, zero actions, two jobs, unchanged module lock.
The retained native test/symbol/runfiles evidence archive SHA-256 is
`c8eb4df701f02eee911eae9d398d188aa4f627dd5b699c01423648a4e38071ac`.
The 2.6 GiB CPU-only task directory, Bazel server, transfer archive and
task-private SSH credentials were removed after evidence retrieval.

On a macOS host with full Xcode selected (including the macOS SDK and XCTest):

```sh
bazel build -c opt //objectivec:onnxruntime_objc
bazel test -c opt //objectivec:inference_tests --test_filter=ORTSessionTest/testInitAndRunOk
bazel test -c opt //objectivec:inference_tests --test_filter=ORTValueTest
bazel test -c opt //objectivec:inference_tests --test_filter=ORTModuleImportTest/testInferenceModuleImport
```

`//objectivec:swift_inference_tests` compiles actual Swift source importing
`onnxruntime_objc` and checks scalar Add inference through its Objective-C API.
It uses the same source-built runtime and model resource, not a prebuilt Swift
package. The explicit `ORTSwiftInferenceTest` Objective-C runtime name keeps the
XCTest selector stable. Native execution is separately required; an Objective-C
module-import or runner-only test is not Swift compiler/inference evidence:

```sh
bazel test -c opt //objectivec:swift_inference_tests --test_filter=ORTSwiftInferenceTest/testAddInference
```

The Swift rule uses the already-resolved `rules_swift` 2.1.1 dependency directly,
without changing its version or the frozen module lock. Windows full wildcard
plus six manual WASM targets passed analysis after this addition: 258 targets,
53.414 seconds, zero actions, two jobs. This deliberately does not claim
compilation of Apple-incompatible Swift targets on Windows.
All eight portable XCTest runner-contract cases also passed, including exact
forwarding of the Swift selector. Those mock-runner checks are not native Swift
compiler or inference evidence.

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
checked-in C API headers using a checksum-pinned public libclang archive on
Windows/Linux and the selected Xcode libclang and builtin headers on macOS.
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
  //:runtime_cuda_smoke_test \
  //:runtime_cuda_shared_smoke_test

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

The CUDA smoke tests disable CPU execution-provider fallback. Passing them
therefore requires GPU execution of Add, MatMul, and Conv, not merely detection of an
installed GPU. GPU tests are tagged `manual`/`requires-gpu` and must be requested
explicitly. The shared variant links only the public runtime API. Its CUDA
import co-locates runtime and provider runfiles using native C++ linking
metadata, so ORT's sibling-library loader works inside Bazel's `_solib`
directory. Providers are not linked eagerly: their initialization requires
the host interface installed by the runtime.

`//:runtime_shared_smoke_test` also exercises the public shared-runtime import
in the default CPU configuration, without requiring CUDA SDKs.

The following combinations have executed the fallback-disabled CUDA test on
an NVIDIA H200 (`sm_90`), driver 580.159.04, on Azure Linux 3. The CUDA 12.0
run used GCC 12.2; the other rows used GCC 13.2:

| Toolkit | cuDNN | Result |
| --- | --- | --- |
| CUDA 12.0.1, NVCC 12.0.140 | 9.14.0.64 | Static/public-shared Add, MatMul, and Conv passed on GPU; 30 selected operator regressions passed on the integrated test graph. |
| CUDA 12.8, NVCC 12.8.93 | 9.14.0.64 | Add, MatMul, and Conv passed on GPU. |
| CUDA 13.0.2, NVCC 13.0.88 | 9.14.0.64 | Add, MatMul, and Conv passed on GPU at `35992fc6ad`; selected existing operator cases also passed. |
| CUDA 13.3, NVCC 13.3.73 | 9.27.0.42 | Add, MatMul, and Conv passed on GPU. |

These are tested combinations, not a claim that every CUDA 12/13 minor release,
GPU architecture, or driver/cuDNN combination has passed. The existing Abseil
NVCC 13.3 source workaround is included.

Existing C++ `test_support` conditionally receives CUDA SDK definitions and
provider runfiles, so CUDA-enabled existing tests no longer silently compile
as CPU-only. `//:cuda_operator_test` is a manual Linux x86-64 CUDA target:

```sh
bazel test //:cuda_operator_test -c opt --jobs=24 --loading_phase_threads=4 --lockfile_mode=error --build_python_zip=false --//bazel:cuda --@rules_cuda//cuda:archs=sm_90 --test_env=LD_LIBRARY_PATH --test_env=CUDA_VISIBLE_DEVICES=0 --test_arg=--gtest_filter=MathOpTest.Abs_Cuda_CanonicalizesSignedZero:MathOpTest.MatMulFloat16:ConvTest.MissingBiasCUDA --test_output=all --nocache_test_results
```

The handed-off CUDA 13.0.2/cuDNN 9.14/H200 validation passed all three actual
GoogleTests, zero skipped; `MissingBiasCUDA` explicitly disables CPU fallback.
The source build executed 781 actions with 992 cached actions in 258.690 seconds;
the test command took 27.763 seconds wall. This validates the static-runtime
operator test, not CUDA provider loading through every shared language binding.

The CUDA 12.0.1 lower-boundary run required source compatibility fixes for
BF16 raw-bit min/max conversions, host-side half bias-scale comparison,
Abseil range-trait specialization syntax, and integral atomic conversion
without newer CCCL `bit_cast`. Range concepts remain enabled. On the
integrated test graph from the dependent PR, the selected Min/Max and Gemm
tests plus eight signed ScatterElements add/multiply wraparound tests passed
(30 tests, no skips). BF16, Gemm, and signed wraparound GPU paths disable CPU
fallback. ScatterND integer reductions remain unsupported; no new kernel
support was added.

The dependent target includes the existing `gemm_test.cc` and
`scatter_op_test.cc` sources, not ScatterND integer reductions. After merging
foundation `a6de7b83a5`, Windows frozen whole-graph analysis including the six
manual WASM labels passed: 254 targets, 74,561 configured targets, 84.438 seconds,
zero actions, `--lockfile_mode=error --jobs=2 --build_python_zip=false -c opt --nobuild`.
The GPU results below are handed-off native validation; no duplicate native
or GPU build was performed for this integration.

The private xPack GCC 12.2.0-1 distribution defaults to global stack canaries
and bundles an assembler too old for existing MLAS instructions. This run
retained stack protection using TLS canaries and a private `as` symlink to
host GNU assembler 2.41. CPU flags alone do not configure NVCC's host compiler;
the matching `rules_cuda` setting is required for this particular profile:

```bash
# AS_PREFIX is a private directory containing the host assembler symlink.
compiler_flags=(
  --copt=-mstack-protector-guard=tls --host_copt=-mstack-protector-guard=tls
  "--copt=-B${AS_PREFIX}/" "--host_copt=-B${AS_PREFIX}/"
  "--@rules_cuda//cuda:host_copts=-mstack-protector-guard=tls,-B${AS_PREFIX}/"
)
bazel test //:runtime_cuda_smoke_test //:runtime_cuda_shared_smoke_test \
  -c opt --jobs=24 --loading_phase_threads=4 --lockfile_mode=error \
  --build_python_zip=false --//bazel:cuda --@rules_cuda//cuda:archs=sm_90 \
  "${compiler_flags[@]}" --local_test_jobs=1 --test_env=LD_LIBRARY_PATH \
  --test_env=CUDA_VISIBLE_DEVICES=0 --test_output=all --nocache_test_results
# On the dependent PR's integrated graph, with the same flags/environment:
# //:cuda_operator_test
# --test_arg='--gtest_filter=MathOpTest.Min_13_Float16_*:MathOpTest.Max_13_Float16_*:CudaGemmBiasBroadcastTest/*.*:CudaScatterElementsReductionWrapTest/*.*'
```

Verified SDK/compiler/Bazel setup took 80.12 s, excluding cuDNN installation.
After earlier compiler/source repairs, successful native compilation/link
took 203.90 s (571 actions, 1,993 cache hits), both uncached GPU smokes
46.57 s, and the corrected 30-test regression run 48.74 s including three
sandbox actions. The unchanged three-target build took 0.911 s with no native
compilation. These are staged, warm-cache timings, not a cold-build benchmark.
The compiler-profile adjustments are not requirements for every GCC 12
installation.

The CUDA 13.0.2 run used a task-private SDK assembled from official,
SHA-256-verified NVIDIA redistributions, GCC 13.2, 24 jobs, and the same H200
host. An initial native compilation took 316.52 s but stopped before linking
because the manually assembled SDK lacked the separate `cuda_culibos` package.
After completing the SDK, the successful incremental compilation/link took
175.19 s, the no-change build took 0.422 s, and the uncached fallback-disabled
GPU test took 23.76 s. Dependency downloads were warm and partial object
outputs were reused; these are staged validation timings, not an uninterrupted
cold-build benchmark. Standard complete toolkit installations do not require
this manual redistribution assembly.

The shared-runtime regression subsequently passed the same three
fallback-disabled CUDA operations on that SDK/H200 configuration in 23.05 s
wall time, using already compiled runtime/provider objects. ELF inspection
confirmed that the executable depends on `libonnxruntime.so`, not eagerly
loaded provider libraries. The default CPU shared-runtime test passed with
`CUDA_PATH`, `CUDNN_PATH`, and `LD_LIBRARY_PATH` unset in 113.45 s including
550 actions (1,127 cache hits), using 24 jobs. These are incremental
regression timings, not cold-build comparisons.

The final handed-off two-target CPU/shared-CUDA regression passed in
138.811 seconds at jobs=24. CPU execution within the CUDA-enabled profile
also passed with `LD_LIBRARY_PATH` unset. A relocated executable, all three
shared libraries and its model fixture passed fallback-disabled GPU inference
outside the Bazel directories in 21.880 seconds. This is native shared-runtime
loading evidence, not a claim that every CUDA language binding or package
consumer has been exercised.

The provider links with `--no-undefined` and exports only its `GetProvider`
entry point, so missing implementation dependencies fail the build rather
than surfacing later during provider loading.

Contrib CUDA kernels and optional FlashAttention, memory-efficient attention,
and XQA paths are not enabled. Some shared attention infrastructure is compiled
because standard ONNX operators use it; this does not register contrib ops.

## Tests and remaining migration

### iOS and React Native follow-up

The Android React Native follow-up adds `//js/react_native:android_java` and
`//js/react_native:android_package`. The former compiles the three existing
bridge classes against pinned ReactAndroid 0.73.11, fbjni 0.5.1, Android API 35,
AndroidX annotation 1.7.1, and Kotlin 1.8.0 SDKs. Windows compilation passed.
The latter assembles those classes with the source-native ORT/JSI libraries
and NDK shared C++ runtime as an ARM64 AAR and npm tarball. Its npm frontend
uses the existing Metro TypeScript sources and compiler-emitted common
JavaScript/declarations; the included Gradle metadata only exposes the finished
AAR, never invokes a legacy ORT build. React Native autolinking declares the
actual Java package. Matching React 18.2/React Native 0.73.11 is supplied by
the host application.

The generated npm peer range is exactly `"react-native": "0.73.11"`, not
a caret or wildcard range. The autolinked Android artifact configuration also
declares strict `com.facebook.react:react-android:0.73.11` and
`com.facebook.fbjni:fbjni:0.5.1` dependencies. Gradle must resolve those exact
SDK versions instead of silently upgrading the precompiled bridge's ABI.
A focused packaging regression checks both emitted boundaries. This does not
claim compatibility with other React Native/fbjni versions.
All six packaging regressions passed on Windows. The metadata-only npm update
also passed the actual pinned-npm offline consumer, including the installed
strict declarations. This check does not run Android Gradle resolution or
execute a React application.

The combined Linux-host Android build passed from an empty native output,
using Bazel 8.4.2, NDK r27d, eight jobs/two loading threads, frozen lock, and
the same Xeon Platinum 8480C/96-logical-CPU VM described below. SDK setup took
17 seconds. An initial analysis-only failure exposed that the common
JavaScript compiler needed the host execution configuration; after fixing
that transition, the first completed build took 316.862 seconds wall,
executing 1,450 actions (1,104 sandboxed), without substituting native binaries.
The final notices/verifier packaging update took 6.218 seconds and 15 actions,
reusing 1,444 actions; no native recompilation was required. The unchanged
repeat took 0.256 seconds, one internal action/no native compilation.

```sh
bazel build --config=android_arm64 -c opt //js/react_native:android_package //js/react_native:android_jsi_native_archive --jobs=8 --loading_phase_threads=2 --lockfile_mode=error --build_python_zip=false
package_dir="$(bazel info --config=android_arm64 -c opt --lockfile_mode=error bazel-bin)/js/react_native"
bazel run -c opt //:react_native_package_consumer --jobs=8 --loading_phase_threads=2 --lockfile_mode=error --build_python_zip=false -- "$package_dir/onnxruntime-react-native-android-arm64.tgz" "$package_dir/onnxruntime-react-native-android-arm64.aar"
bazel test //js/react_native:package_test --lockfile_mode=error --jobs=2 --test_output=errors
```

The real pinned-Node/npm offline consumer passed in 2.303 seconds wall:
installed AAR bytes, autolinking metadata, emitted common JavaScript,
source version, and installed common TypeScript declarations. Peer
installation is intentionally disabled for this package-structure check;
it does not create a React host or substitute fake React APIs. Five package
regressions passed on Windows. The AAR contains exactly the actual three
compiled Java classes, not copied React SDK classes, plus byte-identical
source-built ORT/JSI and NDK shared-runtime payloads. ELF inspection confirmed
AArch64, `JNI_OnLoad`, expected SDK dependencies, and 16 KB load alignment
for both source-built libraries. Source Metro files match the checked-in
TypeScript byte-for-byte; React/fbjni/NDK license notices are retained.
The host app must resolve its single compatible shared C++ runtime when
combining this AAR with React SDK AARs; complete app packaging, Metro execution,
emulator/device installation and inference remain unverified.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| `onnxruntime-react-native-android-arm64.aar` | 8,028,972 | `472570bb924887ba563a31577085b006dc8e9906a876937daea0b5d469f637a5` |
| `onnxruntime-react-native-android-arm64-source-c941.tgz` | 8,047,387 | `81769bf8f303656bfbd2aa58217ff2866bdf87739c92cc967c74d673e0245ec2` |
| `onnxruntime-react-native-android-arm64.tgz` (strict SDK metadata) | 8,047,438 | `f7f7ed48d84cfd9164183ffb5047a85a768cb00255dbd32c1400c3835d05f2ee` |
| `onnxruntime-react-native-android-arm64.zip` | 8,025,508 | `3e91a795e0ecf475a6fe8cde2d5fee418a127fe1d12ea21ffe28283ab84fb2f9` |

The native payloads were built from source `c941097eb7`, not the earlier
native ZIP below. The later strict-SDK npm update changes only
`android/build.gradle`; the AAR, native/Java, Metro and common API bytes are
preserved exactly, without claiming a fresh native build. Final Windows
frozen wildcard/manual-WASM analysis passed 252 targets,
74,497 configured targets and zero build actions at jobs=2. This analysis
also includes the integrated handed-off CUDA test graph, but does not execute
GPU or Apple-incompatible targets. Artifacts and logs were preserved before
stopping the RN Bazel server and removing only its completed private 7.8 GiB
VM root, source-transfer archive, and SSH credentials; `/home` retained 256 GiB
free. The independent CUDA task's files were untouched.

`//js/react_native:ios_adapter_framework` adds the existing Objective-C++ JSI
adapter and shared C++ implementation to a native static framework against
the matching official React Native source headers. Its final application
link requires the host's React runtime. The explicit macOS CI step builds
this target for simulator ARM64; no successful Apple compilation or device
execution is claimed until observed.

`//objectivec:ios_framework` uses native `rules_apple` static framework
assembly with the existing Objective-C inference wrappers, public headers,
and source-built CPU runtime. `//objectivec:ios_inference_tests` reuses the
existing XCTest sources and ORT model resources in an iOS test bundle.
Both require a macOS host with full Xcode and are manual, separate from the
desktop wildcard; they do not invoke CocoaPods or a legacy runtime build.
The deployment minimum is iOS 15.1. For example, on an Apple Silicon Mac:

```sh
bazel build -c opt //objectivec:ios_framework --ios_multi_cpus=arm64
bazel build -c opt //objectivec:ios_framework //objectivec:ios_inference_tests --ios_multi_cpus=sim_arm64 --lockfile_mode=error
```

The second command is included in macOS CI as an explicit native framework
and test-bundle build, not simulator execution. Neither native
iOS compilation, framework import, simulator execution, nor device execution
has been verified locally. Windows explicitly rejects these targets as
incompatible rather than substituting a desktop SDK.

The existing React Native data-loading policy tests now run directly as
`//js/react_native:data_policy_test`. Windows source-native execution passed
all checks (nine build/test actions); this policy test alone does **not**
validate the JSI adapter. The genuine Android bridge build is described
below. Full Java/iOS React Native application integration is not supplied
by the policy target. No header-only React substitute is used.
No mobile application integration, CoreML/NNAPI provider, or device
inference is claimed by this policy-test follow-up. Integrated Windows frozen wildcard
plus manual WASM analysis passed 244 targets with zero actions; this skips
Apple-incompatible targets and is not iOS compilation evidence.

The Android bridge graph additionally provides
`//js/react_native:libonnxruntimejsi.so` and
`//js/react_native:android_jsi_native_archive`. These compile the actual
`cpp-adapter.cpp` and shared JSI implementation against checksum-pinned
official ReactAndroid 0.73.11 and fbjni 0.5.1 prefab SDKs. Those versions match
the repository's React Native development dependency and its published Maven
dependency; newer CallInvoker interfaces are not ABI-compatible. ORT itself
still builds from source. The SDK imports supply no prebuilt ORT runtime.

```sh
bazel build --config=android_arm64 -c opt //js/react_native:android_jsi_native_archive --jobs=8 --loading_phase_threads=2 --lockfile_mode=error --build_python_zip=false
```

The native ZIP contains ARM64 ORT and JSI adapter libraries plus the actual
NDK r27d shared C++ runtime, not a complete
React Native AAR or npm package. An application must supply the matching
React Native/fbjni SDK. The bridge deliberately does
not enable NNAPI, CoreML, QNN, or extensions. Java adapter classes, Android
application installation, iOS React Native linkage, and device inference
remain separate from native library compilation.

The actual Linux-host cross-build passed with NDK r27d and Bazel 8.4.2,
`--jobs=8 --loading_phase_threads=2`, strict frozen lock and a task-private
output/cache root. It compiled the real JNI/JSI adapter and source-native
ORT runtime and linked both with `--no-undefined`. Initial SDK setup took
21 seconds. The first attempt exposed an incompatible newer CallInvoker
interface; using the repository-matching SDK fixed compilation without
changing adapter implementation. A subsequent link failure exposed the NDK
rules' empty dynamic C++ runtime filegroup; the graph now explicitly imports
and packages the actual r27d shared runtime instead of disabling undefined
symbol checks or linking a second static C++ runtime into the adapter.
The successful source build after these corrections took 267.068 seconds
wall and executed 854 actions, reusing 554 unchanged actions from the failed
attempts. This is not a clean-output build timing or device inference result.
The final archive rebuild took 0.691 seconds and its unchanged repeat took
0.294 seconds (one internal action, no native compilation). All three payloads
were verified as AArch64 ELF; the adapter exports `JNI_OnLoad`, uses 16 KB
load-segment alignment, and has the expected ORT/React/fbjni/shared-C++
dependencies. ZIP membership and notices/version were checked.
Artifact SHA256:
`7ddc14837ba31a35f8f603de70738e5add15a7a248e628d218149bcc238192d4`.
The VM was an Intel Xeon Platinum 8480C Linux host with 96 logical CPUs;
this build used eight jobs, not all available CPUs. Artifacts and logs were
preserved before stopping the task's Bazel server and removing only its
private 5.3 GiB VM root and SSH credentials.

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
above; React Native has the bounded Android native bridge described in the
mobile follow-up, not complete application integration. Native Node coverage
alone does not migrate either surface.

### Foundation validation (before the language test layers)

Before cpuinfo integration,
`bazel test //... -c opt --jobs=16 --keep_going --nocache_test_results` passed
all seven default test targets on Azure Linux 3 with GCC 13.2, with
`CUDA_PATH`, `CUDNN_PATH`, and `LD_LIBRARY_PATH` unset. This includes all five
desktop-language smoke/unit targets and 20 existing C++ utility tests.

A fresh Linux source build also passed all seven explicit targets with frozen
locks after the portability fixes below (2,390 build actions, followed by an
uncached seven-target test run). The Go wrapper preserves rules_go's run
environment so its configured test directory and runtime-library path agree.

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
The updated workflow YAML parsed successfully and all Bash steps passed
`bash -n`; the matrix and absence of wildcard test execution were also checked.

Remaining work includes remaining provider/operator/contrib test coverage,
additional EPs, additional architecture variants, complete release packaging,
cross-compilation, sanitizers, and the CUDA/toolchain matrix. Those are separate
native targets/toolchains to add, not reasons to delegate back to the old build
system.

## Optimization and size parity

Correctness results above are not inference-performance or binary-size
comparisons against the release build. The current source-level audit found:

- The pinned rules_cc native Linux toolchain uses `-O2`, `-DNDEBUG`,
  function/data sections, and linker garbage collection in `-c opt` mode.
  Its MSVC toolchain uses `/O2`, `/DNDEBUG`, and
  `/OPT:ICF`/`/OPT:REF`. These are toolchain defaults, not evidence that every
  effective compile/link action matches a particular release configuration.
- MLAS ISA-specific sources retain their separate architecture flags and
  runtime dispatch. The runtime and baseline MLAS archives use `alwayslink`
  for registration and link-order correctness; this is not a minimal build.
- This graph does not enable LTO explicitly. The CMake option
  `onnxruntime_ENABLE_LTO` also defaults to off, but a release pipeline can
  enable it. Comparing against such a build requires matching that setting.
- The audit found missing PyTorch cpuinfo integration. Native Linux, Windows,
  and macOS now compile its existing pinned source with upstream Bazel rules
  and ORT's existing lifecycle/sysfs patches. `CPUINFO_SUPPORTED` and the
  dependency are propagated together. Other target operating systems retain
  their prior fallback detection; this does not enable cpuinfo for WASM or
  Android. The standalone `//:cpuinfo_test` checks vendor/topology detection,
  reference counting, and reinitialization; the runtime smoke test checks ORT's
  detected vendor as well. This closes a source-graph gap, not proof of
  equivalent kernel dispatch or inference performance.
- Contrib operators are disabled in this foundation. Binary sizes must be
  compared with matching operator/provider coverage, architecture, debug
  information, and stripping settings.

The new standalone cpuinfo test passed a fresh Windows MSVC build with two
jobs: 25 actions, 51.35 seconds wall time including startup/dependency analysis,
and 0.2 seconds test execution. No ORT runtime was rebuilt for this isolated
check. The preceding foundation revision `d473c4e677` passed all seven native
runtime/binding targets in Linux, Windows, and macOS CI; those results do not
validate later changes.

### Timed Linux comparison

A controlled CPU-only comparison used source `f268b6bf43`, Azure Linux 3,
GCC 13.2, and an Intel Xeon Platinum 8480C VM with 96 vCPUs and 1.8 TiB RAM.
Both builds used 24 jobs, C++20, `-O2 -DNDEBUG -D_FORTIFY_SOURCE=1`, full
protobuf, cpuinfo, no contrib operators, no CUDA, and no LTO. CMake was invoked
independently with Ninja solely as a measurement baseline; it is not part of
the Bazel graph. This is not a comparison against an official release artifact
or an untouched CMake Release configuration.

| Runtime build at `f268b6bf43` | Bazel 8.4.2 | CMake/Ninja |
| --- | ---: | ---: |
| Fresh native build, wall time | 176.41 s | 164.63 s |
| Separate configure, wall time | Included above | 8.77 s |
| Configure plus fresh build | 176.41 s | 173.40 s |
| No-change build, wall time | 0.559 s | 0.064 s |
| Unstripped shared library | 25,539,768 bytes | 26,288,464 bytes |
| Stripped shared library | 19,466,944 bytes | 20,546,224 bytes |

Bazel performed 1,422 actions for the fresh runtime. Its subsequent
`//:cpuinfo_test //:runtime_smoke_test --nocache_test_results` took 139.20 s,
including 882 additional native compilations for the static test variant;
this is not pure test execution time. Both tests passed, with Intel vendor
detection and complete Add/MatMul/Conv output checks. Build directories and
dependency caches were task-private and initially empty; shared-host OS caches
were not cleared. Tool installation and source extraction are outside these
timings. These single-run build measurements do not establish a build-speed
advantage.

Initial inference trials exposed an approximately 9% Conv latency gap. A
controlled MLAS-only `-fomit-frame-pointer` experiment removed most of it:
the Bazel Linux toolchain otherwise supplies `-fno-omit-frame-pointer`, whereas
the controlled CMake `-O2` build omits frame pointers by default. The native
graph now applies `-fomit-frame-pointer` only to Linux x86-64 MLAS targets in
`-c opt` mode. Debug, Windows, and macOS flags are unchanged; other runtime
code retains the toolchain defaults. For frame-pointer-based profiling, this
MLAS choice can be overridden with
`--per_file_copt='onnxruntime/core/mlas/lib/.*@-fno-omit-frame-pointer'`.

The actual selector-based change, without experimental command-line compiler
overrides, rebuilt and passed the two native tests in 18.40 s (187 actions,
2,125 action-cache hits). Linux optimized/debug action inspection confirmed
the flag is respectively present/absent; Windows optimized analysis confirmed
it is absent, with zero native build actions.

Final inference measurements used five alternating trials per library, pinned
to the same allowed logical CPU, one intra/inter-op thread, graph optimizations
enabled, 20 warm-ups, and 100 measured runs per model. The same independently
compiled `dlopen` harness checked every output element against deterministic
all-ones references before timing. Entries are medians across the five trial
medians or p95s, in microseconds:

| FP32 model | Bazel median | CMake median | Bazel p95 | CMake p95 |
| --- | ---: | ---: | ---: | ---: |
| Add, 1,024 elements | 1.408 | 1.398 | 1.461 | 1.451 |
| MatMul, 512 x 512 | 1,503.95 | 1,494.22 | 1,536.78 | 1,531.19 |
| Conv, 1 x 16 x 64 x 64, 32 filters, 3 x 3 | 361.60 | 358.57 | 421.64 | 380.06 |

The unchanged pre-fix Bazel library measured 396.62 microseconds for Conv in
the same final trial batch. The fixed stripped library is 19,471,680 bytes,
about 5.2% smaller than this CMake baseline. Both final libraries measured
26,112 KiB process peak RSS. Library loading plus environment and first Add
session construction took median 12.42 ms for Bazel and 12.83 ms for CMake;
timing only session construction would misleadingly omit initialization work
performed earlier by CMake.

The three warmed median latencies are within 1% in this bounded experiment;
tail latency varies on the shared VM, and these are not representative-model,
multithreaded, GPU, or cross-platform performance results. The 114 compiled
MLAS runtime source files match, but other definitions still differ, including
RTTI, stream support, and Eigen flags. Smaller size alone therefore does not
prove equivalent feature coverage or better optimization. Release parity
requires broader feature/flag alignment and model benchmarks, not merely
matching `-O2` or passing these microbenchmarks.
