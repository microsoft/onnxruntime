# EP Op Conformance Source Kit

This CI preview builds a public-API-only native operator tester. It is not a published release kit or proof of
WebGPU parity. The accompanying case archive and runtime/plugin fixture have independent hash manifests.

## Build

On Windows x64 with Visual Studio 2022, CMake 3.26+, and Python 3.10+:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DEP_OP_CONFORMANCE_FETCH_DEPS=ON
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

The kit uses only its own public ORT headers under `include/`, plus hash-pinned public ONNX, protobuf, and JSON
dependencies downloaded by CMake. No ORT checkout, source build, private headers, or private libraries are needed.
Python may be used by dependency generation during the build. The license, third-party notices, and version file
accompany these sources. Build dependencies and archive hashes are declared in the top-level CMake file.

## Execute

```powershell
./build/Release/onnxruntime_ep_op_conformance_test.exe --runtime C:/runtime/onnxruntime.dll `
  --profile examples/cpu.json --cases C:/cases/positive --report reports/cpu.json
```

Create the report directory before execution. Native execution needs no Python or source kit. Windows, PowerShell
for the scenario driver, and the MSVC x64 redistributable are host prerequisites. Runtime/provider dependencies must
accompany the runtime. API 24 is the runner minimum; the source-built sample plugin must use its matching runtime,
not the separate ORT 1.24.4 CPU smoke. Profile library paths are relative to the profile file. The sample plugin
profile shows the bounded selection; `registration_name` determines the sample plugin's reported name.

The packaged contract document records the case, profile, CLI, comparison, assignment, exit-code, and report
contracts, and distinguishes implemented facilities from future plans. The source manifest identifies the revision,
header API, and every packaged file hash. The companion case package retains positives and producer/input digests;
binaries are never independently authored goldens. `tools/package_ci.py verify` checks the three packages before
compilation. Native scenario reports retain the three package manifests and exact generation metadata.