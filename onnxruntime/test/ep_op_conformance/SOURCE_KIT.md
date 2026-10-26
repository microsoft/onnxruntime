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
header API, and every packaged file hash. The companion case artifact contains canonical `cases.zip`, a separate
CI-only `negative-fixtures.zip`, and producer/input digests; binaries are never independently authored goldens.
`tools/ci/package_artifacts.py verify` checks the three CI artifacts before compilation. Native scenario reports
retain the three package manifests and exact generation metadata.

## Package Cases

`tools/create_cases_from_json.py --definitions cases/pilot.json --output C:/cases/generated` creates cases from
authored JSON; `--build-cache` selects immutable build-cache output instead. Existing JSON-generated or C++-exported
cases can be packaged independently using Python 3.10+ with ONNX and NumPy:

```powershell
python tools/package_cases.py create --cases C:/cases/generated --output C:/cases/preview-1.zip `
  --version preview-1 --revision SOURCE_REVISION --license LICENSE --notices ThirdPartyNotices.txt
python tools/package_cases.py verify --archive C:/cases/preview-1.zip --revision SOURCE_REVISION
Expand-Archive C:/cases/preview-1.zip -DestinationPath C:/cases/unpacked
```

Pass `C:/cases/unpacked/cases` to the runner. Packaging preserves case/model/tensor bytes, validates the supported
structure and dependency closure, and records provenance, hashes, case IDs, and model IR/opset requirements. Supply
applicable notices for the inputs; `--provenance` optionally retains a producer JSON object. Packaging does not run
inference or establish correctness or provider compatibility. Its public commands cannot bypass case validation.
Model IR/opset versions are recorded rather than checked against the installed ONNX package's schema ceiling;
packaging does not validate operator semantics or runtime support.
Source-kit/runtime packaging and negative-fixture preparation remain CI-only under `tools/ci/`; publication and
release version policy are not implemented.