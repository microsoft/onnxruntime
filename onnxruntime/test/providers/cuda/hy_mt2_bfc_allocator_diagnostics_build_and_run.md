# HY-MT2 Foundry Local BFC Diagnostics Build and Run Guide

## Scope

This guide records the Windows workflow used to build source-matched ONNX
Runtime and ONNX Runtime GenAI CUDA binaries, integrate them with Foundry
Local, run the HY-MT2 translation benchmark, and capture diagnostic BFC
allocator statistics.

The numerical results and their interpretation are in
[HY-MT2 ORT Versus llama.cpp Memory Gap Analysis](hy_mt2_ort_llama_memory_gap_analysis.md).

This is a reproduction guide for the diagnostic branch, not a supported
public Foundry Local deployment recipe. ONNX Runtime and ONNX Runtime GenAI
are built from source. The Foundry Local C++ SDK and
`Microsoft.AI.Foundry.Local.Core.dll` come from the matching Foundry Local
SDK checkout/package.

## Validated Configuration

| Component | Validated value |
|---|---|
| OS | Windows |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| GPU architecture | SM120 |
| NVIDIA driver | 610.62 |
| CUDA | 12.8 |
| cuDNN | 9.5, included in the CUDA 12.8 SDK tree |
| Visual Studio | Visual Studio 2022 Build Tools |
| Windows SDK | 10.0.26100.0 |
| ONNX Runtime branch | `chilo/static-workspace-preallocation-test` |
| ONNX Runtime source base recorded by the benchmark manifest | `e648c9c2fe684082d0c1b786049215ee2dbbd570` plus the then-uncommitted diagnostic changes |
| ONNX Runtime clean reproduction commit | `696d3fbe19aefc3c407a20fcfc95e9ff68c5e4ea` |
| ONNX Runtime GenAI commit | `ed5f4e87147731e5b07810f9f5c90103b3603cdf` |
| Foundry Local Core | `1.3.0-dev-202607231959-200c04bf` |
| llama.cpp comparison build | `b10156` |

Use CUDA 12.8 rather than the system CUDA 13.3 installation. The validated
CUDA 12.8 tree contains the matching cuDNN 9.5 headers, import libraries, and
runtime DLLs used by both builds and the deployed process.

## Directory Variables

The commands below use the paths from the validated machine. Change only the
root variables when reproducing elsewhere.

```powershell
$OrtRoot = "C:\Users\lochi\repos\onnxruntime"
$OrtBuild = "$OrtRoot\build\Windows\Release"
$WorkRoot = "C:\Users\lochi\hy_mt2_wayne_repro"
$OrtInstall = "$WorkRoot\ort-prealloc-install"
$GenAiRoot = "$WorkRoot\onnxruntime-genai"
$GenAiBuild = "$GenAiRoot\build-prealloc\Release"
$FoundrySdkRoot = "C:\foundry-local-july23-customer-like-validation\Foundry-Local\sdk\cpp"
$FoundrySdkBuild = "$FoundrySdkRoot\out\build\x64-release"
$RunnerSource = "$WorkRoot\runner-native\main.cpp"
$RunnerBuild = "$WorkRoot\runner-native\build-diagnostic"
$Deploy = "$WorkRoot\benchmark\translate_coherent_source_diagnostic"
$ModelCache = "$WorkRoot\foundry-cache"
$ModelDir = "$ModelCache\Tencent\hy-mt2-cuda-1\v1"
$FoundryAppData = "$WorkRoot\foundry-appdata-diagnostic"
$FoundryEpDir = "$FoundryAppData\ep\cuda-ep"
$CudaRoot = "C:\agent\_work\1\s\cuda_sdk\v12.8"
```

Run all commands from a regular 64-bit PowerShell unless a section explicitly
requires a Visual Studio Developer PowerShell.

## 1. Build ONNX Runtime with the CUDA EP

Create or activate an isolated Python environment:

```powershell
Set-Location $OrtRoot
if (-not (Test-Path .venv)) {
    py -3.12 -m venv .venv
}
.\.venv\Scripts\Activate.ps1
git checkout 696d3fbe19aefc3c407a20fcfc95e9ff68c5e4ea
```

Configure and build a Release shared-library build:

```powershell
python tools\ci_build\build.py `
    --config Release `
    --build_dir $OrtBuild `
    --use_cuda `
    --cuda_home $CudaRoot `
    --build_shared_lib `
    --parallel `
    --update `
    --build `
    --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=120-real;120-virtual"
```

The architecture value must contain both entries:

```text
120-real;120-virtual
```

The MSVC CUDA LLM target filters out SM120 real code. If the build is
configured with only `120-real`, that target can be left without a device
architecture and later fail with unresolved CUDA kernel symbols.

The expected runtime outputs are:

```text
$OrtBuild\Release\onnxruntime.dll
$OrtBuild\Release\onnxruntime_providers_shared.dll
$OrtBuild\Release\onnxruntime_providers_cuda.dll
```

Install the build so that ORT GenAI can consume one coherent include/library
prefix:

```powershell
cmake --install $OrtBuild --config Release --prefix $OrtInstall
```

The install prefix should contain:

```text
$OrtInstall\include\onnxruntime_c_api.h
$OrtInstall\bin\onnxruntime.dll
$OrtInstall\bin\onnxruntime_providers_shared.dll
$OrtInstall\bin\onnxruntime_providers_cuda.dll
$OrtInstall\lib\cmake\onnxruntime
```

The diagnostic API is experimental and is looked up by this exact name:

```text
OrtApi_DebugLogAndShrinkGpuArenas_SinceV29
```

It is available only when `onnxruntime.dll` was built from a source revision
that includes the diagnostic implementation.

## 2. Build ONNX Runtime GenAI Against That ORT Install

Check out the recorded ORT GenAI revision:

```powershell
Set-Location $GenAiRoot
git checkout ed5f4e87147731e5b07810f9f5c90103b3603cdf
```

Configure the CUDA Release build:

```powershell
cmake `
    -G "Visual Studio 17 2022" `
    -T "cuda=$CudaRoot" `
    -DCMAKE_BUILD_TYPE=Release `
    -S $GenAiRoot `
    -B $GenAiBuild `
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
    -DUSE_CUDA=ON `
    -DUSE_TRT_RTX=OFF `
    -DUSE_DML=OFF `
    -DUSE_WINML=OFF `
    -DENABLE_JAVA=OFF `
    -DBUILD_WHEEL=OFF `
    -DUSE_GUIDANCE=OFF `
    -DPUBLISH_JAVA_MAVEN_LOCAL=OFF `
    -DENABLE_TELEMETRY=ON `
    -DORT_HOME=$OrtInstall `
    -DCMAKE_CUDA_COMPILER="$CudaRoot\bin\nvcc"
```

Build:

```powershell
cmake --build $GenAiBuild --config Release --parallel
```

The required outputs are:

```text
$GenAiBuild\Release\onnxruntime-genai.dll
$GenAiBuild\Release\onnxruntime-genai-cuda.dll
```

Do not combine these DLLs with an ORT NuGet package or a different ORT source
build. ORT GenAI and `onnxruntime.dll` must agree on the C API and runtime
implementation.

## 3. Build the Foundry Local C++ SDK

The validated Foundry SDK uses the `x64-release` preset, Ninja, and the
`x64-windows-static-md` vcpkg triplet. Run this section from a Visual Studio
2022 Developer PowerShell:

```powershell
$env:VCPKG_ROOT = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\vcpkg"
Set-Location $FoundrySdkRoot
cmake --preset x64-release
cmake --build --preset x64-release --parallel
```

The runner uses:

```text
$FoundrySdkRoot\include\foundry_local.h
$FoundrySdkBuild\CppSdk.lib
$FoundrySdkBuild\vcpkg_installed\x64-windows-static-md\include
```

The SDK configure step also downloads packaged ORT and ORT GenAI binaries.
Those packaged runtime DLLs are not used in the coherent diagnostic
deployment. Only `CppSdk.lib` and the matching
`Microsoft.AI.Foundry.Local.Core.dll` are retained.

## 4. Build the Diagnostic Foundry Runner

The runner:

- creates the `bililive_translate_pipe` named pipe expected by the benchmark;
- initializes Foundry Local and loads `hy-mt2-cuda:1`;
- sets maximum output tokens to 100, temperature to 0.7, and top-p to 0.6;
- resolves the ORT diagnostic function through `OrtGetApiBase()` and
  `GetExperimentalFunction()`;
- logs `post_initialize`;
- logs and shrinks at the configured completed-request count; and
- logs `post_cached_run` if another request follows.

Build it from a Visual Studio 2022 Developer PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path $RunnerBuild | Out-Null

cl /nologo /std:c++20 /EHsc /O2 /MD `
    /D_WIN32_WINNT=0x0A00 /DWINVER=0x0A00 `
    /I"$FoundrySdkRoot\include" `
    /I"$FoundrySdkBuild\vcpkg_installed\x64-windows-static-md\include" `
    /I"$OrtInstall\include" `
    $RunnerSource `
    /Fo"$RunnerBuild\main.obj" `
    /Fe"$RunnerBuild\livehime_translate.exe" `
    /link `
    /LIBPATH:"$FoundrySdkBuild" `
    CppSdk.lib d3d11.lib dxgi.lib
```

This exact include/link set was validated by compiling the recorded
`runner-native\main.cpp` from scratch. No ORT import library is needed: the
runner obtains `OrtGetApiBase` from the already loaded `onnxruntime.dll`.
The Foundry SDK itself dynamically loads
`Microsoft.AI.Foundry.Local.Core.dll`.

## 5. Assemble One Coherent Runtime Directory

Create a clean deployment directory:

```powershell
New-Item -ItemType Directory -Force -Path $Deploy | Out-Null
```

Copy the runner, source-built ORT, and source-built ORT GenAI:

```powershell
Copy-Item "$RunnerBuild\livehime_translate.exe" $Deploy

Copy-Item "$OrtBuild\Release\onnxruntime.dll" $Deploy
Copy-Item "$OrtBuild\Release\onnxruntime_providers_shared.dll" $Deploy
Copy-Item "$OrtBuild\Release\onnxruntime_providers_cuda.dll" $Deploy

Copy-Item "$GenAiBuild\Release\onnxruntime-genai.dll" $Deploy
Copy-Item "$GenAiBuild\Release\onnxruntime-genai-cuda.dll" $Deploy
```

Copy the matching Foundry Local Core DLL:

```powershell
$FoundryCore = Get-ChildItem `
    -Path "$FoundrySdkBuild\_native_deps\Microsoft.AI.Foundry.Local.Core.*\runtimes\win-x64\native\Microsoft.AI.Foundry.Local.Core.dll" |
    Select-Object -First 1

if (-not $FoundryCore) {
    throw "Microsoft.AI.Foundry.Local.Core.dll was not found."
}
Copy-Item $FoundryCore.FullName $Deploy
```

Copy the CUDA and cuDNN runtime dependencies used by the validated
deployment:

```powershell
$CudaRuntimeDlls = @(
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudart64_12.dll",
    "cufft64_11.dll",
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll"
)

foreach ($name in $CudaRuntimeDlls) {
    $source = Join-Path "$CudaRoot\bin" $name
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing CUDA runtime DLL: $source"
    }
    Copy-Item -LiteralPath $source -Destination $Deploy
}
```

The resulting directory must contain exactly one selected copy of each
runtime family:

```text
livehime_translate.exe
Microsoft.AI.Foundry.Local.Core.dll
onnxruntime.dll
onnxruntime_providers_shared.dll
onnxruntime_providers_cuda.dll
onnxruntime-genai.dll
onnxruntime-genai-cuda.dll
CUDA and cuDNN runtime DLLs
```

### Verify Runtime Identity

Record hashes before running:

```powershell
$RuntimeFiles = @(
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime-genai.dll",
    "onnxruntime-genai-cuda.dll",
    "Microsoft.AI.Foundry.Local.Core.dll",
    "livehime_translate.exe"
)

Get-FileHash -Algorithm SHA256 `
    -LiteralPath ($RuntimeFiles | ForEach-Object { Join-Path $Deploy $_ }) |
    Sort-Object Path |
    Format-Table Path, Hash -AutoSize
```

For the recorded deployment, the source-built hashes were:

| File | SHA-256 |
|---|---|
| `onnxruntime.dll` | `F2AB6D6CA327C7EB1C0AF8124AD002926913B4EFE1078430F99158DDC3C55905` |
| `onnxruntime_providers_shared.dll` | `098A83E778040863727F6D04306842A741FDEDD0A712A0CF0A91551A0DC95EE7` |
| `onnxruntime_providers_cuda.dll` | `F6785E36707EAD7DB10A38EEDE4CAF7555AED7D43F45596413D6AAC08125AB9D` |
| `onnxruntime-genai.dll` | `44B5E1119665BDF83820CF963318DF4293B894833226DBEE53ABB8721109E1B6` |
| `onnxruntime-genai-cuda.dll` | `BAF2AB223F97B40B7160D57A7AC795271D114BB0EFC7B32ED89E1E536741907A` |

Hashes from a new source revision will differ. The invariant is that every
copy used by the process comes from the same intended ORT/ORT GenAI build.

## 6. Prepare the Foundry Model Cache and EP Directory

The recorded model-cache layout is:

```text
$ModelCache
+-- Tencent
    +-- foundry.modelinfo.json
    +-- hy-mt2-cuda-1
        +-- v1
            +-- chat_template.jinja
            +-- genai_config.json
            +-- inference_model.json
            +-- model.onnx
            +-- model.onnx.data
            +-- tokenizer.json
            +-- tokenizer_config.json
```

`inference_model.json` contains:

```json
{"Name":"hy-mt2-cuda:1","PromptTemplate":null}
```

`Tencent\foundry.modelinfo.json` must contain a cached local entry equivalent
to:

```json
{
  "createdAt": 0,
  "id": "hy-mt2-cuda:1",
  "name": "hy-mt2-cuda",
  "version": 1,
  "alias": "hy-mt2-cuda",
  "displayName": "hy-mt2-cuda",
  "providerType": "Local",
  "uri": "local://hy-mt2-cuda",
  "modelType": "ONNX",
  "cached": true
}
```

The entry belongs in the top-level `models` array. The benchmark passes the
exact model identifier `hy-mt2-cuda:1`.

Create a coherent EP directory for the benchmark PATH prefix:

```powershell
New-Item -ItemType Directory -Force -Path $FoundryEpDir | Out-Null
Copy-Item "$Deploy\onnxruntime_providers_cuda.dll" $FoundryEpDir
Copy-Item "$Deploy\onnxruntime-genai-cuda.dll" $FoundryEpDir
foreach ($name in $CudaRuntimeDlls) {
    Copy-Item (Join-Path $Deploy $name) $FoundryEpDir
}
```

The runner is executed with Foundry EP download/registration disabled. This
prevents Foundry from replacing the selected source-built CUDA binaries with
packaged binaries:

```powershell
$env:ORT_SKIP_FOUNDRY_EP_REGISTRATION = "1"
```

If the same DLL exists in both `$Deploy` and `$FoundryEpDir`, verify that the
hashes match.

## 7. Configure BFC Diagnostics

Set these variables before starting the runner:

```powershell
$env:ORT_SKIP_FOUNDRY_EP_REGISTRATION = "1"
$env:ORT_ARENA_DIAGNOSTICS = "1"
$env:ORT_ARENA_DIAGNOSTIC_WARMUP_REQUESTS = "133"
$env:ORT_BENCH_CAPTURE_RUNNER_LOGS = "1"
```

`ORT_ARENA_DIAGNOSTICS=1` registers each GPU BFC arena when it is created.
Without it, the diagnostic call succeeds but reports zero arenas.

The value 133 means:

```text
3 warmup requests + 130 measured requests = 133 completed requests
```

The runner emits:

1. `post_initialize`, with `phase=snapshot` and no shrink.
2. `post_warmup`, with `phase=before_shrink`.
3. A real `BFCArena::Shrink()` call.
4. `post_warmup`, with `phase=after_shrink`.
5. `post_cached_run` only if request 134 is sent.

For the selected 3-warmup/130-sample workload, `post_warmup
phase=before_shrink` is the final allocator state. `phase=after_shrink`
exists only to prove which complete unused BFC regions are reclaimable.

## 8. Run the 130-Sample Benchmark

The recorded scripts are external experiment artifacts:

```powershell
$BenchmarkRoot = "$WorkRoot\benchmark"
$CompareScript = "$BenchmarkRoot\compare_hy_q4_cuda_serial.ps1"
$LlamaExe = "$BenchmarkRoot\translate_llama_b10156\livehime_translate_llama.exe"
$LlamaModel = "$WorkRoot\models\Hy-MT2-1.8B-Q4_K_M.gguf"
$GenAiConfig = "$ModelDir\genai_config.json"
```

The benchmark protocol is:

- fresh process per engine;
- llama.cpp exits before Foundry starts;
- three warmups;
- 130 identical measured inputs;
- 100 maximum completion tokens;
- temperature 0.7;
- top-p 0.6;
- top-k 20;
- repetition penalty 1.05; and
- no shrink before or during the selected workload.

The runner sets the first three generation values. The comparison wrapper
temporarily sets top-k and repetition penalty in `genai_config.json`, applies
the requested session option, and restores the original file bytes in a
`finally` block.

### Default BFC Initializers

Do not pass `-UseDeviceAllocatorForInitializers`:

```powershell
$DefaultOut = "$BenchmarkRoot\translate_benchmark_results\BFC_DIAGNOSTIC_DEFAULT"

& $CompareScript `
    -TranslateDir $Deploy `
    -LlamaExe $LlamaExe `
    -FoundryExe "$Deploy\livehime_translate.exe" `
    -LlamaModelPath $LlamaModel `
    -FoundryModelCacheDir $ModelCache `
    -FoundryAppDataDir $FoundryAppData `
    -FoundryEpDir $FoundryEpDir `
    -FoundryGenAiConfigPath $GenAiConfig `
    -FoundryModelAlias "hy-mt2-cuda:1" `
    -OutputDir $DefaultOut `
    -WarmupCount 3 `
    -SampleCount 130 `
    -TimeoutSec 300 `
    -KillExisting

Copy-Item "$Deploy\foundry.stdout.log" "$DefaultOut\foundry.arena.stdout.log" -Force
Copy-Item "$Deploy\foundry.stderr.log" "$DefaultOut\foundry.arena.stderr.log" -Force
```

This leaves
`session.use_device_allocator_for_initializers` absent, which is equivalent
to the default value `0`.

### Device-Allocator Initializers

Run again in fresh processes and add
`-UseDeviceAllocatorForInitializers`:

```powershell
$DeviceOut = "$BenchmarkRoot\translate_benchmark_results\BFC_DIAGNOSTIC_DEVICE_INITIALIZERS"

& $CompareScript `
    -TranslateDir $Deploy `
    -LlamaExe $LlamaExe `
    -FoundryExe "$Deploy\livehime_translate.exe" `
    -LlamaModelPath $LlamaModel `
    -FoundryModelCacheDir $ModelCache `
    -FoundryAppDataDir $FoundryAppData `
    -FoundryEpDir $FoundryEpDir `
    -FoundryGenAiConfigPath $GenAiConfig `
    -FoundryModelAlias "hy-mt2-cuda:1" `
    -OutputDir $DeviceOut `
    -WarmupCount 3 `
    -SampleCount 130 `
    -TimeoutSec 300 `
    -UseDeviceAllocatorForInitializers `
    -KillExisting

Copy-Item "$Deploy\foundry.stdout.log" "$DeviceOut\foundry.arena.stdout.log" -Force
Copy-Item "$Deploy\foundry.stderr.log" "$DeviceOut\foundry.arena.stderr.log" -Force
```

The wrapper adds:

```text
session.use_device_allocator_for_initializers=1
```

to the decoder session options for this run.

Copy the runner logs immediately after each configuration. Runner log capture
uses fixed file names beside the executable, so the next run overwrites them.

## 9. Verify That the Configuration Was Restored

Record the original model configuration hash before the run:

```powershell
$ConfigHashBefore = (Get-FileHash -Algorithm SHA256 $GenAiConfig).Hash
```

Verify it afterward:

```powershell
$ConfigHashAfter = (Get-FileHash -Algorithm SHA256 $GenAiConfig).Hash
if ($ConfigHashAfter -ne $ConfigHashBefore) {
    throw "genai_config.json was not restored."
}
```

The recorded restored hash was:

```text
01EEF87C41EE4D059BBAAB16700F9F506C7309BE4A31D43AE185AABD115233DE
```

Each output directory also receives `genai_config.effective.json`, which
records the temporary configuration actually used by that run.

## 10. Output Files

The comparison wrapper writes:

| Output | Contents |
|---|---|
| `comparison.json` | Per-engine timing, process memory snapshots, retained checkpoints, and translations |
| `comparison.csv` | Per-sample aligned output and latency |
| `summary.md` | Human-readable latency, resource, and output comparison |
| `q4_raw` | Raw llama.cpp benchmark output from current script revisions |
| `foundry_raw` | Raw Foundry benchmark output from current script revisions |
| `genai_config.effective.json` | Temporary ORT GenAI configuration used by the run |
| `foundry.arena.stdout.log` | Copied runner stdout containing allocator checkpoints |
| `foundry.arena.stderr.log` | Copied runner stderr |

Quickly verify the diagnostic phases:

```powershell
Select-String `
    -Path "$DeviceOut\foundry.arena.stdout.log" `
    -Pattern "\[ ARENA CHECKPOINT \]|\[ ARENA SHRINK \]|\[ RUNNER ARENA CHECKPOINT \]"
```

## 11. BFC Field Definitions

Each registered CUDA or CUDA-pinned arena writes one line per phase.

| Field | Meaning |
|---|---|
| `total_allocated_bytes` | All device memory currently owned through this arena |
| `reserved_bytes` | Direct allocations made through `BFCArena::Reserve()`, including device-allocator initializers |
| `bfc_region_bytes` | Capacity of normal BFC-managed regions |
| `bytes_in_use` | Occupied BFC chunks, including allocator rounding |
| `bytes_requested_in_use` | Bytes requested by currently live allocations |
| `arena_slack_bytes` | Free reusable bytes inside retained BFC regions |
| `internal_fragmentation_bytes` | Difference between occupied chunk bytes and requested live bytes |
| `max_bytes_in_use` | High-water mark for occupied BFC chunks |
| `max_alloc_size` | Largest individual allocation observed |
| `num_allocs` | Successful allocation count |
| `num_reserves` | Direct reserve count |
| `num_arena_extensions` | Current number of BFC allocation regions |
| `num_arena_shrinkages` | Number of complete BFC regions released |

The accounting identities are:

```text
total_allocated_bytes = reserved_bytes + bfc_region_bytes

bytes_in_use =
    bytes_requested_in_use + internal_fragmentation_bytes

bfc_region_bytes =
    bytes_in_use + arena_slack_bytes
```

Convert bytes to MiB using:

```text
MiB = bytes / 1,048,576
```

Sum `total_allocated_bytes` across all `allocator=Cuda` lines at the same
checkpoint and phase to obtain the total ORT-tracked CUDA allocation.
CUDA-pinned arenas are host-pinned memory and must not be added to dedicated
GPU memory.

The whole-process residual is:

```text
other process-local dedicated GPU memory =
    WDDM process dedicated memory
    - sum of ORT allocator total_allocated_bytes
```

The residual includes CUDA context state, CUDA/cuDNN library allocations,
Foundry Local Core allocations outside the registered ORT arenas, and other
process-local GPU resources.

## 12. Select the Correct Measurement

Use these phases consistently:

| Question | Selected measurement |
|---|---|
| ORT allocator state after model load | `post_initialize phase=snapshot` |
| ORT allocator state after all 3 warmups and 130 requests | `post_warmup phase=before_shrink` |
| Capacity proven reclaimable | Difference between `before_shrink` and `after_shrink` |
| Whole-process post-initialization memory | `comparison.json` `post_initialize.gpu_memory_mb` |
| Whole-process retained inference memory | Last WDDM checkpoint captured before shrink, or the pre-shrink peak |

Do not report `after_shrink` as the benchmark's final memory. Shrink is a
diagnostic intervention performed after the selected workload.

The shrink occurs before the response to completed request 133 is returned.
Consequently, a request-130 WDDM sample or the later `post_generation`
snapshot can observe post-shrink memory. In that case, use the stable
pre-shrink checkpoint, such as request 100, together with allocator
`before_shrink`.

`BFCArena::Shrink()` releases only complete regions that contain no live
allocation. `arena_slack_bytes` is reusable capacity, not necessarily
reclaimable capacity. Long-lived allocations can pin a region even when most
of its bytes are free.

## 13. Expected Diagnostic Signatures

For device-allocator initializers, the decoder/model-session arena should
show:

- approximately 1169.68 MiB in `reserved_bytes` after initialization;
- approximately 1 MiB in normal BFC regions after initialization;
- approximately 169 MiB in BFC regions before the final shrink; and
- approximately 168 MiB reclaimed by the shrink.

The separate ORT GenAI global CUDA allocator should show approximately
65 MiB of BFC capacity at the final checkpoint.

For default BFC initializers, the decoder arena should show:

- zero direct `reserved_bytes`;
- approximately 2049 MiB of BFC capacity after initialization;
- the same capacity before the final shrink; and
- zero reclaimed bytes because long-lived model allocations pin the regions.

These values are diagnostic signatures for the recorded model and runtime,
not general allocator constants.

## 14. Troubleshooting

### `arena_count=0`

`ORT_ARENA_DIAGNOSTICS=1` was not present when the arenas were constructed.
Set it before launching the benchmark process.

### `GPU arena diagnostic function is unavailable`

The process loaded an `onnxruntime.dll` without
`OrtApi_DebugLogAndShrinkGpuArenas_SinceV29`, usually because a packaged ORT
DLL appeared earlier in the runtime search path. Check loaded-module paths
and compare hashes in `$Deploy`, `$FoundryEpDir`, and any directory prepended
to `PATH`.

### CUDA provider load failure

Keep the CUDA 12.8 and cuDNN 9.5 DLLs next to the executable or in the single
selected PATH prefix. Do not allow CUDA 13.3 DLLs to satisfy part of the
dependency set.

### Foundry downloads or selects a packaged EP

Set:

```powershell
$env:ORT_SKIP_FOUNDRY_EP_REGISTRATION = "1"
```

before the process starts. The diagnostic runner deliberately bypasses
`DownloadAndRegisterEps()` when this variable is present.

### Unresolved CUDA LLM kernel symbols

Regenerate the ORT build with:

```text
CMAKE_CUDA_ARCHITECTURES=120-real;120-virtual
```

Do not use only `120-real` on the validated SM120 MSVC build.

### Request 130 reports less WDDM memory than request 100

The diagnostic shrink ran after the 133rd completed request and before the
last response reached the harness. Use `before_shrink` for allocator state
and the last stable pre-shrink WDDM checkpoint for process memory.

### Default and device-initializer runs have the same allocator layout

Inspect `genai_config.effective.json`. The device run must contain:

```json
"session.use_device_allocator_for_initializers": "1"
```

The default run must not contain that property.

### Results change after copying a new DLL

Re-copy all five ORT/ORT GenAI DLLs as one unit and regenerate the hash
manifest. Replacing only `onnxruntime_providers_cuda.dll` or only
`onnxruntime.dll` creates a mixed deployment and invalidates the result.
