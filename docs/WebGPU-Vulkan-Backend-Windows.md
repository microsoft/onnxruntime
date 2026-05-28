# WebGPU Vulkan Backend on Windows

Notes from getting the WebGPU EP to actually run on the Vulkan backend (not D3D12) on Windows. Validated 2026-05-13 on a Windows 11 dev box with an NVIDIA RTX 5060 Ti, NVIDIA driver 591.86, Vulkan loader 1.4.341.1.

## Why this is non-trivial

Two independent issues conspire to make `--use_webgpu --use_dawn_vulkan` (or `dawnBackendType=Vulkan` via session options) silently fall back to D3D12 on Windows when both backends are compiled in.

### Issue 1: `WebGpuContext::Initialize()` is wrapped in `std::call_once`

`onnxruntime/core/providers/webgpu/webgpu_context.cc`:

- The first call to `WebGpuContext::DefaultContext()` (or any `Initialize()` invocation) wins forever via `std::call_once`. Subsequent calls with different `WebGpuContextConfig` are silently ignored.
- The EP plugin factory and several utility paths call `DefaultContext()` early, before user provider options have been parsed. On Windows that defaulted to `WGPUBackendType_D3D12`. By the time the user-requested `Vulkan` backend reaches `Initialize()`, it's too late.
- Symptom: `loaded modules` lists only `d3d12.dll` / `nvwgf2umx.dll`, and `vulkan-1.dll` is absent. Benchmarks for "D3D12 vs Vulkan" return suspiciously identical numbers (within ~1.5%).

**Diagnosis**: temporarily add `LOGS_DEFAULT(WARNING) << "Initialize entry: backend_type=" << static_cast<int>(config.backend_type);` at the top of `WebGpuContext::Initialize()` and another at the start of `DefaultContext()`. You will see `backend_type=4` (D3D12) fire first, then `backend_type=6` (Vulkan) ignored.

**Workaround**: build with D3D12 disabled — `onnxruntime_ENABLE_DAWN_BACKEND_D3D12=OFF` — so the header-level default flips to Vulkan and the call_once race becomes a no-op. A proper fix would require either:
- Removing `call_once` and supporting per-session contexts keyed by `WebGpuContextConfig`, or
- Deferring `DefaultContext()` until after provider options are parsed (so the first call already carries the user's backend choice).

### Issue 2: Dawn's `DynamicLib::Open` cannot load `vulkan-1.dll` by bare filename

`_deps/dawn-src/src/dawn/common/DynamicLib.cpp` (lines ~91-101) uses:

```cpp
HMODULE module = LoadLibraryExA(
    filename.c_str(), nullptr,
    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
```

`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` requires an **absolute path**. When Dawn's Vulkan backend asks for the bare filename `"vulkan-1.dll"` (`BackendVk.cpp:363`), `LoadLibraryExA` returns `ERROR_INVALID_PARAMETER` (Windows error 87).

**Symptom** (from `dawn_native`):
```
Couldn't load Vulkan: DynamicLib.Open: vulkan-1.dll Windows Error: 87
    at BackendVk.cpp:363
```
Process exit: `STATUS_STACK_BUFFER_OVERRUN` (`-1073740791`). The error comes from the Dawn backend factory, propagates as an exception, and crashes through the C API boundary because no one catches it on the way back from `og.Model(...)`.

**Fix**: configure Dawn with `-DDAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON`. This switches `DynamicLib.cpp` to:

```cpp
HMODULE module = LoadLibraryExA(filename.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
```

which accepts bare filenames and resolves `vulkan-1.dll` to `C:\Windows\System32\vulkan-1.dll`. This is the canonical, Khronos-blessed loader (signed by LunarG, dispatches to driver ICDs via the registry).

Equivalent alternatives that were considered but not used:
- **B**: prepend an absolute path (e.g. `C:\Windows\System32`) to Dawn's `runtimeSearchPaths` so `_DLL_LOAD_DIR` can find the file. Requires a code patch.
- **C**: call `ctypes.WinDLL("kernel32").AddDllDirectory(r"C:\Windows\System32")` from Python before importing onnxruntime. Hacky and fragile.

Option A is upstream-supported, requires no source patch, and was added to Dawn precisely for this scenario.

## Build recipe (validated)

```powershell
.\build.bat `
    --config Release `
    --build_dir WGPU9 `
    --compile_no_warning_as_error `
    --parallel `
    --skip_submodule_sync `
    --skip_tests `
    --use_webgpu `
    --build_shared_lib `
    --enable_pybind `
    --build_wheel `
    --cmake_extra_defines `
        FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER `
        onnxruntime_BUILD_UNIT_TESTS=OFF `
        onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=ON `
        onnxruntime_ENABLE_DAWN_BACKEND_D3D12=OFF `
        DAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON `
        CMAKE_INSTALL_PREFIX=D:/tmp/ort `
    --cmake_generator "Visual Studio 17 2022"
```

Key flags explained:
- `onnxruntime_ENABLE_DAWN_BACKEND_D3D12=OFF` — works around Issue 1 (call_once race).
- `onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=ON` — builds the Vulkan backend in Dawn.
- `DAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON` — works around Issue 2 (LoadLibraryExA flag bug). **Required** if Vulkan is enabled on Windows.
- `FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER` — forces FetchContent to use the bundled deps from `cmake/deps.txt` rather than any system packages.
- `onnxruntime_BUILD_UNIT_TESTS=OFF` — saves ~10 minutes; tests don't run on this build anyway because of the wheel-only intent.

Wall time on this machine: ~22 minutes from clean.

## Runtime requirements

- `C:\Windows\System32\vulkan-1.dll` must exist. On most dev machines this is installed by:
  - The Vulkan SDK installer (LunarG, even though the SDK Bin folder no longer keeps a copy in modern versions),
  - A standalone `VulkanRT-x.y.z.w-Installer.exe`, or
  - The GPU vendor driver (NVIDIA / AMD / Intel installers all drop one).

  Check with: `(Get-Item C:\Windows\System32\vulkan-1.dll).VersionInfo.FileVersion`. Anything ≥ 1.1 works for Dawn; 1.3+ is current.

- A Vulkan-capable GPU + ICD driver. Verify with `vulkaninfo` (from Vulkan SDK) — both adapters should show up in `Devices`.

- No SDK files are needed at runtime. Validation layers, `glslc`, etc. are dev-only.

## Verifying Vulkan is actually selected (not silent D3D12 fallback)

Numbers can lie — D3D12 and Vulkan benchmarks land within ~1-2% of each other for many workloads, so identical-looking results don't prove anything. Better: enumerate loaded modules.

```python
import onnxruntime_genai as og
m = og.Model(r"D:\models\Qwen3-1.7B")
g = og.Generator(m, og.GeneratorParams(m))
g.append_tokens([1, 2, 3])
g.generate_next_token()

import psutil
mods = {mm.path for mm in psutil.Process().memory_maps()}
print("vulkan-1.dll loaded:", any('vulkan-1.dll' in p.lower() for p in mods))
print("d3d12.dll loaded:   ", any('d3d12.dll' in p.lower() for p in mods))
print("nvoglv64.dll (NV Vulkan ICD):", any('nvoglv64.dll' in p.lower() for p in mods))
```

Expected on a Vulkan-only build:
```
vulkan-1.dll loaded: True
d3d12.dll loaded:    False
nvoglv64.dll (NV Vulkan ICD): True
```

`dxgi.dll` will load even on a pure-Vulkan build — Dawn uses DXGI for adapter enumeration metadata. That is normal and not a sign of D3D12 use.

For a stronger signal, temporarily reinstate the `WebGpuContext::Initialize()` log probe and confirm `backend_type=6` (Vulkan).

## Selecting the backend at runtime

Once the build is right, the user-facing knob is the provider option `dawnBackendType`. Examples:

**ONNX Runtime GenAI `genai_config.json`:**
```json
"provider_options": [
  {
    "webgpu": {
      "dawnBackendType": "Vulkan"
    }
  }
]
```
or via the older flat key form:
```json
"ep.webgpuexecutionprovider.dawnBackendType": "Vulkan"
```

**Python session options:**
```python
sess_options = ort.SessionOptions()
sess = ort.InferenceSession(
    "model.onnx",
    sess_options,
    providers=[("WebGpuExecutionProvider", {"dawnBackendType": "Vulkan"})],
)
```

Accepted values: `D3D11`, `D3D12`, `Vulkan`, `Metal`, `OpenGL`, `OpenGLES`, `Null`. On a Vulkan-only build, anything other than `Vulkan` will either be ignored or fall back to whatever Dawn's default is for the current platform.

## Multi-GPU on Vulkan

Vulkan exposes every adapter (RTX 5060 Ti = device 1, T1000 = device 0 on this box). Dawn picks adapter 0 by default. To pick a different one:

- `dawnAdapterIndex` (if exposed in the EP option set), or
- Set `VK_LOADER_DEBUG=all` env var to see what adapters Dawn enumerates and in what order, then control via `VK_ICD_FILENAMES` to mask out unwanted ICDs.

This box hits an interesting case: `vulkaninfo` reports T1000 first, but Dawn's preference logic favors discrete GPU → RTX 5060 Ti. Worth verifying per-machine with the module-enumeration probe above plus a memory-bandwidth or shader benchmark.

## Decode performance: D3D12 vs Vulkan, with and without graph capture

Reproduced 2026-05-13 on RTX 5060 Ti, NVIDIA driver 591.86. Qwen3-1.7B (4-bit MatMulNBits), prompt = 3000 tokens, generation = 500 tokens, 5-run average via `D:\onnxruntime-genai\benchmark\python\benchmark_e2e.py` (driven by `D:\ort-web-perf\ort-llm\bench_matrix.py` — see *Reproducing the perf matrix* below).

| Backend                    | GC ON           | GC OFF          | Δ (GC vs no-GC)              |
|----------------------------|----------------:|----------------:|------------------------------|
| **D3D12** (WGPU7)          | **137.96** tok/s| 103.96 tok/s    | **+33 %** (GC helps)         |
| **Vulkan** (WGPU9)         | 72.20 tok/s     | **124.37** tok/s| **−42 %** (GC HURTS)         |

The Vulkan-with-graph-capture regression is real and severe — graph capture **cuts decode throughput by ~42 % on Vulkan**, while it **adds ~33 %** on D3D12. Without GC, Vulkan is competitive with D3D12 (within ~16 %). With GC, Vulkan collapses to roughly **half** of D3D12's GC throughput.

Hypothesised root causes (not yet confirmed):

1. **Dawn Vulkan command-buffer replay overhead**: replaying a recorded command buffer on Vulkan may force pipeline-barrier reconstruction or per-submit validation work that D3D12's bundle/`ExecuteIndirect` path avoids.
2. **Dawn's Vulkan GC implementation may re-record / re-translate** rather than performing a pure `vkQueueSubmit` of the cached `VkCommandBuffer`. Worth checking `Vulkan/CommandBufferVk.cpp` and `Vulkan/QueueVk.cpp` paths under capture-replay.
3. **NVIDIA Vulkan driver heuristic**: NV's Vulkan driver may treat repeated submits of the same `VkCommandBuffer` worse than D3D12's command-list reuse. Cross-check on Intel/AMD Vulkan would isolate this.

Matches the regression a colleague reported for Linux Vulkan-GC; same symptom reproduces on Windows once Vulkan is actually being exercised (per Issues 1 and 2 above).

## Reproducing the perf matrix

Two wheels are needed because the call_once race (Issue 1) means D3D12 and Vulkan can't coexist in one build right now:

- **D3D12 wheel**: `D:\onnxruntime2\WGPU7\Release\Release\dist\onnxruntime_webgpu-1.27.0-cp312-cp312-win_amd64.whl` — dual-backend build, defaults to D3D12 due to call_once.
- **Vulkan wheel**: `D:\onnxruntime2\WGPU9\Release\Release\dist\onnxruntime_webgpu-1.27.0-cp312-cp312-win_amd64.whl` — Vulkan-only build (`onnxruntime_ENABLE_DAWN_BACKEND_D3D12=OFF`, `DAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON`).

Measurement harness: `D:\onnxruntime-genai\benchmark\python\benchmark_e2e.py` (the upstream genai-side benchmark — forces a fixed `min_length` so decode runs the full 500 tokens and is not cut short by EOS). The thin wrapper `D:\ort-web-perf\ort-llm\bench_matrix.py` patches `genai_config.json` for the desired backend and GC flag, runs the benchmark, parses the `Average Token Generation Throughput (per token)` line, and prints a summary table.

`bench_matrix.py` flags:

- `--backend {D3D12|Vulkan}` and `--gc {on|off|both}` for a single backend (use `both` to do GC=on then GC=off back to back without re-typing).
- `--matrix` for the full 2×2 (only meaningful on a build that actually has both backends — currently neither WGPU7 nor WGPU9 do, so on this machine the matrix is run as two separate `--gc both` invocations, one per wheel).
- `--prompt`, `--gen`, `--runs` to override the canonical 3000 / 500 / 5.

NumPy version note: the WGPU7 wheel was compiled against NumPy 1.x and does not load under NumPy 2.x. WGPU9 was compiled against NumPy 2.x and loads under both. Stay on `numpy<2` (e.g. `pip install "numpy<2"`) to allow flipping between the two wheels.

```powershell
# Once: pin numpy
pip install "numpy<2" --quiet

# --- Vulkan cells (WGPU9) ---
pip install --force-reinstall --no-deps `
    D:\onnxruntime2\WGPU9\Release\Release\dist\onnxruntime_webgpu-1.27.0-cp312-cp312-win_amd64.whl
$dst = 'C:\Users\hasesh\AppData\Local\Programs\Python\Python312\Lib\site-packages\onnxruntime\capi'
Copy-Item D:\onnxruntime2\WGPU9\Release\Release\onnxruntime_pybind11_state.pyd, `
          D:\onnxruntime2\WGPU9\Release\Release\onnxruntime.dll $dst -Force
cd D:\ort-web-perf\ort-llm
python bench_matrix.py --backend Vulkan --gc both

# --- D3D12 cells (WGPU7) ---
pip install --force-reinstall --no-deps `
    D:\onnxruntime2\WGPU7\Release\Release\dist\onnxruntime_webgpu-1.27.0-cp312-cp312-win_amd64.whl
python bench_matrix.py --backend D3D12 --gc both
```

Why the patched `.pyd` overlay step matters for the WGPU9 wheel: the wheel as built (timestamp 2026-05-13 14:20) was produced *before* the `DAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON` reconfigure. The wheel file alone will hit the Windows error 87 from Issue 2. The overlaid `.pyd` and `.dll` (timestamp 2026-05-13 14:32) come from the post-reconfigure rebuild and are required for Vulkan to actually load. If you ever re-roll the WGPU9 wheel from scratch with the new flag baked in, the overlay step disappears.

What `bench_matrix.py` writes into `D:\models\Qwen3-1.7B\genai_config.json` (idempotent — the file is rewritten before each cell):

- `model.decoder.session_options["ep.webgpuexecutionprovider.dawnBackendType"]` ← `"D3D12"` or `"Vulkan"`
- `model.decoder.session_options["provider_options"][*]["WebGPU"]["enableGraphCapture"]` ← `"1"` or `"0"`

Sample output:

```
=== D3D12 GC=on ===
  decode = 137.96 tps  (rc=0)

=== D3D12 GC=off ===
  decode = 103.96 tps  (rc=0)

=== summary ===
backend  gc          tps  rc
D3D12    on       137.96  0
D3D12    off      103.96  0
```

To independently confirm a Vulkan run actually ran on Vulkan (not silent fallback) — important because requesting `D3D12` on the Vulkan-only WGPU9 build silently falls back to Vulkan:

```powershell
python d:\temp\canary_probe.py 2>&1 | Select-String "vulkan-1.dll|d3d12.dll|nvogl|nvwgf2"
```

Expected on Vulkan: `vulkan-1.dll` and `nvoglv64.dll`, **no** `d3d12.dll` / `nvwgf2umx.dll`. Inverse on D3D12.

## Open issues / TODO

- The call_once race (Issue 1) should be fixed properly so that a single build can ship with both backends and switch at runtime. Tracking idea: refactor `WebGpuContext` to a per-config map keyed by `(backend_type, adapter_index, ...)`.
- Investigate why `STATUS_STACK_BUFFER_OVERRUN` is the exit code rather than a clean Python exception. Likely an exception thrown from Dawn through the C API boundary in `OrtCreateSession` without `API_IMPL_BEGIN/END` protection on a path specific to EP factory init.
- Root-cause the **Vulkan + GC decode regression** (see perf matrix above). Tracked below in *Debugging the Vulkan + GC regression*.
- Sweep context length (e.g. 1024, 2048, 4096, 8192) to see if the Vulkan-GC regression is context-length-dependent.
- Try the same matrix on a non-NVIDIA GPU (Intel iGPU + Vulkan) to isolate whether the regression is driver-specific.

## Debugging the Vulkan + GC regression — work plan

**Goal**: land a single cherry-pickable commit on a topic branch that closes the −48 % Vulkan-with-graph-capture regression without regressing D3D12 + GC (currently +51 %).

**Branch**: `hari/webgpu_vulkan_gc_fix`, branched from `hari/webgpu_perf_1_full @ d5e208bffe`. The intent is one squashed commit that can be cherry-picked onto a future `main`-based PR branch.

**Boundaries** (agreed up front so the work stays scoped):

1. Iterate against the WGPU9 (Vulkan-only) wheel because the call_once race makes a dual-backend build slower to test. Final validation must happen on a build with **both** backends so the D3D12 + GC number is also re-measured before the commit lands.
2. If the fix turns out to require a Dawn source change (i.e. cannot live entirely under `onnxruntime/core/providers/webgpu/`): **stop, report**, but ship the Dawn diff as a `.patch` under `cmake/patches/dawn/` (or similar) plus a paragraph here explaining what it does. Don't fork Dawn silently.
3. Nsight Systems Vulkan tracing requires admin (queue-side capture). Same launch pattern as the D3D12 trace recorded in `nsight-systems-d3d12-findings.md`.

**Layered approach** (each layer cheap and reversible):

- **Layer 1 — Locate where time goes (read-only).** Capture a Nsight Systems Vulkan trace of WGPU9 + GC (single context, repeat=10) and an identically configured no-GC trace. Diff per-iter:
  - `vkQueueSubmit` count and avg duration
  - `vkBeginCommandBuffer` / `vkEndCommandBuffer` (should be 0 / iter under true replay)
  - `vkAllocateCommandBuffers` / `vkResetCommandBuffer` (also 0 / iter under true replay)
  - `vkCmdPipelineBarrier`
  - `vkMapMemory` / `vkUnmapMemory` per-iter (would confirm staging-copy hypothesis)
  - `vkWaitForFences`

  Output → `tmp/vulkan_gc_trace/` and a summary appended to a new `nsight-systems-vulkan-gc-findings.md` next to the existing D3D12 one.

- **Layer 2 — Per-replay instrumentation in the WebGPU GC path.** Add env-gated timer markers (zero cost when off) in `WebGpuContext::Run()` and the GC submit shim, splitting per-token wall time into "ORT EP overhead", "Dawn submit", "GPU compute". This piece is shipped as a *diagnostic toggle* in the final commit, not as the fix.

- **Layer 3 — Targeted fix.** Driven by what Layers 1 + 2 reveal. Most-likely-shapes:
  - Dawn re-records on every replay → either a local workaround in our GC layer that holds the cached `wgpu::CommandBuffer` and skips Dawn's higher-level wrapper, or a Dawn-side patch (with the boundary rule above).
  - Extra `Submit`s on Vulkan that D3D12 collapses → gate on `backend_type == Vulkan`.
  - Pipeline-barrier reconstruction → mark the cached buffer as having stable resource state.

**Acceptance criteria for the commit:**

- Vulkan + GC at the canonical workload (Qwen3-1.7B, ctx 3000, gen 500, RTX 5060 Ti) ≥ Vulkan no-GC (currently 134.8 tps). Stretch goal: meet or beat D3D12 + GC (147.3 tps).
- D3D12 + GC at the same workload within ±2 % of the pre-change number (147.3 tps). No regression.
- D3D12 no-GC and Vulkan no-GC unaffected (within ±2 %).
- The diagnostic env vars added in Layer 2 default to off; an empty env var run produces zero new log lines and zero new allocations on the steady-state path.

## Reference

- Dawn `DynamicLib.cpp`: `_deps/dawn-src/src/dawn/common/DynamicLib.cpp`
- Dawn `BackendVk.cpp`: `_deps/dawn-src/src/dawn/native/vulkan/BackendVk.cpp`
- ORT WebGPU context: `onnxruntime/core/providers/webgpu/webgpu_context.cc`
- Branch validated on: `hari/webgpu_perf_1_full @ d5e208bffe`
- Build dir validated: `D:\onnxruntime2\WGPU9\Release\`
- Wheel: `D:\onnxruntime2\WGPU9\Release\Release\dist\onnxruntime_webgpu-1.27.0-cp312-cp312-win_amd64.whl`
