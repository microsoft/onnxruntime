# WebGPU EP — Nsight Graphics Profiling

## Overview

This document describes the profiling infrastructure the WebGPU EP exposes for
**NVIDIA Nsight Graphics GPU Trace** on both **Windows** and **Linux**. The
goal is to let a kernel author answer:

> *"For this specific ONNX operator/kernel with these input shapes, what is
> the GPU actually doing, why is it slow, and what changed after my edit?"*

The WebGPU EP already has a built‑in per‑kernel wall‑time profiler (based on
WebGPU `TimestampQuery` — see
[`webgpu_profiler.cc`](../onnxruntime/core/providers/webgpu/webgpu_profiler.cc)).
That answers *how long* a dispatch took. What it cannot answer is *why*:
occupancy, memory throughput, cache hit rates, warp stall reasons,
per‑instruction attribution. Those are the metrics **Nsight Graphics's GPU
Trace activity** is designed to surface.

### Why Nsight Graphics — and not Nsight Compute or Nsight Systems?

Nsight tools share the "Nsight" brand and the NvPerf hardware‑counter
backend, but they hook different layers of the driver stack:

- **Nsight Compute (`ncu`)** hooks the CUDA driver (`cuLaunchKernel`).
  WebGPU dispatches go through D3D12 / Vulkan and *never* touch the CUDA
  driver. `ncu` reports "No kernels were profiled" against a WebGPU
  workload. This is a design limitation of `ncu`, not a build flag we can
  flip. This was our first attempt; it does not work.
- **Nsight Systems (`nsys`)** hooks D3D12 / Vulkan and gives you a
  *timeline* with per‑dispatch durations, plus GPU metrics *sampled at
  ~10 kHz*. Sampling means coarse SM %, DRAM BW, PCIe throughput — never
  per‑dispatch warp stall reasons, per‑dispatch SM occupancy, or
  per‑dispatch cache hit rates. Useful for "which dispatch is slow" or
  "is there an idle bubble," insufficient for "*why* is this specific
  dispatch slow."
- **Nsight Graphics GPU Trace** hooks D3D12 / Vulkan and drives the same
  NvPerf counter backend as `ncu`. It produces the same class of
  per‑dispatch counter data — SM occupancy, warp stalls by reason,
  L1TEX / L2 hit rates, tensor pipe activity, DRAM throughput — but for
  graphics‑API compute shaders instead of CUDA kernels. This is the right
  tool for WebGPU EP kernel perf work.

The counters are the same silicon in every case; only the interception
layer differs. That is why Nsight Graphics can give you the same class of
metrics `ncu` gives for CUDA.

### What this feature adds

Making Nsight Graphics's GPU Trace produce useful reports against the
WebGPU EP requires three small pieces of runtime infrastructure:

1. **Stable, human‑readable dispatch labels**
   (`ComputePipelineDescriptor::label` set unconditionally + per‑dispatch
   WebGPU debug groups). These become the row names / filter keys in GPU
   Trace's Range Profiler view.
2. **Per‑dispatch compute‑pass boundaries** (opt‑in), so GPU Trace's
   Range Profiler can isolate one dispatch cleanly for counter collection.
3. **A runtime override to disable graph capture** for the profiling
   process, so per‑op debug groups are actually emitted (graph capture
   would bypass the per‑dispatch code path).

All three are gated by **two** switches that both must be on to activate:

- **Build flag `onnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP`** (default
  `OFF`) — compiles the profiling code paths into the binary at all. Off
  = zero cost, zero binary‑size impact.
- **Provider option `ep.webgpuexecutionprovider.enableNsightProfiling`**
  (default `"0"`) — turns the code paths on per `InferenceSession`.
  Off = profiling binary behaves exactly like production.

  This matches the shape of the existing PIX capture flag
  (`enablePIXCapture`) and the graph‑capture flag (`enableGraphCapture`).

This two‑gate design means: shipping the profiling build to a colleague or
CI does not silently degrade their sessions; only sessions that explicitly
opt in pay the profiling cost. A user who has both a normal build and a
profiling build can run the same Python without changes — the launcher
sets the provider option for profiling runs, and a WARNING is logged if
someone opts in on a binary that lacks the build flag.

The profiling workflow itself is one Python script,
`tools/perf/webgpu_nsight_profile.py`. It has a flat CLI — no subcommands
— that spawns a single‑op WebGPU workload as a long‑running loop and
prints the exact "Start Activity → GPU Trace" values you paste into
Nsight Graphics's UI. See [Usage](#usage) for the flow.

**The built‑in ORT profiler (`SessionOptions::enable_profiling`) is
unaffected by any of this.** A default release binary with
`enable_profiling=true` continues to produce the standard chrome‑tracing
JSON with per‑kernel wall time, node name, op type, and shapes, exactly
as it does today.

The feature is cross‑platform: on Windows the underlying backend can be
D3D12 or Vulkan; on Linux only Vulkan is available. Nsight Graphics itself
is NVIDIA‑only — see the [Non‑NVIDIA hardware](#nonnvidia-hardware)
caveat for what to use on other vendors.

---

## Non‑goals

- Replacing the built‑in WebGPU EP wall‑time profiler — that is still the
  primary source of "which kernel took how long" numbers, and works on
  every vendor.
- Nsight Compute (`ncu`) integration. `ncu` cannot see WebGPU dispatches
  (see [above](#why-nsight-graphics--and-not-nsight-compute-or-nsight-systems)).
- Nsight Systems (`nsys`) integration beyond what falls out for free from
  the same debug groups. If richer `nsys` support is wanted later, the
  same launcher can be extended.
- macOS / Metal profiling. Nsight tools do not target Metal; that
  platform needs Xcode Instruments Metal System Trace.
- Headless / CLI‑only Nsight Graphics capture (Pattern B). Nsight Graphics
  has an `ngfx-cli` binary, and once Pattern A is proven we can add a
  `--tool nsight-graphics-cli` mode. For now the launcher only supports
  Pattern A (attach from the GUI).

---

## Design

### The three problems, and how each is solved

**Problem 1 — dispatches are unlabeled.** In stock ORT
`ComputePipelineDescriptor::label` is set to `program.Name()` only in
Debug builds (`#ifndef NDEBUG`) in
[`program_manager.cc`](../onnxruntime/core/providers/webgpu/program_manager.cc).
In a Release build, Nsight Graphics's kernel list shows `""` for every
row and any per‑kernel filter matches nothing.

*Fix:* Set `pipeline_descriptor.label` unconditionally. Cost: a few extra
bytes per compiled pipeline. Benefit: pipelines are named in all tooling,
in all build configurations.

**Problem 2 — dispatches cannot be filtered per ONNX node / op.** Even a
labeled pipeline is shared across many nodes (e.g. dozens of `Add` nodes
share one `Add` pipeline). We need per‑dispatch labels carrying node/op
context.

*Fix:* In a build with `ENABLE_NSIGHT_FOR_WEBGPU_EP` defined AND a session
that sets `enableNsightProfiling=1`, wrap every dispatch in
`wgpu::ComputePassEncoder::PushDebugGroup(label)` / `PopDebugGroup()`
where `label` includes op type, node name, program name, cache‑key
prefix, and the dispatch dimensions. This lowers to
`ID3D12GraphicsCommandList::BeginEvent` on D3D12 and
`vkCmdBeginDebugUtilsLabelEXT` on Vulkan. Nsight Graphics, Nsight Systems,
PIX, and RenderDoc all consume these.

*Cost:* one push/pop pair per dispatch and a few dozen bytes of
command‑list storage. Zero GPU cost. Not present unless both gates are on.

**Problem 3 — Nsight Graphics's Range Profiler pollutes counters when
multiple dispatches share a compute pass.** The WebGPU EP intentionally
packs many dispatches into one compute pass (up to
`max_num_pending_dispatches_`, default 16) to amortize submit overhead.
When Nsight Graphics's Range Profiler replays a target dispatch inside
such a pass to collect counters, cache state from preceding dispatches in
the same pass leaks into the counters (inflating L1/L2 hit rate,
under‑stating DRAM bandwidth, etc.).

*Fix:* In a build with `ENABLE_NSIGHT_FOR_WEBGPU_EP` defined AND a session
that sets `enableNsightProfiling=1`, `EndComputePass()` is called after
every dispatch. Each dispatch becomes its own D3D12 / Vulkan compute
pass, so Range Profiler counter replay isolates it cleanly.

*Cost:* extra pass boundaries during a profiling session only. Wall time
increases; counters become correct. Not present unless both gates are on.

### Graph capture and profiling

Graph capture (`enableGraphCapture=1`) makes Nsight Graphics unreliable
because:

- On the capture run, `LaunchComputePipeline` does not issue real
  `Dispatch` calls — it appends captured commands to an in‑memory list.
- On the replay run, dispatches are issued from the cached command list
  with a cached bind group and pipeline — so counters may reflect the
  "last captured state," not what you'd see in a normal live run.
- Per‑op debug groups are recorded on capture but *not* naturally
  interleaved with replay dispatches in a way GPU Trace can filter on.

*Fix:* In a build with `ENABLE_NSIGHT_FOR_WEBGPU_EP` defined AND a session
that sets `enableNsightProfiling=1`,
`WebGpuExecutionProvider::IsGraphCaptureEnabled()` returns `false`. Any
other combination — default build, or profiling build with the opt‑in off
— leaves graph capture behaving exactly as production.

### Two gates: build flag + runtime opt-in

The feature is gated by two independent switches. Both must be on for the
profiling code paths to activate; either one being off leaves the EP
behaving exactly like a normal (production) build.

**1. Build flag: `onnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP` (CMake option,
default `OFF`).** When enabled, defines the compile‑time macro
**`ENABLE_NSIGHT_FOR_WEBGPU_EP`** for the WebGPU provider build. Name and
shape deliberately mirror `onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP` — both
flags select an optional WebGPU‑EP profiling integration. When the flag
is off, the profiling code paths compile out entirely (no branches, no
fields, no bytes).

**2. Runtime opt‑in: WebGPU EP provider option
`ep.webgpuexecutionprovider.enableNsightProfiling` (values `"0"` /
`"1"`, default `"0"`).** When set to `"1"` on a session, the WebGPU EP
consults its compiled‑in profiling paths for the rest of that session's
lifetime. When the option is set to `"1"` on a binary that was built
*without* the build flag, the EP logs a WARNING at session creation to
make the no‑op visible; the session then behaves like a normal
(non‑profiling) session.

Because the WebGpuContext is a process‑wide singleton keyed by context id,
the first session to initialize a given context establishes the runtime
opt‑in state for that context. A later session that disagrees gets a
one‑shot WARNING and its request is ignored. In practice this is only an
issue if multiple concurrent WebGPU EP sessions run inside one process,
which is unusual for profiling workflows.

| Behavior | Default build | Profiling build, opt‑in off | Profiling build, opt‑in on |
|---|---|---|---|
| `ComputePipelineDescriptor::label` set | Yes (unconditional; ~0 cost) | Yes | Yes |
| Per‑dispatch `PushDebugGroup` / `PopDebugGroup` | No | No | Yes |
| `EndComputePass()` after every dispatch | No | No | Yes |
| `IsGraphCaptureEnabled()` honours session config | Yes | Yes | Returns `false` |
| Runtime cost on hot path | Zero | Zero | Extra push/pop + pass boundary per dispatch |

There are no environment variables to set or unset.

### Why do we need EP changes even for a single-op harness?

A reasonable question: the `webgpu_nsight_profile.py` script already
isolates the workload to one ONNX op. If the whole point is to profile
one kernel, why change the EP at all?

Because "one ONNX op" is not "one WebGPU dispatch." A single MatMulNBits
op produces one target dispatch, but ORT also emits `WriteBuffer`
uploads and a `session.run()` may issue helper dispatches around it
depending on layout / type transitions. Even in the best case, the
WebGPU EP batches multiple dispatches into one compute pass. That means,
even for a single‑op model, Nsight Graphics running against a default EP
build gets:

- Unlabeled dispatches (all show as `""` in the Range Profiler view).
- Multiple dispatches sharing one compute pass → cache‑state leakage
  during counter replay (the numbers you'd trust — L1/L2 hit rate, DRAM
  bytes — are wrong).
- Debug groups recorded but hidden behind graph capture replay when the
  session also has graph capture on.

The EP‑side changes fix all three regardless of workload size. The
single‑op harness gives you a *clean* profiling target; the EP changes
make Nsight Graphics's counters *trustworthy* on that target. You need
both.

---

## Usage

### Prerequisites

- **An NVIDIA GPU.** Nsight Graphics is NVIDIA‑only and refuses to
  collect SM counters on other vendors. See
  [Non‑NVIDIA hardware](#nonnvidia-hardware) for what to use instead.
- **Nsight Graphics 2024.x or newer.** Download from
  <https://developer.nvidia.com/nsight-graphics>. The Windows MSI installs
  under `C:\Program Files\NVIDIA Corporation\Nsight Graphics <version>\`;
  the Linux `.run` / `.deb` installs under `/opt/nvidia/nsight-graphics/`.
  The launcher auto‑discovers the latest install; set the `NGFX` env var
  to point at an `ngfx.exe` / `ngfx` explicitly if the auto‑discovery
  picks the wrong one.
- **An ORT build with `onnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP=ON`.**
  Reuse your normal build command with one extra CMake define:
    ```bash
    python tools/ci_build/build.py --config Release --use_webgpu \
        --cmake_extra_defines onnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP=ON
    ```
    Keep this build directory separate from your production build so you
    can switch between them without full rebuilds. The launcher sets the
    per‑session opt‑in (`enableNsightProfiling`) automatically on the
    workload subprocess, so you do not need to edit any session code.
- **The Python `onnxruntime` package installed from that build.** Nsight
  Graphics launches the workload as `python.exe <script>`; unless the
  Python interpreter it launches imports the profiling‑build wheel,
  `enableNsightProfiling=1` is a no‑op. After `build.py` finishes, run
  `pip install --force-reinstall <build_dir>/Release/dist/onnxruntime_*.whl`
  in the same virtual environment you will hand to the launcher. Verify
  with `python -c "import onnxruntime; print(onnxruntime.__file__)"` —
  the path must point inside your profiling build tree.

### Windows‑specific requirements

On Windows the GPU Trace activity has three additional requirements
beyond the cross‑platform list above. Missing any of them produces
symptoms that look like "the profiler ran but captured nothing," which
is much harder to debug than an outright launch failure.

1. **Elevated (admin) PowerShell.** The NVIDIA display driver restricts
   GPU performance counter access to administrators by default. Without
   elevation, `ngfx.exe` launches the workload successfully, dispatches
   run, but every counter reads zero and the `.ngfx-gputrace` opens with
   an empty Metrics panel. Either right‑click PowerShell → *Run as
   administrator* before invoking the launcher, or open NVIDIA Control
   Panel → *Developer* → *Manage GPU Performance Counters* and select
   *Allow access to the GPU performance counters to all users*. The
   admin‑shell route is preferred for one‑off runs; the Control Panel
   route is preferred when the profiling machine is a shared dev box.
   See <https://developer.nvidia.com/ERR_NVGPUCTRPERM> for background.

2. **`ngfx.exe` under the Nsight Graphics install, not `ngfx-ui.exe`.**
   The GUI executable (`nv-nsight-gfx.exe` on some versions, or the
   Start Menu shortcut labeled "NVIDIA Nsight Graphics") cannot be
   scripted; it is a Qt application. What the launcher needs is the
   headless CLI at
   `C:\Program Files\NVIDIA Corporation\Nsight Graphics <version>\host\windows-desktop-nomad-x64\ngfx.exe`.
   The `find_ngfx()` helper in `webgpu_nsight_profile.py` picks the
   lexicographically newest matching directory automatically; set
   `NGFX=C:\...\ngfx.exe` to pin a specific version.

3. **Two environment variables must be set in the parent shell before
   the launcher runs.** The launcher forwards these to the workload
   subprocess:

    | Variable | Purpose | Recommended value |
    |---|---|---|
    | `ORT_WEBGPU_NSIGHT_ONNX_PATH` | Path where the single‑op ONNX model built by the launcher is saved *and* re‑loaded by the workload subprocess. If unset the launcher aborts with a diagnostic. | `$env:TEMP\qwen_nsight.onnx` |
    | `ORT_WEBGPU_EP_SHADER_DUMP_FILE` | Path where the WebGPU EP dumps every WGSL it compiles during the profiling run. Not required for a capture to succeed, but essential for cross‑referencing DXIL / SPIR‑V back to WGSL in the Shader Source view. | `$env:TEMP\qwen_wgsl_dump.wgsl` |

   The launcher writes both alongside the `.ngfx-gputrace` in the
   per‑run output directory if you leave them unset — but on Windows,
   admin shells often have a *different* `%TEMP%` than the caller's
   normal shell, so setting them explicitly avoids surprise about where
   artefacts land.

A minimal admin PowerShell invocation looks like:

```powershell
# Admin PowerShell
$env:ORT_WEBGPU_NSIGHT_ONNX_PATH  = "$env:TEMP\qwen_nsight.onnx"
$env:ORT_WEBGPU_EP_SHADER_DUMP_FILE = "$env:TEMP\qwen_wgsl_dump.wgsl"

# Optional: pin the ngfx.exe version
# $env:NGFX = "C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.2.0\host\windows-desktop-nomad-x64\ngfx.exe"

# Activate the venv that has the profiling-build wheel installed
& d:\onnxruntime2\.venv\Scripts\Activate.ps1

python d:\onnxruntime2\tools\perf\webgpu_nsight_profile.py `
    --op MatMulNBits `
    --shape "M=1,N=8192,K=3072,bits=4,block_size=32,dtype=fp16"
```

If the run succeeded you should see a `python_<timestamp>.ngfx-gputrace`
file in the launcher's output directory, plus (when `--auto-export` is
on in the ngfx CLI args) a `BASE/` subdirectory with `.xls` per‑range
tables. If the `.xls` files are present but very small (a few hundred
bytes) it means the capture window closed before any of the target
dispatches submitted — increase `--warmup-submits` or
`--capture-timeout-ms` in the launcher call.

### The tight loop

```
# 1. Rebuild ORT if needed (with the flag ON).

# 2. Spawn a workload for the kernel of interest. The script prints
#    Nsight Graphics attach instructions and blocks; the workload loops
#    session.run() forever until you press Ctrl+C.
python tools/perf/webgpu_nsight_profile.py \
    --op MatMulNBits \
    --shape "M=1,N=8192,K=3072,bits=4,block_size=32,dtype=fp16"

# 3. In Nsight Graphics: File → New Activity → GPU Trace. Paste the
#    "Application Executable" / "Working Directory" / "Command Line
#    Arguments" values the launcher printed. Click Launch (or Attach if
#    the workload is already running). Filter by debug‑group op=MatMulNBits
#    and take a capture.

# 4. Ctrl+C in the launcher terminal to stop the workload once you have
#    the capture you want.
```

### Supported ops and required shape keys

Each op builder validates its required `--shape` keys up front and fails
with a clear message if any are missing. Sample invocations:

```
# Plain MatMul (M, N, K, dtype required)
python tools/perf/webgpu_nsight_profile.py \
    --op MatMul --shape "M=1,N=8192,K=3072,dtype=fp16"

# MatMulNBits (M, N, K, bits, block_size, dtype required)
python tools/perf/webgpu_nsight_profile.py \
    --op MatMulNBits \
    --shape "M=1,N=8192,K=3072,bits=4,block_size=32,dtype=fp16"

# MatMulNBitsMlp — Qwen-style fused gate/up + RMSNorm kernel.
# Additional required keys: gate_activation, rms_norm, skip.
python tools/perf/webgpu_nsight_profile.py \
    --op MatMulNBitsMlp \
    --shape "M=1,N=8192,K=3072,bits=4,block_size=32,dtype=fp16,\
gate_activation=silu,rms_norm=1,skip=0"

# MatMulNBitsMlp with SkipSimplifiedLayerNormalization (residual + RMSNorm).
# skip=1 requires rms_norm=1.
python tools/perf/webgpu_nsight_profile.py \
    --op MatMulNBitsMlp \
    --shape "M=1,N=8192,K=3072,bits=4,block_size=32,dtype=fp16,\
gate_activation=silu,rms_norm=1,skip=1"
```

Key notes on shape parameters:

- `dtype` is required and must be `fp16` (or `float16`) or `fp32` (or
  `float32`).
- `bits` currently must be `4` for `MatMulNBits` used inside
  `MatMulNBitsMlp` (matches the WebGPU kernel's specialization).
  Standalone `MatMulNBits` accepts `2`, `4`, or `8`.
- `block_size` currently must be `32` for `MatMulNBitsMlp` (matches the
  WebGPU kernel's specialization).
- `gate_activation` currently must be `silu` (only activation implemented
  in the WebGPU kernel).
- `rms_norm=1` fuses `SimplifiedLayerNormalization` (RMSNorm over `K`)
  into the MLP kernel. `skip=1` fuses `SkipSimplifiedLayerNormalization`
  (residual add then RMSNorm); `skip=1` requires `rms_norm=1`.

### Output layout

The launcher writes to `$ORT_PROFILE_OUT` (default `~/perf_runs`) under a
per‑day / per‑op / per‑time directory. Every run produces:

- `<op>.onnx` — the generated one‑node ONNX (self‑describing report
  artefact).
- `metadata.json` — command line, platform, backend, spawned workload
  args, env overrides.
- `wgsl_dump.wgsl` — the WGSL source dumped for each pipeline the
  workload compiles (uses the existing `ORT_WEBGPU_EP_SHADER_DUMP_FILE`
  mechanism; the launcher sets this env var to the run‑local path when
  it is not already set).

The `.nsight-gfxcapture` file produced by Nsight Graphics itself is
saved wherever the GUI is configured (default is under
`%APPDATA%\NVIDIA Corporation\Nsight Graphics <version>\Documents\` on
Windows). Point Nsight Graphics's save path at the launcher's output
directory to keep everything together.

### Flow inside the script

The script has a single entry point. When invoked with `--op` / `--shape`:

1. It allocates the per‑run output directory and writes `metadata.json`
   up front (survives Ctrl+C or crash).
2. It spawns itself as a child process with a hidden `--_workload` flag.
   The child builds a one‑node ONNX for the requested op/shape, saves
   it, opens a WebGPU EP `InferenceSession` with
   `enableNsightProfiling=1`, prints a `[workload] READY` marker, and
   loops `session.run()` indefinitely.
3. The parent process prints the exact "Application Executable",
   "Working Directory", and "Command Line Arguments" you paste into
   Nsight Graphics's "Start Activity → GPU Trace" dialog, then blocks
   on the child.
4. Ctrl+C in the parent forwards to the child, which prints a clean
   `[workload] stopping (Ctrl+C received)` line and exits.

### What you get in the GPU Trace capture

Open the resulting `.nsight-gfxcapture` in the Nsight Graphics UI.
Expect to see, in the Range Profiler view:

- **Range rows** labeled `op=<op>|node=<node>|prog=<program>|key=<hash>|shape=<dims>`
  (or a truncation thereof — GPU Trace usually shows the leading part).
- **Duration** column reflecting true GPU kernel wall time.
- **SM Throughput** and **Memory Throughput** as % of peak (the "which
  side am I bound on" number).
- **SM Active / Warps Active** — occupancy in practice.
- **Warp Stall Reasons** — long/short scoreboard, MIO throttle, wait,
  math pipe throttle. This is *why* an FMA‑heavy kernel isn't
  FMA‑limited.
- **L1TEX hit rate**, **L2 hit rate**, **DRAM read/write throughput**,
  **PCIe traffic**.
- **VRAM vs system‑memory apertures** for each L2 miss.
- **Tensor pipe activity** vs FMA pipe activity — critical for figuring
  out whether MatMulNBits is actually hitting tensor cores.
- **Advanced Mode Analysis** — ranked list of *projected % frame‑time
  reduction* if you fix specific issues.

Cross‑reference with `wgsl_dump.wgsl` for the original WGSL and with
GPU Trace's Shader source view for the compiled DXIL / SPIR‑V.

### Viewing a `.ngfx-gputrace` in the Nsight Graphics UI

The `.ngfx-gputrace` file the launcher produces is a self‑contained
capture: everything the UI needs (per‑dispatch timings, HW counter
values, DXIL / SPIR‑V shader bytecode, debug‑group labels) is inside
that one file. To open it, launch the Nsight Graphics GUI
(`nv-nsight-gfx.exe` on Windows, Start Menu → *NVIDIA Nsight
Graphics*) and either drag the file onto the window or use
*File → Open File…*. No re‑capture is performed and the target machine
does not need to be the same as the machine viewing the report.

Once loaded you are dropped into the GPU Trace view. The parts that
matter for a WebGPU EP kernel investigation, in the order you will
typically use them:

1. **Timeline / Frame view (top).** A horizontal ribbon of every GPU
   submit in the capture window, coloured by API command type. For a
   WebGPU workload the interesting bands are the D3D12 (or Vulkan)
   compute queue segments containing `ExecuteCommandLists → Dispatch`
   entries. Zoom with `Ctrl` + mouse wheel; drag to pan. Every
   `Dispatch(x,y,z)` node is one WGPU dispatch — and because
   `enableNsightProfiling=1` forces one dispatch per compute pass, each
   `Dispatch` node also corresponds to exactly one WebGPU EP kernel
   invocation.

2. **Hierarchical vs Flat toggle (top‑left of the events panel).**
   *Hierarchical* nests events under their parent debug groups so you
   can collapse an entire ONNX op into one row. *Flat* is a linear list
   of dispatches, useful for sorting by duration to find the slowest
   kernel. Toggle freely; the underlying capture data is the same.

3. **Debug Group filter (Events panel search bar).** Type `op=` to see
   only the WebGPU EP dispatches (the `PushDebugGroup` label the EP
   emits for each dispatch starts with `op=<OpType>|node=<NodeName>|prog=<ProgramName>|key=<CacheKey>|disp=<x>,<y>,<z>`).
   Type `op=MatMulNBits` to see only that op's dispatches. This is the
   primary way to answer "which of these 40 dispatches is the one I
   just edited."

4. **Range Info panel (bottom, activated by clicking a Dispatch).**
   For the selected dispatch this panel shows:

    - The full debug‑group label (so you can copy the `key=...` prefix
      out and grep the WGSL dump for the exact shader source that ran).
    - **Duration** in µs — this is the counter‑replay‑corrected GPU
      wall time for exactly one launch of this kernel, not the noisy
      wall time.
    - **Thread groups** (`x*y*z`) and **thread group size** (from the
      shader) — sanity‑check the dispatch geometry matches what the
      kernel author intended.
    - **Pipeline state** — bound compute pipeline label
      (`ComputePipelineDescriptor::label`, which the EP now sets
      unconditionally).

5. **GPU Metrics panel (right side, usually the widest tab).** This is
   the panel that answers "*why* is this dispatch slow." Sections you
   will look at first:

    - **SM Throughput %** and **Memory Throughput %** — high‑level
      "which side am I bound on" numbers. If SM is near 100% and Memory
      is low the kernel is compute‑bound; the reverse means it is
      memory‑bound; both low means it is latency‑bound (usually because
      of warp stalls — see next).
    - **Warp Stall Reasons.** Long scoreboard = waiting on a global
      memory load; short scoreboard = waiting on a shared‑memory load;
      MIO throttle = too much traffic through the memory I/O queue;
      wait = execution dependency; math pipe throttle = ALU pipe
      saturated. These are the single most useful pointer for what to
      change in the WGSL.
    - **L1TEX Hit Rate**, **L2 Hit Rate**, **DRAM Read/Write Throughput
      (GB/s)** — the cache subsystem story. Trustworthy because the EP
      forces one dispatch per pass so no cache state leaks in from
      preceding dispatches.
    - **Tensor Pipe %** vs **FMA Pipe %** — critical for MatMul‑style
      kernels. A MatMulNBits that shows 0% tensor pipe on an Ampere+
      GPU is doing scalar FMAs instead of `mma.sync` — a large
      unrealized perf opportunity.
    - **Occupancy — Active Warps / SM**, **Registers Per Thread**,
      **Shared Memory Per Block** — the "why is occupancy limited"
      breakdown.

6. **Shader Source panel (`Ctrl+Shift+S`, or click on a pipeline in
   Range Info).** Shows the compiled DXIL (D3D12) or SPIR‑V (Vulkan)
   for the currently selected dispatch, with per‑instruction sample
   counts overlaid in the gutter. The instruction lines with the tallest
   sample bars are where the GPU spent most of its time — usually one
   or two hot instructions dominate. Cross‑reference against
   `wgsl_dump.wgsl` for the WGSL source; readable identifiers
   (`accum_reg`, `k_tile`, ...) usually survive into the compiled form.

7. **Advanced Mode Analysis panel (bottom‑right, "Analysis" tab).**
   Nsight Graphics runs the "peak‑performance‑percentage" analysis
   method against the counters and produces a *ranked* list like
   "unfused load bandwidth = 34% of frame time; if this were removed
   the workload would be N% faster." Use this to triage which of the
   several possible perf issues to fix first.

If you also enabled the `--auto-export` argument to `ngfx.exe` in the
launcher call, the same per‑dispatch numbers are dumped as `.xls`
files under the `BASE/` subdirectory of the output folder — grep or
Excel‑pivot those instead of clicking through the UI when doing a
metric diff across many runs.

### Collecting the maximum set of perf‑analysis metrics

The launcher's banner (Option B) prints a ready‑to‑paste ngfx.exe
command that turns on every counter‑collection knob GPU Trace exposes.
For reference, the flags and what each one buys you:

| Flag | What it enables | Cost |
|---|---|---|
| `--activity "GPU Trace Profiler"` | The compute/graphics‑API profiling activity. Frame Debugger is a different activity that does **not** collect HW counters. | none |
| `--nvperf-metric-set "workload_metrics_all"` | Every metric section: SM occupancy, warp stall reasons, LSU throughput, L1TEX / L2 hit rates, DRAM read/write, PCIe, tensor pipe %, FMA pipe %, ALU issue slots, memory pipe utilization. If your Nsight Graphics version rejects this name, try `"full"` or `"advanced_workload"` — the underlying NvPerf backend picks whatever named set is closest. | ~10× replay per dispatch |
| `--advanced-mode-analysis` | Populates the Range Profiler's **Analysis** tab with the Peak‑Performance‑Percentage (PPP) ranked list of "% frame time you'd save by fixing X." **Without this flag the Analysis tab is empty**, which is the most common cause of "I don't see any perf recommendations." | small (post‑capture compute) |
| `--collect-shader-source` | Attaches DXIL / SPIR‑V + per‑instruction sample counts to every dispatch. Populates the Shader Source view; without it, you get the compiled bytecode but no hot‑instruction attribution. | small |
| `--sampling` | Enables PC sampling — the substrate the Shader Source hot‑instruction bars are drawn from. | ~5–10% wall time |
| `--start-after-submits N` | Skip the first `N` GPU submits (typically warmup / init copies). | none |
| `--limit-to-submits M` | Capture up to `M` consecutive submits after the skip window. Match to how many `session.run()` iterations you want profiled. | linear |
| `--max-duration-ms MS` | Hard cap on capture wall time — safety net if the workload stalls. | none |
| `--auto-export` | Dumps `BASE/*.xls` per‑range tables next to the `.ngfx-gputrace` so scripts can grep / diff without opening the UI. | small |

**Trade‑off.** Turning on the full metric set replays each target
dispatch roughly 10 times to collect all HW counter groups. A capture
that would normally take ~5 seconds becomes ~50 seconds, and
throughput/latency numbers reported by the workload's own instrumentation
during the capture window are meaningless. That's expected: the
per‑dispatch "Duration" column in the `.ngfx-gputrace` UI is the
counter‑replay‑corrected GPU time and *is* trustworthy — do not use
wall‑clock time from the terminal to reason about performance during a
profiling capture. For wall‑clock numbers use a separate normal run with
the ORT profiler JSON.

**When to dial back.** For quick iteration (e.g. verifying a shader edit
compiled and dispatched), use `--nvperf-metric-set "standard"` and drop
`--advanced-mode-analysis` / `--collect-shader-source` / `--sampling`.
That gets the capture time back to near‑real wall clock, at the cost of
losing the Analysis tab, hot‑instruction bars in Shader Source, and the
finer‑grained warp stall attribution.


### Baselining and diffing

Nsight Graphics supports side‑by‑side capture comparison in the UI:
`File → Compare Captures`. The launcher stores captures under a
deterministic path (`$ORT_PROFILE_OUT/<date>/<op>/<time>/`) so it is
easy to keep a "latest good" alongside a candidate.

---

## Caveats

### Interaction with the built‑in ORT wall‑time profiler

The WebGPU EP has its own timestamp‑query‑based per‑kernel wall‑time
profiler (see
[`webgpu_profiler.cc`](../onnxruntime/core/providers/webgpu/webgpu_profiler.cc)),
enabled via `SessionOptions::enable_profiling`. **It works identically in
every combination of build flag and runtime opt‑in.** A default release
binary with `enable_profiling=true` produces the standard ORT
chrome‑tracing JSON with per‑kernel wall time, node name, op type,
program name, cache key, and shapes — exactly as it does today.

When both the ORT profiler AND the Nsight session opt‑in are active in a
profiling build, the ORT profiler produces **cleaner per‑dispatch
numbers** than a default build, because:

- Every dispatch runs in its own compute pass, so the two
  `WriteTimestamp` writes bracket exactly one kernel with no other
  dispatches in between. There is no ambiguity about what a "start" and
  "end" tick refer to.
- Graph capture is disabled in this session, so there is no
  run‑0‑is‑silent / runs‑1..N‑1‑have‑events asymmetry. Every
  `session.run()` produces the same set of per‑dispatch events, which
  makes averaging and diffing across runs trivial.
- On backends that only support `TimestampQueryType::AtPasses` (older
  Vulkan/D3D12 devices without the InsidePasses extension), the
  per‑pass timestamps also become per‑dispatch because the pass *is*
  per‑dispatch.

The one thing that is not directly comparable across sessions is
**total run wall time**: a profiling session is slower because of the
extra pass boundaries and the lack of graph capture. Use a profiling
session for shape‑of‑cost analysis; use a normal session for absolute
wall‑time numbers.

### Non‑NVIDIA hardware

Two independent stories:

- **The build.** `onnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP=ON` is a pure
  Dawn/WebGPU addition (extra `PushDebugGroup` / `EndComputePass` calls)
  — it builds and runs on every WebGPU backend (D3D12, Vulkan, Metal).
  No NVIDIA dependency at build time.
- **The runtime.** Setting `enableNsightProfiling=1` on any GPU vendor
  produces correct output — the WebGPU EP just runs slower because it
  uses the profiling code paths (one pass per dispatch, no graph
  capture). Debug groups are still emitted; they are consumed by AMD's
  Radeon GPU Profiler (Vulkan), Intel GPA (Vulkan / D3D12), and Xcode
  Instruments (Metal), so the labels are useful there too.

What is **NVIDIA‑only** is Nsight Graphics's SM counter path itself. On
AMD / Intel / Apple hardware, use the vendor's own profiler:

- Radeon GPU Profiler (RGP) on AMD Vulkan.
- Intel GPA on Intel.
- Xcode Instruments Metal System Trace on Apple silicon.

The launcher prints a warning if it detects a non‑NVIDIA GPU. Set
`ORT_WEBGPU_PROFILE_SKIP_GPU_CHECK=1` to bypass the check (e.g. when
attaching a non‑Nsight profiler intentionally).

### Nsight Graphics is GUI‑driven (Pattern A)

Unlike `ncu` — which is a CLI that takes a target application and
produces a report — Nsight Graphics is designed around its GUI. There is
no direct `nsight-graphics --capture -- python ...` equivalent for GPU
Trace in the standard install. The launcher's flow reflects this:

1. It builds the workload and spawns it as a long‑running loop.
2. It prints exact "Start Activity → GPU Trace" values.
3. You Alt‑Tab to the Nsight Graphics UI, paste, and click Launch (or
   Attach to the already‑running process).

Nsight Graphics does ship an `ngfx-cli` binary (name varies by version)
that can drive GPU Trace headlessly. Once Pattern A is proven, a future
enhancement can add a `--tool nsight-graphics-cli` mode to the launcher.
For now, Pattern A is the tested path.

### Opting in on a binary that was not built with the flag

If you set `enableNsightProfiling=1` on a session but the binary was
built without `onnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP`, the option is
parsed but has no effect. The EP logs a WARNING at session creation to
make this visible:

```
[W:onnxruntime:Default, webgpu_provider_factory.cc:...] WebGPU EP: provider
option "ep.webgpuexecutionprovider.enableNsightProfiling" is set but this
binary was built without ENABLE_NSIGHT_FOR_WEBGPU_EP. The option has no
effect. To enable, rebuild with -Donnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP=ON.
See docs/WebGPU-EP-Nsight-Graphics-Profiling.md.
```

The launcher script sets the option unconditionally on the workload
subprocess, so seeing this WARNING means the workload subprocess is
pointing at a production build — rebuild in a separate build tree.

### DXIL / SPIR‑V correlation

Nsight Graphics's Shader source view shows Dawn's compiled shader (DXIL
on D3D12, SPIR‑V on Vulkan). The `wgsl_dump.wgsl` file in each run has
the original WGSL for correlation. Line‑by‑line mapping is not
automatic; readable WGSL identifiers usually survive into the compiled
form.

### Graph capture is silently disabled for a Nsight-profiling session

A session with `enableNsightProfiling=1` unconditionally disables graph
capture for that session, even when the caller also set
`enableGraphCapture=1`. The runtime prints a one‑shot WARNING to the ORT
log in that case:

```
[W:onnxruntime:Default, webgpu_execution_provider.cc:...] WebGPU EP: graph
capture was requested in provider options but is disabled in this session
because enableNsightProfiling is on. Per-dispatch wall time in the ORT
profiler and in Nsight tools remains correct; total run time will be higher
than a non-profiling session. See docs/WebGPU-EP-Nsight-Graphics-Profiling.md.
```

If `enableNsightProfiling` is *not* set on the session (the default),
graph capture behaves exactly as it does in a non‑profiling build —
even if the binary was built with `ENABLE_NSIGHT_FOR_WEBGPU_EP=ON`.

Per‑dispatch timing (both the ORT profiler JSON and any Nsight capture)
remains correct — you are seeing the true GPU time each kernel spent.
Only the **total** wall time inflates because the host‑side latency win
from graph capture is gone. This is intentional: graph capture would
bypass the per‑dispatch debug groups and pass boundaries that make GPU
Trace's Range Profiler counters trustworthy.

---

## Implementation

Files touched:

- [`cmake/CMakeLists.txt`](../cmake/CMakeLists.txt)
  — CMake option `onnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP` and
  `add_compile_definitions(ENABLE_NSIGHT_FOR_WEBGPU_EP)` guard, next to
  the existing `ENABLE_PIX_FOR_WEBGPU_EP` block.
- [`onnxruntime/core/providers/webgpu/program_manager.cc`](../onnxruntime/core/providers/webgpu/program_manager.cc)
  — drop the `#ifndef NDEBUG` guard around
  `pipeline_descriptor.label`. Kept **unconditional** (not gated on the
  profiling flag) because it costs effectively nothing and helps every
  profiler that ever attaches, regardless of build type.
- [`onnxruntime/core/providers/webgpu/webgpu_provider_options.h`](../onnxruntime/core/providers/webgpu/webgpu_provider_options.h)
  — declare the new WebGPU EP provider option constant
  `kEnableNsightProfiling`
  (`"ep.webgpuexecutionprovider.enableNsightProfiling"`).
- [`onnxruntime/core/providers/webgpu/webgpu_provider_factory.cc`](../onnxruntime/core/providers/webgpu/webgpu_provider_factory.cc)
  — parse the option unconditionally; emit a WARNING when set in a
  build without `ENABLE_NSIGHT_FOR_WEBGPU_EP`; forward the value into
  `WebGpuContextConfig` so the context picks it up.
- [`onnxruntime/core/providers/webgpu/webgpu_execution_provider.h`](../onnxruntime/core/providers/webgpu/webgpu_execution_provider.h) /
  [`webgpu_execution_provider.cc`](../onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc)
  — store the opt‑in on `WebGpuExecutionProvider`; `IsGraphCaptureEnabled()`
  returns `false` only when the opt‑in is on (under the build flag).
- [`onnxruntime/core/providers/webgpu/webgpu_context.h`](../onnxruntime/core/providers/webgpu/webgpu_context.h) /
  [`webgpu_context.cc`](../onnxruntime/core/providers/webgpu/webgpu_context.cc)
  — cache the opt‑in on the shared `WebGpuContext` at first init; add
  `debug_label` to `CapturedCommandInfo`; gate the per‑dispatch
  `PushDebugGroup` / `PopDebugGroup` + `EndComputePass()` in `Run` and
  `Replay` on both the build flag AND the runtime opt‑in.
- `tools/perf/webgpu_nsight_profile.py` — the cross‑platform launcher
  and single‑op harness in one script. Sets `enableNsightProfiling=1`
  unconditionally on the workload subprocess. Works on Windows and
  Linux.
- `.github/workflows/windows_webgpu.yml` — CI job that builds with
  `-Donnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP=ON` and runs the WebGPU
  unit tests without the session opt‑in, catching runtime regressions
  in the "profiling build but opt‑in off" behavior.

EP‑side additions are ~100 LOC net. All new profiling paths are inside
`#if defined(ENABLE_NSIGHT_FOR_WEBGPU_EP)` guards *and* an inner runtime
check, so:

- A default build has zero runtime and binary‑size impact.
- A profiling build with the runtime opt‑in off is byte‑for‑byte
  equivalent to production on the hot path.
- Only the (profiling build AND session opt‑in) combination activates
  the profiling code paths.

## References

- Nsight Graphics product page:
  <https://developer.nvidia.com/nsight-graphics>.
- Nsight Graphics GPU Trace documentation:
  <https://docs.nvidia.com/nsight-graphics/UserGuide/> (see the "GPU
  Trace" section).
- "Peak‑performance‑percentage" analysis method (used by GPU Trace's
  Advanced Mode Analysis):
  <https://developer.nvidia.com/blog/the-peak-performance-analysis-method-for-optimizing-any-gpu-workload/>.
- Dawn WebGPU C++ headers: `dawn-src/include/webgpu/webgpu_cpp.h` (in
  the ORT build tree under `_deps/dawn-src/`).
- WebGPU spec — Debug Groups:
  <https://www.w3.org/TR/webgpu/#debug-groups>.
- Existing ORT WebGPU profiling:
  [`webgpu_profiler.cc`](../onnxruntime/core/providers/webgpu/webgpu_profiler.cc).
