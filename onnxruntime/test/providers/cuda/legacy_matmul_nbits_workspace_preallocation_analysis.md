# Legacy MatMulNBits Workspace Preallocation Analysis

## Purpose

This document explains the runtime behavior and benchmark implications of static
workspace preallocation for the legacy CUDA `MatMulNBits` dequantize-plus-GEMM
path. The key distinction is between:

- declaring a workspace requirement during session initialization;
- recording an activation memory pattern during the first execution of an input
  shape;
- retrieving a workspace slot from a cached memory pattern during later
  executions; and
- releasing memory already retained by the CUDA arena.

Static workspace preallocation changes where the temporary dequantization buffer
comes from. It does not remove the dequantization operation or the following
cuBLAS GEMM.

## Legacy Workspace

The legacy path dequantizes the packed weight matrix into an FP16 or BF16 buffer
before running GEMM. The workspace size is independent of activation dimension
`M`:

```text
workspace bytes = scratch rows * padded K * sizeof(element type)
```

For the HY-MT2 output projection:

```text
N = 120818
K = 2048
scratch rows = 32768
element type = FP16

workspace = 32768 * 2048 * 2
          = 134217728 bytes
          = 128 MiB
```

The output projection uses chunked dequantization because fully dequantizing its
weight matrix would require approximately 472 MiB.

## First Execution Versus Cached Execution

Enabling:

```text
session.enable_static_workspace_preallocation=1
```

adds the declared workspace lifetime to ORT's activation memory plan. It does
not make a usable activation memory pattern available before the first
execution.

The execution sequence is:

1. During session initialization, each eligible legacy `MatMulNBits` kernel
   declares workspace slot 0 and its required size.
2. On the first execution of an input-shape signature, ORT does not yet have a
   cached activation memory pattern for that signature.
3. `GetPreallocatedWorkspace()` cannot provide a pattern-backed address, so the
   kernel falls back to `GetScratchBuffer()`.
4. ORT records the activation and workspace allocation lifetimes while that
   execution runs.
5. After the memory pattern has been cached, a later execution with a matching
   input-shape signature can obtain workspace slot 0 from
   `GetPreallocatedWorkspace()`.
6. If a later execution has a new input-shape signature, that signature must be
   traced separately and initially uses dynamic scratch again.

This distinction matters for autoregressive generation. Prompt lengths and
past-sequence lengths can produce multiple input-shape signatures. A warmup
request therefore does not necessarily prepare every shape that a different
translation will encounter.

## Expected Latency Behavior

Workspace preallocation is a memory-lifetime optimization, not a compute
optimization:

- Weight dequantization still runs.
- The cuBLAS GEMM still runs.
- Only the source of the temporary buffer changes.
- Avoiding a CUDA arena allocation can remove some allocator overhead, but that
  overhead is normally small relative to dequantization and GEMM.
- Recording a new memory pattern can be substantially more expensive than
  retrieving a cached pattern.

Historical benchmark values previously reported here were produced by a
superseded binary-incoherent deployment and have been removed. They must not be
used to quantify cold-start or steady-state behavior. Current coherent
benchmark results are maintained in
`hy_mt2_ort_llama_memory_gap_analysis.md`.

## Why Retained VRAM Did Not Decrease

The Foundry translation runner does not explicitly shrink the CUDA arena after
warmup. Consequently:

1. The first execution allocates dynamic legacy scratch.
2. The CUDA arena grows to satisfy that allocation.
3. The allocation is released logically after the kernel finishes, but the
   arena retains its backing region.
4. Later executions may use the planned workspace slot, but the arena region
   acquired during the first trace remains reserved.

Static workspace reuse therefore cannot lower the process's retained high-water
mark in this lifecycle.

The obsolete memory and latency tables have been removed rather than retained
as historical baselines. See the current analysis document for coherent
default-initializer, device-initializer, and static-workspace A/B results.

## Interpretation and Next Engineering Steps

The current implementation proves that legacy workspaces can participate in
activation memory planning and that matching cached executions can retrieve the
planned slot. It does not prove a retained-VRAM improvement for applications
that keep the first-run CUDA arena high-water mark.

A meaningful retained-memory reduction requires one of the following:

1. Make the planned workspace usable on the first execution, avoiding the
   initial dynamic scratch allocation.
2. Shrink the CUDA arena after warmup so completely free first-run regions can
   be returned before measurement or steady-state service.
3. Change the application lifecycle to perform warmup and shrink explicitly.

Option 2 is useful for measuring steady-state live memory, but it does not
reduce the instantaneous memory required by the first execution. It must
therefore be reported separately from first-request peak VRAM.

The remaining inference-growth gap should also be decomposed into activation
patterns, KV/generator buffers, CUDA library workspaces and caches, and retained
allocator regions. The 128 MiB legacy workspace alone cannot explain the full
ORT-versus-llama.cpp difference.



## From the copilot cli chat

after a memory pattern has been recorded for a particular input-shape signature, subsequent executions matching that pattern obtain the legacy dequantization workspace from the preallocated workspace slot instead of calling  GetScratchBuffer() .

However, lower latency is not guaranteed:

• Preallocation only changes where the 128 MiB dequantization buffer comes from. It does not eliminate weight dequantization or the cuBLAS GEMM.
• The first execution of each new shape still uses dynamic scratch while tracing. Autoregressive decode can encounter multiple shapes as the past sequence length grows.
• Repeated requests with previously observed shapes should avoid the expensive tracing path.
• Any cold-to-warm reduction includes tracing and pattern-cache reuse; it cannot be attributed to a 128 MiB allocation optimization alone.
• In steady state, preallocation may be neutral or slower and must be measured with a coherent deployment.

So the intended benefit is primarily workspace lifetime overlap and lower peak memory, not faster computation. In the current Foundry lifecycle, the first dynamic allocation remains retained by the CUDA arena, preventing that memory benefit from appearing even though later matching executions use the preallocated slot



During normal autoregressive decode, the  MatMulNBits  activation shape usually does not grow with the past sequence length.

For HY-MT2:

Prefill:
A = [batch, prompt_length, 2048]
M = batch * prompt_length

Decode:
A = [batch, 1, 2048]
M = batch

With batch 1, every decode-step  MatMulNBits  normally sees:

A = [1, 1, 2048]
M = 1

The past sequence length grows in the attention-related inputs, such as the attention mask and KV cache—not in the hidden-state sequence dimension passed to the linear projections.

The important nuance is that ORT’s activation memory pattern is session-wide, not keyed only by the  MatMulNBits   A  shape. For example, decode steps may have:

Step 1:
A                 = [1, 1, 2048]
attention_mask    = [1, prompt_length + 1]

Step 2:
A                 = [1, 1, 2048]
attention_mask    = [1, prompt_length + 2]

Even though the  MatMulNBits  input remains  [1, 1, 2048] , the overall session input-shape signature may differ because the attention mask or KV-cache shapes changed. ORT may therefore need a new memory-pattern trace.

 past_present_share_buffer=true  can keep KV-cache buffer shapes fixed, reducing this variation, but the attention-mask shape may still grow. Therefore:

• Different prompt lengths produce different prefill  MatMulNBits  shapes.
• Standard decode normally keeps  MatMulNBits  at  M=1 .
• Growing attention/KV-related inputs can still create new session memory-pattern signatures.
• The legacy 128 MiB workspace itself remains the same size because it depends on  N  and  K , not  M  or past-sequence length.
memory-pattern trace is per kernel or the whole session?
The memory-pattern trace is for the whole session execution, not independently cached per kernel.

For one complete  InferenceSession::Run() :

1. ORT observes allocations and frees for all planned tensors and eligible kernel workspaces.
2. It records their lifetimes and sizes.
3. It creates one memory pattern describing offsets within larger arena-backed blocks.
4. The pattern is cached using the session’s input-shape signature.
5. A later  Run()  with the same signature can reuse the complete pattern.

Each  MatMulNBits  workspace has its own entry—identified by the node and workspace slot—but that entry belongs to the session-wide pattern:

Session memory pattern
├── activation tensor A       offset X, size ...
├── attention output          offset Y, size ...
├── MatMulNBits node 14 slot 0
├── MatMulNBits node 27 slot 0
└── other temporary tensors

This session-wide planning is what allows workspace memory to overlap with activation memory whose lifetime does not intersect that kernel invocation.

Consequently, even if a particular  MatMulNBits  always receives  [1, 1, 2048] , a changed attention-mask or KV-cache input shape can produce a different session input-shape signature. The whole  Run()  then needs a new pattern trace, and that  MatMulNBits  may temporarily use dynamic scratch again while the new session pattern is recorded



So if mem pattern is enable, for each Run() if the input shape signature is not recognized, ORT will run trace and then record this input shape and then stop and waiting for the next Run()?
Almost, except ORT does not stop the current  Run()  or wait for another  Run()  to produce results.

For an unrecognized input-shape signature:

1. ORT starts tracing allocation and release events.
2. The complete  Run()  executes normally using dynamic arena allocations and scratch buffers.
3. ORT produces and returns the requested outputs.
4. The recorded allocation lifetimes are converted into a memory pattern and cached.
5. A future  Run()  with the same signature can use that cached pattern.

Conceptually:

Run 1, shape signature A:
  No cached pattern
  → execute normally with dynamic allocations
  → record allocation lifetimes
  → cache pattern A
  → return outputs

Run 2, shape signature A:
  Cached pattern A found
  → allocate/use pattern-backed memory
  → return outputs

Run 3, shape signature B:
  No cached pattern B
  → execute normally with dynamic allocations
  → record and cache pattern B
  → return outputs

During autoregressive generation, each decode step is typically a separate  Run() . If growing attention-mask or cache shapes produce signatures A, B, C, and so on, each new signature can require its own tracing execution. The token from that execution is still produced normally; only a later matching signature benefits from the cached pattern
