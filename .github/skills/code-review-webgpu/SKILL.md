---
name: code-review-webgpu
description: Use only for changes with a direct or traceable effect on the native WebGPU EP, including shared tests, optimizers, dependencies, packaging, and native defects exposed by JSEP fixes. Cover new kernel authoring, WGSL, numerics, packing, caches, thread safety, runtime, devices, plugin ABI, Dawn rolls, builds, profiling, registration, and validation, locally or in GitHub Copilot. Apply during implementation too. Exclude unrelated CPU/CUDA or shared-file edits without WebGPU impact. Deprecated JSEP follows its own policy; use native checks only where affected. Use code-review for review workflow.
---

# Native WebGPU EP review

## Scope gate

Select this skill only when the changed behavior directly belongs to native
WebGPU or has a concrete WebGPU consumer/call path, build/dependency input,
test configuration, or deployed artifact affected by the diff. Shared-file
location, a historical PR link, and a passing mention of WebGPU are insufficient.
An unrelated CPU/CUDA kernel, optimizer rule excluding WebGPU, packaging job
without a WebGPU artifact, or shared-test edit that cannot affect WebGPU does
not use this skill or its references.
If selected in error, stop applying it. For mixed PRs, apply these checks only
to the WebGPU-related portion and affected shared consumers.

JSEP is deprecated. Follow the [JSEP review policy](../../instructions/jsep-deprecated.instructions.md)
and its authoritative [deprecation notice](../../../docs/JSEP_Deprecation.md).
For a JSEP kernel fix, inspect the native counterpart as that policy requires;
if the same defect applies, use this skill for the affected native behavior,
including a missing native fix/regression test. Pure JSEP maintenance does not
load native references. Identify the implementation from the build/import/WASM
artifact: the JavaScript backend name `webgpu` alone does not distinguish them.

For reviews, use [code-review](../code-review/SKILL.md) for scope and reporting.
For implementation, apply the same checks to the resulting behavior. Pin the
base and head or working state; inspect changed functions in full, their base
versions, controlling code, callers, and tests. Do not review only diff hunks.

## Select the checks

Read the live table and load **every matching row**. Routes are cumulative:
one kernel change can require several references. Classify by behavior, not
just filenames; follow shared helpers to all affected operators. If relevance
is unclear, inspect the controlling code before marking a row inapplicable.
Re-evaluate the table when the scope grows.

Open the applicable reference files before judging the change; the table only
selects checks. If a required reference cannot be read, name it as a review gap.

| When to read | Reference |
| --- | --- |
| Authoring a new kernel, or changing host inputs, shape computation, program setup, GPU-derived indices, or host/GPU input placement | [Kernel contract](references/kernel-contract.md) |
| Changing generated WGSL, template parameters, program metadata, specialization, provider options, or program-cache behavior | [Cache identity](references/cache-identity.md) |
| Changing shader execution, indexing, synchronization, dispatch, workgroup/subgroup sizing, or bindings | [Dispatch and WGSL](references/dispatch-and-wgsl.md) |
| Changing arithmetic, reductions, casts, quantization, masks, ordering, or accuracy options | [Numerical semantics](references/numerical-semantics.md) |
| Changing vectorization, tensor views, packed/quantized storage, layout, or prepacking | [Tensor layout and prepacking](references/tensor-layout-and-prepacking.md) |
| Adding multi-thread support or changing thread safety, callbacks, shared mutable state, allocation/cache policy, transfers, external data, capture, or resource lifetime | [Runtime and memory](references/runtime-and-portability.md) |
| Changing provider-option parsing, context reuse, adapter selection, feature enable/disable, or hardware fast-path eligibility | [Devices and features](references/devices-and-features.md) |
| Changing plugin adapters, capabilities, API/version negotiation, exported callbacks, or static/shared plugin integration | [Plugin contracts](references/plugin-contracts.md) |
| Rolling Dawn or changing native/WASM build boundaries, Dawn/DXC, patches/manifests, deployment artifacts, or release automation | [Builds and dependencies](references/build-and-dependencies.md) |
| Changing graph transformations, fusion, partitioning, layout conversion, or optimizer/provider support predicates | [Graph transformations](references/graph-transformations.md) |
| Changing an operator's supported behavior, domain, opset, types, host-read inputs, or registration | [Registration](references/registration.md) |
| Changing profiling, timestamps, instrumentation, performance claims, or hardware workarounds | [Profiling and performance](references/profiling-and-performance.md) |
| Selecting or assessing validation for any WebGPU behavior/build change, or changing WebGPU/shared OpTester tests, exclusions, or tolerances | [Validation](references/validation.md) |

Typical infrastructure owners include `webgpu_context.*`, `program_manager.*`,
`buffer_manager.*`, `allocator.*`, `data_transfer.*`, `session_buffer_pool.*`,
`compute_context.*`, `ep/`, and `webgpu_provider_factory.cc`. Build changes include
`cmake/onnxruntime_providers_webgpu.cmake`, `cmake/patches/dawn/`, and
`plugin-ep-webgpu/`. These are discovery hints, not an exhaustive scope.

## Complete the review

Before reporting, account for every reading-table row: checked with source/test
evidence, inapplicable with a reason, or unverified with the check that would
settle it. Keep this brief in the review or handoff; do not dump all checklists
into the report. A green build, skipped test, CPU fallback, or one vendor's run
does not establish coverage of another shader variant or target.

Use the `code-review` finding format: file and line, evidence or trigger,
impact, and correction. For WebGPU findings:

- Name the triggering dtype, rank, shape, component count, and relevant device
  feature or provider option.
- Mark as blocking: wrong results (including cache-key collisions), out-of-bounds
  access, crashes, hangs, WebGPU validation errors on a supported backend, a
  broken build target, and missing tests for changed behavior.
- Report performance concerns with a concrete mechanism, such as per-run
  pipeline creation, host readback, or lost vectorization. Redundant keys need
  measurable compile churn; performance claims need measurements.
- Separate pre-existing defects from regressions introduced by the change.
  State unverified behavior and the evidence needed to resolve it.

## Maintain the guidance

This skill owns recurring native WebGPU implementation/review checks. Each
reference states its boundary and when its meaning needs updating. Put a rule
with one owner and link to it from other routes. Add or change the reading row
alongside its reference; an unrouteable check is incomplete guidance. Keep
PR status, raw evidence, and experiments in the task handoff rather than here.
When extending categories, consult the source/evidence map and limits in
[WebGPU review coverage](../../../docs/WebGPU_Review_Coverage.md); ordinary
reviews do not need to load that audit or the historical corpus.
