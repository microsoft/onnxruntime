---
name: code-review-webgpu
description: Review native WebGPU changes through operators and computation, execution and memory, and platform support and release. Use only for direct or traceable WebGPU effects, including affected shared code and native defects exposed by JSEP fixes. Apply during implementation too. Exclude unrelated CPU/CUDA or shared-file edits without WebGPU impact. Deprecated JSEP follows its own policy. Use code-review for review workflow.
---

# Native WebGPU EP review

Derived from [Devang Patel's contribution](https://github.com/microsoft/onnxruntime/commit/4453ec99843d2561e1f99071525db8a2c5ce48eb).

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

Reviewers do **not** configure or build ORT/Dawn, install dependencies, run tests
or validation workloads, or trigger CI jobs. CI bots own build/test execution.
Inspect source, test coverage, and existing CI evidence; report missing, failed,
or insufficient validation.
Validation commands in references identify evidence to assess, not actions to
execute during review. Implementation tasks follow their separately authorized
workflow; this review skill never initiates a build.

## Select the checks

The three categories separate responsibilities, not C++ versus WGSL files or
quality concerns. Kernel host code and shaders implement one computation and
belong together. Testing and performance evidence are shared review steps.

| Category | When to read | Reference |
| --- | --- | --- |
| Operators and computation | Operator/schema support, host inputs/shapes, registration and native/plugin acceptance, WGSL/templates, dispatch/indexing/barriers, cache keys, numerics, layouts/prepacking, graph rewrites/fusion/partitioning | [Computation](references/computation.md) |
| Execution and memory | Submission/order, threads and shared state, callbacks, buffers/allocators/transfers/external data, capture/replay, cache storage/lifetime, profiling/timestamp/trace implementation | [Execution](references/execution.md) |
| Platform support and release | Public options, context/device selection, enabled features and hardware restrictions, host/plugin API/ABI, native/WASM/static/shared integration, Dawn/DXC rolls/patches/manifests, builds/packages/release tools | [Platform](references/platform.md) |

Select **every affected category**, including indirect shared consumers. A PR
can change several responsibilities; do not assign it exactly one label.
Using an unchanged allocator/device API does not alone select another category;
check that the changed caller obeys its contract, and expand scope if it does not.
Inspect controlling code before excluding an uncertain category; reselect when
scope grows. Read relevant sections within selected references, rather than all
sections of every large category. Name unreadable required sections as gaps.

Keep boundaries explicit: cache-key correctness belongs to computation, cache
storage/thread safety to execution; native/plugin operator acceptance belongs
to computation, host ABI compatibility to platform. Layout math belongs to
computation, buffer lifetime to execution. Each check has one owner.

For every WebGPU behavior/build review, also assess the relevant sections of
[test and performance evidence](references/validation.md): regression coverage,
existing CI, shader-key evidence for codegen changes, and measurements for
performance claims. These steps do not launch tests or builds and are not a
fourth subsystem category. Test-only changes can use these shared steps without
loading an unrelated implementation category.

## Complete the review

Before reporting, account for all three categories: checked with source/test
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
with one owner and link to it from other routes. Update the category route or
shared-step trigger alongside its reference; an unrouteable check is incomplete guidance. Keep
PR status, raw evidence, and experiments in the task handoff rather than here.
Replace superseded checks as the implementation evolves. Keep examples only
when they clarify an otherwise ambiguous rule; historical PR lists and audit
snapshots belong in maintenance evidence, outside the active references.
Organize by responsibility; keep strongly coupled parts of one computation
together. Keep shared evidence steps outside the category list. Write concise
triggers, invariants, corrections, and regression checks, preserving conditions
and exceptions when shortening text.
