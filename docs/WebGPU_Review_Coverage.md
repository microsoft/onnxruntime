# WebGPU Review Coverage

This document records the boundaries and evidence behind
[`code-review-webgpu`](../.github/skills/code-review-webgpu/SKILL.md).
Consult it when maintaining the guidance or auditing coverage; routine reviews
start with the skill's scope gate and cumulative reading table.

## Scope and category design

The skill applies only to changes with a direct or traceable effect on the
native WebGPU EP. It also covers that EP's browser/WASM and plugin deployment
paths. It does not activate merely because a shared file, old PR, or comment
mentions WebGPU. CPU/CUDA-only work, shared optimizer/test changes without an
affected WebGPU consumer, and packaging without WebGPU artifacts retain their
existing review guidance. JSEP is deprecated and uses
[its own review policy](../.github/instructions/jsep-deprecated.instructions.md),
with [the deprecation notice](JSEP_Deprecation.md) as the authoritative source.
Mixed PRs apply these checks only to the affected WebGPU portion.

JSEP correctness/security fixes require focused regression coverage and a check
of the corresponding native kernel; apply native review checks when the same
defect affects it, even if the PR omitted the native fix. New operators,
features, and performance-only work belong in the native EP. Deprecation,
migration, and removal follow the linked JSEP policy and migration plan.
Deprecation does not mean JSEP has already been removed or that a removal date
is fixed. Both implementations use the JavaScript backend key `webgpu`, so
identify the build/import/WASM artifact before selecting checks.

The initial kernel-centered grouping combined too many independent contracts
under runtime/portability and shader semantics. The following 13 categories
separate those contracts without creating 13 always-loaded skills. Each has one
owning reference, an explicit reading trigger, corrective checks, and validation
prompts. Cross-cutting changes select multiple rows.

Paths below are relative to `onnxruntime/` unless otherwise stated. Source was
inspected against commit
[`527971bcece3d944cf18147472ba1e5010289c2e`](https://github.com/microsoft/onnxruntime/commit/527971bcece3d944cf18147472ba1e5010289c2e).
They identify controlling implementation, not a claim that every line or
operator was independently proven correct.

| Category and owning checks | Source boundary inspected | Evidence and regression focus |
| --- | --- | --- |
| [Kernel contract](../.github/skills/code-review-webgpu/references/kernel-contract.md) | `core/providers/webgpu/compute_context.*`, kernel `ComputeInternal`, host-input placement | [#28721](https://github.com/microsoft/onnxruntime/pull/28721), [#29255](https://github.com/microsoft/onnxruntime/pull/29255): validated shapes, safe narrowing, GPU-derived indices |
| [Cache identity](../.github/skills/code-review-webgpu/references/cache-identity.md) | `program_cache_key.cc`, `program_manager.*`, `program.*`, templates | [#33153](https://github.com/microsoft/onnxruntime/pull/33153), [#33693](https://github.com/microsoft/onnxruntime/pull/33693): complete keys, output components, overrides, same-context both-order tests |
| [Dispatch and WGSL](../.github/skills/code-review-webgpu/references/dispatch-and-wgsl.md) | dispatch normalization, `shader_helper.*`, `shader_variable.*`, generated kernels | [#32343](https://github.com/microsoft/onnxruntime/pull/32343), [#32796](https://github.com/microsoft/onnxruntime/pull/32796): normalized indices, partial tiles, subgroup ownership, binding limits |
| [Numerical semantics](../.github/skills/code-review-webgpu/references/numerical-semantics.md) | `math/gemm_utils.cc`, normalization and quantization kernels, contrib attention | [#33057](https://github.com/microsoft/onnxruntime/pull/33057), [#32528](https://github.com/microsoft/onnxruntime/pull/32528), [#33074](https://github.com/microsoft/onnxruntime/pull/33074): partial precision, int64, quantization ties, masks and selection |
| [Tensor layout/prepacking](../.github/skills/code-review-webgpu/references/tensor-layout-and-prepacking.md) | `nn/conv*`, `math/matmul*`, views/accessors, contrib `quantization/matmul_nbits*` | [#32688](https://github.com/microsoft/onnxruntime/pull/32688), [#33231](https://github.com/microsoft/onnxruntime/pull/33231): logical versus physical shape, groups, initializer/dynamic/disabled modes |
| [Runtime and memory](../.github/skills/code-review-webgpu/references/runtime-and-portability.md) | context/recording, allocator, buffer manager, session pool, data transfer and external loaders | [#32874](https://github.com/microsoft/onnxruntime/pull/32874), [#32587](https://github.com/microsoft/onnxruntime/pull/32587), [#33509](https://github.com/microsoft/onnxruntime/pull/33509): streamless ordering, reuse lifetime, capture failure/cleanup |
| [Devices and features](../.github/skills/code-review-webgpu/references/devices-and-features.md) | `webgpu_provider_factory.cc`, options, context creation/reuse, `ep/factory.cc` | [#32319](https://github.com/microsoft/onnxruntime/pull/32319), [#33481](https://github.com/microsoft/onnxruntime/pull/33481), [#33724](https://github.com/microsoft/onnxruntime/pull/33724): selector identity, enabled features, compile-only versus execution |
| [Plugin contracts](../.github/skills/code-review-webgpu/references/plugin-contracts.md) | `ep/`, public API/adapters, session `plugin_ep/ep_factory_webgpu.*` | [#32587](https://github.com/microsoft/onnxruntime/pull/32587): ABI exceptions, ownership, capability parity; source `ep/runtime_compatibility.*` and `ep/api.cc`: host negotiation and exports |
| [Builds and dependencies](../.github/skills/code-review-webgpu/references/build-and-dependencies.md) | repository `cmake/`, Dawn patches/manifests, `plugin-ep-webgpu/paths.txt`, staging/package/update tools | [#32686](https://github.com/microsoft/onnxruntime/pull/32686), [#33396](https://github.com/microsoft/onnxruntime/pull/33396): DXC redistribution, exact patch/DEPS coordinates, rollback and package consumers |
| [Graph transformations](../.github/skills/code-review-webgpu/references/graph-transformations.md) | native/plugin capability discovery, `core/optimizer/` Conv/MatMulNBits/GQA fusions and transformer registration | [#33363](https://github.com/microsoft/onnxruntime/pull/33363), [#32929](https://github.com/microsoft/onnxruntime/pull/32929): preserved attributes/outputs, accepted/rejected rewrites, actual partition/layout |
| [Registration](../.github/skills/code-review-webgpu/references/registration.md) | native and contrib registry tables, supported types, NHWC domains, generated kernel docs | Domain/opset/type/host-input consistency; exercise the advertised contract in native and plugin paths |
| [Profiling and performance](../.github/skills/code-review-webgpu/references/profiling-and-performance.md) | `webgpu_profiler.*`, context query/flush/readback paths, Dawn platform and PIX integration | [#33781](https://github.com/microsoft/onnxruntime/pull/33781): clock domains; legacy [#18894](https://github.com/microsoft/onnxruntime/pull/18894) and [#19791](https://github.com/microsoft/onnxruntime/pull/19791): dispatch metadata and scoped workarounds |
| [Validation](../.github/skills/code-review-webgpu/references/validation.md) | provider/shared tests, test helpers, shader-key validator and current CI definitions | [#33242](https://github.com/microsoft/onnxruntime/pull/33242), [#33269](https://github.com/microsoft/onnxruntime/pull/33269): skip before tester construction, optional features, no fallback, test migration |

This map covers standard kernels plus contrib attention/transformer, diffusion,
MoE, and quantization families through their behavior. A new operator family
does not automatically require a new category. Add one when its contract cannot
be expressed by existing ownership boundaries and reading predicates.

## Dawn rolls, threading, and new kernels

These workflows have explicit entry-table triggers and stay within the existing
13 owners. A follow-up inspected source at
[`3e531e98c4117679b8b8497be5f65a336e3eb278`](https://github.com/microsoft/onnxruntime/commit/3e531e98c4117679b8b8497be5f65a336e3eb278):

- [Dawn rolls](../.github/skills/code-review-webgpu/references/build-and-dependencies.md#dawn-roll-workflow)
  use `plugin-ep-webgpu/tools/dawn/README.md` and `update_dawn.py`, the component
  policy, active CMake patches, and downstream consumers. Review the upstream
  revision range as well as generated metadata; tool success is not runtime
  validation. Source pins, redistributed DXC, Agility SDK, and package DLL lists
  have separate update boundaries.
- [Multi-thread support and thread safety](../.github/skills/code-review-webgpu/references/runtime-and-portability.md#thread-safety-contract)
  use `ep/README.md`, `ep/runtime_compatibility.*`, native/plugin concurrent-Run
  callbacks, `CommandRecordingState`, and `ProgramManager` cache ownership.
  Distinguish supported concurrent plugin sessions, serialized same-session
  Runs, caller-serialized compatibility mode, and built-in/WASM contracts.
  `plugin/test_webgpu_concurrency.cc` provides concrete regression scenarios;
  the callback and allocator tests cover separate lifetime/order boundaries.
- [New-kernel authoring](../.github/skills/code-review-webgpu/references/kernel-contract.md#author-a-new-native-kernel)
  composes schema/semantics, host validation, program/WGSL generation,
  accessors, identity, numerics/layout/features, registration, source/template
  integration, and tests. Native Pad and the template README demonstrate the
  integration shape. Authoring still selects all applicable category rows;
  it does not duplicate their detailed checks in the entry point.

These are source-backed guidance additions. No Dawn update, concurrency test,
new kernel implementation, or GPU run was performed for this documentation
follow-up. Unavailable execution evidence remains a review gap.

## Historical audit method and limits

The 2026-11-06 discovery queried public `microsoft/onnxruntime` PRs with
`is:pr webgpu in:title,body,comments`, split by creation year 2018–2026 and fully
paginated. Results were 180 in 2023, 260 in 2024, 463 in 2025, and 832 in 2026:
1,735 keyword matches. The union with 668 PR numbers extracted from native
WebGPU path history and 209 existing review records added 44 PRs, yielding
1,779 discovered PRs, including 65 authored by `gyagp`. An archive entry for
issue #32716 was excluded from PR counts. The path inventory contains 679 tracked files across the
provider, contrib, native tests, public headers, plugin, CMake/patch, and WGSL
generator roots used for history discovery.

The collection refreshes PR descriptions, current base/head/merge revisions,
general discussion, submitted review bodies, and inline comments including
replies, with pagination: 7,888 general comments, 10,965 submitted reviews
(including empty approvals), and 12,177 inline comments. Raw API responses and a heuristic category index stay
in local audit evidence, outside repository startup context. Keyword labels
assist triage; they are not accepted findings. Workflow approvals, bot commands,
empty approvals, and resolved/outdated flags do not establish technical review
or a verified fix.

Source-guided semantic inspection prioritizes the author's PRs and the review
threads linked above, including their replies and affected code. Rules in the
references are preventive checks corroborated by controlling implementation or
scoped historical evidence, not assertions that an open PR is fixed or merged.
Deprecated JSEP discussions contribute only an invariant that also makes sense for
the native implementation; their obsolete APIs and backend assumptions are not
imported as native rules or used to justify new performance work in JSEP.

Discovery and collection are broad; semantic validation is selective. This is
not a line-by-line re-review of every historical patch/comment, a proof that
every relevant PR is discoverable, or a guarantee of complete bug detection.
Shared-only changes with no WebGPU keyword/path-history signal can be missed;
deleted/private discussions and later updates are outside the snapshot. No new
GPU runs or live hosted Copilot review were performed for this documentation
audit. These limits must remain explicit when describing category completeness.

The current result is complete routing and maintained checks for each category
in this source/evidence map. Further verified lessons extend their owning
reference; an unverified suggestion stays in task evidence until checked against
its affected revision. Re-audit when a new subsystem, deployment contract, or
failure mode invalidates the map.

## Context retention and review walkthroughs

Local and GitHub Copilot reviews use the same committed checks. Local task
handoffs and raw logs preserve exact revisions, tested cases, unresolved work,
and links to evidence outside the skill. Hosted Copilot receives repository
guidance on the PR head branch; local history/personal skills do not transfer
automatically. Promote a verified reusable lesson into its owning reference
before expecting a hosted reviewer to use it. See
[agent guidance](Agent_Coding_Guidance.md#copilot-pull-request-reviews-on-github)
for platform settings and adoption evidence.

The following are manual selection walkthroughs, not measured hosted-model
evaluations. They check both exclusion and cumulative inclusion:

| Change and established effect | Expected selection |
| --- | --- |
| CPU-only kernel fix, CUDA-only attention change, or unrelated Python tooling | Existing review guidance; no WebGPU skill/references |
| Shared optimizer rule that excludes WebGPU, or shared test edit with no affected WebGPU execution | No WebGPU skill/references |
| Documentation that merely links an old WebGPU PR | No WebGPU skill/references |
| JSEP correctness/security fix; native counterpart checked and unaffected | JSEP policy and focused regression coverage; no native WebGPU references |
| JSEP kernel fix exposing the same native defect | JSEP policy plus matching native checks; flag a missing native fix/regression |
| New JSEP operator, feature, or performance-only optimization | Report under JSEP policy and direct work to native EP; do not treat it as accepted JSEP maintenance |
| JSEP migration/removal touching native integration | JSEP migration policy plus native checks for the affected native portion |
| Shared Gemm helper changing native WebGPU arithmetic and generated types | Kernel, cache, dispatch, numerics, validation; other rows only if their behavior changes |
| New ConvTranspose prepack layout | Kernel, layout/prepacking, cache, dispatch, numerics, registration, validation; runtime if ownership changes |
| Plugin streamless allocator/copy change | Runtime/memory, plugin contracts, validation; builds if target boundaries change |
| Dawn/DXC artifact removal | Builds/dependencies and validation; plugin/device/runtime rows for affected ABI/features/ownership |
| Dawn pin roll changing Tint validation or native callbacks | Builds/dependencies and validation; inspect upstream range, then select shader/runtime/device checks for affected behavior |
| Plugin multi-thread support or host-version concurrency gate change | Runtime/memory, plugin contracts, validation; devices/builds where required synchronization features or target guards change; keep same-session and compatibility-mode serialization distinct |
| New native shader kernel | Kernel authoring workflow, cache, dispatch, numerics, registration, validation; runtime/layout/devices/builds/graph/performance rows for the contracts introduced or changed |
| Shared GQA fusion changing WebGPU partition and epsilons | Graph transformations, numerics, registration, validation; kernel/cache rows if kernel behavior/codegen changes |
| Native timestamp correlation change | Profiling/performance and validation; runtime, device, plugin or dependency rows according to changed integration |

After selecting the skill, account for every reading-table row as checked with
evidence, inapplicable with a reason, or unverified with the check needed. Keep
the review report brief. Only selected references enter the active context;
the complete historical corpus and this maintenance audit need not.
