---
name: workspace-estimation-shared-header
description: "Use when editing a boundary-safe shared header meant to be included by both in-tree and shared-provider/plugin-bridge code (e.g. include/onnxruntime/core/framework/workspace_requirement.h, or any future header supporting OrtKernelWorkspaceEstimateFunc / DeclareWorkspaceRequirements per issue #29775's Phase-A workspace-estimation roadmap), or when a workspace-estimation math helper needs to be callable from a future plugin EP DLL. Covers a Node forward-declaration ODR trap and the math-helper vs. graph-parsing-wrapper reuse boundary discovered while implementing PR #29811 (MatMulNBits pilot)."
---

# Workspace-Estimation Shared Header: DLL-Boundary Pitfalls

Lessons from implementing the two-level (`EstimateWorkspace` / `DeclareWorkspaceRequirements`)
workspace-size estimation pilot for `MatMulNBits` (issue #29775 Phase-A, PR #29811). The
`WorkspaceRequirement` struct in `include/onnxruntime/core/framework/workspace_requirement.h` is
designed to be included by both in-tree kernel code and future plugin-EP adapter code — this is exactly
the kind of dual-included, DLL-boundary-crossing header where these gotchas apply.

## 1. Never forward-declare `Node` in a header included from both worlds

**Symptom:** a compile failure or, worse, a silent type mismatch that depends on include order —
because `onnxruntime::Node` is not one type across the whole codebase. In-tree code sees `class Node`
from `core/graph/graph.h`. Shared-provider/plugin-bridge code (anything reachable from a plugin DLL)
sees a *different* `Node` type from a provider-bridge header (e.g. `struct Node final`). These are two
distinct types that happen to share a name — different "ODR worlds."

**The trap:** writing `class Node;` (or any forward-declaration of `Node`) in a header that might be
`#include`d from both worlds. Whichever world's real definition gets included later in the same
translation unit can clash with your forward-declaration's class-key (`class` vs `struct`), or — worse
— the header can compile fine in isolation and only fail (or silently pick the wrong type) once combined
with a specific set of other includes.

**The fix:** omit the forward-declaration entirely. Don't try to avoid the `#include` of the real `Node`
header for compile-time savings in a shared, dual-included header — let each translation unit's own
real includes provide whatever `Node` type it needs. If you only need a pointer/reference and think a
forward-declare is a safe optimization, it is not safe here specifically because the two worlds disagree
on the underlying type.

**How to confirm you're clear:** the header must build cleanly when included from an in-tree-only
translation unit AND (once such a target exists) from a plugin-DLL translation unit, without relying on
which one happens to be included first.

## 2. Keep the shared/reusable half of a workspace-estimation function free of graph types

When a kernel's workspace-size computation needs to be callable from both in-tree and (eventually)
plugin code, split it into two pieces:

- **Pure-math core** — takes only plain shape/arch integers (e.g.
  `ComputeFpAIntBGemmWorkspaceSize(int m, int n, int k, int sm, int multiProcessorCount)`). No
  `Node&`, no `NodeArg`, no `TensorShape` parsing, no ORT graph types at all. This is the part a future
  plugin implementation can call *verbatim* — plugin kernels don't have `Node&` access, only the C-ABI
  shape representation (`OrtNode`, `Node_GetInputShape`).
- **Graph-parsing wrapper** — extracts the plain integers from `const Node&`/`NodeArg`/`TensorShape` (or
  from the plugin's C-ABI shape accessors). This half is inherently different per build configuration and
  is NOT reusable across the DLL boundary; it must be reimplemented against whichever shape
  representation the caller has.

**Why this matters concretely:** if you accidentally let the pure-math core accept or touch an in-tree
graph type (even just to read one field), you've made it impossible to reuse from the plugin path without
either (a) linking in-tree graph headers into the plugin DLL (defeats the purpose of a plugin boundary) or
(b) duplicating the whole function. Keep the split clean from the start.

**How to confirm you're clear:** grep the pure-math function's signature and body for any ORT graph type
(`Node`, `NodeArg`, `TensorShape`, `GraphViewer`, etc.) — it should only ever see plain scalars
(`int`/`int64_t`/`size_t`) and, at most, opaque device-property values already extracted by the caller.
If you find a graph type anywhere in that function, the split has leaked and needs to be pulled apart
before a plugin implementation can reuse it.

Related, not yet built: `docs/annotated_partitioning/future_directions_constrained_env.md` (Phase A /
plugin-ABI sections) describes the intended plugin-side C ABI surface for `DeclareWorkspaceRequirements`,
but as of PR #29811 no plugin-side implementation exists yet — see that doc's "Cost of a Real
Plugin-Side Override (Deferred)" note for what's still missing beyond just following this split.
