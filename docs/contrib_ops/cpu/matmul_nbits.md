# MatMulNBits CPU pre-packed weight sharing

This document describes how pre-packed `MatMulNBits` weights are shared **across inference sessions**
on the CPU execution provider, and the design that makes such sharing correct by construction.

## Background

`MatMulNBits` stores its quantized weight `B`, the per-block `scales`, and the optional
`zero_points` as constant initializers. At session initialization, the CPU kernel "pre-packs" these
into an MLAS-specific layout (`packed_b_`) for fast GEMM. Pre-packing is CPU- and time-intensive,
and the packed buffer can be large.

ONNX Runtime already supports sharing a single pre-packed buffer across multiple sessions through an
`OrtPrepackedWeightsContainer` (a.k.a. the *shared container*). When several sessions load the *same*
model and a container is provided to all of them, the first session pre-packs a weight and writes it
into the container; later sessions look it up and adopt the existing buffer instead of re-packing.

Historically this only worked for initializers the user explicitly enrolled (via `AddInitializer`).
The `MatMulNBits` weights produced by the **DQ → MatMulNBits graph fusion** are synthesized during
graph optimization and have transformer-generated names, so they could not be enrolled this way.

## Design: transformer-tagged stable sharing identity

The transformer that synthesizes the `MatMulNBits` weight tags the generated `B` initializer with a
stable **sharing identity**. SessionState enrolls *only* tagged initializers into the shared
container, keyed by that identity instead of by a hash of the packed bytes.

### Why key on a source-derived identity (not the packed bytes)?

A natural but fragile alternative is to content-address every `MatMulNBits` initializer by
`hash(packed_bytes)`. That makes sharing correctness depend on the kernel's packed representation
*fully* capturing the compute semantics. If two weights with different compute semantics (for
example, different `zero_points`) ever produce the same packed bytes — because some semantic input is
applied at compute time rather than baked into the buffer, or because of uninitialized padding — the
second session would silently adopt the wrong buffer and compute incorrect results, with no crash and
no warning.

Keying on a **source-derived identity** removes that dependency:

- The identity is a pure function of the *source* tensors — the transposed/packed quantized weight,
  the scales, and the optional zero points (plus their shapes) — see
  `ComputeMatMulNBitsSharingIdentity` in
  `onnxruntime/core/optimizer/matmul_nbits_sharing_identity.cc`.
- The **same** logical weight in two sessions of the **same** model yields the **same** identity, so
  the buffer is shared.
- **Any** difference that changes the result (different quantized weight, scales, or zero points)
  yields a **different** identity, so distinct weights can never collide.

Sharing is therefore correct by construction, independent of whether the packed bytes happen to
encode every semantic input.

### Components

1. **`Graph` side-map** (`include/onnxruntime/core/graph/graph.h`):
   `Graph::AddSharedInitializerIdentity(name, identity)` records a runtime-only map from an
   initializer name to its sharing identity. `Graph::GetSharedInitializerIdentity(name)` reads it.
   This state is **not** serialized into the `GraphProto`.

2. **Identity computation** (`core/optimizer/matmul_nbits_sharing_identity.{h,cc}`):
   `ComputeMatMulNBitsSharingIdentity(weight, scale, zero_point)` returns an FNV-1a content hash over
   the quantized tensors and their shapes.

3. **Transformer tagging**: the fusion sites tag the generated `B` initializer immediately after
   adding it to the graph, computing the identity from the source-derived tensors *before* they are
   moved into the graph:
   - `DQMatMulToMatMulNBitsAction::ProcessNewNode`
     (`core/optimizer/qdq_transformer/selectors_actions/qdq_actions.cc`)
   - `ApplyReshapeTransposeFusions` and `ApplyDirectDQFusions`
     (`core/optimizer/dq_matmulnbits_fusion.cc`)

   Only `B` is tagged; `scales` and `zero_points` are folded into `packed_b_` (see below) and are not
   shared separately.

4. **SessionState enrollment** (`core/framework/session_state.cc`,
   `PrepackConstantInitializedTensors`): an initializer is eligible for cross-session sharing if it is
   in `initializers_to_share_map` (the existing explicit path) **or** it is tagged. For tagged
   initializers the container key is `op_type + "+sid+" + identity`; for explicitly shared
   initializers the existing `op_type + "+" + hash(packed_bytes)` key is unchanged. As before, sharing
   is only active when a shared container is present (`prepacked_weights_container_ != nullptr`); no
   new session option is introduced.

## Kernel requirement: a finalized, self-contained `packed_b_`

Because another session may **adopt** the shared `B` buffer and skip its own scales/zero-point
packing, the buffer placed in the container must already be fully compute-ready. The CPU kernel
(`onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc`) is therefore changed so that, for the
`CompInt8` path, the scales and (constant) zero points are folded into `packed_b_` **during the `B`
PrePack**, rather than being deferred to the later `scales`/`zero_points` PrePack calls:

- `PrePack(InputIndex::B)` zero-fills the buffer (so alignment padding does not perturb the content
  hash), packs `B`, then issues one additional `MlasQNBitGemmPackQuantBData` call with
  `QuantBData == nullptr` to fold in the scales and zero points. It sets `packed_b_finalized_ = true`
  and pushes the finalized buffer into `PrePackedWeights`.
- The block sum / KleidiAI `BZpCorr` depends on the zero points, so it must be folded in **before**
  the framework content-hashes the buffer. This guarantees the buffer is self-contained.
- The later `scales`/`zero_points` PrePack calls are guarded with `!packed_b_finalized_`; they report
  `is_packed = false` and do not pack again (the `CompInt8` packing is single-shot, and the buffer may
  by then be one shared from another session).

This finalization is required regardless of the keying strategy: a shared buffer must be complete
before another session adopts it. The identity-based keying additionally guarantees that the buffer a
session adopts is the *correct* one for its weights.

## Summary

| Aspect | Content-hash-of-packed-bytes | Transformer-tagged identity (this design) |
| --- | --- | --- |
| What is hashed | Kernel's packed output bytes | Source-derived quantized tensors |
| Correctness depends on | Packed bytes capturing all semantics | Nothing extra — correct by construction |
| Opt-in surface | Broad session option | Existing shared-container presence |
| Scope | Every `MatMulNBits` initializer | Only transformer-generated, tagged weights |
