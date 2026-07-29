# Design: WebGPU Fused SwiGLU MLP with Subgroup Matrix Operations

**Status**: Draft
**Target**: WebGPU EP, `MatMulNBits` MLP fusion for prefill (`M >= 32`)
**Owner**: TBD
**Primary hardware target**: NVIDIA Ampere / Ada / Blackwell (RTX 30/40/50 series)
via the Vulkan backend and Dawn's `ChromiumExperimentalSubgroupMatrix` (implemented
on top of `VK_KHR_cooperative_matrix`). Verified on RTX 5060 Ti (Blackwell,
`config_index = 0`, subgroup size 32, 16×16×16 sub-tile shape).

**Extension targets (PR 3)**: Intel Xe2/Xe3 (`config_index = 1`, 8×16×16),
Apple M-series (`config_index = 2`, 8×8×8, Metal backend). Same code structure,
different tile-shape WGSL template.

---

## 1. Motivation

### 1.1 Current state

ONNX Runtime's WebGPU EP already has a fused SwiGLU MLP contrib op — see
[matmul_nbits_mlp.cc](../../onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_mlp.cc).
It fuses SkipLayerNorm + Gate MatMul + Up MatMul + SiLU + Multiply into a
single dispatch, and it works.

But: the fused fast path (`MatMulNBitsMlpDecodeProgram`) only activates when
**none of the specialized MatMulNBits kernels would win in the unfused case**.
That gate is in the `ComputeInternal` cascade:

```cpp
const bool can_use_decode_fast_path =
    is_decode_fast_path_candidate &&
    !would_use_subgroup_unfused &&
    !would_use_dp4a_unfused &&
    !would_use_wide_tile_unfused;
```

The consequence is that **the hottest platforms — Apple M-series (SubgroupMatrix),
NVIDIA/AMD desktop (DP4A) — bypass fusion entirely and run three separate
MatMuls with intermediate VRAM writes**. The fused path is essentially a
consolation prize for Intel iGPU / mobile / odd shapes.

This design adds a **prefill-only** fused MLP kernel that uses subgroup matrix
operations for the two matmuls, so Apple / NVIDIA / Intel Xe get fusion *and*
their specialized matmul acceleration at the same time.

### 1.2 Expected impact (honest)

For a typical FFN prefill (Llama-2-7B, `S=512`, `D_model=4096`, `D_ffn=11008`,
4-bit weights, block_size=32):

| Comparison baseline | Expected speedup |
|---|---|
| Unfused with SubgroupMatrix per-matmul (3 dispatches) | **1.10 – 1.25×** |
| Current `MatMulNBitsMlpDecodeProgram` extended to prefill (no subgroup matrix) | **1.4 – 1.8×** |
| Decode workloads | **No change** — this kernel does not activate at `M=1` |

**Why the win exists on NVIDIA specifically.** Two intermediate tensors —
`gate_tile` and `up_tile` — each `M × D_ffn` fp16 = ~11 MB per layer per prefill
step at these shapes. Eliminating both round-trips through global memory saves
~22 MB / layer, ~700 MB across 32 layers. On RTX 5060 Ti with ~450 GB/s memory
bandwidth, that's ~1.5 ms saved per prefill step. Prefill of Llama-2-7B
typically runs 15–40 ms on this class of GPU, so the win is **5–12% end-to-end**
on a bandwidth-bound prefill.

**Where the win is smaller.** On higher-bandwidth desktop cards (RTX 4090:
~1000 GB/s; RTX 5090: ~1800 GB/s) the intermediate roundtrip cost is
proportionally smaller. On the 5060 Ti the ratio is favorable because
Blackwell tensor cores are fast relative to the memory bandwidth — the kernel
is memory-bound and fusion directly attacks the bottleneck.

**In Wasm / browser builds**, per-dispatch overhead adds ~50 µs of CPU cost;
saving 2 dispatches per layer across 32 layers is an additional ~3 ms — real
but secondary to the VRAM savings above.

**Not applicable to decode.** Subgroup matrix instructions expect a tile shape
of 8×8 or 16×16 in the M dimension; at M=1 they waste 87–94% of the tile. The
existing decode fast path (`MatMulNBitsMlpDecodeProgram`) remains the correct
choice for M=1 and is out of scope for this document.

### 1.3 Precedent in the codebase

This design mirrors two existing patterns:

- **`SubgroupMatrixMatMulNBitsProgram`** in
  [subgroup_matrix_matmul_nbits.cc](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.cc) —
  the single-MatMul subgroup-matrix kernel. This design essentially extends it
  to fuse two co-located MatMuls (Gate and Up) that share the same A input.
- **Prefill vs. decode split in FA** in
  [flash_attention.cc](../../onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc):
  `FlashAttentionProgram` for prefill (M-axis / Q-axis parallelism, no cross-workgroup
  reduction), `FlashAttentionDecodeQKV + VxReduce` for decode (K-axis Split-K with
  reduction). Same split of responsibilities applies here.

---

## 2. Semantic contract (v1)

### 2.1 Inputs

Same as the existing `MatMulNBitsMlp` op, minus the SkipLayerNorm inputs (v1
requires SkipLayerNorm to be a separate preceding node — the fused version is
future work):

| Input | Shape | Dtype | Notes |
|---|---|---|---|
| `a` | `[M, K]` | fp16 or fp32 | Activation input |
| `gate_B` | packed `[N, K/8]` u32 | uint8/uint32 packed | 4-bit blocked weights, block_size=32 |
| `gate_scales` | `[N, K/block_size]` | Same as `a` | Per-block scale |
| `up_B` | packed `[N, K/8]` u32 | uint8/uint32 packed | Same layout as gate_B |
| `up_scales` | `[N, K/block_size]` | Same as `a` | |
| `gate_bias` (optional) | `[N]` | Same as `a` | |
| `up_bias` (optional) | `[N]` | Same as `a` | |

Where `M = batch × seq_length`, `K = D_model`, `N = D_ffn`.

### 2.2 Output

```
y[m, n] = SiLU(a[m, :] · gate_B[n, :] + gate_bias[n])
         * (a[m, :] · up_B[n, :] + up_bias[n])
```

Shape `[M, N]`, same dtype as `a`.

### 2.3 Non-inputs (deferred)

Future PRs may add:
- `skip` (residual add before norm)
- `norm_scale` (RMSNorm weight)
- `epsilon` (RMSNorm epsilon)
- `input_skip_bias_sum` output (residual passthrough for the next block)

For v1 these must be handled by preceding graph nodes (`SkipLayerNorm` runs as
a separate dispatch). The fusion win is preserved for the MLP body.

---

## 3. Algorithm

### 3.1 High-level structure

Each workgroup:
1. Owns one output tile `[TILE_M, TILE_N]` of `y`.
2. Iterates over the K dimension in blocks of `TILE_K`.
3. Per K-block:
   - Cooperatively loads a shared A tile into workgroup memory (single load; both matmuls consume it).
   - Cooperatively loads and dequantizes a Gate B tile.
   - Cooperatively loads and dequantizes an Up B tile.
   - Barrier.
   - Each subgroup accumulates gate and up matmuls via `subgroupMatrixMultiplyAccumulate`.
   - Barrier before next K-block.
4. Epilogue: add biases, apply SiLU * Multiply directly on subgroup matrix registers, write to output.

### 3.2 Pseudocode

```wgsl
// Workgroup: (num_workgroups_x, num_workgroups_y, 1) = (N / TILE_N, M / TILE_M, 1)
// Each workgroup owns tile [tile_m_start : tile_m_start + TILE_M, tile_n_start : tile_n_start + TILE_N]

var<workgroup> A_tile: array<f16, TILE_M * TILE_K>;
var<workgroup> gate_B_tile: array<f16, TILE_K * TILE_N>;
var<workgroup> up_B_tile:   array<f16, TILE_K * TILE_N>;

fn main(...) {
    // Per-subgroup accumulator registers (fp32 accumulate, fp16 result)
    var gate_acc: subgroup_matrix<f32, SUB_TILE_M, SUB_TILE_N>;
    var up_acc:   subgroup_matrix<f32, SUB_TILE_M, SUB_TILE_N>;
    subgroup_matrix_fill(&gate_acc, 0.0);
    subgroup_matrix_fill(&up_acc, 0.0);

    for (var k = 0u; k < K; k += TILE_K) {
        // Phase 1: cooperative load and dequantize
        cooperative_load_A(A_tile, a, tile_m_start, k, TILE_M, TILE_K);
        cooperative_load_and_dequant_B(gate_B_tile, gate_B, gate_scales, tile_n_start, k, TILE_N, TILE_K);
        cooperative_load_and_dequant_B(up_B_tile,   up_B,   up_scales,   tile_n_start, k, TILE_N, TILE_K);
        workgroupBarrier();

        // Phase 2: subgroup matrix multiplies (each subgroup owns one sub-tile)
        for (var kk = 0u; kk < TILE_K; kk += SUB_TILE_K) {
            let A_sub  = subgroup_matrix_load(A_tile, ...);
            let gBsub  = subgroup_matrix_load(gate_B_tile, ...);
            let uBsub  = subgroup_matrix_load(up_B_tile, ...);
            gate_acc = subgroupMatrixMultiplyAccumulate(A_sub, gBsub, gate_acc);
            up_acc   = subgroupMatrixMultiplyAccumulate(A_sub, uBsub, up_acc);
        }
        workgroupBarrier();
    }

    // Epilogue (in subgroup matrix registers)
    if (has_gate_bias) { gate_acc += broadcast_bias(gate_bias, tile_n_start); }
    if (has_up_bias)   { up_acc   += broadcast_bias(up_bias,   tile_n_start); }
    let result = silu(gate_acc) * up_acc;  // element-wise on the matrix tile

    // Write to global y
    subgroup_matrix_store(y, result, tile_m_start, tile_n_start);
}
```

The critical properties:
- **A is loaded once per K-block** (shared with both matmuls) — this is the primary fusion win.
- **Gate and up accumulators live in subgroup matrix registers** for the entire K sweep — no intermediate VRAM writes.
- **SiLU + Multiply happens directly on registers** — never touches workgroup or global memory.

### 3.3 Tile shape choices

Follow the existing `SubgroupMatrixMatMulNBits` selector in
[subgroup_matrix_config.h](../../onnxruntime/core/providers/webgpu/math/subgroup_matrix_config.h) —
four tile shapes are supported today, matching the `supported_subgroup_matrix_configs`
array:

| Config index | Sub-tile M × N × K | Component type | Subgroup size | Needs prepack | Target |
|---|---|---|---|---|---|
| **0** | **16 × 16 × 16** | **F16 → F16** | **32** | **yes** | **NVIDIA Ampere/Ada/Blackwell** (RTX 30/40/50 series); 128×128 workgroup tile |
| 1 | 8 × 16 × 16 | F16 → F16 | 16–32 | yes | Intel Xe2/Xe3 |
| 2 | 8 × 8 × 8 | F16 → F16 | 32 | no | Apple M-series (fp16 output) |
| 3 | 8 × 8 × 8 | F32 → F32 | 32 | no | Any adapter with fp32 output |

Reuse the same `IsSubgroupMatrixConfigSupported` and `config_index` mechanism.
Each config gets its own WGSL template (mirroring the existing three MatMulNBits
templates). The corresponding existing single-matmul templates are:

- [subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template) — NVIDIA (config 0)
- [subgroup_matrix_matmul_nbits_8x16x16.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_8x16x16.wgsl.template) — Intel Xe (config 1)
- [subgroup_matrix_matmul_nbits_8x8x8.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_8x8x8.wgsl.template) — Apple (configs 2 and 3)

---

## 4. Memory budget

### 4.1 Workgroup shared memory

**Primary config: NVIDIA `16x16x16_128` (TILE_M=128, TILE_K=32, TILE_N=128).**
This matches the 128×128 workgroup output tile used by the existing NVIDIA
subgroup-matrix MatMulNBits kernel:

| Buffer | fp16 elements | Bytes |
|---|---|---|
| `A_tile` | 128 × 32 = 4096 | 8 KB |
| `gate_B_tile` (dequantized) | 32 × 128 = 4096 | 8 KB |
| `up_B_tile` (dequantized) | 32 × 128 = 4096 | 8 KB |
| **Total workgroup memory (NVIDIA)** | | **24 KB** |

Fits within NVIDIA's `maxComputeWorkgroupStorageSize` (typically 48 KB via
Vulkan). At pipeline creation time, verify
`context.DeviceLimits().maxComputeWorkgroupStorageSize >= 32 * 1024` before
dispatching this variant; fall back otherwise.

**Extension configs** (PR 3):

| Config | TILE_M × TILE_K × TILE_N | Workgroup memory |
|---|---|---|
| Intel `8x16x16` | 64 × 32 × 128 (proposed) | 20 KB |
| Apple `8x8x8` | 32 × 32 × 64 | 10 KB |

All fit within typical `maxComputeWorkgroupStorageSize` on their respective
adapters (Apple ~32 KB, Intel Xe ~32 KB via Vulkan/D3D12).

### 4.2 Per-subgroup register footprint

Two subgroup matrix accumulators (fp16 result on all supported configs, per
`resultComponentType` in `supported_subgroup_matrix_configs`) at sub-tile shape:

| Config | Accumulator elements per subgroup | Bytes |
|---|---|---|
| **NVIDIA `16x16x16`** | **16 × 16 × 2 accumulators = 512 fp16** | **1024** |
| Intel `8x16x16` | 8 × 16 × 2 = 256 fp16 | 512 |
| Apple `8x8x8` (fp16) | 8 × 8 × 2 = 128 fp16 | 256 |

All comfortably within the register file of a modern GPU SIMD/warp (NVIDIA
Blackwell ~2 KB per warp effective, Apple ~4 KB per SIMD-32, Intel ~2 KB per
SIMD-16). No spill risk.

Note: multiple subgroups (typically 4) per workgroup, each owning a sub-tile of
the output; so the *per-invocation* register footprint scales with the number
of output sub-tiles a single invocation is responsible for. For the NVIDIA
128×128 workgroup tile with 16×16 sub-tiles: 8×8 = 64 sub-tiles distributed
across 4 subgroups = 16 sub-tiles per subgroup. That's manageable but tight —
verify no register spill at Stage 3 by inspecting compiled SPIR-V.

---

## 5. Prepack strategy

The existing `SubgroupMatrixMatMulNBitsProgram` prepacks 4-bit weights via
[subgroup_matrix_matmul_nbits_prepack.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_prepack.wgsl.template)
into a subgroup-matrix-friendly layout. This runs once at op initialization
(the `WebGpuKernel::PrePack` override) — never per Run.

For the fused MLP, we prepack **both** `gate_B` and `up_B` using the same shader
against the same target layout. Two prepack dispatches at op init.

### 5.1 PrePack implementation sketch

Add to `MatMulNBitsMlp`:

```cpp
Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
               /*out*/ bool& is_packed,
               /*out*/ PrePackedWeights* prepacked_weights) override {
  is_packed = false;
  // Only prepack when the subgroup matrix path will apply for this session.
  if (!SubgroupMatrixMlpFusedWillApply(...)) {
    return Status::OK();
  }

  if (input_idx == kGateBIdx) {
    // Reuse the existing prepack shader; store into gate_B_packed_.
    ORT_RETURN_IF_ERROR(PrepackWeightForSubgroupMatrix(tensor, gate_B_packed_, alloc, ...));
    is_packed = true;
  } else if (input_idx == kUpBIdx) {
    ORT_RETURN_IF_ERROR(PrepackWeightForSubgroupMatrix(tensor, up_B_packed_, alloc, ...));
    is_packed = true;
  }
  return Status::OK();
}
```

Note: PrePack is called synchronously during model load, so the two prepack
dispatches happen at load time and add ~1–10 ms per layer to load latency
(negligible).

---

## 6. Dispatch cascade integration

### 6.1 New activation gate

Add to `MatMulNBitsMlp::ComputeInternal`, **before** the existing
`can_use_decode_fast_path` check:

```cpp
#if !defined(__wasm__)
int32_t sm_config_index = -1;
const bool can_use_subgroup_matrix_mlp_fused =
    M >= kMinMForTileOptimization &&              // typically 32
    (bits_ == 4) &&                                // v1: 4-bit only
    block_size == 32 &&
    K % 32 == 0 && N % 64 == 0 &&
    context.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix) &&
    IsSubgroupMatrixConfigSupported(context, is_fp16, sm_config_index) &&
    // v1: only NVIDIA config (16x16x16); other configs added in PR 3.
    sm_config_index == 0 &&
    !has_skip_input && !has_norm_input &&          // v1: no norm/skip fusion
    context.DeviceLimits().maxComputeWorkgroupStorageSize >= 32 * 1024 &&
    !env_disable_fused_mlp_subgroup_matrix();      // ORT_WEBGPU_DISABLE_FUSED_MLP_SUBGROUP_MATRIX

if (can_use_subgroup_matrix_mlp_fused) {
    return ApplySubgroupMatrixMlpFused(a, gate_B_packed_.get(), gate_scales,
                                        up_B_packed_.get(), up_scales,
                                        gate_bias, up_bias,
                                        M, N, K, block_size, sm_config_index,
                                        activation_kind_, context, y);
}
#endif
```

The `sm_config_index == 0` gate is temporary — it restricts v1 to NVIDIA-style
16×16×16 subgroup matrix hardware. Remove this line in PR 3 when the Intel
(config 1) and Apple (config 2/3) templates land.

// ... existing dispatch cascade continues (can_use_decode_fast_path, unfused fallback)
```

### 6.2 Cascade order (post-integration)

1. **New: SubgroupMatrix fused MLP** (`M >= 32`, subgroup matrix available)
2. **Existing: DecodeProgram fast path** (`M == 1`, bits=4, block_size=32, no subgroup-matrix win for unfused)
3. **Existing: unfused fallback** (`ApplyUnfusedMlp` — separate gate/up matmuls + tiny SiLU*mul kernel)

The unfused fallback continues to route each individual matmul through
`SubgroupMatrixMatMulNBits` / `DP4AMatMulNBits` / `WideTileMatMulNBits` as it
does today. Nothing changes for platforms/shapes outside the new gate.

---

## 7. File layout

Create these files:

```
onnxruntime/contrib_ops/webgpu/quantization/
├── matmul_nbits_mlp_subgroup_matrix.h        (NEW)
├── matmul_nbits_mlp_subgroup_matrix.cc       (NEW)
├── matmul_nbits_mlp_subgroup_matrix_16x16x16_128.wgsl.template   (NEW, PR 1)  ← NVIDIA
├── matmul_nbits_mlp_subgroup_matrix_8x16x16.wgsl.template        (NEW, PR 3)  ← Intel
└── matmul_nbits_mlp_subgroup_matrix_8x8x8.wgsl.template          (NEW, PR 3)  ← Apple
```

The `.h` and `.cc` follow the same structure as
[subgroup_matrix_matmul_nbits.h](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h)
and
[subgroup_matrix_matmul_nbits.cc](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.cc)
respectively. Public API:

```cpp
class SubgroupMatrixMlpFusedProgram final : public Program<SubgroupMatrixMlpFusedProgram> {
 public:
  SubgroupMatrixMlpFusedProgram(uint32_t nbits, int32_t config_index,
                                bool has_gate_bias, bool has_up_bias,
                                MlpActivationKind activation_kind);
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"m_tiles_per_wg", ProgramUniformVariableDataType::Uint32});
 private:
  uint32_t nbits_;
  int32_t config_index_;
  bool has_gate_bias_;
  bool has_up_bias_;
  MlpActivationKind activation_kind_;
};

Status ApplySubgroupMatrixMlpFused(const Tensor* a,
                                    const Tensor* gate_b_packed, const Tensor* gate_scales,
                                    const Tensor* up_b_packed,   const Tensor* up_scales,
                                    const Tensor* gate_bias, const Tensor* up_bias,
                                    uint32_t M, uint32_t N, uint32_t K,
                                    uint32_t block_size, int32_t config_index,
                                    MlpActivationKind activation_kind,
                                    onnxruntime::webgpu::ComputeContext& context,
                                    Tensor* y);
```

---

## 8. Stage-by-stage plan

Each stage is a landable PR. Merge in order.

### Stage 1 — Scaffolding (PR 1a, ~2 days)

**Deliverable**: Files created, dispatch cascade wired, guarded by env var so
default behavior is unchanged.

**Tasks**:
- Create the three files listed in §7 with empty/stub implementations. **PR 1
  starts with the NVIDIA `16x16x16_128` template only**.
- Copy the existing
  [subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template)
  into the new fused-MLP template as a starting point (single matmul, will be
  extended in Stage 2).
- Wire `can_use_subgroup_matrix_mlp_fused` gate into
  `MatMulNBitsMlp::ComputeInternal`, guarded by
  `ORT_WEBGPU_ENABLE_FUSED_MLP_SUBGROUP_MATRIX` (opt-in for now — flip to
  opt-out `ORT_WEBGPU_DISABLE_...` at Stage 6).
- Add the two new `.cc` files to the WebGPU quantization build glob (verify in
  the CMake config for the quantization folder — usually automatic).
- Add empty `PrePack` override; `gate_B_packed_`/`up_B_packed_` member fields.

**Success criterion**: Builds on Linux with Vulkan and Windows with D3D12/Vulkan;
existing tests pass; env var flips the code path but produces a "not implemented"
stub error message so callers know they're on the new path. Verify the RTX GPU
adapter reports `config_index = 0` in a small standalone probe.

### Stage 2 — Two-matmul WGSL, no fusion yet (PR 1b, ~1 week)

**Deliverable**: The kernel produces correct output but with two independent A
loads (no fusion win yet). Parity established.

**Tasks**:
- Extend the WGSL template to declare two weight buffers (`gate_B`, `up_B`) and
  two scale buffers.
- Two matmul loops sequentially:
  ```
  for k_tile: load A → matmul into gate_acc
  for k_tile: load A → matmul into up_acc  (same A tile shape, second time)
  ```
- Epilogue: SiLU + multiply, write output.
- Add C++ parity test at `onnxruntime/test/contrib_ops/matmul_nbits_mlp_subgroup_matrix_test.cc`:
  Generate random inputs; run both fused and unfused paths; assert
  `max_abs_diff < 1e-3` (fp16 tolerance).

**Success criterion**: Parity test passes; kernel is likely slower than
unfused baseline (that's expected without the fusion refactor).

### Stage 3 — A-tile reuse (PR 1c, ~1 week)

**Deliverable**: The fusion win — single A load per K-tile, shared by both
matmuls. This is what the kernel is *for*.

**Tasks**:
- Restructure the K-loop:
  ```
  for k_tile:
      cooperative_load_A_tile()
      cooperative_load_gate_B_tile()
      cooperative_load_up_B_tile()
      barrier
      for sub_k in TILE_K:
          gate_acc += subgroup_matrix_multiply_accumulate(A_sub, gate_B_sub, gate_acc)
          up_acc   += subgroup_matrix_multiply_accumulate(A_sub, up_B_sub, up_acc)
      barrier
  ```
- Verify parity is maintained.
- Add benchmark comparing:
  - Fused (this kernel)
  - Unfused with SubgroupMatrix per-matmul (current best baseline)
  - Unfused with manual dot-product WGSL (older baseline)

**Success criterion**: Parity holds. **Measurable speedup vs. unfused
SubgroupMatrix baseline on RTX 5060 Ti (or any Blackwell/Ada NVIDIA GPU):
1.10–1.25×** on `M=512, K=4096, N=11008` (Llama-2-7B FFN prefill shape). If
speedup < 1.10×, investigate:
- Suboptimal A-tile layout in workgroup memory (padding, bank conflicts)
- Extra `workgroupBarrier` calls between gate and up matmuls
- Accumulator spilling to threadgroup memory (inspect compiled SPIR-V
  via `chrome://gpu` or Vulkan tools)
- Prepack layout mismatch — gate_B and up_B must be in the same layout the
  subgroup matrix `subgroupMatrixLoad` expects

**PR 1 lands after Stage 3.** Total for PR 1: ~500 lines, well-scoped.

### Stage 4 — Bias fusion (PR 2, ~2 days)

**Deliverable**: Optional `gate_bias` and `up_bias` inputs supported.

**Tasks**:
- Add uniform bindings for biases (guarded by `has_gate_bias` and `has_up_bias`
  template params).
- In the epilogue, add bias to the accumulator before SiLU/Multiply. This is a
  register-level add — no additional VRAM traffic.
- Extend parity tests to cover biased shapes.

**Success criterion**: Parity holds with biases; perf unchanged vs Stage 3
(bias add is free — pure register op).

### Stage 5 — Cross-vendor templates (PR 3, ~2 weeks)

**Deliverable**: Kernel works on Intel Xe and Apple M-series in addition to
NVIDIA.

**Tasks**:
- Add `matmul_nbits_mlp_subgroup_matrix_8x16x16.wgsl.template` for Intel Xe2/Xe3
  (config 1).
- Add `matmul_nbits_mlp_subgroup_matrix_8x8x8.wgsl.template` for Apple M-series
  (config 2 fp16, and config 3 fp32).
- Extend `ApplySubgroupMatrixMlpFused` to select the right template based on
  `config_index` (returned by `IsSubgroupMatrixConfigSupported`).
- Remove the `sm_config_index == 0` restriction from
  `can_use_subgroup_matrix_mlp_fused`.
- On Apple, apply the existing `accuracy_level == 4` restriction from
  `CanApplySubgroupMatrixMatMulNBits` (see §10.1).
- Add per-vendor benchmark measurements to §12 of this file.

**Success criterion**: Parity across NVIDIA / Intel Xe / Apple. Perf table
showing speedup on each. NVIDIA is expected to remain the biggest single-GPU
win (tensor core throughput is very high on Ada/Blackwell).

### Stage 6 — Rollout (PR 4, ~2 days)

**Deliverable**: Path enabled by default; env var flipped to opt-out.

**Tasks**:
- Rename `ORT_WEBGPU_ENABLE_FUSED_MLP_SUBGROUP_MATRIX` →
  `ORT_WEBGPU_DISABLE_FUSED_MLP_SUBGROUP_MATRIX`.
- Extend `benchmark_matmul_nbits.py` (or the model builder benchmark) to include
  a fused MLP toggle for regression tracking.
- Update this design doc's "Measured performance" section with real numbers.

**Success criterion**: Model-level benchmarks show speedup on Phi-4 Mini /
Llama-3.1-8B / Gemma-2 prefill on Apple M-series and one NVIDIA GPU; no
regressions elsewhere.

---

## 9. Testing plan

### 9.1 Unit tests (C++)

Location: `onnxruntime/test/contrib_ops/matmul_nbits_mlp_subgroup_matrix_test.cc`.

- **Parity vs. unfused**: 20+ shape combinations covering:
  - `M ∈ {32, 64, 128, 256, 512, 1024}`
  - `K ∈ {2048, 4096, 8192}` (multiples of 32)
  - `N ∈ {5504, 11008, 14336, 28672}` (multiples of 64)
  - fp16 and fp32
  - `bits ∈ {4}` (add 8 in PR 2 or later)
  - With and without biases
- **Env var toggle**: run the same test with `ORT_WEBGPU_DISABLE_FUSED_MLP_SUBGROUP_MATRIX=1`
  to force the unfused path.

### 9.2 Vulkan-specific testing

Since the primary target is NVIDIA discrete hardware via Vulkan:

- **NVIDIA Ampere+ / Ada / Blackwell with Vulkan (`VK_KHR_cooperative_matrix`)**:
  primary target. Verify `IsSubgroupMatrixConfigSupported` returns `config_index = 0`
  (16×16×16 shape). Minimum driver version: NVIDIA 525.60 on Linux, 528.24 on
  Windows (first release with cooperative matrix; earlier drivers report the
  extension but with limited shape support).
- **AMD RDNA3+ discrete with Vulkan**: RDNA3+ exposes
  `VK_KHR_cooperative_matrix` with MFMA-like intrinsics. Verify
  `IsSubgroupMatrixConfigSupported` returns any of configs 0/1/2. Fall back
  gracefully if not.
- **Intel Arc (Alchemist/Battlemage) with Vulkan**: exposes cooperative matrix
  via `VK_KHR_cooperative_matrix` on driver v101.5xxx+; targets config 1
  (8×16×16). Deferred to PR 3.
- **Apple M-series (Metal, not Vulkan)**: targets config 2 (8×8×8). Deferred
  to PR 3.
- **lavapipe**: `ChromiumExperimentalSubgroupMatrix` is **not exposed**. Kernel
  path will not activate; existing fallback runs. The `webgpu-local-testing`
  skill's lavapipe MatMul crash is not a concern here because lavapipe will
  never dispatch this kernel.

Testing checklist per-vendor:
```
□ IsSubgroupMatrixConfigSupported returns expected config_index
□ Kernel compiles without shader errors (check Dawn logs at
  ORT_LOGGING_LEVEL=Verbose)
□ Parity vs. unfused: max_abs_diff < 1e-3
□ Perf > 1.10× vs. unfused with SubgroupMatrix
□ No workgroup memory or register spill (inspect compiled SPIR-V via
  `spirv-dis` or Vulkan validation layer output)
```

### 9.3 End-to-end model tests

Once PR 1 lands:
- Phi-4 Mini prefill on Apple M3/M4 — expect 10–20% end-to-end prefill speedup
- Llama-3.1-8B prefill on NVIDIA RTX 4090 (via Vulkan) — expect similar

Run with `ORT_WEBGPU_DISABLE_FUSED_MLP_SUBGROUP_MATRIX=1` to confirm the
speedup delta.

### 9.4 CI

Add a job to run the parity test suite on:
- macOS-arm64 Metal (existing coverage)
- Windows Vulkan (if a Vulkan CI runner exists; if not, defer to manual testing
  and document the gap)

---

## 10. Risks and mitigations

### 10.1 Subgroup matrix precision on Apple (deferred concern for PR 3)

The existing `SubgroupMatrixMatMulNBits` restricts Apple usage to
`accuracy_level == 4` because fp16 subgroup matmul has "some precision issues"
per the comment in `CanApplySubgroupMatrixMatMulNBits`. **When PR 3 adds Apple
support, mirror the same restriction** — restrict Apple activation to
`accuracy_level == 4`.

Does not apply to v1 (NVIDIA-only). NVIDIA cooperative-matrix fp16
accumulation is well-behaved on Ampere/Ada/Blackwell hardware.

### 10.2 Workgroup memory overshoot on some Intel adapters

Some older Intel Xe adapters may report `maxComputeWorkgroupStorageSize = 16 KB`,
which is below the 18 KB required by `16x16x16_128`. **Mitigation**: check
`context.DeviceLimits().maxComputeWorkgroupStorageSize` in the activation
gate; fall back to unfused if insufficient. Same pattern as the existing
Metal 10-buffer workaround in `MatMulNBitsMlp`.

### 10.3 Prepack + shared weights across sessions

If the same MatMulNBits weight tensor is shared across multiple ops or
sessions, prepacking twice is wasted work. **Mitigation**: use ORT's existing
`PrePackedWeights` cache mechanism (same pattern as
`SubgroupMatrixMatMulNBits::PrePack`). The framework deduplicates.

### 10.4 Wasm build

`ChromiumExperimentalSubgroupMatrix` is not available in browser WebGPU today.
**Mitigation**: guard the entire new kernel with `#if !defined(__wasm__)`
(same pattern as existing subgroup matrix code). Wasm path continues to use
the existing decode fast path and unfused fallback.

### 10.5 Numerical accumulation over long K

Accumulating `gate_acc` and `up_acc` in fp16 across a 4096-element K sweep can
overflow the fp16 range for pathological inputs. **Mitigation**: accumulate in
fp32 even when inputs are fp16 (subgroup matrix operations natively support
fp32 accumulator with fp16 A/B). Register cost: ~2× the accumulator footprint
(still well within budget, per §4.2).

---

## 11. Addendum (2026-07-22): Fusion-shape taxonomy, prepack semantics, and NVIDIA-Vulkan ShaderF16 toggle

This addendum captures design discussion that occurred after the original doc
was written. Nothing above is invalidated; this refines and extends §§ 3–5.

### 11.1 What the `PrepackProgram` actually does — and its scope

The prepack kernel
([subgroup_matrix_matmul_nbits_prepack.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_prepack.wgsl.template))
is **not** a general matmul optimization. It exists exclusively because the
WGSL `subgroupMatrixLoad` intrinsic (which lowers to `VK_KHR_cooperative_matrix`
`OpCooperativeMatrixLoad`) expects the matrix fragment source memory to be laid
out in a specific tiled order — each `(sg_mat_m × sg_mat_k)` fragment sitting
contiguously — so all subgroup lanes can coalesce their fragment loads.

Row-major A does not satisfy this layout. The prepack rearranges A from
row-major `[M, K]` into a "blocks-of-blocks" layout where fragment tiles are
contiguous. For the NVIDIA 16×16×16 config that means every 16×16 sub-tile
becomes 256 consecutive fp16 elements. Padding: the prepack output is padded
to `padded_M = ceil(M / tile_size_a) * tile_size_a` so all subgroups read
valid data at tile boundaries.

**Applicability**: 100% specific to the subgroup-matrix matmul path. The
`config.needsPrepack` flag is checked at
[subgroup_matrix_matmul_nbits.cc:181](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.cc#L181)
and may be `false` on some configs (e.g., certain Apple Metal configs where
the tile-shape matches Metal's native SIMD-group layout without reshaping).
Other MatMulNBits kernels — DP4A int8, wide-tile fp16, the small-M kernel —
use conventional vec4/scalar A-loads and never need this prepack. Therefore
**the "share the prepack" optimization discussed below only benefits the
subgroup-matrix path**; it is invisible to DP4A/wide-tile/decode paths.

### 11.2 Fusion-shape taxonomy for the prefill MLP

The original doc (§§ 3–5) describes a single fused-kernel target. In
implementation we identified **four viable shapes**, with different risk /
reward profiles. Ordered from safest-first:

#### Shape A — Share prepack; keep gate/up matmuls separate; keep silu*mul as its own kernel

```
prepack(normalized_a)                       → prepacked_a   (dispatch 1)
matmul(prepacked_a, gate_b, gate_scales)    → gate_out      (dispatch 2)
matmul(prepacked_a, up_b,   up_scales)      → up_out        (dispatch 3)
silu*mul(gate_out, up_out)                  → y             (dispatch 4)
```

- Total dispatches: **4** (down from 6 in current unfused prefill: 1 SLN + 2 prepack + 2 matmul + 1 epilogue).
- Refactor: split `ApplySubgroupMatrixMatMulNBits` (subgroup_matrix_matmul_nbits.cc:152) into two functions:
  - `PrepackAForSubgroupMatrixMatMulNBits(a, ...) → prepacked_a`
  - `ApplyPrepackedSubgroupMatrixMatMulNBits(prepacked_a, b, scales, ...)`
  Existing call sites in `matmul_nbits.cc` become `PrepackA(...) → ApplyPrepacked(...)`
  (semantics-preserving); the new MLP path prepacks once and calls ApplyPrepacked twice.
- Kernel body of the actual matmul: **unchanged**. Correctness risk: ~zero.
- Expected uplift on prefill throughput: **5–10%**, from eliminating one redundant M·K prepack write+read pass. At M=2048, K=2048 that's ~16 MB of VRAM traffic per MLP layer.

#### Shape B — Shape A + fuse silu into gate matmul, fuse mul into up matmul

```
prepack(normalized_a)                                    → prepacked_a   (dispatch 1)
matmul(prepacked_a, gate_b) + silu epilogue              → silu_gate_out (dispatch 2)
matmul(prepacked_a, up_b)   + "load silu_gate * mul" ep. → y             (dispatch 3)
```

- Total dispatches: **3**.
- Eliminates the standalone silu*mul kernel and its 3-buffer `M·N` traffic pattern (read gate_out + read up_out + write y ≈ 3·M·N·2 bytes = ~72 MB at M=2048, N=6144).
- Kernel modification is confined to the epilogue emitter of the existing subgroup matmul kernel. Two epilogue variants become template parameters (or use two shader variants). Accumulate loop is untouched, so tuning is preserved.
- Register pressure: unchanged per kernel — silu is scalar-in-place; the up kernel's extra fp16 load hits the same tile coordinates it's writing to (cache-friendly).
- Additional uplift over Shape A: **10–15%** at mid-M (M∈[128, 1024]), tapering above M≈1500 where matmul compute dominates the epilogue traffic.

#### Shape C-dualB — Single fused matmul; two B bindings; two accumulators; silu*mul at output write

```
prepack(normalized_a)                                                       → prepacked_a   (dispatch 1)
one kernel:
  for k_tile in K:
    A_tile = load prepacked_a[m_tile, k_tile]
    subgroupMatMulAccumulate(gate_acc, A_tile, gate_b[k_tile, n_tile])
    subgroupMatMulAccumulate(up_acc,   A_tile, up_b  [k_tile, n_tile])
  y[m_tile, n_tile] = silu(gate_acc) * up_acc                               → y             (dispatch 2)
```

- Total dispatches: **2**.
- **No offline concat required.** This is structurally the same idea as the existing decode fused kernel (`MatMulNBitsMlpDecodeProgram`), which already takes gate_b, gate_scales, up_b, up_scales as four separate bindings and fuses without any weight preprocessing. Scaled to prefill's tile geometry.
- A-load amortization: `A_tile` is loaded once per K-iteration and consumed by both matmul-accumulate calls. Cuts A-side subgroup-matrix load traffic in half compared to Shape B.
- **Main risk: 2× accumulator fragment storage per workgroup.** On the 16×16×16 config with 128×128 output tiles, each workgroup holds two `128×128` fp32 accumulator tiles (distributed across subgroup lanes, but the aggregate register/shared-mem footprint doubles). This may reduce occupancy on drivers where the single-matmul kernel is already register-pressure-limited. Whether that costs perf is workload- and driver-dependent — measure before assuming.
- Additional uplift over Shape B (if occupancy holds): **5–10%**, primarily from halving A-load traffic and eliminating one dispatch's launch overhead. If occupancy drops one workgroup/SM, this shape can lose to Shape B.

#### Shape C-fatN — Single fused matmul over concatenated B = (gate_b ‖ up_b)

Same runtime shape as C-dualB, but B is a single tensor of shape `[2N, K]` and
the kernel treats `n_tile < N_gate` as gate rows and `n_tile ≥ N_gate` as up
rows, then merges the two halves in a second-pass epilogue over each output
row.

- **Requires an offline graph rewrite** to build `B_concat`, `scales_concat`,
  and (if present) `zero_points_concat` — otherwise a runtime concat cost
  negates the fusion gain. This mirrors the QKV-concat pattern already used in
  the WebGPU EP's attention fusion.
- Provides no correctness/perf advantage over C-dualB except cleaner uniform
  packing (single N-dimension). C-dualB is strictly preferred unless the graph
  rewrite already exists for other reasons.

### 11.3 Recommended sequencing

1. **Shape A first** — safest, ~guaranteed measurable win, prerequisite refactor for everything downstream.
2. **Shape B second** — highest ROI extension, no kernel-body changes, no dependencies.
3. **Shape C-dualB third**, conditional on profiling data. If Nsight shows A-load traffic dominating after Shape B, pursue it. If B-weight loads and cooperative-matrix throughput dominate, stop — Shape B is enough.
4. **Shape C-fatN**: only if offline weight concat is being introduced anyway for other reasons (e.g., a broader "merge sibling matmul weights" graph pass). Not a first-class target.

### 11.4 SkipLayerNorm: keep separate for prefill, keep fused for decode

The decode fused kernel (`MatMulNBitsMlpDecodeProgram`) fuses SkipSLN into the
matmul via the `has_skip_input` / `has_norm_input` / `has_skip_output` shader
flags. **This is right for decode, wrong for prefill.** Reasoning:

1. **Cost proportion.** At prefill M=2048 K=2048, SkipSLN costs ~24 MB of memory traffic and ~4·M·K flops. The two matmuls together are ~2·M·N·K·(bit-adjusted) flops — orders of magnitude more work. SLN is a small percentage of total MLP wall time. For decode M=1, matmul flops collapse to ~2·N·K, matmul is memory-bound on weights, and SLN's per-dispatch overhead is a real fraction of MLP wall time.
2. **Fusion architecturally awkward at prefill scale.** SkipSLN needs the full K-row to compute per-row mean/std before any matmul K-tile can consume normalized values. Options:
   - Two-pass A-load per matmul workgroup (reduce first, then matmul) — doubles A-side traffic, kills the fusion gain.
   - K-wide shared-memory reduction inside the matmul kernel — blows up shared-memory pressure and hurts occupancy on the exact 16×16×16 128×128 tile config that gives peak.
   Neither is a good trade for prefill.
3. **Decode's fusion pays because it doesn't have this problem.** With M=1, each workgroup owns one full row of A. Norm stats reduce trivially over the workgroup's own K-tile scan. There's no cross-workgroup dependency to work around.

**Practical outcome for the fused prefill MLP kernel**:
`MatMulNBitsMlp::ComputeInternal` already has the right dispatch cascade shape.
Under the `skip != nullptr` and `norm_scale != nullptr` branches, the existing
code runs `RunSkipLayerNormProgram` / `RunLayerNormProgram` first, then calls
`ApplyUnfusedMlp(normalized_a, ...)`. Adding the fused prefill kernel is one
new function `ApplyFusedPrefillMlp(normalized_a, ...)` and one new dispatch
condition (route to it when the matmul would use the subgroup-matrix path).
**No changes to the SkipLayerNorm plumbing are required or desirable.**

### 11.5 Where the silu*mul epilogue lives (Shape B split)

For Shape B, the epilogue is split across the two matmul dispatches:

- **Gate matmul's epilogue applies `silu(x) = x / (1 + exp(-x))`** to its accumulator before store. Cost: ~4 flops per output element, executed in-register on values already loaded for the store. Effectively free.
- **Up matmul's epilogue reads `silu_gate_out[m, n]` from VRAM and multiplies** it into the up accumulator before the final store. Cost: one fp16 load per output element at the same coordinates the up kernel is about to write to — read-modify-write within L2 residency, cache-friendly.

Rationale for this split (vs. putting both silu and mul in the up kernel):

- Keeps each kernel's epilogue minimal and independently testable — you can
  validate `silu(gate_matmul_out)` against a reference by multiplying by 1.
- Symmetric split of the intermediate materialization — one full-shape write
  (silu_gate_out) instead of two (gate_out and up_out) as in the current
  unfused epilogue.
- Register footprint per kernel unchanged.

### 11.6 Benchmark-harness Dawn toggle drift (fixed 2026-07-22)

**Symptom**: All prefill and decode benchmarks failed on the WGPU10 (Vulkan-
only) build with `Program SkipLayerNorm requires f16 but the device does not
support it`, despite the same source tree previously producing valid Vulkan
fp16 numbers on an older NVIDIA driver.

**Root cause**: Dawn's `PhysicalDeviceVk::ValidateFeatureSupportedWithTogglesImpl`
[unconditionally strips `Feature::ShaderF16` on NVIDIA vendors](../../WGPU10/Release/_deps/dawn-src/src/dawn/native/vulkan/PhysicalDeviceVk.cpp)
unless the `Toggle::VulkanEnableF16OnNvidia` (`"vulkan_enable_f16_on_nvidia"`)
is enabled at adapter/device creation. Cite: crbug.com/42251215
(CTS test failures). All four underlying VK physical-device bits
(`VK_KHR_shader_float16_int8`, `shaderFloat16`, `shaderInt16`,
`storageBuffer16BitAccess`) were `true` on the NVIDIA driver — the strip
happens purely inside Dawn's feature validator.

**Why prior runs worked**: the production `WebGpuContext::Initialize`
(`webgpu_context.cc`) has always included `"vulkan_enable_f16_on_nvidia"` in
`GetEnabledAdapterToggles()`. The microbenchmark harness in
`test/onnx/microbenchmark/webgpu_matmul_nbits_decode.cc` created its own Dawn
context without any toggles — so it was silently accumulating drift from the
EP's toggle list. This surfaced now, not earlier, because the harness's
Vulkan+NVIDIA+fp16 path only became a routine benchmark target recently.

**Fix**: chained a `wgpu::DawnTogglesDescriptor` to both `RequestAdapterOptions.nextInChain`
and `DeviceDescriptor.nextInChain` in `CreateSelectedWebGpuContext()`, mirroring
the full toggle set from `WebGpuContext::GetEnabledAdapterToggles()` and
`GetEnabledDeviceToggles()`. Enabled adapter toggles: `use_dxc`,
`allow_unsafe_apis`, `decompose_uniform_buffers`, and (under `DAWN_ENABLE_VULKAN`)
`use_vulkan_memory_model` + `vulkan_enable_f16_on_nvidia`. Enabled device
toggles: `skip_validation`, `disable_robustness`, `d3d_disable_ieee_strictness`.

**Non-obvious C API detail**: on the current Dawn header revision,
`WGPURequestAdapterOptions.nextInChain` is `WGPUChainedStruct*` (non-const), so
the cast from `wgpu::DawnTogglesDescriptor*` must be `reinterpret_cast<WGPUChainedStruct*>`,
not `reinterpret_cast<const WGPUChainedStruct*>`.

**Long-term maintenance note**: the harness toggle arrays are a copy-paste of
`WebGpuContext`'s toggle strings. Future toggle changes in the EP will not
automatically propagate to the benchmark. Two remediations, in order of preference:
1. Expose `WebGpuContext::GetEnabledAdapterToggles()` and `GetEnabledDeviceToggles()`
   as public static utility functions, then have the harness call them directly.
2. Move the toggle string arrays into a shared header
   (`onnxruntime/core/providers/webgpu/webgpu_dawn_toggles.h`) that both
   `webgpu_context.cc` and the benchmark harness include.

### 11.7 What changed from the "requires offline concat" claim

An earlier iteration of this design (and the pre-implementation gut-check
discussion) asserted that any "single fused matmul dispatch" version of the
kernel would require offline concatenation of gate_b and up_b. **That was
overstated.** Shape C-dualB (§ 11.2) is a legitimate single-dispatch fusion
that takes gate_b, gate_scales, up_b, up_scales as four separate bindings —
directly mirroring what `MatMulNBitsMlpDecodeProgram` already does for
decode. The only Shape C variant that actually needs offline concat is the
fat-N variant (C-fatN), which is strictly inferior to C-dualB.

The correct constraint statement: *Shape C-dualB doubles per-workgroup
accumulator fragment storage and therefore risks reducing occupancy on the
16×16×16 128×128 config.* That's an occupancy question, not a graph-rewrite
question.

### 11.8 Shape D — SLN fused into the matmul K-loop (considered and rejected for prefill)

For completeness, we considered a fifth fusion shape that pushes SkipLayerNorm
into the K-loop of the fused matmul kernel itself — i.e., no separate
SkipLayerNorm dispatch at all, even for prefill. **We are explicitly not
pursuing this for the v1 prefill MLP fusion.** Documenting it here so future
readers do not re-derive it and assume it's a good idea.

**What Shape D would look like**:

```
one kernel:
  // Pass 1 over K (per output row): reduce for norm stats
  for k_tile in K:
    A_row_tile = load a[m_tile, k_tile] + skip[m_tile, k_tile]
    accumulate sum, sum_sq for norm
  mean, rstd = finalize(sum, sum_sq)

  // Pass 2 over K (per output tile): matmul-accumulate with on-the-fly norm
  for k_tile in K:
    A_row_tile   = load a[m_tile, k_tile] + skip[m_tile, k_tile]
    A_norm_tile  = (A_row_tile - mean) * rstd * norm_scale[k_tile]
    subgroupMatMulAccumulate(gate_acc, A_norm_tile, gate_b[k_tile, n_tile])
    subgroupMatMulAccumulate(up_acc,   A_norm_tile, up_b  [k_tile, n_tile])
  y[m_tile, n_tile] = silu(gate_acc) * up_acc
```

- Total dispatches: **1** (down from 2 in Shape C-dualB).
- Removes the intermediate `normalized_a` and its `M·K·2` bytes of VRAM write+read (~16 MB per MLP layer at Qwen3-1.7B prefill M=2048).

**Why we are rejecting it for prefill v1**:

1. **Double A-load traffic** (fundamental, not tunable). Norm requires a
   full K-row reduction *before* the matmul K-loop can consume normalized
   values. That means every workgroup reads A twice — once for stats, once
   for matmul — doubling A-side prepack loads from `M_tile·K·2` bytes to
   `2·M_tile·K·2` bytes. The saved intermediate write is `M_tile·K·2`. **Net
   traffic is worse, not better.**
2. **Shared-memory pressure on the peak-throughput tile config.** Storing
   mean+rstd per row of the M_tile requires `2·M_tile` fp32 values in
   shared memory or across subgroup registers. Manageable at M_tile=128
   (~1 KB), but combined with the 2× accumulator fragment storage from
   Shape C-dualB (see §11.2), the workgroup's aggregate register + shared
   footprint grows enough to drop one workgroup per SM at occupancy on
   the 16×16×16 128×128 config. That single-workgroup occupancy loss
   typically costs 8–15% throughput on RTX 40/50 series — more than the
   ~5% traffic win it's trying to buy.
3. **Kernel complexity explodes.** The pass-1/pass-2 structure requires a
   full workgroup barrier between reduction finalization and the matmul
   K-loop. It also means the kernel's WGSL source doubles in size, with
   two separate K-loop bodies that must stay in sync as tuning parameters
   change. Testing surface roughly triples (norm-only mode, matmul-only
   mode, fused mode).
4. **Debugging pathology.** When numerics go wrong, you cannot bisect —
   there's no intermediate tensor to `printf`-inspect between the norm
   step and the matmul step. Every other shape in §11.2 leaves at least
   one materialized intermediate.
5. **The problem it solves is already small at prefill.** SkipLayerNorm at
   prefill scale is ~1–2% of MLP wall time (see §11.4). Even a *perfect*
   fusion of SLN into the matmul, ignoring the traffic and occupancy
   issues above, has a hard ceiling in the low single digits.

**Conclusion**: Shape D is not a preferred v1 target. It **may** become
interesting in the future *if* one of the following changes:

- The subgroup-matrix path adopts a tile geometry that leaves substantial
  spare register/shared-memory budget on NVIDIA (unlikely — the current
  budget is already tight).
- A future WGSL / cooperative-matrix extension exposes a "streaming A-tile
  transform" hook that would let the norm apply lazily on-the-fly without
  a separate pass-1 (would eliminate the double-A-load penalty).
- Prefill workload distributions shift toward much smaller M where norm's
  wall-time share rises. Even then, the decode fused kernel is the more
  natural place to solve it — see §11.4.

Cross-reference: this section supersedes any earlier suggestion (implicit or
explicit) that the fused prefill kernel might absorb SkipLayerNorm.
Non-goal §11-below already listed "SkipLayerNorm + skip input fusion" as a
v2 candidate; Shape D is the concrete architecture that was evaluated and
declined for v1.


## 11. Non-goals for v1 (explicit)

The following are deferred, in decreasing priority:

1. **SkipLayerNorm + skip input fusion** — v2 project. Requires solving the
   Metal 10-buffer limit *and* the norm reduction pattern within the fused
   kernel.
2. **DP4A fused MLP for decode (M=1)** — separate kernel entirely, targeting
   NVIDIA/AMD decode where SubgroupMatrix doesn't apply. Different
   architecture (uses `dot4I8Packed`).
3. **DownProj fusion** — the "full FFN megakernel." Estimated impact is small
   (~5–10% on prefill on top of this, ~0% on native decode, ~10% on browser
   decode). Only pursue if this v1 succeeds and there's remaining prefill
   headroom.
4. **8-bit and 2-bit quantization** — same structure, different dequant
   arithmetic. Straightforward extension in a follow-up PR.
5. **Non-SwiGLU activations** — GELU (Gemma-style gated MLP), ReLU². Add via
   `MlpActivationKind` enum extension in `matmul_nbits_mlp.h`.
6. **Support for non-power-of-2 K or N** — the divisibility gates
   (`K % 32 == 0`, `N % 64 == 0`) match the existing subgroup matrix kernel.
   Do not relax them in v1.

---

## 12. Measured performance

*This section will be filled in after Stage 3 lands. Placeholder:*

| Adapter | Config | Baseline (unfused w/ SubgroupMatrix, ms) | Fused (this design, ms) | Speedup |
|---|---|---|---|---|
| **NVIDIA RTX 5060 Ti (Vulkan)** | **16x16x16_128 (0)** | **TBD** | **TBD** | **TBD** |
| NVIDIA RTX 4090 (Vulkan) | 16x16x16_128 (0) | TBD | TBD | TBD |
| Intel Arc B580 (Vulkan) — PR 3 | 8x16x16 (1) | TBD | TBD | TBD |
| Apple M3 (Metal) — PR 3 | 8x8x8 (2) | TBD | TBD | TBD |
| Apple M4 (Metal) — PR 3 | 8x8x8 (2) | TBD | TBD | TBD |

Model shapes measured (Llama-2-7B FFN prefill):
- `M=512, K=4096, N=11008`
- `M=1024, K=4096, N=11008`
- `M=2048, K=4096, N=11008`

Note the RTX 5060 Ti row is the primary v1 target — this is the one that must
land green before PR 1 merges.

---

## 13. References

### Existing code to study

- **Fused MLP host code (already fuses SLN + Gate + Up + SiLU + Mul, missing subgroup matrix)**:
  [matmul_nbits_mlp.cc](../../onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_mlp.cc)
  and [matmul_nbits_mlp.h](../../onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_mlp.h).
  The decode fast path shader is
  [matmul_nbits_mlp.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_mlp.wgsl.template).

- **Single-MatMul subgroup matrix kernel (structural template)**:
  [subgroup_matrix_matmul_nbits.h](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h),
  [subgroup_matrix_matmul_nbits.cc](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.cc).

- **Three tile-shape WGSL templates to mirror**:
  - [subgroup_matrix_matmul_nbits_8x8x8.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_8x8x8.wgsl.template) (Apple)
  - [subgroup_matrix_matmul_nbits_8x16x16.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_8x16x16.wgsl.template) (NVIDIA WMMA)
  - [subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template) (Intel Xe DPAS)

- **Weight prepack shader (for gate_B and up_B)**:
  [subgroup_matrix_matmul_nbits_prepack.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_prepack.wgsl.template).

- **Config selector**:
  [subgroup_matrix_config.h](../../onnxruntime/core/providers/webgpu/math/subgroup_matrix_config.h)
  / [subgroup_matrix_config.cc](../../onnxruntime/core/providers/webgpu/math/subgroup_matrix_config.cc).
  `IsSubgroupMatrixConfigSupported` returns a `config_index` (0/1/2) matching the
  three templates above.

- **Dispatch cascade pattern (prefill vs decode routing)**:
  [flash_attention.cc](../../onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc) `ApplyFlashAttention`,
  specifically the `use_split_reduce = (sequence_length_ < 32)` decision. Same
  spirit as the M>=32 gate proposed here.

### External

- **WebGPU spec — `ChromiumExperimentalSubgroupMatrix` feature**:
  Dawn's implementation lives in `dawn/native/ChromiumExperimentalSubgroupMatrix`.
- **Vulkan cooperative matrix (`VK_KHR_cooperative_matrix`)**: the underlying
  Vulkan mechanism used by Dawn on NVIDIA/AMD.
- **Metal `simdgroup_matrix`**: Apple's underlying subgroup matrix hardware
  (M-series only, exposed by Dawn via the Chromium experimental feature).
- **Tri Dao's FlashDecoding paper** (for context on the Q-axis vs. K-axis
  parallelism distinction that motivates why this design targets prefill
  and defers decode to a separate DP4A design):
  https://crfm.stanford.edu/2023/10/12/flashdecoding.html

### Related PRs

- **PR #28109 (qjia7)**: vendor-agnostic MatMulNBits config refactor that
  established the current subgroup matrix dispatch structure.
- **PR #29557 (xiaofeihan1)**: deferred-dispatch parallel shader compilation.
  Orthogonal but relevant — reduces the first-Run compile cost of the multi-template
  approach here.

---

## 14. Open questions

For the implementer / another agent to resolve during Stage 1–3:

1. **PrePack cache scope**: Does ORT deduplicate prepacked tensors across
   ops that share the same underlying weight tensor? If yes, no
   extra work. If not, we may double-prepack in models where multiple
   `MatMulNBitsMlp` ops share weight init tensors. Check
   `PrePackedWeights` docs.

2. **`components_b` from the existing weight layout**: The existing MatMulNBits
   uses `components_b_with_u32` = 4 or 8 depending on `blob_size_in_words`.
   Confirm the prepack shader outputs weights in the layout the subgroup matrix
   template expects — read the prepack template carefully before Stage 2.

3. **Vulkan `VK_KHR_cooperative_matrix` availability across drivers**: on
   Windows, the WebGPU-via-Dawn Vulkan backend exposes subgroup matrix only if
   the underlying driver supports the extension. Check NVIDIA driver versions
   and AMD driver versions where the feature is reliable. Document
   minimum required driver version in Stage 5.

4. **Workgroup size choice**: The existing `SubgroupMatrixMatMulNBits` uses
   `work_group_size = subgroup_size * subgroups_per_wg`. For the fused MLP,
   the same formula applies, but each subgroup now does 2× the matmul work,
   so occupancy considerations may differ. Empirically tune in Stage 3.

5. **Dispatch shape when `M` is not a multiple of `TILE_M`**: Handle the
   ragged tail via masked stores (write only the valid rows) or a fallback
   dispatch for the tail rows. The existing `SubgroupMatrixMatMulNBits` uses
   `m_tiles_per_wg` uniform to let workgroups process multiple M-tiles when
   over-decomposed. Follow the same pattern.

---

## 15. Handoff checklist for the implementing agent

Before starting:

- [ ] Read this doc end-to-end.
- [ ] Read the three "existing code to study" files in §13 in full — especially
      [subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template](../../onnxruntime/contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template)
      (the NVIDIA-target template that PR 1 forks from).
- [ ] Verify your NVIDIA GPU adapter is picked up as `config_index = 0`:
      ```bash
      # Linux/Windows with NVIDIA discrete GPU (Ampere+) and recent Vulkan driver.
      # Build with WebGPU EP:
      ./build.sh --config Release --use_webgpu --build_wheel
      # Run the existing subgroup-matrix MatMulNBits tests:
      ./build/Release/onnxruntime_test_all --gtest_filter="*SubgroupMatrix*MatMulNBits*"
      ```
      If those tests skip because the feature is unreported, check driver
      version (§9.2 minimums) and Dawn feature exposure. If tests skip on
      Ampere+ hardware with a modern driver, coordinate with the design owner
      before proceeding.
- [ ] Set up a standalone MLP benchmark (isolated from full-model runs) so
      Stage 3 perf claims are verifiable. See
      [benchmark_matmul_nbits.py](../../onnxruntime/test/python/transformers/benchmark_matmul_nbits.py)
      as a starting pattern (or its MLP equivalent if present).
- [ ] Read the [ort-build](../../.agents/skills/ort-build/SKILL.md),
      [ort-test](../../.agents/skills/ort-test/SKILL.md), and
      [webgpu-local-testing](../../.agents/skills/webgpu-local-testing/SKILL.md)
      skills. Note: lavapipe **does not** expose subgroup matrix — it is not a
      substitute for real NVIDIA hardware for this design.

During implementation:

- [ ] Preserve parity with unfused path at every stage; never merge a stage
      that regresses correctness.
- [ ] Every PR includes:
      - The code change
      - Updated parity test coverage
      - Perf measurement showing the expected speedup on at least one target
        adapter
      - A benchmark log excerpt in the PR description
- [ ] Update this design doc's §12 "Measured performance" table as Stage 3, 5,
      6 complete.

---

## 16. Change log

| Date | Author | Change |
|---|---|---|
| 2026-07-21 | (initial draft) | Initial design |
